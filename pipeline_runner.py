import asyncio
import httpx
import random
import time
from tqdm.asyncio import tqdm_asyncio

from fetch_officers import OfficerListFetcher
from fetch_officer_details import OfficerDetailFetcherAsync
from embedding_docs import FullPDFEmbedder
from qdrant_client import QdrantClient
from config import QDRANT_COLLECTION_NAME
from logger_config import setup_logger

logger = setup_logger()


def get_officers_to_update(fetched_officers: list[dict], qdrant_client: QdrantClient) -> list[dict]:
    """
    Compare freshly fetched officers with what's in Qdrant.
    Return only new or changed officers.
    """

    # 1. Scroll through all existing points in Qdrant
    existing_map = {}
    offset = None

    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=None,
            with_payload=True,
            with_vectors=False,
            limit=1000,  # safe batch size
            offset=offset,
        )
        for p in points:
            supremo_url = p.payload.get("supremo_url")
            current_posting = p.payload.get("current_posting")
            if supremo_url:
                existing_map[supremo_url] = current_posting

        if next_offset is None:  # ✅ stop condition
            break
        offset = next_offset

    logger.info(f"📋 Loaded {len(existing_map)} existing officers from Qdrant")

    # 2. Compare with freshly fetched
    to_update = []
    for officer in fetched_officers:
        supremo_url = officer.get("supremo_url")
        current_posting = officer.get("current_posting")

        if supremo_url not in existing_map:
            to_update.append(officer)  # new officer
        elif existing_map[supremo_url] != current_posting:
            to_update.append(officer)  # posting changed
        # else: unchanged → skip

    return to_update


class AsyncPipelineRunner:
    def __init__(self, qdrant_client: QdrantClient, max_retries: int = 3, concurrency_limit: int = 20):
        self.list_fetcher = OfficerListFetcher()
        self.detail_fetcher = OfficerDetailFetcherAsync()
        self.embedder = FullPDFEmbedder()
        self.qdrant = qdrant_client
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(concurrency_limit)  # ✅ concurrency guard

    async def process_officer(self, officer: dict, cadre_code: str):
        async with self.semaphore:  # ✅ limit concurrency
            retries = 0
            while retries < self.max_retries:
                try:
                    officer["scraped_from_cadre"] = cadre_code

                    raw_year = officer.get("allotment_year")
                    try:
                        allotment_year = int(raw_year) if raw_year else None
                    except ValueError:
                        logger.warning(f"⚠️ Invalid allotment year for {officer['name']}: {raw_year}")
                        allotment_year = None

                    enriched = {
                        "name": officer.get("name", ""),
                        "supremo_url": officer.get("supremo_url", ""),
                        "identity_no": officer.get("identity_no", ""),
                        "allotment_year": allotment_year,
                        "recruitment_source": officer.get("recruitment_source", ""),
                        "qualification": officer.get("qualification", ""),
                        "pay_scale": officer.get("pay_scale", ""),
                        "remarks": officer.get("remarks", ""),
                        "cadre_domicile": officer.get("cadre_domicile", ""),
                        "current_posting": officer.get("current_posting", ""),
                        "scraped_from_cadre": cadre_code,
                        "personal": {
                            "name": None,
                            "identity_no": None,
                            "cadre": None,
                            "dob": None,
                            "gender": None,
                            "allotment_year": None,
                            "domicile": None,
                            "mother_tongue": None,
                            "languages_known": None,
                            "retirement_reason": None
                        },
                        "education": [], "experience": [],
                        "training": [], "awards": [],
                        "deputation": {}
                    }

                    details = await self.detail_fetcher.fetch_details(officer)
                    enriched["personal"] = details.get("personal", {})
                    enriched["education"] = details.get("education", [])
                    enriched["experience"] = details.get("experience", [])
                    enriched["training"] = details.get("training", {})
                    enriched["awards"] = details.get("awards", [])
                    enriched["deputation"] = details.get("deputation", {})

                    payload = self.embedder.build_vector_payload(enriched)

                    logger.info(f"✅ Prepared vector for {officer['name']} (ID: {payload['id']})")
                    return payload

                except Exception as e:
                    retries += 1
                    logger.warning(f"⚠️ Retry {retries} for officer {officer.get('name')} due to error: {e}")
                    await asyncio.sleep(0.5 * retries + random.random())

            logger.error(f"❌ Skipped officer {officer.get('name')} after {self.max_retries} retries")
            return None

    async def run_for_cadre(self, cadre_code: str):
        logger.info(f"🚀 Starting async pipeline for cadre: {cadre_code}")

        officer_list = self.list_fetcher.fetch_by_cadre(cadre_code)
        logger.info(f"✅ Fetched {len(officer_list)} officers for cadre {cadre_code}")
        logger.debug(f"[DEBUG] First officer from list: {officer_list[0]}")

        officer_list = get_officers_to_update(officer_list, self.qdrant)
        logger.info(f"🔍 {len(officer_list)} officers need processing (new or changed)")

        if not officer_list:
            logger.info("✅ Nothing new to process.")
            return

        tasks = [self.process_officer(officer, cadre_code) for officer in officer_list]
        results = await tqdm_asyncio.gather(*tasks, desc="Processing officers")

        payloads = [res for res in results if res]
        logger.info(f"📦 {len(payloads)} vectors ready for upsert. Uploading in batches...")

        batch_size = 100
        for i in range(0, len(payloads), batch_size):
            batch = payloads[i:i + batch_size]
            try:
                self.qdrant.upsert(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points=batch
                )
                logger.info(f"✅ Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")
            except Exception as e:
                logger.error(f"❌ Failed to upsert batch {i // batch_size + 1}: {e}")

        logger.info(f"🎉 Finished processing and upserting for cadre {cadre_code}")
