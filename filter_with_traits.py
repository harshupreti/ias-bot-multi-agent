# tools/filter_tool.py

import os
import random
from itertools import combinations
from dotenv import load_dotenv
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range

# --- Load environment ---
load_dotenv(dotenv_path="QDRANT.env")
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- Qdrant client ---
client = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY)

COLLECTION_NAME = "ias_officers"
MAX_RESULTS_PER_SCROLL = 50
RETURN_LIMIT = 3

# --- Allotment year operation mapping ---
ALLOTMENT_YEAR_MAPPING = {
    "before": lambda year: Range(lte=float(year)),
    "from": lambda year: Range(gte=float(year)),
    "after": lambda year: Range(gt=float(year))
}

@tool
def filter_officers(filters: dict) -> list:
    """
    Filters IAS officers using input metadata filters and returns 3 officers randomly.

    Input Format:
    {
        "cadre": "Gujarat",
        "gender": "Female",
        "allotment_year_operation": "after",  # or "before", "from"
        "allotment_year": 2005
    }

    Applies combinatorial logic to attempt partial matches if full filters yield no results.
    """

    available_keys = {
        "cadre": lambda val: FieldCondition(key="cadre", match=MatchValue(value=val)),
        "gender": lambda val: FieldCondition(key="gender", match=MatchValue(value=val)),
        "allotment_year": lambda val: None  # Handled separately
    }

    # Build traits list
    trait_items = []
    if "cadre" in filters:
        trait_items.append(("cadre", filters["cadre"]))
    if "gender" in filters:
        trait_items.append(("gender", filters["gender"]))
    if "allotment_year" in filters and "allotment_year_operation" in filters:
        op = filters["allotment_year_operation"]
        val = filters["allotment_year"]
        if op in ALLOTMENT_YEAR_MAPPING:
            trait_items.append(("allotment_year", (op, val)))

    # Try all combinations of available traits (combinatorial fallback)
    for r in range(len(trait_items), 0, -1):
        for combo in combinations(trait_items, r):
            conditions = []
            for key, value in combo:
                try:
                    if key == "allotment_year":
                        op, year = value
                        condition = FieldCondition(
                            key="allotment_year",
                            range=ALLOTMENT_YEAR_MAPPING[op](year)
                        )
                    else:
                        condition = available_keys[key](value)

                    if condition:
                        conditions.append(condition)
                except Exception as e:
                    print(f"[WARN] Skipped {key}: {e}")
                    continue

            if not conditions:
                continue

            query_filter = Filter(must=conditions)
            print("[DEBUG] Trying filter:", query_filter)

            try:
                results, _ = client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=query_filter,
                    limit=MAX_RESULTS_PER_SCROLL,
                    with_payload=True,
                    with_vectors=False
                )

                all_payloads = [hit.payload for hit in results if hasattr(hit, "payload")]

                if all_payloads:
                    random.shuffle(all_payloads)
                    return all_payloads[:RETURN_LIMIT]

            except Exception as e:
                return [{"error": str(e)}]

    return []
