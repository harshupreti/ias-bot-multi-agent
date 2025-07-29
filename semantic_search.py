from langchain_core.tools import tool
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)
from qdrant_client import QdrantClient
from embedding import get_embedding_model
import os
from dotenv import load_dotenv
from typing import Union, List, Dict, Any
from rapidfuzz import process

load_dotenv(dotenv_path="QDRANT.env")

TITLES = {
    "Junior Scale" : 1,
    "Under Secretary": 2,
    "Deputy Secretary" : 3,
    "Director" : 4,
    "Joint Secretary": 5,
    "Additional Secretary": 6,
    "Secretary": 7,
}

from rapidfuzz import fuzz

def match_titles(user_title: str) -> List[str]:
    # Step 1: Find best matching known title
    best_title = None
    best_score = 0
    for title in TITLES:
        score = fuzz.ratio(user_title.lower(), title.lower())
        if score > best_score:
            best_score = score
            best_title = title

    if best_title is None:
        return []

    # Step 2: Get pay matrix of matched title
    matched_level = TITLES[best_title]

    # Step 3: Get all titles with same or one-below pay matrix
    allowed_titles = [
        title for title, level in TITLES.items()
        if level == matched_level or level == matched_level - 1
    ]

    return allowed_titles


# Add missing ALLOTMENT_YEAR_MAPPING, EMBEDDING_FUNC, COLLECTION_NAME, client

ALLOTMENT_YEAR_MAPPING = {
    "before": lambda year: Range(lte=float(year)),
    "from": lambda year: Range(gte=float(year)),
    "after": lambda year: Range(gt=float(year))
}
load_dotenv(dotenv_path="QDRANT.env")
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "ias_officers"
client = QdrantClient(
    url=QDRANT_CLOUD_URL,
    api_key=QDRANT_API_KEY
)
EMBEDDING_FUNC = get_embedding_model()

@tool
def semantic_search(
    query: Union[str, List[str]],
    filters: dict,
    top_k: int = None,
) -> list:
    """
    Perform semantic search over IAS officer data.
    - query: Single or list of semantic queries.
    - top_k: Number of top results to return (default is 5).
    - current_title: The current title to filter the search.
    Returns: List of deduplicated officer payloads with _vector_score field.
    """

    if not filters or "current_title" not in filters:
        raise ValueError("current_title is required in filters.")

    if isinstance(query, str):
        queries = [query]
        k = top_k or 5
    elif isinstance(query, list):
        queries = query
        k = top_k or 1
    else:
        raise ValueError("Invalid query type. Must be string or list of strings.")

    op = filters.pop("allotment_year_operation", None)
    if op and "allotment_year" in filters and op in ALLOTMENT_YEAR_MAPPING and op is not None:
        filters["allotment_year"] = {"operator": op, "value": filters["allotment_year"]}

    allowed = {"cadre", "gender", "current_title", "allotment_year"}
    clean_filters = {k: v for k, v in filters.items() if k in allowed}

    def build_filter(fdict: Dict[str, Any]) -> Filter:
        must_conditions = []
        should_conditions = []

        for key, value in fdict.items():
            if key == "allotment_year" and value is not None:
                if isinstance(value, dict):
                    op = value.get("operator")
                    val = value.get("value")
                    if op in ALLOTMENT_YEAR_MAPPING and val is not None:
                        must_conditions.append(
                            FieldCondition(key=key, range=ALLOTMENT_YEAR_MAPPING[op](float(val)))
                        )
                else:
                    must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

            elif key == "gender" and value is not None:
                must_conditions.append(FieldCondition(key="gender", match=MatchValue(value=value)))

            elif key == "cadre" and value is not None:
                must_conditions.append(FieldCondition(key="cadre", match=MatchValue(value=value)))

            elif key == "current_title" and value:
                # Enhanced fuzzy title matching: match + titles with same or one-below level
                allowed_titles = match_titles(value)
                if allowed_titles:
                    should_conditions = [
                        FieldCondition(key="current_title", match=MatchValue(value=title))
                        for title in allowed_titles
                    ]

        if not must_conditions and not should_conditions:
            return None

        return Filter(
            must=must_conditions,
            should=should_conditions if should_conditions else None
        )


    qdrant_filter = build_filter(clean_filters)
    print(f"Qdrant filter: {qdrant_filter}")

    seen = set()
    combined_results = []

    for q in queries:
        query_vector = EMBEDDING_FUNC.embed_query(q)

        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=min(k, 5),
            with_payload=True,
            with_vectors=False,
            query_filter=qdrant_filter
        )

        for hit in hits:
            officer = hit.payload
            officer["_vector_score"] = hit.score
            identity = (
                officer.get("officer_name", ""),
                officer.get("cadre", ""),
                officer.get("allotment_year", ""),
            )
            if identity not in seen:
                seen.add(identity)
                combined_results.append(officer)

    return combined_results