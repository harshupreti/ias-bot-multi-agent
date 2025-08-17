# config.py
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from qdrant_client.http.models import Distance, VectorParams


load_dotenv(dotenv_path="QDRANT.env")

# === General Settings ===
MAX_WORKERS = 4
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds

# === Directory Structure ===
BASE_DIR = Path(__file__).parent.resolve()
PDF_DIR = BASE_DIR / "pdfs"
LOG_DIR = BASE_DIR / "logs"

# === BASE URL ===
CIVIL_LIST_URL = "https://iascivillist.dopt.gov.in/Home/ViewList"

# === Invalid Filename Characters ===
INVALID_FILENAME_CHARS = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '(', ')', '.', ',']
REPLACE_CHAR = "_"

# === Model Config ===
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768

# === Qdrant Cloud Settings ===
QDRANT_URL = os.getenv("QDRANT_CLOUD_URL")  # replace with actual URL
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")              # keep secure
QDRANT_COLLECTION_NAME = "ias_officers"

STATE_CODES = {
    "UT": "AGMUT",
    "AP": "Andhra Pradesh",
    "AM": "Assam Meghalaya",
    "BH": "Bihar",
    "CG": "Chhattisgarh",
    "GJ": "Gujarat",
    "HY": "Haryana",
    "HP": "Himachal Pradesh",
    "JH": "Jharkhand",
    "KN": "Karnataka",
    "KL": "Kerala",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TG": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UD": "Uttarakhand",
    "WB": "West Bengal"
}

# === Headers ===
USER_AGENT = "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Mobile Safari/537.36"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Referer": "https://iascivillist.dopt.gov.in/",
    "Origin": "https://iascivillist.dopt.gov.in",
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

# === Config Validation ===
def validate_config():
    from logging import getLogger
    logger = getLogger("IASPipeline")

    # Create necessary directories
    for dir_path in [PDF_DIR, LOG_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Directory ready: {dir_path}")

    # Check embedding model availability
    try:
        SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        logger.info(f"✅ Embedding model loaded: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        logger.error(f"❌ Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        raise

    # Check Qdrant Cloud connectivity
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        collections = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION_NAME not in collections:
            logger.warning(f"⚠️ Collection '{QDRANT_COLLECTION_NAME}' not found. Creating...")
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✅ Created Qdrant collection: {QDRANT_COLLECTION_NAME}")
        else:
            logger.info(f"✅ Connected to Qdrant. Collection found: {QDRANT_COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"❌ Qdrant Cloud connection error: {e}")
        raise

