# full_pdf_embedder.py

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME
from logger_config import setup_logger
from metadata_utils import MetadataUtils
import json

logger = setup_logger()

class FullPDFEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            trust_remote_code=True
            )
        self.utils = MetadataUtils()

    def format_officer_as_text(self, officer: dict) -> str:
        """
        Flattens structured officer data (education, experience, training, awards, etc.)
        into a readable text block.
        Handles multiple training types grouped under their respective keys.
        """
        import json

        parts = []

        # Personal Info
        personal = officer.get("personal") or {}
        for k, v in personal.items():
            if v:
                parts.append(f"{k.title().replace('_', ' ')}: {v}")

        # Education
        education_list = officer.get("education") or []
        for edu in education_list:
            q = edu.get("qualification", "")
            s = edu.get("subject", "")
            d = edu.get("division", "")
            if q or s or d:
                parts.append(f"Education: {q} in {s} ({d})")

        # Experience
        experience_list = officer.get("experience") or []
        for exp in experience_list:
            desg = exp.get("designation", "")
            org = exp.get("organization", "")
            min_ = exp.get("ministry", "")
            area = exp.get("experience_area", "")
            period = exp.get("period", "")
            parts.append(f"Worked as {desg} in {org} ({min_}, {area}) during {period}")

        # Training (dict of sections)
        training_sections = officer.get("training") or {}
        for section_name, rows in training_sections.items():
            if not rows:
                continue
            parts.append(f"{section_name.replace('_', ' ').title()} Training:")
            for row in rows:
                row_text = "; ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in row.items() if v)
                parts.append(f"  - {row_text}")

        # Awards (extended structure)
        awards_list = officer.get("awards") or []
        for aw in awards_list:
            line = "; ".join(
                f"{k.replace('_', ' ').title()}: {v}"
                for k, v in aw.items() if v
            )
            parts.append(f"Award: {line}")

        # Deputation
        deputation = officer.get("deputation") or {}
        if deputation:
            parts.append("Deputation Details:")
            parts.append(json.dumps(deputation, indent=2))

        return "\n".join(parts)


    def extract_current_title(self, experience: list[dict]) -> str | None:
        """
        Extracts the most recent officer title (e.g., 'Secretary') from the designation list.
        The last part after <br/> is usually the title.
        """
        if not experience:
            return None

        designation = experience[0].get("designation", [])
        if isinstance(designation, str):
            designation = [designation]

        # Return the last meaningful part
        for part in reversed(designation):
            clean = part.strip()
            if clean:
                return clean

        return None



    def embed_text(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def build_vector_payload(self, officer: dict) -> dict:
        personal = officer.get("personal", {})
        identity_no = personal.get("identity_no") or officer.get("identity_no")
        cadre = personal.get("cadre") or officer.get("scraped_from_cadre")
        year = personal.get("allotment_year")

        # Validate required fields
        missing_keys = []
        if not identity_no:
            missing_keys.append("identity_no")
        if not cadre:
            missing_keys.append("scraped_from_cadre")
        if year is None:
            missing_keys.append("allotment_year")

        if missing_keys:
            raise ValueError(f"Missing {missing_keys} for officer: {officer.get('name')}")

        # Generate ID, current title, embedding text
        vector_id = self.utils.generate_vector_id(officer["supremo_url"])
        current_title = self.extract_current_title(officer.get("experience", []))
        full_text = self.format_officer_as_text(officer)
        vector = self.embed_text(full_text)

        payload = {
            "id": vector_id,
            "vector": vector,
            "payload": {
                "name": personal.get("name") or officer.get("name", ""),
                "identity_no": identity_no,
                "cadre": cadre,
                "scraped_from_cadre": officer.get("scraped_from_cadre", ""),
                "allotment_year": year,
                "recruitment_source": officer.get("recruitment_source", ""),
                "qualification": officer.get("qualification", ""),
                "current_posting": officer.get("current_posting", ""),
                "current_title": current_title,
                "supremo_url": officer.get("supremo_url", ""),
                "gender": personal.get("gender", ""),
                "dob": personal.get("dob", ""),
                "has_training": bool(officer.get("training")),
                "has_awards": bool(officer.get("awards")),
                "education_count": len(officer.get("education") or []),
                "experience_count": len(officer.get("experience") or []),
                "pdf_path": None,
                "text": full_text
            }
        }

        logger.info(f"✅ Embedded vector for {payload['payload']['name']} ({vector_id})")
        return payload


