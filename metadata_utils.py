# metadata_utils.py

import base64
from urllib.parse import urlparse, parse_qs

class MetadataUtils:
    

    def generate_vector_id(self, supremo_url: str) -> int:
        """
        Extracts internal officer ID from supremo_url and returns it as an integer.
        Example: https://supremo.nic.in/ERSheetHtml.aspx?OffIDErhtml=MTQ4OTA=&PageId=
        → vector_id: 14890 (int)
        """
        try:
            parsed = urlparse(supremo_url)
            encoded_id = parse_qs(parsed.query).get("OffIDErhtml", [None])[0]

            if not encoded_id:
                raise ValueError("Missing OffIDErhtml in supremo_url")

            internal_id_str = base64.b64decode(encoded_id).decode().strip()
            internal_id = int(internal_id_str)
            return internal_id
        except Exception as e:
            print(f"[WARN] Could not decode internal ID from supremo_url: {e}")
            return -1  # fallback invalid ID
