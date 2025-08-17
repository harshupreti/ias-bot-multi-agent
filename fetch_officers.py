# officer_list_fetcher.py

import requests
from bs4 import BeautifulSoup
from config import CIVIL_LIST_URL, HEADERS
from logger_config import setup_logger

logger = setup_logger()

class OfficerListFetcher:
    def __init__(self):
        self.session = requests.Session()

    def fetch_by_cadre(self, cadre_code: str) -> list[dict]:
        """
        Fetch officers listed under a specific cadre code (state) from DoPT website.
        """
        payload = {
            "ViewCadreCode": cadre_code,
            "btn_submit": "Submit"
        }

        try:
            response = self.session.post(CIVIL_LIST_URL, headers=HEADERS, data=payload)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"❌ Failed to fetch officer list for {cadre_code}: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.select("table#IASList tbody tr")

        if not rows:
            logger.warning(f"⚠️ No officer table found for cadre {cadre_code}. HTML may have changed.")
            return []

        officers = []
        for row in rows:
            card = row.select_one(".IAS_cardCont")
            if not card:
                continue

            def get_text_after_label(label):
                for p in card.find_all("p"):
                    if p.text.strip().startswith(label):
                        return p.text.strip().split(":", 1)[-1].strip()
                return None

            def get_cadre_domicile():
                tag = card.find("b", string="Cadre & Domicile:")
                if tag:
                    return tag.parent.get_text(strip=True).split(":", 1)[-1].replace("&", " & ")
                return None

            def get_posting():
                posting_b = card.find("b", string="Posting:-")
                if posting_b:
                    span = posting_b.find_next_sibling("span")
                    return span.get_text(" ", strip=True) if span else None
                return None

            name_tag = card.select_one("h2 a")
            name = name_tag.get_text(strip=True).replace("Name:", "") if name_tag else None
            link = name_tag["href"] if name_tag else None

            officer = {
                # Basic info (scraped in this layer)
                "name": name,
                "supremo_url": link,  # 🔄 renamed from supremo_url for consistency
                "identity_no": get_text_after_label("Identity No."),
                "allotment_year": get_text_after_label("Allotment Year"),
                "recruitment_source": get_text_after_label("Source of Recruitment"),
                "qualification": get_text_after_label("Qualification(Subject):"),
                "pay_scale": get_text_after_label("Pay Scale"),
                "remarks": get_text_after_label("Remarks"),
                "cadre_domicile": get_cadre_domicile(),
                "current_posting": get_posting(),
                "scraped_from_cadre": cadre_code,

                # Reserved full structure
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
                "education": [],
                "experience": [],
                "training": [],
                "awards": [],
                "deputation": {},
            }


            officers.append(officer)

        logger.info(f"✅ Fetched {len(officers)} officers for cadre {cadre_code}")
        return officers
