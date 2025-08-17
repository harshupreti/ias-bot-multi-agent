# fetch_officer_details_async.py

import httpx
from bs4 import BeautifulSoup
from logger_config import setup_logger

logger = setup_logger()

class OfficerDetailFetcherAsync:
    def __init__(self):
        self.session: httpx.AsyncClient | None = None

    def default_headers(self) -> dict:
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Referer": "https://iascivillist.dopt.gov.in/",
        }

    async def init_session(self):
        if self.session is None:
            self.session = httpx.AsyncClient(timeout=20, headers=self.default_headers(), http2=False)
            await self.session.get("https://supremo.nic.in/")  # Warm-up

    async def fetch_details(self, officer: dict) -> dict:
        await self.init_session()

        supremo_url = officer.get("supremo_url")
        if not supremo_url:
            raise ValueError(f"No supremo_url for officer: {officer.get('name')}")

        logger.info(f"🔍 Fetching supremo info for {officer['name']}")

        try:
            response = await self.session.get(supremo_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            tables = soup.find_all("table", class_="tbl_border")

            info_rows = tables[0].find_all("tr")
            personal_data = {}

            def extract(label):
                for tr in info_rows:
                    tds = tr.find_all("td")
                    if len(tds) == 2 and label in tds[0].text:
                        return tds[1].get_text(separator=" ", strip=True)
                return None

            # ⛏️ Parse cadre and allotment_year from service_cadre_year
            service_cadre_year = extract("Service/ Cadre/ Allotment Year")
            cadre = None
            allotment_year = None
            if service_cadre_year:
                parts = [p.strip() for p in service_cadre_year.split("/")]
                if len(parts) == 3:
                    cadre = parts[1]
                    allotment_year = parts[2]

            personal_data.update({
                "name": extract("Name"),
                "identity_no": extract("Identity No."),
                "cadre": cadre,
                "allotment_year": allotment_year,
                "recruitment_source": extract("Source of Recruitment"),
                "dob": extract("Date of Birth"),
                "gender": extract("Gender"),
                "domicile": extract("Place of Domicile"),
                "mother_tongue": extract("Mother Tongue"),
                "languages_known": extract("Languages Known"),
                "retirement_reason": extract("Retirement Reason")
            })
            print(f"Personal Data: {personal_data}")

            # Deputation
            deputation_data = {}
            if len(tables) >= 2:
                for row in tables[1].find_all("tr"):
                    tds = row.find_all("td")
                    if len(tds) == 2:
                        key = tds[0].text.strip().replace("?", "")
                        val = tds[1].get_text(separator=" ", strip=True)
                        deputation_data[key] = val

            # Remaining tables
            rounded_tables = soup.find_all("table", id="rounded-cornerA")
            education, experience, mid_training, awards = [], [], [], []
            # Education
            if len(rounded_tables) >= 1:
                for row in rounded_tables[0].find_all("tr")[2:]: # Skip header rows
                    tds = row.find_all("td")
                    if len(tds) == 4:
                        education.append({
                            "qualification": tds[1].get_text(separator=" ", strip=True),
                            "subject": tds[2].get_text(separator=" ", strip=True),
                            "division": tds[3].get_text(separator=" ", strip=True)
                        })
            # Experience
            if len(rounded_tables) >= 2:
                for row in rounded_tables[1].find_all("tr")[2:]:
                    tds = row.find_all("td")
                    if len(tds) > 1:
                        experience.append({
                            "designation": [
                                part.strip() for part in tds[1].decode_contents().split("<br/>") if part.strip()
                            ] or [" "],
                            "ministry": tds[2].get_text(separator=" ", strip=True) or "",
                            "organization": tds[3].get_text(separator=" ", strip=True) or "",
                            "experience_area": tds[4].get_text(separator=" ", strip=True) or "",
                            "period": tds[5].get_text(separator=" ", strip=True) or ""
                        })
            # Mid career training
            if len(rounded_tables) >= 3:
                for row in rounded_tables[2].find_all("tr")[2:]:
                    tds = row.find_all("td")
                    if len(tds) > 1:
                        mid_training.append({
                            "year": tds[1].get_text(separator=" ", strip=True) or "",
                            "training_name": tds[2].get_text(separator=" ", strip=True) or "",
                            "date_from": tds[3].get_text(separator=" ", strip=True) or "",
                            "date_to": tds[4].get_text(separator=" ", strip=True) or ""
                        })
            # In-service training
            in_service_training = []
            if len(rounded_tables) >= 4:
                for row in rounded_tables[3].find_all("tr")[2:]:
                    tds = row.find_all("td")
                    if len(tds) > 1:
                        in_service_training.append({
                            "year": tds[1].get_text(separator=" ", strip=True) or "",
                            "training_name": tds[2].get_text(separator=" ", strip=True) or "",
                            "institute": tds[3].get_text(separator=" ", strip=True) or "",
                            "city": tds[4].get_text(separator=" ", strip=True) or "",
                            "duration": tds[5].get_text(separator=" ", strip=True) or ""
                        })
            # Domestic training
            domestic_training = []
            if len(rounded_tables) >= 5:
                for row in rounded_tables[4].find_all("tr")[2:]:
                    tds = row.find_all("td")
                    if len(tds) > 1:
                        domestic_training.append({
                            "year": tds[1].get_text(separator=" ", strip=True) or "",
                            "training_name": tds[2].get_text(separator=" ", strip=True) or "",
                            "subject": tds[3].get_text(separator=" ", strip=True) or "",
                            "duration": tds[4].get_text(separator=" ", strip=True) or ""
                        })
            
            # Foreign training
            foreign_training = []
            if len(rounded_tables) >= 6:
                for row in rounded_tables[5].find_all("tr")[2:]:
                    tds = row.find_all("td")
                    if len(tds) > 1:
                        foreign_training.append({
                            "year": tds[1].get_text(separator=" ", strip=True) or "",
                            "training_name": tds[2].get_text(separator=" ", strip=True) or "",
                            "subject": tds[3].get_text(separator=" ", strip=True) or "",
                            "duration": tds[4].get_text(separator=" ", strip=True) or "",
                            "country": tds[5].get_text(separator=" ", strip=True) or ""
                        })

            # Awards
            if len(rounded_tables) >= 7:
                for row in rounded_tables[6].find_all("tr")[2:]:
                    tds = row.find_all("td")
                    if len(tds) > 1:
                        awards.append({
                            "type": tds[1].get_text(separator=" ", strip=True) or "",
                            "area": tds[2].get_text(separator=" ", strip=True) or "",
                            "year": tds[3].get_text(separator=" ", strip=True) or "",
                            "award_name_book_title": tds[4].get_text(separator=" ", strip=True) or "",
                            "award_by_publisher_name": tds[5].get_text(separator=" ", strip=True) or "",
                            "subject": tds[6].get_text(separator=" ", strip=True) or "",
                            "level": tds[7].get_text(separator=" ", strip=True) or ""
                        })
            # Combine all training sections
            training = {
                "mid_career": mid_training,
                "in_service": in_service_training,
                "domestic": domestic_training,
                "foreign": foreign_training
            }
            # ✅ Return clean structured object — no duplication
            return {
                "personal": personal_data,
                "education": education,
                "experience": experience,
                "training": training,
                "awards": awards,
                "deputation": deputation_data
            }

        except Exception as e:
            logger.error(f"❌ Failed to fetch officer {officer.get('name')}: {e}")
            raise