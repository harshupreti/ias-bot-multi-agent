# tools/reasoning_tool.py

from langchain_core.tools import tool
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from typing import List, Optional

load_dotenv("GITHUB_TOKEN.env")

client = OpenAI(api_key=os.getenv("GITHUB_TOKEN"))

class Officer(BaseModel):
    name: str
    identity_no: str
    cadre: Optional[str] = "Unknown"
    scraped_from_cadre: Optional[str] = "Unknown"
    allotment_year: Optional[int] = None
    recruitment_source: Optional[str] = "Unknown"
    qualification: Optional[str] = "Unknown"
    current_posting: Optional[str] = "Unknown"
    current_title: Optional[str] = "Unknown"
    supremo_url: Optional[str] = None
    gender: Optional[str] = "Unknown"
    dob: Optional[str] = "Unknown"
    has_training: Optional[bool] = False
    has_awards: Optional[bool] = False
    education_count: Optional[int] = 0
    experience_count: Optional[int] = 0
    pdf_path: Optional[str] = None

    @field_validator("allotment_year")
    def convert_year(cls, v):
        return int(v) if v is not None else None

@tool
def reasoning_tool(query: str, officers: List[Officer], filters: Optional[dict] = None) -> str:
    """
    Final reasoning tool that evaluates IAS officer profiles for a specific query.
    Removes only completely irrelevant officers and ranks the rest with reasoning.
    """

    system_msg = """
You are a reasoning agent tasked with evaluating IAS officers for a given query.

Your job is to:
1. Examine the list of OFFICERS (in JSON format).
2. Remove officers who are completely unrelated to the query — i.e., match NONE of the criteria (like experience, title, education, etc).
3. For the remaining officers, rank them from most to least suitable.
4. Provide a short reasoning for why each officer was ranked that way.

Return results in this format:

write a human-readable summary in markdown:
- Introduce the top candidates by name, cadre, allotment year, and current title/posting.
- For each, explain briefly (3–4 lines) why they’re a strong fit (experience, education, seniority, etc.).
- Highlight any unique or standout traits.
- Link to their full profile if available.

If multiple officers, display atleast 3–5 top candidates and why they are suitable.
Avoid being too strict — include anyone who seems even somewhat relevant.
Keep the explanation natural, readable, and helpful for a senior official.
"""

    officers_json = json.dumps([o.dict() for o in officers], indent=2)
    filters_block = f"\nFILTERS:\n{json.dumps(filters, indent=2)}" if filters else ""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg.strip()},
            {
                "role": "user",
                "content": f"QUERY: {query}{filters_block}\n\nOFFICERS:\n{officers_json}"
            }
        ],
        temperature=0.5
    )

    return response.choices[0].message.content.strip()
