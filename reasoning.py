# tools/reasoning_tool.py

from langchain_core.tools import tool
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from typing import List, Optional, Union

load_dotenv("GITHUB_TOKEN.env")

client = OpenAI(api_key=os.getenv("GITHUB_TOKEN"))

# Officer schema
class Officer(BaseModel):
    officer_name: str
    cadre: Optional[str] = None
    allotment_year: Optional[Union[str, int]] = None
    education: Optional[str] = None
    postings: Optional[str] = None
    training_details: Optional[str] = None
    awards_publications: Optional[str] = None
    source_file: Optional[str] = None
    source: Optional[str] = None
    blob_path: Optional[str] = None

    @field_validator("allotment_year")
    def convert_allotment_year_to_str(cls, v):
        return str(v) if v is not None else v


@tool
def reasoning_tool(query: str, officers: List[Officer]) -> str:
    """
    Use this tool to analyze the query and retrieved officer list before generating a final answer.

    The LLM will:
    - Evaluate whether enough relevant information has been found.
    - Reject any officers that do not match the query requirements.
    - If no suitable candidates exist, trigger a web search or re-query instruction.
    - If suitable candidates are found, return their full original information block (unmodified).
    """

    system_msg = """
You are a critical reasoning agent that evaluates IAS officer profiles for a given user query.

Your job is to:
1. Carefully analyze the list of OFFICERS (in JSON format).
2. For each officer, decide whether they are a good match for the user's query (based on experience, role, domain, seniority, etc.).
3. Strictly return officers that are just below the target role or ready for promotion.
4. ONLY keep the officers who are relevant.
5. Return your result in this exact format:

- If at least one relevant match is found:
    ✅ Proceed with final answer.
    Relevant officers:
    <full original JSON objects for relevant officers>

- If NONE are relevant:
    ❌ Insufficient data.
    Recommend: <web search | retry | no match found in DB | etc.>

Rules:
- Do not summarize or reformat officer data.
- Only include exact JSON blocks of matching officers.
- Be strict. If unsure, exclude the officer and suggest fallback.

- After identifying relevant officers, write a short, clean final answer in Markdown format:
  - Briefly list up to 3 matching officers with relevant details (name, cadre, year, key role)
  - If one officer clearly stands out, highlight them
  - Their key experience/education
  - A source link to their full profile

Format the summary like a human-written answer, not JSON. Keep it concise.
"""

    officers_json = json.dumps([o.dict() for o in officers], indent=2)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg.strip()},
            {"role": "user", "content": f"QUERY: {query}\n\nOFFICERS:\n{officers_json}"}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()
