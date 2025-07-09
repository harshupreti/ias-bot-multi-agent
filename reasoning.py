# tools/reasoning_tool.py

from langchain_core.tools import tool
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional

load_dotenv("GITHUB_TOKEN.env")

client = OpenAI(
    api_key=os.getenv("GITHUB_TOKEN")
)

# Officer schema
class Officer(BaseModel):
    officer_name: str
    cadre: Optional[str]
    allotment_year: Optional[str]
    education: Optional[str]
    postings: Optional[str]
    training_details: Optional[str]
    awards_publications: Optional[str]
    source_file: Optional[str]
    source: Optional[str]
    blob_path: Optional[str]

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
3. ONLY keep the officers who are relevant.
4. Return your result in this exact format:

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

    # Convert list of Pydantic objects to raw JSON
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
