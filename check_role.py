from langchain_core.tools import tool
import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load credentials
load_dotenv(dotenv_path="GITHUB_TOKEN.env")
token = os.environ["GITHUB_TOKEN"]

# Setup OpenAI client
model = "gpt-4o"
client = OpenAI(api_key=token)

TITLES = [
    "Junior Scale",
    "Under Secretary",
    "Deputy Secretary",
    "Director",
    "Joint Secretary",
    "Additional Secretary",
    "Secretary"
]

def safe_json_parse(text, fallback="{}"):
    """Parse JSON robustly from response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        return json.loads(fallback)

@tool
def check_role_intent(query: str) -> dict:
    """
    called when the user is requesting officer recommendations for a specific IAS role.

    Returns:
      - `queries`: semantic search prompts for vector similarity.
      - `current_title`: The position for which the recommendations are being made.
    """
    titles_str = "\n".join(f"- {title}" for title in TITLES)

    system_prompt = f"""
You are an expert assistant that identifies IAS role recommendation requests and generates semantic search prompts.

Here is the available titles for current_title. Only use these titles:
{titles_str}

These titles are not in correct order, so you need to look to look at all the titles to find the suitable ones.

Your task:
1. Understand the user's query to identify the target IAS role. For example, if the user asks ias officer for the role of Joint Secretary in MeitY, you should infer the target role as "Joint Secretary".
2. Return:
  - 3 to 4 rich semantic queries.
    Example: if the user asks recommend 3 officers for Joint Secretary in MeitY, return queries like:
    - "Officers with experience in IT, electronics, digital governance, e-governance"
    - "Officers with B.Tech or technical background"
    - "Officers with postings in MeitY or Niti Aayog or NIC or similar"
    etc.
   queries should include things like educational qualifications such as btech, b.a. etc. or experiences like digital governance, e-governance, or ministries like finance ministry, dpot, ministry of electronics, or domain like IT, finance, administration etc.
  - the inferred current title that the user is asking for, e.g. "Director, Joint Secretary" etc.

Only return this JSON format:
{{
  "queries": [...],
  "current_title": "...
}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": query}
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    return safe_json_parse(raw, fallback='{{"queries": [], "current_title": null}}')