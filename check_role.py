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
model = "gpt-4o-mini"
client = OpenAI(api_key=token)

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
    Detects if the query is about a specific IAS role.
    If yes, returns:
      - checklist: markdown string of info to gather
      - queries: list of semantic search prompts (for vector similarity)
      - filters: rules like "exclude officers already at this level"
    """

    system_prompt = """
Your job is to interpret the user's query and determine if it targets a specific IAS role.

If YES:
- Identify what type of officer would be suitable for that role.
- List the kind of background indicators to match (education, policy experience, domain expertise).
- Generate 2–4 short semantic search queries that are suitable for *vector similarity search*.
  (Avoid web-style phrasing like "recommend me officers for..."; instead use trait-based prompts like:
   "IAS officers with experience in digital governance" or "IAS officers with B.Tech degree".)

- Also include filters:
  - Whether to exclude officers already at or above the level of the role (e.g., don't suggest Secretaries for a Joint Secretary role).
  - What seniority level is ideal (e.g., "Joint Secretary", "Director", etc.)
  - Any required background keywords like ["finance", "digital", "infrastructure"]

If the query is NOT about a role, return:
{
  "checklist": "❌ No role intent detected.",
  "queries": [],
  "filters": {}
}

Only return filters if clearly implied by the query.
Return clean JSON only.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": query}
        ],
        temperature=0.2
    )

    raw = response.choices[0].message.content.strip()
    return safe_json_parse(raw, fallback="{}")
