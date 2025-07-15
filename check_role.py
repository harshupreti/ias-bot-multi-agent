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
      - position_below: one level below the target position (to help widen search scope)

    Filters are for internal reference and not meant for direct use in filtering tools.
    After this tool, use semantic_search to find officers matching the queries.
    If no role intent, try filter_officers() to find officers based on traits.
    """

    system_prompt = """
You are an expert assistant that identifies when the user's query is asking for IAS officer recommendations for a specific role or position.

If the user is **clearly asking for officers to fill a specific IAS position**, such as "recommend 3 officers for Joint Secretary in MeitY", "Officers with technical background for a Director role" then:

✅ This **is a role intent**.

If YES (role intent is detected), return a JSON object with:
- `checklist`: A short markdown bullet list of what info is needed to make a good recommendation.
- `queries`: 2–4 **trait-based semantic search prompts** suitable for vector similarity. In each query, include the role just below the target role, e.g., if target is "Joint Secretary", use the terms "Director" or "Deputy Secretary" in every query. Avoid web-style phrasing. Write short prompts like:
  - "(Position just below) with experience in digital governance"
  - "(Position just below) who worked in finance ministry"
  - "(Position just below) with B.Tech or technical background"
- `filters`: Internal reference only (not for hard filtering). Include:
  - Any domain background implied by the query (e.g., 'finance', 'cybersecurity')

If NO (not about any role recommendation), return:

```json
{
  "checklist": "❌ No role intent detected.",
  "queries": [],
  "filters": {},
  "position_below": null
}
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
    return safe_json_parse(raw, fallback="{}")
