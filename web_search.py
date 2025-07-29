# tools/web_tools.py

from langchain_core.tools import tool
import requests

# --- Constants ---
SEARCH_API_KEY = "k48oAMpAYHB6XixnKYJG8WLK"
SEARCH_API_URL = "https://www.searchapi.io/api/v1/search"

def search_web_google(query: str, num_results: int = 5) -> list[str]:
    """
    Internal utility for performing web search using SearchAPI.io.
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": SEARCH_API_KEY,
    }

    response = requests.get(SEARCH_API_URL, params=params)
    data = response.json()

    snippets = []
    if "organic_results" in data:
        for item in data["organic_results"][:num_results]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            combined = f"{title} — {snippet}\n🔗 {link}"
            snippets.append(combined)

    return snippets

@tool
def web_trait_search(officer_name: str) -> str:
    """
    Search web for missing IAS officer traits (e.g. education, postings, training).
    """
    query = f"IAS officer {officer_name} education career postings training details"
    snippets = search_web_google(query)
    return "\n\n".join(snippets) if snippets else "No information found."

@tool
def general_web_search(query: str) -> str:
    """
    Search the web to answer general user queries about IAS officers.
    """
    snippets = search_web_google(query)
    return "\n\n".join(snippets) if snippets else "No results found."
