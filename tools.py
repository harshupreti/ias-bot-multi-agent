from semantic_search import semantic_search
from filter_with_traits import filter_officers
from web_search import general_web_search, web_trait_search
from reasoning import reasoning_tool
from check_role import check_role_intent

# List of all tools to be passed to your agent
ALL_TOOLS = [
    semantic_search,
    filter_officers,
    general_web_search,
    web_trait_search,
    reasoning_tool,
    check_role_intent
]
