import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableConfig
import io
import asyncio

# --- Import update checker ---
from pipeline_runner import AsyncPipelineRunner
from qdrant_client import QdrantClient
from config import validate_config, QDRANT_URL, QDRANT_API_KEY, STATE_CODES

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

runner = AsyncPipelineRunner(qdrant_client=client)

async def process_all_states():
    for state_code in STATE_CODES.keys():
        try:
            print(f"▶ Processing cadre/state: {state_code}")
            await runner.run_for_cadre(state_code)
        except Exception as e:
            print(f"❌ Error in {state_code}: {e}")

# --- Streamlit page setup ---
st.set_page_config(page_title="IAS Officer Bot (Multi-Agent)", layout="wide")
st.title("🎯 IAS Officer Search Bot (v5.1)")
st.write("Now powered by LangChain agent + tools")

# --- Run update check once at startup ---
@st.cache_resource(show_spinner="🔄 Checking for updates in dataset...")
def check_for_updates():
    validate_config()
    asyncio.run(process_all_states())
    return True

check_for_updates()

# --- Cached Graph Loader ---
@st.cache_resource(show_spinner="⚙️ Warming up reasoning engine...")
def load_workflow():
    from main import app  # app is the compiled StateGraph workflow
    return app

workflow = load_workflow()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# --- Sidebar with Filters ---
st.sidebar.header("🧰 Filters")
st.sidebar.markdown(
    "**1)** For just filtration tasks using filters, choose the filters you like and in the prompt write officers and hit enter.\n\n"
    "**2)** For role recommendation, choose the filters you require and in prompt describe the role for which you want the officer including the level and department/ministry."
)

# Filter options
cadre_options = [
    "Any", "A G M U T", "Andhra Pradesh", "Assam Meghalya", "Bihar", "Chhattisgarh", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal"
]
gender_options = ["Any", "Male", "Female"]
year_ops = ["Any", "before", "from", "after"]
years = list(range(1987, 2025))

selected_cadre = st.sidebar.selectbox("Cadre", cadre_options)
selected_gender = st.sidebar.selectbox("Gender", gender_options)
selected_year_op = st.sidebar.selectbox("Allotment Year Operation", year_ops)
selected_year = st.sidebar.selectbox("Allotment Year", ["Any"] + years)

filters = {
    "cadre": None if selected_cadre == "Any" else selected_cadre,
    "gender": None if selected_gender == "Any" else selected_gender,
    "allotment_year_operation": None if selected_year_op == "Any" else selected_year_op,
    "allotment_year": None if selected_year == "Any" else selected_year
}

# --- Custom Callback Handler ---
class StreamToBufferHandler(BaseCallbackHandler):
    def __init__(self, buffer):
        self.buffer = buffer

    def on_tool_start(self, tool, input_str, **kwargs):
        self.buffer.write(f"\n🔧 Tool {tool} started with input: {input_str}\n")

    def on_tool_end(self, output, **kwargs):
        self.buffer.write(f"✅ Tool finished. Output: {output}\n")

    def on_llm_start(self, *args, **kwargs):
        self.buffer.write("🤖 LLM started\n")

    def on_llm_end(self, output, **kwargs):
        try:
            self.buffer.write(f"📝 LLM finished. Output: {output.generations[0][0].text}\n")
        except Exception:
            self.buffer.write(f"📝 LLM finished. Output: {output}\n")

    def on_chain_start(self, *args, **kwargs):
        self.buffer.write("🔗 Chain started\n")

    def on_chain_end(self, outputs, **kwargs):
        self.buffer.write(f"🔚 Chain finished. Outputs: {outputs}\n")

# --- Display Previous Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "logs" in msg:
            with st.expander("🔍 View internal logic"):
                st.markdown(msg["logs"])
        st.markdown(msg["content"])

# --- Input Box ---
user_input = st.chat_input("Ask about IAS officers...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    reasoning_buffer = io.StringIO()
    callback_handler = StreamToBufferHandler(reasoning_buffer)

    # Prepare initial state for the workflow
    initial_state = {
        "input": user_input,
        "filters": filters,
        "queries": [],
        "below_position": "",
        "search_results": [],
        "filtered_officers": [],
        "reasoning_output": "",
        "steps": []
    }

    try:
        with st.spinner("🧠 Thinking..."):
            # Run the workflow with callbacks
            response = workflow.invoke(
                initial_state,
                config=RunnableConfig(callbacks=[callback_handler])
            )
            final_output = response.get("reasoning_output", "") or response.get("output", "") or str(response)
    except Exception as e:
        final_output = f"⚠️ Error: {str(e)}"

    logs = reasoning_buffer.getvalue()

    with st.chat_message("assistant"):
        with st.expander("🔍 View internal logic"):
            st.markdown(logs or "*No internal reasoning captured.*")
        st.markdown("### 📢 Final Answer")
        st.markdown(final_output)

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_output,
        "logs": logs
    })

    st.rerun()
