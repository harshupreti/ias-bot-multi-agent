import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableConfig
import io

st.set_page_config(page_title="IAS Officer Bot (Multi-Agent)", layout="wide")
st.title("🎯 IAS Officer Search Bot (v4)")
st.write("Now powered by LangChain agent + tools")

# --- Cached Agent Loader ---
@st.cache_resource(show_spinner="⚙️ Warming up reasoning engine...")
def load_agent():
    from main import agent  # `agent` should be built via initialize_agent
    return agent

agent = load_agent()  # Preloaded and cached

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# --- Custom Callback Handler ---
class StreamToBufferHandler(BaseCallbackHandler):
    def __init__(self, buffer):
        self.buffer = buffer

    def on_tool_start(self, tool, input_str, **kwargs):
        self.buffer.write(f"\n🔧 Tool `{tool}` started with input: {input_str}\n")

    def on_tool_end(self, output, **kwargs):
        self.buffer.write(f"✅ Tool finished. Output: {output}\n")

    def on_llm_start(self, *args, **kwargs):
        self.buffer.write("🤖 LLM started\n")

    def on_llm_end(self, output, **kwargs):
        self.buffer.write(f"📝 LLM finished. Output: {output.generations[0][0].text}\n")

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

    # Reasoning + Callbacks
    reasoning_buffer = io.StringIO()
    callback_handler = StreamToBufferHandler(reasoning_buffer)

    try:
        with st.spinner("🧠 Thinking..."):
            response = agent.invoke(
                input=user_input,
                config=RunnableConfig(callbacks=[callback_handler])
            )
            final_output = response["output"] if isinstance(response, dict) and "output" in response else str(response)
    except Exception as e:
        final_output = f"⚠️ Error: {str(e)}"

    logs = reasoning_buffer.getvalue()

    # Display Assistant Response
    with st.chat_message("assistant"):
        with st.expander("🔍 View internal logic"):
            st.markdown(logs or "*No internal reasoning captured.*")
        st.markdown("### 📢 Final Answer")
        st.markdown(final_output)

    # Save to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_output,
        "logs": logs
    })

    st.rerun()
