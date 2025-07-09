# streamlit_app.py

import streamlit as st
import io
import sys
from contextlib import redirect_stdout
import time

st.set_page_config(page_title="IAS Officer Bot (Multi-Agent)", layout="wide")
st.title("🎯 IAS Officer Search Bot (v4)")
st.write("Now powered by LangChain agent + tools")

# --- Cached Agent Loader ---
@st.cache_resource(show_spinner="⚙️ Warming up reasoning engine...")
def load_agent():
    from main import agent
    return agent

agent = load_agent()  # Preloaded and cached

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thinking_logs" not in st.session_state:
    st.session_state.thinking_logs = []

# --- Display Previous Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Input Box ---
user_input = st.chat_input("Ask about IAS officers...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    thinking_buffer = io.StringIO()
    with redirect_stdout(thinking_buffer):
        try:
            with st.spinner("🧠 Thinking..."):
                response = agent.run(user_input)
        except Exception as e:
            response = f"⚠️ Error: {str(e)}"

    raw_logs = thinking_buffer.getvalue()
    st.session_state.thinking_logs.append(raw_logs)

    with st.chat_message("assistant"):
        st.markdown("### 🤔 Internal Reasoning Steps")
        st.code(raw_logs.strip(), language="text")
        st.markdown("### 📢 Final Answer")
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
