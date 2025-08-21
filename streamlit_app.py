import os
import streamlit as st
from google import genai
from google.genai import types
import tempfile
import base64
import pickle

# ---------- CONFIG ----------
st.set_page_config(page_title="Invision AIF Solutions", layout="wide")
st.title("Invision AIF Solutions")

# ---------- ENVIRONMENT: GOOGLE GEMINI ----------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else os.environ.get("GEMINI_API_KEY")

def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

def gemini_generate(input_text):
    client = get_gemini_client()
    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input_text),
            ],
        ),
    ]
    tools = [
        types.Tool(code_execution=types.ToolCodeExecution),
        types.Tool(googleSearch=types.GoogleSearch()),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        tools=tools,
    )
    output = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        part = chunk.candidates[0].content.parts[0]
        if part.text:
            output += part.text
        if part.executable_code:
            output += f"\n[Executable Code]\n{part.executable_code}\n"
        if part.code_execution_result:
            output += f"\n[Code Output]\n{part.code_execution_result}\n"
    return output

def gemini_chat(history):
    client = get_gemini_client()
    model = "gemini-2.5-flash"
    contents = []
    for entry in history:
        contents.append(
            types.Content(
                role="user" if entry["role"] == "user" else "model",
                parts=[types.Part.from_text(text=entry["content"])]
            )
        )
    tools = [
        types.Tool(code_execution=types.ToolCodeExecution),
        types.Tool(googleSearch=types.GoogleSearch()),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        tools=tools,
    )
    output = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        part = chunk.candidates[0].content.parts[0]
        if part.text:
            output += part.text
        if part.executable_code:
            output += f"\n[Executable Code]\n{part.executable_code}\n"
        if part.code_execution_result:
            output += f"\n[Code Output]\n{part.code_execution_result}\n"
    return output

# ---------- LOCAL "DB" HELPER ----------
def get_dashboard_data():
    if "dashboard_data" not in st.session_state:
        st.session_state["dashboard_data"] = {
            "analyses": [],    # list of {"name", "report"}
            "chat_turns": 0,   # int
            "chatbot_usage": [], # list of {"prompt", "response"}
        }
    return st.session_state["dashboard_data"]

def save_dashboard_data(data):
    st.session_state["dashboard_data"] = data

# ---------- FILE HANDLING ----------
def save_uploaded_file(uploadedfile):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploadedfile.getbuffer())
        return tmp_file.name

def encode_file_to_base64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ---------- TABS ----------
tabs = st.tabs(["Compliance Analysis", "RegOS Chatbot", "Dashboard"])
dashboard_data = get_dashboard_data()

# 1. Compliance Analysis Tab
with tabs[0]:
    st.header("Compliance Analysis of AIF Documents")
    st.write("Upload your AIF-related documents for compliance analysis powered by AI.")
    uploaded_files = st.file_uploader(
        "Upload document(s) (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"Analysing: {uploaded_file.name}")
            file_path = save_uploaded_file(uploaded_file)
            file_b64 = encode_file_to_base64(file_path)
            input_text = (
                f"You are a world-class compliance analyst. Analyze the following document for AIF (Alternative Investment Fund) compliance, risks, regulatory breaches, and summarize key findings. Document (base64): {file_b64}. "
                "If the file is not text, extract and analyze its content first."
            )
            with st.spinner(f"AI analysing {uploaded_file.name}..."):
                report = gemini_generate(input_text)
            st.subheader(f"AI Compliance Report for {uploaded_file.name}")
            st.write(report)
            dashboard_data["analyses"].append({"name": uploaded_file.name, "report": report})
            save_dashboard_data(dashboard_data)

# 2. RegOS Chatbot Tab
with tabs[1]:
    st.header("RegOS Chatbot")
    st.write(
        "An advanced AI chatbot for regulatory, legal, and compliance queries. Your conversations are remembered for ongoing context."
    )
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    user_input = st.text_input("You:", key="chat_input")
    send_btn = st.button("Send", key="send_btn")
    if send_btn and user_input.strip():
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.spinner("AI thinking..."):
            bot_response = gemini_chat(st.session_state["chat_history"])
        st.session_state["chat_history"].append({"role": "model", "content": bot_response})
        dashboard_data["chat_turns"] += 1
        dashboard_data["chatbot_usage"].append({"prompt": user_input, "response": bot_response})
        save_dashboard_data(dashboard_data)

    # Show chat history
    for entry in st.session_state["chat_history"]:
        if entry["role"] == "user":
            st.markdown(f"**You:** {entry['content']}")
        else:
            st.markdown(f"**RegOS AI:** {entry['content']}")

    if st.button("Clear Chat", key="clear_chat"):
        st.session_state["chat_history"] = []

# 3. Dashboard Tab
with tabs[2]:
    st.header("Dashboard: Metrics, Results & KPIs")
    st.write(
        "Visualize compliance analysis results, AI performance metrics, chatbot usage, and key performance indicators."
    )
    num_analyses = len(dashboard_data["analyses"])
    num_chats = dashboard_data["chat_turns"]
    st.metric("Documents Analyzed", num_analyses)
    st.metric("Chatbot Interactions", num_chats)
    st.subheader("Recent Analyses")
    for a in dashboard_data["analyses"][-3:][::-1]:
        with st.expander(a["name"]):
            st.write(a["report"])
    st.subheader("Recent Chatbot Usage")
    for c in dashboard_data["chatbot_usage"][-3:][::-1]:
        with st.expander(f"Prompt: {c['prompt'][:50]}..."):
            st.markdown(f"**AI Response:** {c['response']}")
    st.info("All data is stored in-memory for your session only.")

st.markdown("---")
st.caption("Powered by Google Gemini & Streamlit. Confidential & Secure.")
