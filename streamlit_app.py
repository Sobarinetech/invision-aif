import os
import streamlit as st
from google import genai
from google.genai import types
import tempfile
import PyPDF2
import docx

# ---------- CONFIG ----------
st.set_page_config(page_title="Invision AIF Solutions", layout="wide")
st.title("Invision AIF Solutions")

# ---------- ENVIRONMENT: GOOGLE GEMINI ----------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))

def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

def gemini_generate(input_text):
    client = get_gemini_client()
    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=input_text)],
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

# ---------- FILE TEXT EXTRACTION ----------
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.strip()

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def extract_text(uploadedfile):
    suffix = uploadedfile.name.lower().split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
        tmp_file.write(uploadedfile.getbuffer())
        tmp_file.flush()
        file_path = tmp_file.name
    if suffix == "pdf":
        text = extract_text_from_pdf(file_path)
    elif suffix == "docx":
        text = extract_text_from_docx(file_path)
    elif suffix == "txt":
        text = extract_text_from_txt(file_path)
    else:
        text = ""
    os.unlink(file_path)
    return text

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
            st.write(f"Analyzing: {uploaded_file.name}")
            extracted_text = extract_text(uploaded_file)
            if not extracted_text or extracted_text.strip() == "":
                st.error("No text extracted; file may be scanned or image-based. Please upload a text-based document.")
                continue
            input_text = (
                "You are a world-class compliance analyst. Analyze the following AIF (Alternative Investment Fund) document for compliance, "
                "risks, regulatory breaches, and summarize key findings. "
                "Return your analysis in the following structure:\n\n"
                "1. **Summary of Document**\n2. **Key Compliance Risks**\n3. **Detected Regulatory Breaches**\n"
                "4. **Recommendations**\n5. **Any Other Notable Observations**\n\n"
                f"---\n\nDOCUMENT TEXT:\n{extracted_text[:8000]}\n"  # limit to 8000 chars for prompt safety
            )
            with st.spinner(f"AI analyzing {uploaded_file.name}..."):
                report = gemini_generate(input_text)
            st.subheader(f"AI Compliance Report for {uploaded_file.name}")
            st.markdown(report)
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
            bot_response = gemini_generate(user_input)
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

# ---------- REQUIREMENTS ----------
# pip install streamlit google-genai PyPDF2 python-docx
