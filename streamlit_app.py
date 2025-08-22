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

def gemini_chat(history, files=None):
    """
    history: List of dicts: [{"role": "user"/"model", "content": "..."}]
    files: List of tuples: [(file_name, file_bytes)]
    """
    client = get_gemini_client()
    model = "gemini-2.5-flash"
    # Build contents with history
    contents = []
    for entry in history:
        parts = [types.Part.from_text(text=entry["content"])]
        contents.append(types.Content(role="user" if entry["role"] == "user" else "model", parts=parts))

    # Add file to latest user message
    if files:
        # Gemini expects files as types.Part.from_data
        latest_content = contents[-1]
        for file_name, file_bytes in files:
            # Guess mime-type
            ext = file_name.lower().split(".")[-1]
            if ext == "pdf":
                mime = "application/pdf"
            elif ext == "docx":
                mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif ext == "txt":
                mime = "text/plain"
            else:
                mime = "application/octet-stream"
            latest_content.parts.append(types.Part.from_data(mime_type=mime, data=file_bytes, file_name=file_name))

    tools = [
        types.Tool(code_execution=types.ToolCodeExecution),
        types.Tool(googleSearch=types.GoogleSearch()),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        tools=tools,
    )
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue
        part = chunk.candidates[0].content.parts[0]
        if part.text:
            yield part.text
        if part.executable_code:
            yield f"\n[Executable Code]\n{part.executable_code}\n"
        if part.code_execution_result:
            yield f"\n[Code Output]\n{part.code_execution_result}\n"

def gemini_generate(input_text, files=None):
    # For non-chat, single-turn usage with optional files
    client = get_gemini_client()
    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=input_text)],
        ),
    ]
    if files:
        for file_name, file_bytes in files:
            ext = file_name.lower().split(".")[-1]
            if ext == "pdf":
                mime = "application/pdf"
            elif ext == "docx":
                mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif ext == "txt":
                mime = "text/plain"
            else:
                mime = "application/octet-stream"
            contents[0].parts.append(types.Part.from_data(mime_type=mime, data=file_bytes, file_name=file_name))

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
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
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
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
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
    try:
        if suffix == "pdf":
            text = extract_text_from_pdf(file_path)
        elif suffix == "docx":
            text = extract_text_from_docx(file_path)
        elif suffix == "txt":
            text = extract_text_from_txt(file_path)
        else:
            text = ""
    finally:
        os.unlink(file_path)
    return text

# ---------- LOCAL "DB" HELPER ----------
def get_dashboard_data():
    if "dashboard_data" not in st.session_state:
        st.session_state["dashboard_data"] = {
            "analyses": [],
            "chat_turns": 0,
            "chatbot_usage": [],
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
    st.write("Upload your AIF-related documents for direct AI analysis. The AI will read and analyze the file itself.")
    uploaded_files = st.file_uploader(
        "Upload document(s) (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        files_data = [(f.name, f.read()) for f in uploaded_files]
        prompt = (
            "You are a world-class compliance analyst. Analyze the uploaded AIF (Alternative Investment Fund) documents for compliance, risks, regulatory breaches, and summarize key findings. "
            "Return your analysis in the following structure:\n\n"
            "1. **Summary of Document(s)**\n2. **Key Compliance Risks**\n3. **Detected Regulatory Breaches**\n"
            "4. **Recommendations**\n5. **Any Other Notable Observations**\n\n"
            "If more than one document is uploaded, compare and contrast where relevant."
        )
        with st.spinner("AI analyzing uploaded document(s)..."):
            report = gemini_generate(prompt, files=files_data)
        st.subheader("AI Compliance Report")
        st.markdown(report)
        dashboard_data["analyses"].append({"name": ", ".join(f[0] for f in files_data), "report": report})
        save_dashboard_data(dashboard_data)

# 2. RegOS Chatbot Tab (LLM with document upload)
with tabs[1]:
    st.header("RegOS Chatbot")
    st.write(
        "An advanced AI chatbot for regulatory, legal, and compliance queries. You can also upload documents for the AI to analyze and reference in your chat."
    )
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Document upload for chat
    chat_uploaded_files = st.file_uploader(
        "Upload documents for this chat (optional, PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="chat_files"
    )
    chat_files_data = [(f.name, f.read()) for f in chat_uploaded_files] if chat_uploaded_files else []

    # Show chat history as bubbles
    for entry in st.session_state["chat_history"]:
        if entry["role"] == "user":
            st.markdown(
                f"<div style='background-color:#222; color:#fff; border-radius:16px; padding:12px 18px; margin-top:10px; margin-bottom:2px; max-width:85%; align-self:flex-end; margin-left:auto;'><b>You:</b> {entry['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background-color:#013a63; color:#8fd6ff; border-radius:16px; padding:12px 18px; margin-top:2px; margin-bottom:10px; max-width:85%; align-self:flex-start; margin-right:auto;'><b>RegOS AI:</b> {entry['content']}</div>",
                unsafe_allow_html=True,
            )

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("You:", key="chat_input", placeholder="Ask a regulatory, legal, or compliance question or request document analysis...")
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.spinner("AI is typing..."):
            response_text = ""
            response_placeholder = st.empty()
            try:
                for chunk in gemini_chat(st.session_state["chat_history"], files=chat_files_data):
                    response_text += chunk
                    response_placeholder.markdown(
                        f"<div style='background-color:#013a63; color:#8fd6ff; border-radius:16px; padding:12px 18px; margin-top:2px; margin-bottom:10px; max-width:85%; align-self:flex-start; margin-right:auto;'><b>RegOS AI:</b> {response_text}</div>",
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                response_text = f"Sorry, there was an error with the AI: {e}"
                response_placeholder.markdown(response_text)
            st.session_state["chat_history"].append({"role": "model", "content": response_text})
        dashboard_data["chat_turns"] += 1
        dashboard_data["chatbot_usage"].append({"prompt": user_input, "response": response_text})
        save_dashboard_data(dashboard_data)
        st.rerun()

    if st.button("Clear Chat", key="clear_chat"):
        st.session_state["chat_history"] = []
        st.rerun()

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
