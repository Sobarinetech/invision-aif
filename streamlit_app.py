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

def gemini_chat(history, doc_text=None):
    """
    history: List of dicts: [{"role": "user"/"model", "content": "..."}]
    doc_text: str, document text to inject into prompt
    """
    client = get_gemini_client()
    model = "gemini-2.5-flash"
    contents = []
    # Inject document text (if any) into the latest user message
    for idx, entry in enumerate(history):
        content = entry["content"]
        # Only add doc_text to the latest user message
        if doc_text and idx == len(history) - 1 and entry["role"] == "user":
            content += (
                f"\n\n---\n\n"
                f"Attached document text (for your analysis, reference, or to answer questions):\n"
                f"{doc_text[:8000]}\n"
            )
        parts = [types.Part.from_text(text=content)]
        contents.append(types.Content(role="user" if entry["role"] == "user" else "model", parts=parts))

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
def extract_text_from_pdf(pdf_file):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file.flush()
        file_path = tmp_file.name
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    finally:
        os.unlink(file_path)
    return text.strip()

def extract_text_from_docx(docx_file):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(docx_file.read())
        tmp_file.flush()
        file_path = tmp_file.name
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    finally:
        os.unlink(file_path)
    return text.strip()

def extract_text_from_txt(txt_file):
    txt_file.seek(0)
    return txt_file.read().decode(errors="ignore").strip()

def extract_uploaded_files_text(uploaded_files):
    """Returns a summary string if multiple files, or all text if one file"""
    if not uploaded_files:
        return None
    all_texts = []
    for file in uploaded_files:
        ext = file.name.lower().split(".")[-1]
        if ext == "pdf":
            text = extract_text_from_pdf(file)
        elif ext == "docx":
            text = extract_text_from_docx(file)
        elif ext == "txt":
            text = extract_text_from_txt(file)
        else:
            text = ""
        if text:
            all_texts.append(f"---\nFile: {file.name}\n{text[:8000]}")
    if not all_texts:
        return None
    return "\n\n".join(all_texts)

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
        key="compliance_files"
    )
    if uploaded_files:
        doc_text = extract_uploaded_files_text(uploaded_files)
        if not doc_text:
            st.error("No usable text extracted from your document(s). Please upload valid PDF, DOCX, or TXT files.")
        else:
            prompt = (
                "You are a world-class compliance analyst. Analyze the following AIF (Alternative Investment Fund) document(s) for compliance, risks, regulatory breaches, and summarize key findings. "
                "Return your analysis in the following structure:\n\n"
                "1. **Summary of Document(s)**\n2. **Key Compliance Risks**\n3. **Detected Regulatory Breaches**\n"
                "4. **Recommendations**\n5. **Any Other Notable Observations**\n\n"
                f"{doc_text}"
            )
            with st.spinner("AI analyzing uploaded document(s)..."):
                report = gemini_generate(prompt)
            st.subheader("AI Compliance Report")
            st.markdown(report)
            dashboard_data["analyses"].append({"name": ", ".join(f.name for f in uploaded_files), "report": report})
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
    chat_doc_text = extract_uploaded_files_text(chat_uploaded_files) if chat_uploaded_files else None

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
                for chunk in gemini_chat(st.session_state["chat_history"], doc_text=chat_doc_text):
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
