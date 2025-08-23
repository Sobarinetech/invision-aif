import os
import streamlit as st
from google import genai
from google.genai import types
from supabase import create_client, Client
import tempfile
import PyPDF2
import docx

# ---------- CONFIG ----------
st.set_page_config(page_title="Invision AIF Solutions", layout="wide")
st.title("Invision AIF Solutions")

# ---------- SECRETS ----------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))

# ---------- SUPABASE CLIENT ----------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---------- LOGIN PAGE ----------
def login():
    if "user" not in st.session_state:
        st.session_state["user"] = None

    if st.session_state["user"] is None:
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Login")
        if login_btn:
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                if hasattr(res, "user") and res.user is not None:
                    st.session_state["user"] = {
                        "id": res.user.id,           # This is auth.uid()
                        "email": res.user.email
                    }
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Login failed. Please check your credentials or contact admin.")
            except Exception as e:
                st.error(f"Login failed: {e}")
        st.stop()

login()
user = st.session_state["user"]
st.info(f"Logged in as: {user['email']}")

# ---------- GEMINI ----------
def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

def gemini_chat(history, doc_text=None):
    client = get_gemini_client()
    model = "gemini-2.5-flash"
    contents = []
    for idx, entry in enumerate(history):
        content = entry["content"]
        if doc_text and idx == len(history) - 1 and entry["role"] == "user":
            content += (
                f"\n\n---\n\n"
                f"Attached document text (for your analysis, reference, or to answer questions):\n"
                f"{doc_text[:8000]}\n"
            )
        parts = [types.Part.from_text(text=content)]
        contents.append(types.Content(role=entry["role"], parts=parts))

    tools = [types.Tool(code_execution=types.ToolCodeExecution), types.Tool(googleSearch=types.GoogleSearch())]
    config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=-1), tools=tools)

    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
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
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=input_text)])]
    tools = [types.Tool(code_execution=types.ToolCodeExecution), types.Tool(googleSearch=types.GoogleSearch())]
    config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=-1), tools=tools)

    output = ""
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
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

# ---------- FILE HANDLING ----------
def extract_text_from_pdf(pdf_file):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
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
    return "\n\n".join(all_texts) if all_texts else None

# ---------- DASHBOARD DATA ----------
def get_dashboard_data():
    if "dashboard_data" not in st.session_state:
        st.session_state["dashboard_data"] = {"analyses": [], "chat_turns": 0, "chatbot_usage": []}
    return st.session_state["dashboard_data"]

def save_dashboard_data(data):
    st.session_state["dashboard_data"] = data

def save_analysis_to_supabase(user_id, doc_name, report):
    try:
        response = supabase.table("analyses").insert({
            "user_id": user_id,
            "analysis_report": report,
            "document_id": None,
        }).execute()
        if "error" in str(response).lower():
            st.error(f"Supabase insert failed: {response}")
    except Exception as e:
        st.error(f"Supabase insert failed: {e}")

def save_chat_to_supabase(user_id, prompt, response):
    try:
        response = supabase.table("analyses").insert({
            "user_id": user_id,
            "analysis_report": f"Prompt: {prompt}\n\nResponse: {response}",
            "document_id": None,
        }).execute()
        if "error" in str(response).lower():
            st.error(f"Supabase chat insert failed: {response}")
    except Exception as e:
        st.error(f"Supabase chat insert failed: {e}")

# ---------- TABS ----------
tabs = st.tabs(["Compliance Analysis", "RegOS Chatbot", "Dashboard"])
dashboard_data = get_dashboard_data()

# --- Compliance Analysis ---
with tabs[0]:
    st.header("Compliance Analysis of AIF Documents")
    uploaded_files = st.file_uploader("Upload AIF Document(s)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_files:
        doc_text = extract_uploaded_files_text(uploaded_files)
        if not doc_text:
            st.error("No usable text found.")
        else:
            prompt = (
                "You are a world-class compliance analyst. Analyze the following AIF document(s) for compliance, risks, regulatory breaches, and summarize key findings.\n\n"
                "Return your analysis in this format:\n"
                "1. Summary\n2. Key Compliance Risks\n3. Detected Regulatory Breaches\n4. Recommendations\n5. Other Observations\n\n"
                f"{doc_text}"
            )
            with st.spinner("Analyzing document(s)..."):
                report = gemini_generate(prompt)
            st.subheader("AI Compliance Report")
            st.markdown(report)
            dashboard_data["analyses"].append({"name": ", ".join(f.name for f in uploaded_files), "report": report})
            save_dashboard_data(dashboard_data)
            save_analysis_to_supabase(user["id"], ", ".join(f.name for f in uploaded_files), report)

# --- RegOS Chatbot ---
with tabs[1]:
    st.header("RegOS Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    chat_uploaded_files = st.file_uploader("Upload documents (optional)", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="chat_files")
    chat_doc_text = extract_uploaded_files_text(chat_uploaded_files) if chat_uploaded_files else None

    for entry in st.session_state["chat_history"]:
        if entry["role"] == "user":
            st.markdown(f"<div style='background:#222; color:#fff; padding:12px; border-radius:10px;'><b>You:</b> {entry['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#013a63; color:#8fd6ff; padding:12px; border-radius:10px;'><b>RegOS AI:</b> {entry['content']}</div>", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("You:", placeholder="Ask a legal or compliance question...")
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.spinner("AI is typing..."):
            response_text = ""
            response_placeholder = st.empty()
            try:
                for chunk in gemini_chat(st.session_state["chat_history"], doc_text=chat_doc_text):
                    response_text += chunk
                    response_placeholder.markdown(f"<div style='background:#013a63; color:#8fd6ff; padding:12px; border-radius:10px;'><b>RegOS AI:</b> {response_text}</div>", unsafe_allow_html=True)
            except Exception as e:
                response_text = f"Error: {e}"
                response_placeholder.markdown(response_text)
            st.session_state["chat_history"].append({"role": "model", "content": response_text})
        dashboard_data["chat_turns"] += 1
        dashboard_data["chatbot_usage"].append({"prompt": user_input, "response": response_text})
        save_dashboard_data(dashboard_data)
        save_chat_to_supabase(user["id"], user_input, response_text)
        st.rerun()

    if st.button("Clear Chat"):
        st.session_state["chat_history"] = []
        st.rerun()

# --- Dashboard ---
with tabs[2]:
    st.header("Dashboard")
    st.metric("Documents Analyzed", len(dashboard_data["analyses"]))
    st.metric("Chatbot Interactions", dashboard_data["chat_turns"])

    st.subheader("Recent Analyses")
    for a in dashboard_data["analyses"][-3:][::-1]:
        with st.expander(a["name"]):
            st.write(a["report"])

    st.subheader("Recent Chatbot Usage")
    for c in dashboard_data["chatbot_usage"][-3:][::-1]:
        with st.expander(f"Prompt: {c['prompt'][:50]}..."):
            st.markdown(f"**AI Response:** {c['response']}")

st.markdown("---")
st.caption("Powered by Google Gemini, Streamlit & Supabase.")
