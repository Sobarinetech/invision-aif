import os
import streamlit as st
from google import genai
from google.genai import types
import tempfile
import PyPDF2
import docx
import pandas as pd
import plotly.express as px
import base64
from datetime import datetime

# ---------- CONFIG ----------
st.set_page_config(page_title="Invision AIF Solutions", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    "<style> .css-1v0mbdj.eknhn3m9{background: linear-gradient(90deg,#013a63,#00509e 60%,#2d82b7);} .block-container {padding-top: 2rem;} </style>",
    unsafe_allow_html=True,
)
st.markdown("<h1 style='color:#013a63;'>Invision AIF Solutions</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='font-size:1.1rem;color:#222;background:#e9f5fe;border-radius:12px;padding:10px 18px;margin-bottom:1.5rem;'>
    🔍 <b>Analyze, chat, and monitor with next-gen compliance AI for Alternative Investment Funds.<br>
    <span style='color:#013a63;'>Upload documents, ask questions, track insights with an advanced dashboard.</span></b>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- ENVIRONMENT: GOOGLE GEMINI ----------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))

def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

def gemini_chat(history, doc_text=None, model="gemini-2.5-flash"):
    """
    history: List of dicts: [{"role": "user"/"model", "content": "..."}]
    doc_text: str, document text to inject into prompt
    """
    client = get_gemini_client()
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

def gemini_generate(input_text, model="gemini-2.5-flash"):
    client = get_gemini_client()
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

def downloadable_report(report, filename="AIF-Compliance-Report.txt"):
    b64 = base64.b64encode(report.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}" style="color:#00509e;">⬇️ Download Report</a>'

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

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("<h3 style='color:#00509e;'>Quick Actions</h3>", unsafe_allow_html=True)
    st.write("• [Contact Support](mailto:support@invisionaif.com)")
    st.write("• [Product Roadmap](https://github.com/Sobarinetech)")
    st.write("• [Docs & FAQ](https://github.com/Sobarinetech)")
    st.write("• [Request New Feature](mailto:features@invisionaif.com)")
    st.markdown("---")
    st.write("**Current User:**", os.getenv("USER", "Guest"))
    st.write(f"**Session Start:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------- TABS ----------
tabs = st.tabs(
    [
        "📄 Compliance Analysis",
        "🤖 RegOS Chatbot",
        "📊 Dashboard & Insights",
    ]
)
dashboard_data = get_dashboard_data()

# 1. Compliance Analysis Tab
with tabs[0]:
    st.header("📄 AIF Compliance Analysis")
    st.write(
        "Upload your AIF-related documents for direct AI analysis. The AI will extract, analyze, and summarize compliance, risks, breaches, and more. "
        "Get detailed, structured reporting with AI-powered insights."
    )
    uploaded_files = st.file_uploader(
        "Upload document(s) (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="compliance_files"
    )
    col1, col2 = st.columns([2,1])
    with col1:
        if uploaded_files:
            doc_text = extract_uploaded_files_text(uploaded_files)
            if not doc_text:
                st.error("No usable text extracted from your document(s). Please upload valid PDF, DOCX, or TXT files.")
            else:
                analysis_prompt = (
                    "You are a world-class compliance analyst AI. Analyze the following AIF (Alternative Investment Fund) document(s) for compliance, risks, regulatory breaches, and summarize key findings. "
                    "Return your analysis in the following structure:\n\n"
                    "1. **Summary of Document(s)**\n"
                    "2. **Key Compliance Risks**\n"
                    "3. **Detected Regulatory Breaches**\n"
                    "4. **Recommendations**\n"
                    "5. **Any Other Notable Observations**\n"
                    "6. **Breakdown of Risk by Section (if possible)**\n"
                    "7. **AI Confidence Level (0-100%) and Reasoning**\n"
                    "8. **Potential Red Flags (list)**\n"
                    "9. **Suggested Next Steps for Compliance Team**\n\n"
                    f"{doc_text}"
                )
                with st.spinner("AI analyzing uploaded document(s)..."):
                    report = gemini_generate(analysis_prompt)
                st.subheader("AI Compliance Report")
                st.markdown(report)
                st.markdown(downloadable_report(report, filename="AIF-Compliance-Report.txt"), unsafe_allow_html=True)
                dashboard_data["analyses"].append({
                    "name": ", ".join(f.name for f in uploaded_files),
                    "report": report,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                save_dashboard_data(dashboard_data)
    with col2:
        st.info("**Pro Tip**: Upload multiple documents for cross-comparison. The AI will aggregate and highlight common risks, patterns, and unique findings.")

# 2. RegOS Chatbot Tab (LLM with document upload)
with tabs[1]:
    st.header("🤖 RegOS Chatbot")
    st.write(
        "An advanced AI chatbot for regulatory, legal, and compliance queries. "
        "You can upload documents for the AI to analyze, reference, or answer questions about your specific files."
    )
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

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
                f"<div style='background:linear-gradient(90deg,#00509e,#2d82b7 60%); color:#fff; border-radius:16px; padding:12px 18px; margin-top:10px; margin-bottom:2px; max-width:85%; align-self:flex-end; margin-left:auto;'><b>You:</b> {entry['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background-color:#e9f5fe; color:#013a63; border-radius:16px; padding:12px 18px; margin-top:2px; margin-bottom:10px; max-width:85%; align-self:flex-start; margin-right:auto;'><b>RegOS AI:</b> {entry['content']}</div>",
                unsafe_allow_html=True,
            )

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "You:",
            key="chat_input",
            placeholder="Ask a regulatory, legal, or compliance question, or request document analysis...",
        )
        model_choice = st.selectbox(
            "AI Model",
            options=["gemini-2.5-flash", "gemini-1.5-pro"],
            index=0,
            help="Select 'pro' for more reasoning, 'flash' for speed."
        )
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        st.session_state["chat_history"].append({"role": "user", "content": user_input, "time": datetime.now().isoformat()})
        with st.spinner("AI is typing..."):
            response_text = ""
            response_placeholder = st.empty()
            try:
                for chunk in gemini_chat(st.session_state["chat_history"], doc_text=chat_doc_text, model=model_choice):
                    response_text += chunk
                    response_placeholder.markdown(
                        f"<div style='background-color:#e9f5fe; color:#013a63; border-radius:16px; padding:12px 18px; margin-top:2px; margin-bottom:10px; max-width:85%; align-self:flex-start; margin-right:auto;'><b>RegOS AI:</b> {response_text}</div>",
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                response_text = f"Sorry, there was an error with the AI: {e}"
                response_placeholder.markdown(response_text)
            st.session_state["chat_history"].append({"role": "model", "content": response_text, "time": datetime.now().isoformat()})
        dashboard_data["chat_turns"] += 1
        dashboard_data["chatbot_usage"].append({
            "prompt": user_input,
            "response": response_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_choice,
        })
        save_dashboard_data(dashboard_data)
        st.rerun()

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Clear Chat", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()
    with colB:
        if st.button("Download Chat History"):
            chat_hist = "\n\n".join(
                f"{e['role'].capitalize()} ({e['time'][:19]}): {e['content']}" for e in st.session_state["chat_history"]
            )
            st.markdown(downloadable_report(chat_hist, filename="RegOS-Chat-History.txt"), unsafe_allow_html=True)

# 3. Dashboard & Insights Tab
with tabs[2]:
    st.header("📊 Dashboard: Metrics, Trends & AI Insights")
    st.write(
        "Visualize compliance analysis results, AI performance metrics, chatbot usage, and key performance indicators. "
        "Track trends, download results, and get advanced breakdowns."
    )

    num_analyses = len(dashboard_data["analyses"])
    num_chats = dashboard_data["chat_turns"]
    st.metric("Documents Analyzed", num_analyses)
    st.metric("Chatbot Interactions", num_chats)
    # More metrics
    uniq_files = set()
    for a in dashboard_data["analyses"]:
        for name in a["name"].split(","):
            uniq_files.add(name.strip())
    st.metric("Unique Files Uploaded", len(uniq_files))

    # Timeline chart of usage
    if dashboard_data["analyses"]:
        df_analyses = pd.DataFrame(dashboard_data["analyses"])
        df_analyses["timestamp"] = pd.to_datetime(df_analyses["timestamp"])
        fig = px.bar(
            df_analyses,
            x="timestamp",
            y=df_analyses.index+1,
            hover_data=["name"],
            labels={"y":"Cumulative Analyses", "timestamp":"Time"},
            title="Document Analysis Timeline",
            color_discrete_sequence=["#00509e"],
        )
        st.plotly_chart(fig, use_container_width=True)
    if dashboard_data["chatbot_usage"]:
        df_chats = pd.DataFrame(dashboard_data["chatbot_usage"])
        df_chats["timestamp"] = pd.to_datetime(df_chats["timestamp"])
        chats_per_day = df_chats.groupby(df_chats["timestamp"].dt.date).size()
        fig2 = px.line(
            chats_per_day,
            markers=True,
            labels={"value":"# Interactions", "timestamp":"Date"},
            title="Chatbot Usage Over Time",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # AI Insights
    st.subheader("Recent Analyses")
    for a in dashboard_data["analyses"][-3:][::-1]:
        with st.expander(f"{a['name']} ({a['timestamp']})"):
            st.write(a["report"])
            st.markdown(downloadable_report(a["report"], filename=f"{a['name']}-AIF-Report.txt"), unsafe_allow_html=True)

    st.subheader("Recent Chatbot Usage")
    for c in dashboard_data["chatbot_usage"][-3:][::-1]:
        with st.expander(f"Prompt: {c['prompt'][:50]}... ({c['timestamp']})"):
            st.markdown(f"**AI Response:** {c['response']}")
            st.markdown(f"<span style='font-size:0.9em;'>Model: {c.get('model','n/a')}</span>", unsafe_allow_html=True)

    st.info("All data is stored in-memory for your session only. Download reports and chat logs for your records.")

st.markdown("---")
st.caption("Powered by Google Gemini, Streamlit, and Plotly. Confidential & Secure. 💡")

# ---------- REQUIREMENTS ----------
# pip install streamlit google-genai PyPDF2 python-docx pandas plotly
