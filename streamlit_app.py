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
st.set_page_config(page_title="Invision AIF Solutions", layout="wide")
st.markdown(
    "<h1 style='color:#013a63;'>Invision AIF Solutions</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div style='font-size:1.08rem;color:#222;background:#e9f5fe;border-radius:12px;padding:10px 18px;margin-bottom:1.5rem;'>
    🔍 <b>Analyze, chat, and monitor with next-gen SEBI AIF compliance AI for Alternative Investment Funds.<br>
    <span style='color:#013a63;'>Upload documents, ask questions, track insights with an advanced dashboard.</span></b>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- ENVIRONMENT: GOOGLE GEMINI ----------
def gemini_generate(input_text):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=input_text)],
        ),
    ]
    tools = [
        types.Tool(googleSearch=types.GoogleSearch()),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0,
        ),
        tools=tools,
    )
    output = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        # Use .text for main output
        if hasattr(chunk, "text") and chunk.text:
            output += chunk.text
        # For compatibility if the chunk structure changes
        elif hasattr(chunk, "candidates") and chunk.candidates:
            part = chunk.candidates[0].content.parts[0]
            if part.text:
                output += part.text
    return output

def gemini_chat(history, doc_text=None):
    """
    history: List of dicts: [{"role": "user"/"model", "content": "..."}]
    doc_text: str, document text to inject into prompt
    """
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
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
        contents.append(types.Content(role="user" if entry["role"] == "user" else "model", parts=parts))
    tools = [
        types.Tool(googleSearch=types.GoogleSearch()),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0,
        ),
        tools=tools,
    )
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if hasattr(chunk, "text") and chunk.text:
            yield chunk.text
        elif hasattr(chunk, "candidates") and chunk.candidates:
            part = chunk.candidates[0].content.parts[0]
            if part.text:
                yield part.text

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

# ---------- TABS ----------
tabs = st.tabs(
    [
        "Compliance Analysis (SEBI AIF)",
        "RegOS Chatbot (SEBI AIF)",
        "Dashboard & Insights",
    ]
)
dashboard_data = get_dashboard_data()

# 1. Compliance Analysis Tab
with tabs[0]:
    st.header("Compliance Analysis of AIF Documents (SEBI focused)")
    st.write(
        "Upload your AIF-related documents for direct SEBI compliance AI analysis. "
        "The AI will extract, analyze, and summarize compliance, risks, breaches, and more, referencing the latest SEBI/Indian regulations and providing citations."
    )
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
            analysis_prompt = (
                "Act as a SEBI AIF compliance expert. Analyze the following AIF (Alternative Investment Fund) document(s) strictly for compliance with current SEBI AIF Regulations and Indian regulatory context. "
                "For every point, provide proper grounded citations from SEBI official regulations, circulars, or Indian law. Use real-time search to ensure all referenced regulations are current. "
                "Return your analysis in the following structure:\n\n"
                "1. **Summary of Document(s)** (with citations)\n"
                "2. **Key Compliance Risks** (with grounded SEBI citations)\n"
                "3. **Detected Regulatory Breaches** (cite SEBI section/circular)\n"
                "4. **Recommendations** (reference the specific SEBI compliance to address)\n"
                "5. **Any Other Notable Observations** (with citations)\n"
                "6. **Breakdown of Risk by Section** (reference relevant SEBI requirements)\n"
                "7. **AI Confidence Level (0-100%) and Reasoning**\n"
                "8. **Potential Red Flags** (cite regulation)\n"
                "9. **Suggested Next Steps for Compliance Team** (with references)\n"
                "10. **All references must cite relevant SEBI AIF regulation/circular with section number and date, if available.**\n\n"
                f"{doc_text}"
            )
            with st.spinner("AI analyzing uploaded document(s) as per latest SEBI AIF regulatory context..."):
                report = gemini_generate(analysis_prompt)
            st.subheader("SEBI AIF Compliance Report (with citations)")
            st.markdown(report)
            st.markdown(downloadable_report(report, filename="AIF-SEBI-Compliance-Report.txt"), unsafe_allow_html=True)
            dashboard_data["analyses"].append({
                "name": ", ".join(f.name for f in uploaded_files),
                "report": report,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            save_dashboard_data(dashboard_data)

# 2. RegOS Chatbot Tab (LLM with document upload)
with tabs[1]:
    st.header("RegOS Chatbot for SEBI AIF Compliance")
    st.write(
        "Ask regulatory, legal, or compliance questions about AIFs in India. "
        "AI is grounded in the latest SEBI/Indian regulatory context and always provides citations."
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
            placeholder="Ask a SEBI AIF regulatory, legal, or compliance question, or request analysis of your uploaded document...",
        )
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        # Enhance user input for SEBI compliance expert with citations and grounded search
        user_query = (
            "You are a SEBI AIF compliance expert. Answer strictly with reference to the latest SEBI AIF Regulations and Indian regulatory context. "
            "For every point, provide grounded citations from the current SEBI regulations, circulars, or Indian law (include section number/date). "
            "Use real-time search to ensure all referenced regulations are current and provide links/citations where possible. "
            f"User query: {user_input}"
        )
        st.session_state["chat_history"].append({"role": "user", "content": user_query, "time": datetime.now().isoformat()})
        with st.spinner("RegOS AI (SEBI expert) is typing..."):
            response_text = ""
            response_placeholder = st.empty()
            try:
                for chunk in gemini_chat(st.session_state["chat_history"], doc_text=chat_doc_text):
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
        })
        save_dashboard_data(dashboard_data)
        st.rerun()

    if st.button("Clear Chat", key="clear_chat"):
        st.session_state["chat_history"] = []
        st.rerun()
    if st.button("Download Chat History"):
        chat_hist = "\n\n".join(
            f"{e['role'].capitalize()} ({e['time'][:19]}): {e['content']}" for e in st.session_state["chat_history"]
        )
        st.markdown(downloadable_report(chat_hist, filename="RegOS-Chat-History.txt"), unsafe_allow_html=True)

# 3. Dashboard & Insights Tab
with tabs[2]:
    st.header("Dashboard: Metrics, Trends & AI Insights")
    st.write(
        "Visualize compliance analysis results, AI performance metrics, chatbot usage, and key performance indicators. "
        "Track trends, download results, and get advanced breakdowns."
    )

    num_analyses = len(dashboard_data["analyses"])
    num_chats = dashboard_data["chat_turns"]
    st.metric("Documents Analyzed", num_analyses)
    st.metric("Chatbot Interactions", num_chats)
    uniq_files = set()
    for a in dashboard_data["analyses"]:
        for name in a["name"].split(","):
            uniq_files.add(name.strip())
    st.metric("Unique Files Uploaded", len(uniq_files))

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

    st.subheader("Recent Analyses")
    for a in dashboard_data["analyses"][-3:][::-1]:
        with st.expander(f"{a['name']} ({a['timestamp']})"):
            st.write(a["report"])
            st.markdown(downloadable_report(a["report"], filename=f"{a['name']}-AIF-SEBI-Report.txt"), unsafe_allow_html=True)

    st.subheader("Recent Chatbot Usage")
    for c in dashboard_data["chatbot_usage"][-3:][::-1]:
        with st.expander(f"Prompt: {c['prompt'][:50]}... ({c['timestamp']})"):
            st.markdown(f"**AI Response:** {c['response']}")

    st.info("All data is stored in-memory for your session only. Download reports and chat logs for your records.")

st.markdown("---")
st.caption("Powered by Google Gemini, Streamlit, and Plotly. Confidential & Secure. 💡")

# ---------- REQUIREMENTS ----------
# pip install streamlit google-genai PyPDF2 python-docx pandas plotly
