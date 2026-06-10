import streamlit as st
import tempfile
import os
from rag_engine import process_pdf, get_answer

st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="🌊",
    layout="centered"
)


BG_FROM           = "#e0f7fa"
BG_TO             = "#b2ebf2"

HEADER_FROM       = "#0077b6"
HEADER_TO         = "#00b4d8"
HEADER_BORDER     = "#0096c7"
HEADER_TITLE      = "#ffffff"
HEADER_SUB        = "#caf0f8"

STAT_BG           = "rgba(0,119,182,0.15)"
STAT_BORDER       = "#0096c7"
STAT_NUMBER       = "#0077b6"
STAT_LABEL        = "#023e8a"

UPLOAD_BG         = "rgba(0,180,216,0.1)"
UPLOAD_BORDER     = "#0096c7"

CHAT_BG           = "rgba(0,119,182,0.1)"
CHAT_BORDER       = "rgba(0,150,199,0.4)"
CHAT_TEXT         = "#03045e"
CHAT_SOURCE       = "#0077b6"

INPUT_BG          = "rgba(0,180,216,0.1)"
INPUT_BORDER      = "#0096c7"
INPUT_TEXT        = "#03045e"
INPUT_PLACEHOLDER = "#0096c7"

SUCCESS_BG        = "rgba(0,119,182,0.15)"
INFO_BG           = "rgba(0,180,216,0.1)"
BOX_BORDER        = "#0096c7"

SCROLL_TRACK      = "#caf0f8"
SCROLL_THUMB      = "#0077b6"

FOOTER_TEXT       = "#0077b6"
FOOTER_BORDER     = "rgba(0,119,182,0.3)"

ALL_TEXT          = "#03045e"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    * {{ font-family: 'Poppins', sans-serif; }}

    .stApp {{
        background: linear-gradient(135deg, {BG_FROM} 0%, {BG_TO} 100%);
        min-height: 100vh;
    }}

    .header-box {{
        background: linear-gradient(135deg, {HEADER_FROM}, {HEADER_TO});
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-bottom: 25px;
        border: 1px solid {HEADER_BORDER};
        box-shadow: 0 8px 32px rgba(0,119,182,0.3);
    }}
    .header-box h1 {{
        color: {HEADER_TITLE};
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }}
    .header-box p {{
        color: {HEADER_SUB};
        font-size: 1rem;
        margin: 8px 0 0 0;
    }}

    .stats-bar {{
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
    }}
    .stat-card {{
        flex: 1;
        background: {STAT_BG};
        border: 1px solid {STAT_BORDER};
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }}
    .stat-number {{
        color: {STAT_NUMBER};
        font-size: 1.4rem;
        font-weight: 700;
    }}
    .stat-label {{
        color: {STAT_LABEL};
        font-size: 0.75rem;
        margin-top: 2px;
        font-weight: 500;
    }}

    p, span, label, div, h1, h2, h3, h4, h5, h6 {{
        color: {ALL_TEXT} !important;
    }}

    [data-testid="stFileUploader"] {{
        background: {UPLOAD_BG} !important;
        border-radius: 12px !important;
        border: 2px dashed {UPLOAD_BORDER} !important;
        padding: 15px !important;
    }}
    [data-testid="stFileUploader"] * {{
        color: {ALL_TEXT} !important;
    }}

    [data-testid="stSuccess"] {{
        background: {SUCCESS_BG} !important;
        border: 1px solid {BOX_BORDER} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stSuccess"] * {{
        color: {ALL_TEXT} !important;
    }}

    [data-testid="stInfo"] {{
        background: {INFO_BG} !important;
        border: 1px solid {BOX_BORDER} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stInfo"] * {{
        color: {ALL_TEXT} !important;
    }}

    [data-testid="stChatMessage"] {{
        background: {CHAT_BG} !important;
        border: 1px solid {CHAT_BORDER} !important;
        border-radius: 16px !important;
        margin-bottom: 12px !important;
        padding: 16px !important;
    }}
    [data-testid="stChatMessage"] * {{
        color: {CHAT_TEXT} !important;
    }}

    [data-testid="stChatInput"] {{
        background: {INPUT_BG} !important;
        border: 2px solid {INPUT_BORDER} !important;
        border-radius: 16px !important;
    }}
    [data-testid="stChatInput"] textarea {{
        color: {INPUT_TEXT} !important;
        background: transparent !important;
        font-size: 0.95rem !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {INPUT_PLACEHOLDER} !important;
    }}

    [data-testid="stSpinner"] * {{
        color: {ALL_TEXT} !important;
    }}

    [data-testid="stCaptionContainer"] * {{
        color: {CHAT_SOURCE} !important;
        font-style: italic !important;
        font-size: 0.82rem !important;
    }}

    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {SCROLL_TRACK}; }}
    ::-webkit-scrollbar-thumb {{
        background: {SCROLL_THUMB};
        border-radius: 3px;
    }}

    .footer {{
        text-align: center;
        color: {FOOTER_TEXT} !important;
        font-size: 0.8rem;
        margin-top: 30px;
        padding: 15px;
        border-top: 1px solid {FOOTER_BORDER};
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1> Document Q&A Chatbot</h1>
    <p>Upload any PDF and ask questions from it using AI</p>
</div>
""", unsafe_allow_html=True)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


uploaded_file = st.file_uploader("📂 Upload your PDF here", type="pdf")

if uploaded_file is not None:
    if st.session_state.pdf_name != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with st.spinner("🌊 Reading and indexing your PDF..."):
            st.session_state.vectorstore = process_pdf(tmp_path)
            st.session_state.chat_history = []
            st.session_state.pdf_name = uploaded_file.name

        st.success(f"✅ {uploaded_file.name} is ready! Ask your questions below.")


    st.markdown(f"""
    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-number">📄</div>
            <div class="stat-label">{uploaded_file.name[:20]}</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(st.session_state.chat_history)}</div>
            <div class="stat-label">Questions Asked</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">🤖</div>
            <div class="stat-label">AI Powered</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    for chat in st.session_state.chat_history:
        with st.chat_message("user", avatar="🧑"):
            st.write(chat["question"])
        with st.chat_message("assistant", avatar="🌊"):
            st.write(chat["answer"])
            st.caption(f"📖 Source: Page(s) {', '.join(chat['pages'])}")


    question = st.chat_input("💬 Ask a question about your document...")

    if question:
        with st.chat_message("user", avatar="🧑"):
            st.write(question)
        with st.chat_message("assistant", avatar="🌊"):
            with st.spinner("🔍 Searching your document..."):
                answer, pages = get_answer(
                    st.session_state.vectorstore, question
                )
            st.write(answer)
            st.caption(f"📖 Source: Page(s) {', '.join(pages)}")

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "pages": pages
        })

else:
    if st.session_state.vectorstore is not None:
        st.session_state.vectorstore = None
        st.session_state.pdf_name = None
    st.info("👆 Please upload a PDF file to get started.")


st.markdown("""
<div class="footer">
     RAG Document Chatbot &nbsp;|&nbsp; Built with LangChain, FAISS & Groq &nbsp;|&nbsp; AI Powered
</div>
""", unsafe_allow_html=True)