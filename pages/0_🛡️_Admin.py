import os
import tempfile
import hmac
import streamlit as st

try:
    from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    PyPDFLoader = None
    WebBaseLoader = None
    Chroma = None
    OpenAIEmbeddings = None
    RecursiveCharacterTextSplitter = None

# region <--------- Streamlit Page Configuration --------->

st.set_page_config(
    layout="centered",
    page_title="Admin",
    page_icon="🛡️",
)

# endregion <--------- Streamlit Page Configuration --------->


def unlock_admin_page():
    """Return True if the admin password is correct."""
    entered_password = st.session_state.get("admin_password", "")
    expected_password = st.secrets.get("admin_password", "")

    if hmac.compare_digest(entered_password, expected_password):
        st.session_state["admin_password_correct"] = True
        return True

    st.session_state["admin_password_correct"] = False
    return False


def load_and_split_pdf(uploaded_file):
    """Load a PDF from a Streamlit uploaded file and return text chunks."""
    if uploaded_file is None:
        return []

    if any(obj is None for obj in [PyPDFLoader, RecursiveCharacterTextSplitter]):
        raise ImportError(
            "LangChain dependencies are not installed. Please install the packages from requirements.txt."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return text_splitter.split_documents(documents)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_and_split_webpage(web_url):
    """Load a web page URL and return text chunks."""
    if not web_url:
        return []

    if any(obj is None for obj in [WebBaseLoader, RecursiveCharacterTextSplitter]):
        raise ImportError(
            "LangChain dependencies are not installed. Please install the packages from requirements.txt."
        )

    loader = WebBaseLoader(web_url)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)


@st.cache_resource
def build_vectorstore(_chunks):
    """Create a Chroma vector store from LangChain document chunks."""
    if not _chunks:
        return None

    if any(obj is None for obj in [Chroma, OpenAIEmbeddings]):
        raise ImportError(
            "LangChain dependencies are not installed. Please install the packages from requirements.txt."
        )

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    return Chroma.from_documents(documents=_chunks, embedding=embeddings)


if st.session_state.get("admin_password_correct", False):
    st.title("🛡️ Admin")
    st.write("""
    This page is used to configure the knowledge base for the RAG chatbot.

    ### Admin Controls
    - Upload a PDF knowledge base document
    - Add a web page as an alternative knowledge source
    - Build and store the vector index for retrieval
    - Prepare the chatbot for grounded Q&A
    """)

    source_type = st.radio(
        "Choose a knowledge source",
        ["📄 PDF Document", "🌐 Web Page"],
        horizontal=True,
    )

    if source_type == "📄 PDF Document":
        uploaded_document = st.file_uploader(
            "📄 Upload a Document",
            type=["pdf"],
        )

        if uploaded_document is not None:
            with st.spinner("Loading and indexing document..."):
                chunks = load_and_split_pdf(uploaded_document)
                vectorstore = build_vectorstore(chunks)
                st.session_state.vectorstore = vectorstore
                st.session_state.chunks_count = len(chunks)
                st.session_state.source_name = uploaded_document.name
                st.success(f"Loaded: {uploaded_document.name}")
                st.success(f"Ready! Indexed {len(chunks)} chunks.")
        else:
            st.info("Upload a PDF to enable document Q&A.")

    else:
        web_url = st.text_input(
            "🌐 Enter a web page URL",
            placeholder="https://example.com/hr-policy",
        )

        if st.button("Index Web Page") and web_url:
            with st.spinner("Loading and indexing web page..."):
                chunks = load_and_split_webpage(web_url)
                vectorstore = build_vectorstore(chunks)
                st.session_state.vectorstore = vectorstore
                st.session_state.chunks_count = len(chunks)
                st.session_state.source_name = web_url
                st.success(f"Loaded web page: {web_url}")
                st.success(f"Ready! Indexed {len(chunks)} chunks.")
        else:
            st.info("Enter a web page URL to enable web-based knowledge retrieval.")

    if "chunks_count" in st.session_state:
        st.metric("Indexed Chunks", st.session_state.chunks_count)
        if "source_name" in st.session_state:
            st.caption(f"Active knowledge source: {st.session_state.source_name}")
else:
    st.title("🛡️ Admin")

    with st.form("admin_unlock_form"):
        admin_password = st.text_input(
            "Admin Password",
            type="password",
            key="admin_password",
        )
        submitted = st.form_submit_button("Unlock")

    if submitted:
        if unlock_admin_page():
            st.rerun()
        else:
            st.error("😕 Password incorrect")
    else:
        st.info("Enter the admin password to access this page.")

    st.stop()
