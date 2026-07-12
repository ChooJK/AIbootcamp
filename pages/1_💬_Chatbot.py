import os
import tempfile
import streamlit as st
from helper_functions.llm import get_completion_stream, count_tokens
from helper_functions.utility import check_password

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    PyPDFLoader = None
    Chroma = None
    OpenAIEmbeddings = None
    RecursiveCharacterTextSplitter = None


def load_and_split(uploaded_file):
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


@st.cache_resource
def build_vectorstore(_chunks):
    """Create a Chroma vector store from LangChain document chunks."""
    if not _chunks:
        return None

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    return Chroma.from_documents(documents=_chunks, embedding=embeddings)


def retrieve_context(vectorstore, query, k=4):
    """Retrieve and combine the most relevant document chunks for a query."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    documents = retriever.invoke(query)
    return "\n\n---\n\n".join(doc.page_content for doc in documents)


def build_rag_system_prompt(context):
    """Create a system prompt that instructs the model to answer only from the provided context."""
    return (
        "You are a helpful assistant. Answer ONLY from the provided context. "
        "If the answer is not present in the context, say: "
        '"I could\'t find that information in the uploaded document."\n\n'
        f"Context:\n{context}"
    )


system_prompt_no_doc = "You are a helpful assistant."


def build_rag_vector_store(uploaded_files):
    """Build a Chroma vector store from uploaded PDF files."""
    if not uploaded_files:
        return None

    if any(obj is None for obj in [PyPDFLoader, Chroma, OpenAIEmbeddings, RecursiveCharacterTextSplitter]):
        raise ImportError(
            "LangChain dependencies are not installed. Please install the packages from requirements.txt."
        )

    documents = []
    temp_paths = []

    try:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            temp_paths.append(temp_path)
            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())

        if not documents:
            raise ValueError("No text could be read from the uploaded PDFs.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(documents)

        persist_directory = os.path.join(os.getcwd(), "chroma_db")
        os.makedirs(persist_directory, exist_ok=True)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = Chroma.from_documents(
            documents=split_documents,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        return vector_store
    finally:
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# region <--------- Streamlit Page Configuration --------->

st.set_page_config(
    layout="centered",
    page_title="Chatbot",
    page_icon="💬",
)

# Do not continue if check_password is not True.
if not check_password():
    st.stop()

# endregion <--------- Streamlit Page Configuration --------->

st.title("💬 AI Chatbot")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    uploaded_document = st.file_uploader(
        "📄 Upload a Document",
        type=["pdf"],
    )

    if uploaded_document is not None:
        st.success(f"Loaded: {uploaded_document.name}")
        chunks = load_and_split(uploaded_document)
        vectorstore = build_vectorstore(chunks)
        st.session_state.vectorstore = vectorstore
        st.session_state.chunks_count = len(chunks)
        st.success(f"Ready! Indexed {len(chunks)} chunks.")
    else:
        st.info("Upload a PDF to enable document Q&A.")

    k_value = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=4)

    st.header("⚙️ Settings")
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant.",
        height=120,
    )

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.get("messages"):
        total_tokens = sum(
            count_tokens(m["content"]) for m in st.session_state.messages
        )
        st.metric("Estimated Tokens Used", total_tokens)

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_vector_store" not in st.session_state:
    st.session_state.rag_vector_store = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        vector_store = st.session_state.get("rag_vector_store")
        augmented_prompt = system_prompt

        if vector_store is not None:
            try:
                retrieved_docs = vector_store.similarity_search(user_input, k=4)
                if retrieved_docs:
                    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                    augmented_prompt = (
                        f"{system_prompt}\n\n"
                        f"Use the following context when it is relevant:\n{context_text}"
                    )
            except Exception as exc:
                st.warning(f"Retrieval failed: {exc}")

        full_messages = [{"role": "system", "content": augmented_prompt}] + st.session_state.messages
        response = st.write_stream(get_completion_stream(full_messages))

    st.session_state.messages.append({"role": "assistant", "content": response})
