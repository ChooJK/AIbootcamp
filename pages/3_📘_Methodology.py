import streamlit as st
from helper_functions.utility import check_password

# region <--------- Streamlit Page Configuration --------->

st.set_page_config(
    layout="centered",
    page_title="Methodology",
    page_icon="📘",
)

# Do not continue if check_password is not True.
if not check_password():
    st.stop()

# endregion <--------- Streamlit Page Configuration --------->

st.title("📘 Methodology")

st.write("""
This page outlines the data flow and implementation details of the chatbot application.

### Data Flow
- 📄 **Document Upload** — Users upload a PDF document through the sidebar.
- 🔍 **Text Extraction** — The PDF content is loaded and split into smaller chunks.
- 🧠 **Embedding & Vector Storage** — Chunks are embedded and stored in a Chroma vector store.
- 💬 **Query Handling** — User prompts are matched against the indexed chunks to build a relevant context.
- 🤖 **Response Generation** — The model generates answers using the retrieved context.

### Implementation Details
- **Streamlit** is used for the web app interface.
- **LangChain** handles document loading, chunking, and retrieval.
- **OpenAI embeddings** are used to create vector representations.
- **Chroma** stores the document vectors for similarity search.
""")
