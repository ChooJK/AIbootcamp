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
This page explains how the chatbot works end to end, including the data pipeline, retrieval logic, and the implementation choices that make the experience feel conversational and grounded in source material.

### Overview
The chatbot follows a Retrieval-Augmented Generation (RAG) pattern. Instead of relying only on the model's pretrained knowledge, it first searches a knowledge base for the most relevant passages and then uses those passages as context when generating the final answer.

### End-to-End Data Flow
1. **User Interaction**  
   The user enters a question in the chat interface. This question becomes the starting point for the retrieval process.

2. **Query Embedding**  
   The question is converted into a vector embedding using an embedding model. This allows the system to compare the question semantically with the stored document chunks.

3. **Similarity Search**  
   The system searches the vector database for the chunks that are most similar to the user's question. These chunks are treated as the most relevant evidence.

4. **Context Assembly**  
   The retrieved chunks are combined into a compact context window. This context is passed to the language model alongside the original user question.

5. **Answer Generation**  
   The LLM uses the retrieved passages to generate a response that is grounded in the source material rather than guessed from memory alone.

6. **Response Display**  
   The answer is returned to the user in the Streamlit interface, often with the option to track token usage or inspect the flow of the request.

### Implementation Details
- **Streamlit** provides the web application interface for chat input, page navigation, and user interaction.
- **Python** is used to coordinate the app logic, file handling, and request flow.
- **OpenAI embeddings** convert text into numerical vectors that capture semantic meaning.
- **A vector store** such as Chroma stores the embedded document chunks and supports similarity-based retrieval.
- **A language model** such as GPT-4o-mini generates the final response using the retrieved context.
- **Chunking** is important because it breaks large documents into smaller, more relevant pieces that can be searched efficiently.

### Why This Works Well
This design improves reliability because the model is not forced to answer from memory alone. Instead, it uses retrieved evidence, which helps reduce hallucinations and makes the output more relevant to the provided documents.

### Generic RAG Flowchart
```mermaid
flowchart LR
    A[Documents] --> B[Chunking]
    B --> C[Embedding]
    C --> D[Vector Store]
    D --> E[User Question]
    E --> F[Embed Question]
    F --> G[Similarity Search]
    G --> H[Retrieve Relevant Chunks]
    H --> I[Prompt + Context]
    I --> J[LLM Response]
    J --> K[Answer to User]
```

### Ingestion Pipeline Flowchart
```mermaid
flowchart TD
    A[Upload or Load Source Documents] --> B[Extract Text]
    B --> C[Clean and Normalize Text]
    C --> D[Split into Chunks]
    D --> E[Generate Embeddings]
    E --> F[Store in Vector Database]
    F --> G[Ready for Retrieval]
```
""")
