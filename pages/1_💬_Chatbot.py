import streamlit as st
from helper_functions.llm import get_completion_stream, count_tokens
from helper_functions.utility import check_password


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
        if "vectorstore" in st.session_state:
            try:
                context = retrieve_context(st.session_state.vectorstore, user_input)
                system_message = build_rag_system_prompt(context)
            except Exception as exc:
                st.warning(f"Retrieval failed: {exc}")
                system_message = system_prompt_no_doc
        else:
            system_message = system_prompt_no_doc

        full_messages = [{"role": "system", "content": system_message}] + st.session_state.messages
        response = st.write_stream(get_completion_stream(full_messages))

    st.session_state.messages.append({"role": "assistant", "content": response})
