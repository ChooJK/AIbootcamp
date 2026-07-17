import streamlit as st
from helper_functions.utility import check_password

# region <--------- Streamlit Page Configuration --------->

st.set_page_config(
    layout="centered",
    page_title="About",
    page_icon="ℹ️",
)

# Do not continue if check_password is not True.
if not check_password():
    st.stop()

# endregion <--------- Streamlit Page Configuration --------->

st.title("ℹ️ About This App")

st.write("""
This is an AI Chatbot built with **Streamlit** and **OpenAI**.

### Project Scope
- 📄 **Document AI Assistant**

### Problem Statement
Employees struggle to find accurate HR-related information because it is spread across multiple and lengthy documents, as well as web pages. This often results in inconsistent answers and wasted time searching for the right resources. As a result, HR team spends significant time responding to questions where the answers can be found these documents and web pages instead of focusing on higher-value tasks.

### Solution
This app is enabled with a Retrieval-Augmented Generation (RAG) application that gives employees instant, accurate answers strictly based on the knowledge base. When the knowledge base is added, the sources are automatically chunked and embedded into a vector store. When an employee prompts a question from this knowledge base, the system:

- Converts the question into a semantic embedding
- Retrieves the most relevant passages from the vector store
- Passes those passages to an LLM as context
- Returns an answer grounded only in the retrieved content

### Features
- 💬 **Conversational AI** — Chat naturally with GPT-4o-mini
- 🔒 **Password Protected** — Secure access with a shared password
- 🔄 **Streaming Responses** — See the AI's reply as it's generated
- 📊 **Token Tracking** — Monitor estimated token usage in the sidebar

### Tech Stack
| Component | Tool |
|---|---|
| UI Framework | [Streamlit](https://streamlit.io/) |
| Language Model | [OpenAI GPT-4o-mini](https://openai.com/) |
| Token Counting | [tiktoken](https://github.com/openai/tiktoken) |
| Backend | Python 3.10+ |

### Disclaimer
*This application is for educational and demonstration purposes. The information provided by the AI should not be taken as professional advice.*
""")
