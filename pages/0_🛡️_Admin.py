import hmac
import streamlit as st

# region <--------- Streamlit Page Configuration --------->

st.set_page_config(
    layout="centered",
    page_title="Admin",
    page_icon="🛡️",
)

# endregion <--------- Streamlit Page Configuration --------->


def unlock_admin_page():
    """Return True if the admin password is correct."""
    entered_password = str(st.session_state.get("admin_password", "")).strip()
    expected_password = "admin123"

    if hmac.compare_digest(entered_password, expected_password):
        st.session_state["admin_password_correct"] = True
        return True

    st.session_state["admin_password_correct"] = False
    return False


if st.session_state.get("admin_password_correct", False):
    st.title("🛡️ Admin")
    st.write("""
    This page is for administrative configuration.

    ### Admin Controls
    - Upload a PDF knowledge base document
    - Configure the RAG source behavior
    - Review administrative details
    """)

    uploaded_document = st.file_uploader(
        "📄 Upload a Document",
        type=["pdf"],
    )

    if uploaded_document is not None:
        st.session_state["uploaded_document"] = uploaded_document
        st.success(f"Loaded: {uploaded_document.name}")
    else:
        st.info("Upload a PDF to enable document Q&A.")
else:
    st.title("🛡️ Admin")

    with st.form("admin_unlock_form"):
        st.text_input(
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
