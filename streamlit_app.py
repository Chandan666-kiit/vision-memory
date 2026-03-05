"""
Streamlit frontend for the vision + long-term memory (FAISS) pipeline.
Run: streamlit run streamlit_app.py
"""
import tempfile
import os

import streamlit as st

from main import app, get_initial_state, memory

st.set_page_config(
    page_title="Vision & Memory",
    page_icon="👁️",
    layout="centered",
)

st.title("👁️ Vision & Memory")
st.caption(
    "Upload an image → we run person, glasses, emotion & age detection in parallel, "
    "then recognize or learn your name and greet you."
)

# Persist uploaded image path and last result across reruns
if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "need_name_submitted" not in st.session_state:
    st.session_state.need_name_submitted = False
if "submitted_name" not in st.session_state:
    st.session_state.submitted_name = ""
if "last_run_path" not in st.session_state:
    st.session_state.last_run_path = None
if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None

# ---------------------------------------------------------------------------
# Image upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload an image (person face works best)",
    type=["png", "jpg", "jpeg", "webp"],
    help="Image is analyzed in parallel (person, glasses, emotion, age), then we recognize or ask your name.",
)

if uploaded is not None:
    file_id = (uploaded.name, uploaded.size)
    is_new_upload = file_id != st.session_state.get("uploaded_file_id")
    if is_new_upload:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.getvalue())
            st.session_state.uploaded_path = tmp.name
        st.session_state.uploaded_file_id = file_id
        st.session_state.last_result = None
        st.session_state.need_name_submitted = False
        st.session_state.submitted_name = ""
        st.session_state.last_run_path = None
    # else: same file as before, keep existing uploaded_path

# ---------------------------------------------------------------------------
# Run pipeline: either first run (with image) or resume with name
# ---------------------------------------------------------------------------
def run_pipeline(image_path: str, person_name: str = ""):
    state = get_initial_state(image_path=image_path, person_name=person_name)
    return app.invoke(state)

# If we have a path, run (first time or after name submit)
if st.session_state.uploaded_path and os.path.isfile(st.session_state.uploaded_path):
    path = st.session_state.uploaded_path
    # Second run: user just submitted their name — use it exactly
    if st.session_state.need_name_submitted and st.session_state.last_result:
        name_to_use = (st.session_state.get("submitted_name") or "").strip()
        if not name_to_use:
            st.session_state.need_name_submitted = False
        else:
            with st.spinner("Adding you to memory and greeting…"):
                result = run_pipeline(path, person_name=name_to_use)
            st.session_state.last_result = result
            st.session_state.need_name_submitted = False
            st.session_state.submitted_name = ""
            st.rerun()
    # First run: no result yet (new upload or new session)
    elif st.session_state.last_result is None:
        with st.spinner("Running parallel vision (person, glasses, emotion, age) and checking memory…"):
            result = run_pipeline(path)
        st.session_state.last_result = result
        st.session_state.last_run_path = path

# ---------------------------------------------------------------------------
# Display: image, analysis, name form (if need_name), greeting, state
# ---------------------------------------------------------------------------
result = st.session_state.last_result

if st.session_state.uploaded_path and os.path.isfile(st.session_state.uploaded_path):
    st.subheader("Image")
    st.image(st.session_state.uploaded_path, use_container_width=True)

if result is not None:
    st.subheader("Analysis (parallel: person, glasses, emotion, age)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Person", "Yes" if result.get("person") else "No")
    with col2:
        st.metric("Glasses", "Yes" if result.get("glasses") else "No")
    with col3:
        st.metric("Emotion", result.get("emotion") or "—")
    with col4:
        st.metric("Age estimate", result.get("age_estimate") or "—")
    if result.get("response_text"):
        with st.expander("Raw model response (JSON)"):
            st.code(result["response_text"], language="json")

    if result.get("need_name"):
        st.info("We don’t recognize you yet. Enter your name below and click **Submit name** so we can remember you.")
        with st.form("name_form"):
            name_from_form = st.text_input("Your name", placeholder="e.g. Alex")
            submitted = st.form_submit_button("Submit name")
        if submitted and name_from_form and name_from_form.strip():
            st.session_state.need_name_submitted = True
            st.session_state.submitted_name = name_from_form.strip()
            st.rerun()
    else:
        if result.get("person_name"):
            st.success(f"Recognized: **{result['person_name']}**")

    if result.get("greeting_message"):
        st.subheader("Greeting")
        st.write(result["greeting_message"])

    with st.expander("Full state (debug)"):
        st.json({k: v for k, v in result.items() if v not in ("", False)})

# Sidebar: optional memory peek
with st.sidebar:
    st.subheader("Memory")
    if st.button("Preview stored memories (last 5)"):
        preview = memory.retrieve_memory("Name:", k=5)
        st.text(preview or "(none yet)")
