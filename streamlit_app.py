"""
streamlit_app.py  —  Vision & Memory frontend
Run:  streamlit run streamlit_app.py

Stage machine
─────────────
  "idle"      → waiting for upload / not yet analysed
  "analysed"  → analyse_image() done; person unknown → show name form
  "done"      → recognised or just registered → show greeting
"""
import os
import tempfile

import streamlit as st
from main import analyse_image, register_and_greet, get_people_col

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Vision & Memory", page_icon="👁️", layout="centered")
st.title("👁️ Vision & Memory")
st.caption("Upload a face photo → recognised instantly, or enter a name to be remembered forever.")

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in {
    "stage":      "idle",
    "file_key":   None,
    "image_path": None,
    "analysis":   None,
    "result":     None,
}.items():
    st.session_state.setdefault(k, v)

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a photo (clear, front-facing face works best)",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded:
    key = (uploaded.name, uploaded.size)
    if key != st.session_state.file_key:
        ext = os.path.splitext(uploaded.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded.getvalue())
        st.session_state.update({
            "file_key":   key,
            "image_path": tmp.name,
            "stage":      "idle",
            "analysis":   None,
            "result":     None,
        })

# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — analyse once per new upload
# ─────────────────────────────────────────────────────────────────────────────
path = st.session_state.image_path

if path and os.path.isfile(path) and st.session_state.stage == "idle":
    with st.spinner("Detecting face, running attributes, checking memory…"):
        analysis = analyse_image(path)

    st.session_state.analysis = analysis

    if not analysis.get("person"):
        st.session_state.stage = "idle"          # no face — let user re-upload
    elif analysis.get("need_name"):
        st.session_state.stage = "analysed"      # unknown → show name form
    else:
        st.session_state.result = analysis
        st.session_state.stage  = "done"         # recognised immediately

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY — image + attribute tiles (always visible once we have data)
# ─────────────────────────────────────────────────────────────────────────────
if path and os.path.isfile(path):
    st.image(path, use_container_width=True)

display = st.session_state.result or st.session_state.analysis
if display:
    if not display.get("person"):
        st.error(display.get("greeting_message", "No face detected."))
    else:
        st.subheader("🔍 Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Person",       "Yes")
        c2.metric("Glasses",      "Yes" if display.get("glasses") else "No")
        c3.metric("Emotion",      display.get("emotion")       or "—")
        c4.metric("Age estimate", display.get("age_estimate")  or "—")
        st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE: analysed → unknown person → name form
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.stage == "analysed":
    a = st.session_state.analysis
    if a and a.get("need_name"):
        st.warning("🤔 I don't recognise this person. Enter their name to save them!")

        with st.form("name_form", clear_on_submit=True):
            entered  = st.text_input("Full name", placeholder="e.g. Alice Smith")
            save_btn = st.form_submit_button("💾 Save & Greet")

        if save_btn:
            name = entered.strip()
            if not name or name.lower() == "unknown":
                st.error("Please enter a valid name.")
            else:
                with st.spinner(f"Saving '{name}' to memory…"):
                    result = register_and_greet(
                        image_path   = path,
                        embedding    = a["embedding"],
                        face_b64     = a["face_b64"],
                        name         = name,
                        glasses      = a["glasses"],
                        emotion      = a["emotion"],
                        age_estimate = a["age_estimate"],
                    )
                if result.get("error"):
                    st.error(result["error"])
                else:
                    st.session_state.result = result
                    st.session_state.stage  = "done"
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE: done → greeting
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.stage == "done":
    res    = st.session_state.result
    is_new = (st.session_state.analysis or {}).get("need_name", False)

    if res and res.get("person_name"):
        label = "✨ Saved & registered!" if is_new else "✅ Recognised"
        st.success(f"{label}: **{res['person_name']}**")
        st.info(res.get("greeting_message", ""))
    else:
        st.warning("Could not identify person — please try again.")

    with st.expander("🛠 Debug"):
        if res:
            st.json({k: v for k, v in res.items()
                     if k not in ("embedding", "face_b64")
                     and v not in ("", False, [])})

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — stored profiles (lazy DB call via get_people_col())
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("🗃️ Stored Profiles")
    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        col  = get_people_col()
        docs = list(
            col.find(
                {"name": {"$nin": ["", None, "unknown"]}},
                {"name": 1, "age": 1, "emotion": 1, "glasses": 1, "last_seen": 1, "_id": 0},
            ).sort("_id", -1).limit(20)
        )
        if not docs:
            st.info("No profiles stored yet.")
        else:
            for doc in docs:
                with st.container(border=True):
                    st.markdown(f"**{doc.get('name', '?')}**")
                    c1, c2, c3 = st.columns(3)
                    c1.caption(f"Age: {doc.get('age') or '—'}")
                    c2.caption(f"Mood: {doc.get('emotion') or '—'}")
                    c3.caption(f"Glasses: {'Yes' if doc.get('glasses') else 'No'}")
                    ls = doc.get("last_seen")
                    if ls:
                        ts = ls.strftime("%d %b %Y %H:%M") if hasattr(ls, "strftime") else str(ls)
                        st.caption(f"Last seen: {ts}")
    except Exception as e:
        st.error(f"Could not load profiles: {e}")
