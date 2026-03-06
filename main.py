"""
main.py  —  Vision & Memory Pipeline

GOLDEN RULE
───────────
GPT-4o compares the LIVE face crop against every STORED face crop from MongoDB.
It is the sole judge of identity. FAISS only pre-filters candidates cheaply.

Flow
────
1. face_recognition  →  128-d embedding + 224x224 face crop (JPEG base64)
2. GPT-4o-mini       →  glasses / emotion / age
3. FAISS             →  top-3 nearest embeddings (generous pre-filter)
4. GPT-4o            →  compare live face vs each stored face image
                         "same_person true + high/medium confidence" = match
5. Match   → greet by name, update MongoDB last_seen
   No match → need_name=True (Streamlit shows name form)
6. Name entered → save to MongoDB + add to live FAISS index

IMPORTANT: Nothing touches MongoDB at module import time.
           All DB access is inside @st.cache_resource functions,
           which Streamlit only calls after its runtime is fully started.
"""

import base64
import io
import json
import os
from datetime import datetime, timezone

import bson
import face_recognition
import numpy as np
import faiss
from PIL import Image
from pymongo import MongoClient
from openai import OpenAI
import streamlit as st

DIM             = 128
FAISS_THRESHOLD = 0.9   # generous — GPT-4o makes the real decision
TOP_K           = 3


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETONS  —  created once per Streamlit session, never at import time
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _openai() -> OpenAI:
    return OpenAI()


@st.cache_resource
def _get_col():
    """Return the MongoDB 'people' collection. Called lazily, never at import."""
    uri = os.environ.get(
        "MONGO_URI",
        "mongodb+srv://chandansrinethvickey_db_user:test1234@cluster0.5cahbo9.mongodb.net/"
        "?retryWrites=true&w=majority&appName=Cluster0",
    )
    client = MongoClient(uri)
    return client["vision_memory"]["people"]


@st.cache_resource
def _get_faiss() -> dict:
    """
    Build FAISS index from all valid MongoDB profiles.
    Cleans up corrupt docs first (empty name, no embedding, no face_b64).
    Called lazily — only when first needed inside analyse_image().
    """
    col = _get_col()

    # Remove corrupt docs saved by older broken runs
    bad = col.delete_many({
        "$or": [
            {"name": {"$in": ["", "unknown", None]}},
            {"embedding": {"$exists": False}},
            {"embedding": []},
            {"face_b64": {"$exists": False}},
            {"face_b64": ""},
        ]
    })
    if bad.deleted_count:
        print(f"[main] 🧹 Removed {bad.deleted_count} corrupt profile(s).")

    index = faiss.IndexFlatL2(DIM)
    ids: list[str] = []

    for doc in col.find(
        {"embedding": {"$exists": True},
         "face_b64":  {"$exists": True},
         "name":      {"$nin": ["", None, "unknown"]}},
    ):
        vec = np.array(doc["embedding"], dtype="float32")
        index.add(vec.reshape(1, -1))
        ids.append(str(doc["_id"]))

    print(f"[main] FAISS ready — {index.ntotal} valid profile(s).")
    return {"index": index, "ids": ids}


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _crop_face(image_path: str, location: tuple) -> str:
    """Crop face with padding, resize to 224x224, return base64 JPEG string."""
    img              = Image.open(image_path).convert("RGB")
    top, right, bottom, left = location
    pad_y = int((bottom - top)  * 0.25)
    pad_x = int((right  - left) * 0.25)
    top    = max(0,          top    - pad_y)
    left   = max(0,          left   - pad_x)
    bottom = min(img.height, bottom + pad_y)
    right  = min(img.width,  right  + pad_x)
    face   = img.crop((left, top, right, bottom)).resize((224, 224))
    buf    = io.BytesIO()
    face.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _full_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# GPT-4o FACE COMPARISON  —  THE GOLDEN RULE
# ─────────────────────────────────────────────────────────────────────────────

def _same_person(live_b64: str, stored_b64: str) -> bool:
    """
    Ask GPT-4o: are the two face images the same person?
    Returns True only on high or medium confidence.
    """
    try:
        res = _openai().chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a face verification system.\n"
                            "Image 1 = NEW photo.  Image 2 = STORED reference.\n\n"
                            "Question: Are these the SAME person?\n\n"
                            "Rules:\n"
                            "- Compare ONLY facial structure: face shape, eyes, nose, "
                            "mouth, jawline, cheekbones, skin tone.\n"
                            "- IGNORE: hair, glasses, lighting, expression, background.\n"
                            "- Be strict. Only say true if you are genuinely sure.\n\n"
                            "Respond with JSON only (no markdown):\n"
                            '{"same_person": true/false, '
                            '"confidence": "high/medium/low", '
                            '"reason": "one short sentence"}'
                        ),
                    },
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{live_b64}",
                                   "detail": "high"}},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{stored_b64}",
                                   "detail": "high"}},
                ],
            }],
            response_format={"type": "json_object"},
            max_tokens=150,
        )
        data       = json.loads(res.choices[0].message.content)
        same       = bool(data.get("same_person", False))
        confidence = data.get("confidence", "low")
        reason     = data.get("reason", "")
        print(f"[main] GPT-4o → same={same}  conf={confidence}  | {reason}")
        return same and confidence in ("high", "medium")
    except Exception as e:
        print(f"[main] GPT-4o error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ATTRIBUTE DETECTION  (GPT-4o-mini)
# ─────────────────────────────────────────────────────────────────────────────

def _vj(img_b64: str, prompt: str) -> dict:
    try:
        res = _openai().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": [
                {"type": "text",  "text": prompt + "\nJSON only. No markdown."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]}],
            response_format={"type": "json_object"},
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"[main] attr error: {e}")
        return {}


def _detect_attrs(img_b64: str) -> dict:
    return {
        "glasses":      bool(_vj(img_b64, '{"glasses": true/false} — wearing glasses?').get("glasses", False)),
        "emotion":      _vj(img_b64, '{"emotion": "..."} — primary emotion on face?').get("emotion", "neutral"),
        "age_estimate": _vj(img_b64, '{"age_estimate": "25-30"} — estimated age range?').get("age_estimate", "unknown"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RECOGNITION CORE
# ─────────────────────────────────────────────────────────────────────────────

def _recognise(embedding: list, live_face_b64: str) -> tuple[str, str]:
    """
    Step 1: FAISS finds top-K nearest neighbours.
    Step 2: GPT-4o visually confirms each candidate.
    Returns (name, mongo_id) or ("", "").
    """
    state = _get_faiss()
    index = state["index"]
    ids   = state["ids"]

    if index.ntotal == 0:
        print("[main] FAISS empty — no profiles stored yet.")
        return "", ""

    k    = min(TOP_K, index.ntotal)
    vec  = np.array(embedding, dtype="float32").reshape(1, -1)
    D, I = index.search(vec, k)

    candidates = []
    for dist, pos in zip(D[0].tolist(), I[0].tolist()):
        dist = float(dist)
        pos  = int(pos)
        if dist <= FAISS_THRESHOLD and pos < len(ids):
            candidates.append((dist, ids[pos]))
            print(f"[main] FAISS candidate: dist={dist:.4f}  id={ids[pos]}")

    if not candidates:
        print(f"[main] No FAISS candidates ≤ {FAISS_THRESHOLD} — unknown person.")
        return "", ""

    candidates.sort(key=lambda x: x[0])  # closest first

    for dist, mongo_id in candidates:
        doc = _get_col().find_one({"_id": bson.ObjectId(mongo_id)})
        if not doc:
            continue

        stored_name = (doc.get("name") or "").strip()
        stored_face = doc.get("face_b64", "")

        if not stored_name or not stored_face:
            print(f"[main] Skipping doc {mongo_id} — missing name or face image.")
            continue

        # ── GOLDEN RULE: GPT-4o compares both face images ───────────────────
        if _same_person(live_face_b64, stored_face):
            print(f"[main] ✅ Confirmed match: '{stored_name}'")
            return stored_name, mongo_id
        else:
            print(f"[main] ❌ GPT-4o rejected candidate '{stored_name}'")

    print("[main] All candidates rejected — unknown person.")
    return "", ""


# ─────────────────────────────────────────────────────────────────────────────
# GREETING BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _greeting(name: str, emotion: str, age: str, glasses: bool, is_new: bool) -> str:
    parts = [f"Hello, {name}! 👋"]
    if emotion and emotion not in ("", "unknown", "neutral"):
        parts.append(f"You look {emotion} today.")
    if age and age not in ("", "unknown"):
        parts.append(f"You appear to be around {age} years old.")
    if glasses:
        parts.append("I see you're wearing glasses.")
    parts.append("Nice to meet you — I'll remember you! ✨" if is_new else "Welcome back! 🎉")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def analyse_image(image_path: str) -> dict:
    """
    Pass 1 — detect face, attributes, attempt recognition.

    Returns dict:
      person, embedding, face_b64, glasses, emotion, age_estimate,
      need_name (True = unknown person), person_name, greeting_message
    """
    out = {
        "person": False, "embedding": [], "face_b64": "",
        "glasses": False, "emotion": "unknown", "age_estimate": "unknown",
        "need_name": False, "person_name": "", "greeting_message": "",
    }

    # 1. Detect face
    image     = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image, model="hog")
    encodings = face_recognition.face_encodings(image, locations)

    if not encodings:
        out["greeting_message"] = "No face detected in the image."
        print("[main] No face found.")
        return out

    out["person"]    = True
    out["embedding"] = encodings[0].tolist()
    out["face_b64"]  = _crop_face(image_path, locations[0])
    print(f"[main] Face found. FAISS has {_get_faiss()['index'].ntotal} profile(s).")

    # 2. Attributes
    out.update(_detect_attrs(_full_b64(image_path)))

    # 3. Recognise
    name, mongo_id = _recognise(out["embedding"], out["face_b64"])

    if not name:
        out["need_name"] = True
        print("[main] → Unknown person. Showing name form.")
        return out

    # 4. Known person — update last_seen in DB
    _get_col().update_one(
        {"_id": bson.ObjectId(mongo_id)},
        {"$set": {"emotion": out["emotion"],
                  "last_seen": datetime.now(timezone.utc)}},
    )
    out["person_name"]      = name
    out["greeting_message"] = _greeting(
        name, out["emotion"], out["age_estimate"], out["glasses"], is_new=False)
    print(f"[main] → Recognised: '{name}'")
    return out


def register_and_greet(
    image_path: str, embedding: list, face_b64: str,
    name: str, glasses: bool, emotion: str, age_estimate: str,
) -> dict:
    """
    Pass 2 — save new person to MongoDB + live FAISS index.
    Only called after user submits name in Streamlit form.
    """
    name = (name or "").strip()
    if not name or name.lower() == "unknown":
        return {"error": "Please enter a valid name."}

    doc = {
        "name":       name,
        "embedding":  embedding,
        "face_b64":   face_b64,
        "glasses":    glasses,
        "emotion":    emotion,
        "age":        age_estimate,
        "created_at": datetime.now(timezone.utc),
        "last_seen":  datetime.now(timezone.utc),
    }
    inserted = _get_col().insert_one(doc)

    state = _get_faiss()
    vec   = np.array(embedding, dtype="float32").reshape(1, -1)
    state["index"].add(vec)
    state["ids"].append(str(inserted.inserted_id))

    print(f"[main] ✅ Saved '{name}'  _id={inserted.inserted_id}  "
          f"FAISS total={state['index'].ntotal}")

    return {
        "person": True, "need_name": False,
        "person_name": name, "glasses": glasses,
        "emotion": emotion, "age_estimate": age_estimate,
        "greeting_message": _greeting(
            name, emotion, age_estimate, glasses, is_new=True),
    }


def get_people_col():
    """Use this in streamlit_app.py instead of importing people_col directly."""
    return _get_col()