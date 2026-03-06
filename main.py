"""
main.py  —  Vision & Memory Pipeline
No bson/ObjectId imports — queries MongoDB by plain string doc_id field.
"""
import base64
import io
import json
import os
import uuid
from datetime import datetime, timezone

import face_recognition
import numpy as np
import faiss
from PIL import Image
from pymongo import MongoClient
from openai import OpenAI
import streamlit as st

DIM             = 128
FAISS_THRESHOLD = 0.9
TOP_K           = 3

@st.cache_resource
def _openai():
    return OpenAI()

@st.cache_resource
def _get_col():
    uri = os.environ.get(
        "MONGO_URI",
        "mongodb+srv://chandansrinethvickey_db_user:test1234@cluster0.5cahbo9.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
    )
    return MongoClient(uri)["vision_memory"]["people"]

@st.cache_resource
def _get_faiss():
    col = _get_col()
    bad = col.delete_many({"$or": [
        {"name": {"$in": ["", "unknown", None]}},
        {"embedding": {"$exists": False}},
        {"embedding": []},
    ]})
    if bad.deleted_count:
        print(f"[main] Cleaned {bad.deleted_count} corrupt doc(s).")
    index = faiss.IndexFlatL2(DIM)
    ids = []
    for doc in col.find({"embedding": {"$exists": True}, "name": {"$nin": ["", None, "unknown"]}}):
        vec = np.array(doc["embedding"], dtype="float32")
        index.add(vec.reshape(1, -1))
        ids.append(doc.get("doc_id") or str(doc["_id"]))
    print(f"[main] FAISS ready — {index.ntotal} profile(s).")
    return {"index": index, "ids": ids}

def get_people_col():
    return _get_col()

def _crop_face(image_path, location):
    img = Image.open(image_path).convert("RGB")
    top, right, bottom, left = location
    pad_y = int((bottom - top) * 0.25)
    pad_x = int((right - left) * 0.25)
    top    = max(0, top - pad_y)
    left   = max(0, left - pad_x)
    bottom = min(img.height, bottom + pad_y)
    right  = min(img.width, right + pad_x)
    face   = img.crop((left, top, right, bottom)).resize((224, 224))
    buf    = io.BytesIO()
    face.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def _full_b64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def _same_person(live_b64, stored_b64):
    try:
        res = _openai().chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": (
                    "You are a face verification system.\n"
                    "Image 1 = NEW photo. Image 2 = STORED reference.\n\n"
                    "Are these the SAME person?\n"
                    "Compare ONLY: face shape, eyes, nose, mouth, jawline, skin tone.\n"
                    "IGNORE: hair, glasses, lighting, expression, background.\n"
                    "Be strict — only true if genuinely sure.\n\n"
                    'JSON only: {"same_person": true/false, "confidence": "high/medium/low", "reason": "one sentence"}'
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{live_b64}", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{stored_b64}", "detail": "high"}},
            ]}],
            response_format={"type": "json_object"},
            max_tokens=150,
        )
        data = json.loads(res.choices[0].message.content)
        same = bool(data.get("same_person", False))
        confidence = data.get("confidence", "low")
        print(f"[main] GPT-4o → same={same} conf={confidence} | {data.get('reason','')}")
        return same and confidence in ("high", "medium")
    except Exception as e:
        print(f"[main] GPT-4o error: {e}")
        return False

def _vj(img_b64, prompt):
    try:
        res = _openai().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt + "\nJSON only. No markdown."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]}],
            response_format={"type": "json_object"},
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"[main] attr error: {e}")
        return {}

def _detect_attrs(img_b64):
    return {
        "glasses":      bool(_vj(img_b64, '{"glasses": true/false} — wearing glasses?').get("glasses", False)),
        "emotion":      _vj(img_b64, '{"emotion": "..."} — primary emotion?').get("emotion", "neutral"),
        "age_estimate": _vj(img_b64, '{"age_estimate": "25-30"} — estimated age range?').get("age_estimate", "unknown"),
    }

def _recognise(embedding, live_face_b64):
    state = _get_faiss()
    index = state["index"]
    ids   = state["ids"]
    if index.ntotal == 0:
        print("[main] FAISS empty.")
        return "", ""
    k    = min(TOP_K, index.ntotal)
    vec  = np.array(embedding, dtype="float32").reshape(1, -1)
    D, I = index.search(vec, k)
    candidates = []
    for dist, pos in zip(D[0].tolist(), I[0].tolist()):
        dist, pos = float(dist), int(pos)
        if dist <= FAISS_THRESHOLD and pos < len(ids):
            candidates.append((dist, ids[pos]))
            print(f"[main] FAISS candidate dist={dist:.4f}")
    if not candidates:
        print("[main] No candidates — unknown person.")
        return "", ""
    candidates.sort(key=lambda x: x[0])
    for dist, doc_id in candidates:
        doc = _get_col().find_one({"doc_id": doc_id})
        if not doc:
            continue
        stored_name = (doc.get("name") or "").strip()
        stored_face = doc.get("face_b64", "")
        if not stored_name or not stored_face:
            continue
        if _same_person(live_face_b64, stored_face):
            print(f"[main] ✅ Confirmed: '{stored_name}'")
            return stored_name, doc_id
        print(f"[main] ❌ Rejected: '{stored_name}'")
    print("[main] All rejected — unknown.")
    return "", ""

def _greeting(name, emotion, age, glasses, is_new):
    parts = [f"Hello, {name}! 👋"]
    if emotion and emotion not in ("", "unknown", "neutral"):
        parts.append(f"You look {emotion} today.")
    if age and age not in ("", "unknown"):
        parts.append(f"You appear to be around {age} years old.")
    if glasses:
        parts.append("I see you're wearing glasses.")
    parts.append("Nice to meet you — I'll remember you! ✨" if is_new else "Welcome back! 🎉")
    return " ".join(parts)

def analyse_image(image_path):
    out = {
        "person": False, "embedding": [], "face_b64": "",
        "glasses": False, "emotion": "unknown", "age_estimate": "unknown",
        "need_name": False, "person_name": "", "greeting_message": "",
    }
    image     = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image, model="hog")
    encodings = face_recognition.face_encodings(image, locations)
    if not encodings:
        out["greeting_message"] = "No face detected in the image."
        return out
    out["person"]    = True
    out["embedding"] = encodings[0].tolist()
    out["face_b64"]  = _crop_face(image_path, locations[0])
    print(f"[main] Face found. FAISS has {_get_faiss()['index'].ntotal} profile(s).")
    out.update(_detect_attrs(_full_b64(image_path)))
    name, doc_id = _recognise(out["embedding"], out["face_b64"])
    if not name:
        out["need_name"] = True
        return out
    _get_col().update_one(
        {"doc_id": doc_id},
        {"$set": {"emotion": out["emotion"], "last_seen": datetime.now(timezone.utc)}},
    )
    out["person_name"]      = name
    out["greeting_message"] = _greeting(name, out["emotion"], out["age_estimate"], out["glasses"], is_new=False)
    return out

def register_and_greet(image_path, embedding, face_b64, name, glasses, emotion, age_estimate):
    name = (name or "").strip()
    if not name or name.lower() == "unknown":
        return {"error": "Please enter a valid name."}
    doc_id = str(uuid.uuid4())
    doc = {
        "doc_id":     doc_id,
        "name":       name,
        "embedding":  embedding,
        "face_b64":   face_b64,
        "glasses":    glasses,
        "emotion":    emotion,
        "age":        age_estimate,
        "created_at": datetime.now(timezone.utc),
        "last_seen":  datetime.now(timezone.utc),
    }
    _get_col().insert_one(doc)
    state = _get_faiss()
    vec   = np.array(embedding, dtype="float32").reshape(1, -1)
    state["index"].add(vec)
    state["ids"].append(doc_id)
    print(f"[main] ✅ Saved '{name}' FAISS total={state['index'].ntotal}")
    return {
        "person": True, "need_name": False,
        "person_name": name, "glasses": glasses,
        "emotion": emotion, "age_estimate": age_estimate,
        "greeting_message": _greeting(name, emotion, age_estimate, glasses, is_new=True),
    }
