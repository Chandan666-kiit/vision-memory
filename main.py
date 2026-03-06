import os
import base64
import json
import time
from typing import TypedDict

import face_recognition
import numpy as np

from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from openai import OpenAI

client = OpenAI()
store = InMemoryStore()

# ==================================
# FAISS DIRECTORY
# ==================================

def _faiss_index_dir():
    path = os.environ.get("FAISS_INDEX_DIR", "faiss_index")
    os.makedirs(path, exist_ok=True)
    return path


# ==================================
# LONG TERM MEMORY
# ==================================

class LongTermMemory:

    def __init__(self):

        self.embeddings = OpenAIEmbeddings()

        if os.path.exists(_faiss_index_dir()):
            self.vectorstore = FAISS.load_local(
                _faiss_index_dir(),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vectorstore = FAISS.from_texts(
                ["System initialized"],
                embedding=self.embeddings
            )
            self.vectorstore.save_local(_faiss_index_dir())

    def add_memory(self, text):
        doc = Document(page_content=text)
        self.vectorstore.add_documents([doc])
        self.vectorstore.save_local(_faiss_index_dir())

    def retrieve(self, query):
        docs = self.vectorstore.similarity_search(query, k=5)
        return [d.page_content for d in docs]


memory = LongTermMemory()


# ==================================
# FACE DATABASE
# ==================================

class FaceDB:

    def __init__(self):

        self.path = "face_db.json"

        if os.path.exists(self.path):
            with open(self.path) as f:
                self.db = json.load(f)
        else:
            self.db = {}

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.db, f)

    def add_face(self, name, embedding):

        self.db[name] = embedding.tolist()
        self.save()

    def find_match(self, embedding):

        best_name = None
        best_distance = 1.0

        for name, vec in self.db.items():

            dist = np.linalg.norm(embedding - np.array(vec))

            if dist < best_distance:
                best_distance = dist
                best_name = name

        if best_distance < 0.55:
            return best_name

        return None


face_db = FaceDB()


# ==================================
# STATE
# ==================================

class VisionState(TypedDict):

    image_path: str

    face_embedding: list

    person: bool
    glasses: bool
    emotion: str
    age_estimate: str

    person_name: str
    need_name: bool

    greeting_message: str


# ==================================
# UTIL
# ==================================

def encode_image(path):

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_face_embedding(image_path):

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return None

    return encodings[0]


# ==================================
# FACE RECOGNITION NODE
# ==================================

def detect_face(state: VisionState):

    emb = get_face_embedding(state["image_path"])

    if emb is None:
        state["person"] = False
        return state

    state["person"] = True
    state["face_embedding"] = emb.tolist()

    name = face_db.find_match(emb)

    if name:
        state["person_name"] = name
        state["need_name"] = False
    else:
        state["need_name"] = True

    return state


# ==================================
# VISION CALL
# ==================================

def vision_call(base64_image, prompt, key, default):

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        response_format={"type": "json_object"}
    )

    raw = res.choices[0].message.content

    try:
        parsed = json.loads(raw)
        return parsed.get(key, default)
    except:
        return default


# ==================================
# PARALLEL VISION NODES
# ==================================

def detect_glasses(state):

    img = encode_image(state["image_path"])

    prompt = "Is the person wearing glasses? Return JSON {\"glasses\":true/false}"

    state["glasses"] = vision_call(img, prompt, "glasses", False)

    return state


def detect_emotion(state):

    img = encode_image(state["image_path"])

    prompt = "What emotion does the person show? Return JSON {\"emotion\":\"happy\"}"

    state["emotion"] = vision_call(img, prompt, "emotion", "")

    return state


def detect_age(state):

    img = encode_image(state["image_path"])

    prompt = "Estimate the person's age range. JSON {\"age_estimate\":\"20-30\"}"

    state["age_estimate"] = vision_call(img, prompt, "age_estimate", "")

    return state


# ==================================
# SAVE PERSON
# ==================================

def save_person(state):

    if state["need_name"] and state.get("person_name"):

        emb = np.array(state["face_embedding"])

        face_db.add_face(state["person_name"], emb)

    return state


# ==================================
# GREETING
# ==================================

def greet(state):

    if not state["person"]:
        state["greeting_message"] = "No person detected"
        return state

    name = state.get("person_name") or "there"

    parts = []

    if state["glasses"]:
        parts.append("nice glasses")

    if state["emotion"]:
        parts.append(f"you look {state['emotion']}")

    if state["age_estimate"]:
        parts.append(f"around {state['age_estimate']}")

    extra = ", ".join(parts)

    state["greeting_message"] = f"Hello {name}! {extra}"

    return state


# ==================================
# GRAPH
# ==================================

graph = StateGraph(VisionState)

graph.add_node("detect_face", detect_face)
graph.add_node("detect_glasses", detect_glasses)
graph.add_node("detect_emotion", detect_emotion)
graph.add_node("detect_age", detect_age)
graph.add_node("save_person", save_person)
graph.add_node("greet", greet)

graph.add_edge(START, "detect_face")

graph.add_edge("detect_face", "detect_glasses")
graph.add_edge("detect_face", "detect_emotion")
graph.add_edge("detect_face", "detect_age")

graph.add_edge("detect_glasses", "save_person")
graph.add_edge("detect_emotion", "save_person")
graph.add_edge("detect_age", "save_person")

graph.add_edge("save_person", "greet")

graph.add_edge("greet", END)

app = graph.compile(store=store)


# ==================================
# RUN
# ==================================

def get_initial_state(image_path, name=""):

    return {
        "image_path": image_path,
        "face_embedding": [],
        "person": False,
        "glasses": False,
        "emotion": "",
        "age_estimate": "",
        "person_name": name,
        "need_name": False,
        "greeting_message": ""
    }


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py image.jpg")
        exit()

    img = sys.argv[1]

    state = get_initial_state(img)

    result = app.invoke(state)

    print(result["greeting_message"])
