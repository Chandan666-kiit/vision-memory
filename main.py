import base64
import json
import os
import shutil
import time
from typing import List, Optional, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, END, START
from langgraph.store.memory import InMemoryStore
from openai import OpenAI

client = OpenAI()
store = InMemoryStore()


# ==================================
#  UTIL
# ==================================
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _api_call_with_retry(messages, max_retries=6, base_wait=3.0):
    """API call with rate-limit retry. Returns the parsed JSON string from the model."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            is_rate_limit = (
                getattr(e, "code", None) == "rate_limit_exceeded"
                or getattr(e, "status_code", None) == 429
                or "rate limit" in str(e).lower()
                or "429" in str(e)
            )
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            time.sleep(base_wait * (2 ** attempt))


def _vision_api_call(base64_image: str, prompt: str, parse_key: str, default):
    """Single-image vision call. Returns (value, raw_json_string)."""
    raw = _api_call_with_retry([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }
    ])
    try:
        parsed = json.loads(raw)
        return parsed.get(parse_key, default), raw
    except Exception:
        return default, raw


# ==================================
#  LONG TERM MEMORY (FAISS - text memories)
# ==================================
def _faiss_index_dir():
    if os.environ.get("FAISS_INDEX_DIR"):
        path = os.environ.get("FAISS_INDEX_DIR").strip()
        os.makedirs(path, exist_ok=True)
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")


class LongTermMemory:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        index_dir = _faiss_index_dir()
        index_file = os.path.join(index_dir, "index.faiss")

        if os.path.isfile(index_file):
            self.vectorstore = FAISS.load_local(
                index_dir, self.embeddings, allow_dangerous_deserialization=True,
            )
        else:
            os.makedirs(index_dir, exist_ok=True)
            self.vectorstore = FAISS.from_texts(
                ["System initialized"], embedding=self.embeddings,
            )
            self.vectorstore.save_local(index_dir)

    def add_memory(self, text):
        self.vectorstore.add_documents([Document(page_content=text)])
        self.vectorstore.save_local(_faiss_index_dir())

    def retrieve_memory(self, query, k=10):
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])


memory = LongTermMemory()


# ==================================
#  FACE STORE (saves actual images, uses GPT to compare faces)
# ==================================
_FACE_REGISTRY = "face_registry.json"


def _faces_dir() -> str:
    d = os.path.join(_faiss_index_dir(), "faces")
    os.makedirs(d, exist_ok=True)
    return d


def _registry_path() -> str:
    return os.path.join(_faiss_index_dir(), _FACE_REGISTRY)


def _compare_faces(b64_new: str, b64_stored: str) -> bool:
    """Ask GPT-4o-mini if two face images show the same person."""
    prompt = (
        "Look at these two photos. Are they the SAME person? "
        "Ignore differences in expression, angle, lighting, glasses, clothing, or background. "
        "Focus ONLY on whether the face/identity is the same person. "
        "Return ONLY valid JSON: {\"same_person\": true or false}"
    )
    raw = _api_call_with_retry([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_new}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_stored}"}},
            ],
        }
    ])
    try:
        return json.loads(raw).get("same_person", False) is True
    except Exception:
        return False


class FaceStore:
    """Stores face images on disk. Identifies people by asking GPT to compare faces."""

    def __init__(self):
        self.registry: List[dict] = []
        self._load()

    def _load(self) -> None:
        path = _registry_path()
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.registry = json.load(f)
            except Exception:
                self.registry = []

    def _save(self) -> None:
        os.makedirs(_faiss_index_dir(), exist_ok=True)
        with open(_registry_path(), "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2)

    def add_face(self, name: str, image_path: str) -> None:
        """Copy the face image and register it with the given name."""
        filename = f"{name}_{int(time.time())}.jpg"
        dest = os.path.join(_faces_dir(), filename)
        shutil.copy2(image_path, dest)
        self.registry.append({"name": name.strip(), "file": filename})
        self._save()

    def find_name(self, image_path: str) -> Optional[str]:
        """Compare the new face against all stored faces. Returns name if match found."""
        if not self.registry:
            return None
        b64_new = encode_image(image_path)
        seen_names = set()
        for entry in self.registry:
            name = entry.get("name", "")
            if name in seen_names:
                continue
            seen_names.add(name)
            stored_path = os.path.join(_faces_dir(), entry["file"])
            if not os.path.isfile(stored_path):
                continue
            b64_stored = encode_image(stored_path)
            if _compare_faces(b64_new, b64_stored):
                return name
        return None


face_store = FaceStore()


# ==================================
#  STATE
# ==================================
class VisionState(TypedDict):
    image_path: str
    person: bool
    glasses: bool
    emotion: str
    age_estimate: str
    person_name: str
    response_text: str
    need_name: bool
    greeting_message: str


# ==================================
#  PARALLEL VISION NODES
# ==================================
def detect_person(state: VisionState):
    base64_image = encode_image(state["image_path"])
    prompt = (
        "Look at this image. Is there a clear human face/person visible? "
        "Return ONLY valid JSON, no markdown. Format: {\"person\": true or false}"
    )
    value, _ = _vision_api_call(base64_image, prompt, "person", False)
    return {"person": value}


def detect_glasses(state: VisionState):
    base64_image = encode_image(state["image_path"])
    prompt = (
        "Look at this image. Does the person wear glasses (or sunglasses)? "
        "Return ONLY valid JSON, no markdown. Format: {\"glasses\": true or false}"
    )
    value, _ = _vision_api_call(base64_image, prompt, "glasses", False)
    return {"glasses": value}


def detect_emotion(state: VisionState):
    base64_image = encode_image(state["image_path"])
    prompt = (
        "Look at the person's face in this image. What is the dominant emotion? "
        "Return ONLY valid JSON, no markdown. "
        "Format: {\"emotion\": \"one word or short phrase, e.g. happy, neutral, surprised\"}"
    )
    value, _ = _vision_api_call(base64_image, prompt, "emotion", "")
    return {"emotion": value if isinstance(value, str) else str(value)}


def detect_age(state: VisionState):
    base64_image = encode_image(state["image_path"])
    prompt = (
        "Look at the person in this image. Estimate their age range. "
        "Return ONLY valid JSON, no markdown. "
        "Format: {\"age_estimate\": \"e.g. 20-30, 30-40, or a single number\"}"
    )
    value, _ = _vision_api_call(base64_image, prompt, "age_estimate", "")
    return {"age_estimate": value if isinstance(value, str) else str(value)}


def merge_vision(state: VisionState):
    state["response_text"] = json.dumps(
        {
            "person": state.get("person", False),
            "glasses": state.get("glasses", False),
            "emotion": state.get("emotion", ""),
            "age_estimate": state.get("age_estimate", ""),
        },
        indent=2,
    )
    return state


# ==================================
#  IDENTIFY PERSON
# ==================================
def identify_person(state: VisionState):
    if not state.get("person"):
        return state

    # User submitted their name → save face image + text memory
    raw_name = state.get("person_name")
    if raw_name is not None and str(raw_name).strip():
        name_to_store = str(raw_name).strip()
        memory.add_memory(
            f"Name: {name_to_store} | Glasses: {state.get('glasses', False)} | "
            f"Emotion: {state.get('emotion', '') or ''} | Age: {state.get('age_estimate', '') or ''}"
        )
        face_store.add_face(name_to_store, state["image_path"])
        state["person_name"] = name_to_store
        state["need_name"] = False
        return state

    # Try to recognize by comparing face images
    matched_name = face_store.find_name(state["image_path"])
    if matched_name:
        state["person_name"] = matched_name
        state["need_name"] = False
        return state

    state["need_name"] = True
    state["person_name"] = ""
    return state


# ==================================
#  GREET
# ==================================
def greet(state: VisionState):
    if not state.get("person"):
        state["greeting_message"] = "Please provide a valid person image."
        return state
    name = state.get("person_name") or "there"
    parts = []
    if state.get("glasses"):
        parts.append("Looking sharp with specs")
    else:
        parts.append("No specs today")
    emotion = (state.get("emotion") or "").strip()
    age = (state.get("age_estimate") or "").strip()
    if emotion:
        parts.append(f"you look {emotion}")
    if age:
        parts.append(f"around {age}")
    extra = (" — " + ", ".join(parts)) if parts else ""
    state["greeting_message"] = f"Good morning {name}!{extra}."
    return state


# ==================================
#  BUILD GRAPH
# ==================================
graph = StateGraph(VisionState)

graph.add_node("detect_person", detect_person)
graph.add_node("detect_glasses", detect_glasses)
graph.add_node("detect_emotion", detect_emotion)
graph.add_node("detect_age", detect_age)
graph.add_node("merge_vision", merge_vision)
graph.add_node("identify", identify_person)
graph.add_node("greet", greet)

graph.add_edge(START, "detect_person")
graph.add_edge(START, "detect_glasses")
graph.add_edge(START, "detect_emotion")
graph.add_edge(START, "detect_age")
graph.add_edge("detect_person", "merge_vision")
graph.add_edge("detect_glasses", "merge_vision")
graph.add_edge("detect_emotion", "merge_vision")
graph.add_edge("detect_age", "merge_vision")
graph.add_edge("merge_vision", "identify")
graph.add_edge("identify", "greet")
graph.add_edge("greet", END)

app = graph.compile(store=store)


# ==================================
#  RUN
# ==================================
def get_initial_state(image_path: str = "", person_name: str = ""):
    return {
        "image_path": image_path or "",
        "person": False,
        "glasses": False,
        "emotion": "",
        "age_estimate": "",
        "person_name": person_name or "",
        "response_text": "",
        "need_name": False,
        "greeting_message": "",
    }


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not image_path or not os.path.isfile(image_path):
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    result = app.invoke(get_initial_state(image_path))
    print("Final State:", result)
