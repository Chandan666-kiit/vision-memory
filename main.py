import os
import base64
import json
import time
from typing import TypedDict

from langgraph.graph import StateGraph, END, START
from langgraph.store.memory import InMemoryStore
from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

client = OpenAI()
store = InMemoryStore()


# ==================================
#  LONG TERM MEMORY (FAISS)
# ==================================
def _faiss_index_dir():
    """Absolute path to faiss_index. Use FAISS_INDEX_DIR env for persistent volume in deployment."""
    if os.environ.get("FAISS_INDEX_DIR"):
        path = os.environ.get("FAISS_INDEX_DIR").strip()
        os.makedirs(path, exist_ok=True)
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")


class LongTermMemory:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        index_dir = _faiss_index_dir()

        if os.path.exists(index_dir):
            self.vectorstore = FAISS.load_local(
                index_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            self.vectorstore = FAISS.from_texts(
                ["System initialized"],
                embedding=self.embeddings,
            )
            self.vectorstore.save_local(index_dir)

    def add_memory(self, text):
        doc = Document(page_content=text)
        self.vectorstore.add_documents([doc])
        self.vectorstore.save_local(_faiss_index_dir())

    def retrieve_memory(self, query, k=10):
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])

    def retrieve_memory_with_scores(self, query, k=5):
        """Return list of (page_content, score). Lower score = more similar (L2 distance)."""
        try:
            docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            return [(doc.page_content, float(score)) for doc, score in docs_and_scores]
        except Exception:
            return []


memory = LongTermMemory()


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
# 🛠 UTIL
# ==================================
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ==================================
#  PARALLEL VISION NODES (run in parallel, then merge)
# ==================================
def _vision_api_call(base64_image: str, prompt: str, parse_key: str, default):
    """Single vision API call with retry on rate limit; returns value for the given key."""
    max_retries = 4
    base_wait = 2.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},
            )
            break
        except Exception as e:
            is_rate_limit = (
                getattr(e, "code", None) == "rate_limit_exceeded"
                or getattr(e, "status_code", None) == 429
                or "rate limit" in str(e).lower()
                or "429" in str(e)
            )
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            wait = base_wait * (2**attempt)
            if hasattr(e, "retry_after") and e.retry_after:
                try:
                    wait = max(wait, float(e.retry_after))
                except (TypeError, ValueError):
                    pass
            time.sleep(wait)
    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
        return parsed.get(parse_key, default), raw
    except Exception:
        return default, raw


def detect_person(state: VisionState):
    """Parallel node: person detection only."""
    base64_image = encode_image(state["image_path"])
    prompt = (
        "Look at this image. Is there a clear human face/person visible? "
        "Return ONLY valid JSON, no markdown. Format: {\"person\": true or false}"
    )
    value, _ = _vision_api_call(base64_image, prompt, "person", False)
    return {"person": value}


def detect_glasses(state: VisionState):
    """Parallel node: glasses detection only."""
    base64_image = encode_image(state["image_path"])
    prompt = (
        "Look at this image. Does the person wear glasses (or sunglasses)? "
        "Return ONLY valid JSON, no markdown. Format: {\"glasses\": true or false}"
    )
    value, _ = _vision_api_call(base64_image, prompt, "glasses", False)
    return {"glasses": value}


def detect_emotion(state: VisionState):
    """Parallel node: emotion detection."""
    base64_image = encode_image(state["image_path"])
    prompt = (
        "Look at the person's face in this image. What is the dominant emotion? "
        "Return ONLY valid JSON, no markdown. "
        "Format: {\"emotion\": \"one word or short phrase, e.g. happy, neutral, surprised\"}"
    )
    value, _ = _vision_api_call(base64_image, prompt, "emotion", "")
    return {"emotion": value if isinstance(value, str) else str(value)}


def detect_age(state: VisionState):
    """Parallel node: age estimation."""
    base64_image = encode_image(state["image_path"])
    prompt = (
        "Look at the person in this image. Estimate their age range. "
        "Return ONLY valid JSON, no markdown. "
        "Format: {\"age_estimate\": \"e.g. 20-30, 30-40, or a single number\"}"
    )
    value, _ = _vision_api_call(base64_image, prompt, "age_estimate", "")
    return {"age_estimate": value if isinstance(value, str) else str(value)}


def merge_vision(state: VisionState):
    """Merge node: runs after all 4 parallel nodes; combine results into response_text."""
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
def _parse_memory_line(line: str):
    """Parse 'Name: X | Glasses: Y | Emotion: Z | Age: W' or legacy 'Name: X | Glasses: Y'."""
    if "Name:" not in line or "System initialized" in line:
        return None
    out = {"name": "", "glasses": False, "emotion": "", "age": ""}
    parts = line.split("|")
    for p in parts:
        p = p.strip()
        if p.startswith("Name:"):
            out["name"] = p.replace("Name:", "").strip()
        elif "Glasses:" in p or "glasses:" in p.lower():
            val = p.split(":", 1)[-1].strip().lower()
            out["glasses"] = val in ("true", "yes", "1")
        elif "Emotion:" in p or (p.lower().startswith("emotion:") and ":" in p):
            out["emotion"] = p.split(":", 1)[-1].strip().lower()
        elif "Age:" in p or (p.lower().startswith("age:") and ":" in p):
            out["age"] = p.split(":", 1)[-1].strip().lower()
    return out if out["name"] else None


def identify_person(state: VisionState):
    if not state.get("person"):
        return state

    # Name provided in this request (e.g. user just submitted the form) → use it and store in memory
    raw_name = state.get("person_name")
    if raw_name is not None and str(raw_name).strip():
        name_to_store = str(raw_name).strip()
        memory_text = (
            f"Name: {name_to_store} | Glasses: {state.get('glasses', False)} | "
            f"Emotion: {state.get('emotion', '') or ''} | Age: {state.get('age_estimate', '') or ''}"
        )
        memory.add_memory(memory_text)
        state["person_name"] = name_to_store
        state["need_name"] = False
        return state

    # Try to recognize from FAISS memory: query by current glasses/emotion/age, then strict match
    current_glasses = state.get("glasses", False)
    current_emotion = (state.get("emotion") or "").strip().lower()
    current_age = (state.get("age_estimate") or "").strip().lower()
    query = (
        f"Name: | Glasses: {current_glasses} | Emotion: {current_emotion} | Age: {current_age}"
    )
    candidates = memory.retrieve_memory_with_scores(query, k=5)
    # Only use best match if it's close enough (L2 distance < 0.6) and fields match
    best_name = None
    for content, score in candidates:
        if score > 0.6:
            continue
        parsed = _parse_memory_line(content)
        if not parsed:
            continue
        if parsed["glasses"] != current_glasses:
            continue
        if current_emotion and parsed["emotion"] and parsed["emotion"] != current_emotion:
            continue
        if current_age and parsed["age"] and parsed["age"] != current_age:
            continue
        best_name = (parsed.get("name") or "").strip()
        if best_name:
            break
    if best_name:
        state["person_name"] = best_name
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
        parts.append("👓 Looking sharp with specs")
    else:
        parts.append("🌞 No specs today")
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
# 🔁 BUILD GRAPH (parallel vision → merge → identify → greet)
# ==================================
graph = StateGraph(VisionState)

# Parallel nodes: all 4 run at once from START
graph.add_node("detect_person", detect_person)
graph.add_node("detect_glasses", detect_glasses)
graph.add_node("detect_emotion", detect_emotion)
graph.add_node("detect_age", detect_age)
graph.add_node("merge_vision", merge_vision)
graph.add_node("identify", identify_person)
graph.add_node("greet", greet)

# Fan-out from START to 4 parallel nodes
graph.add_edge(START, "detect_person")
graph.add_edge(START, "detect_glasses")
graph.add_edge(START, "detect_emotion")
graph.add_edge(START, "detect_age")
# Fan-in: all 4 feed into merge, then linear
graph.add_edge("detect_person", "merge_vision")
graph.add_edge("detect_glasses", "merge_vision")
graph.add_edge("detect_emotion", "merge_vision")
graph.add_edge("detect_age", "merge_vision")
graph.add_edge("merge_vision", "identify")
graph.add_edge("identify", "greet")
graph.add_edge("greet", END)

app = graph.compile(store=store)


# ==================================
# ▶ RUN (CLI: pass image path as first arg; or use Streamlit frontend)
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