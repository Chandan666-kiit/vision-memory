import base64, io, json, os, uuid
from datetime import datetime, timezone
import face_recognition
import numpy as np
import faiss
from PIL import Image
from pymongo import MongoClient
from openai import OpenAI
import streamlit as st

DIM=128; FAISS_THRESHOLD=0.9; TOP_K=3

@st.cache_resource
def _openai(): return OpenAI()

@st.cache_resource
def _get_col():
    uri=os.environ.get("MONGO_URI","mongodb+srv://chandansrinethvickey_db_user:test1234@cluster0.5cahbo9.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    return MongoClient(uri)["vision_memory"]["people"]

@st.cache_resource
def _get_faiss():
    col=_get_col()
    col.delete_many({"$or":[{"name":{"$in":["","unknown",None]}},{"embedding":{"$exists":False}}]})
    index=faiss.IndexFlatL2(DIM); ids=[]
    for doc in col.find({"embedding":{"$exists":True},"name":{"$nin":["",None,"unknown"]}}):
        index.add(np.array(doc["embedding"],dtype="float32").reshape(1,-1))
        ids.append(doc.get("doc_id") or str(doc["_id"]))
    print(f"[main] FAISS ready {index.ntotal} profiles")
    return {"index":index,"ids":ids}

def get_people_col(): return _get_col()

def _crop(path,loc):
    img=Image.open(path).convert("RGB")
    t,r,b,l=loc
    py=int((b-t)*.25); px=int((r-l)*.25)
    t=max(0,t-py); l=max(0,l-px); b=min(img.height,b+py); r=min(img.width,r+px)
    buf=io.BytesIO(); img.crop((l,t,r,b)).resize((224,224)).save(buf,format="JPEG",quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def _b64(path):
    return base64.b64encode(open(path,"rb").read()).decode()

def _same(a,b):
    try:
        res=_openai().chat.completions.create(model="gpt-4o",messages=[{"role":"user","content":[
            {"type":"text","text":'Are these two face images the same person? Compare face shape, eyes, nose, mouth only. Ignore hair/glasses/lighting. JSON only: {"same_person":true/false,"confidence":"high/medium/low"}'},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{a}","detail":"high"}},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b}","detail":"high"}}]}],
            response_format={"type":"json_object"},max_tokens=100)
        d=json.loads(res.choices[0].message.content)
        return bool(d.get("same_person")) and d.get("confidence") in ("high","medium")
    except Exception as e:
        print(f"[main] gpt4o err {e}"); return False

def _vj(b64,p):
    try:
        res=_openai().chat.completions.create(model="gpt-4o-mini",messages=[{"role":"user","content":[
            {"type":"text","text":p+"\nJSON only."},{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}],
            response_format={"type":"json_object"})
        return json.loads(res.choices[0].message.content)
    except: return {}

def _attrs(b64):
    return {
        "glasses":bool(_vj(b64,'{"glasses":true/false}').get("glasses",False)),
        "emotion":_vj(b64,'{"emotion":"..."}').get("emotion","neutral"),
        "age_estimate":_vj(b64,'{"age_estimate":"25-30"}').get("age_estimate","unknown")}

def _greet(name,emotion,age,glasses,is_new):
    p=[f"Hello, {name}!"]
    if emotion and emotion not in ("","unknown","neutral"): p.append(f"You look {emotion}.")
    if age and age not in ("","unknown"): p.append(f"Around {age} years old.")
    if glasses: p.append("Wearing glasses!")
    p.append("Nice to meet you!" if is_new else "Welcome back!")
    return " ".join(p)

def analyse_image(image_path):
    out={"person":False,"embedding":[],"face_b64":"","glasses":False,"emotion":"unknown",
         "age_estimate":"unknown","need_name":False,"person_name":"","greeting_message":""}
    img=face_recognition.load_image_file(image_path)
    locs=face_recognition.face_locations(img,model="hog")
    encs=face_recognition.face_encodings(img,locs)
    if not encs: out["greeting_message"]="No face detected."; return out
    out["person"]=True; out["embedding"]=encs[0].tolist(); out["face_b64"]=_crop(image_path,locs[0])
    out.update(_attrs(_b64(image_path)))
    state=_get_faiss(); index=state["index"]; ids=state["ids"]
    if index.ntotal==0: out["need_name"]=True; return out
    k=min(TOP_K,index.ntotal)
    vec=np.array(out["embedding"],dtype="float32").reshape(1,-1)
    D,I=index.search(vec,k)
    cands=[(float(d),ids[int(p)]) for d,p in zip(D[0],I[0]) if float(d)<=FAISS_THRESHOLD and int(p)<len(ids)]
    cands.sort(key=lambda x:x[0])
    for dist,did in cands:
        doc=_get_col().find_one({"doc_id":did})
        if not doc: continue
        n=(doc.get("name") or "").strip(); f=doc.get("face_b64","")
        if not n or not f: continue
        if _same(out["face_b64"],f):
            _get_col().update_one({"doc_id":did},{"$set":{"emotion":out["emotion"],"last_seen":datetime.now(timezone.utc)}})
            out["person_name"]=n; out["greeting_message"]=_greet(n,out["emotion"],out["age_estimate"],out["glasses"],False)
            return out
    out["need_name"]=True; return out

def register_and_greet(image_path,embedding,face_b64,name,glasses,emotion,age_estimate):
    name=(name or "").strip()
    if not name or name.lower()=="unknown": return {"error":"Enter a valid name."}
    did=str(uuid.uuid4())
    _get_col().insert_one({"doc_id":did,"name":name,"embedding":embedding,"face_b64":face_b64,
        "glasses":glasses,"emotion":emotion,"age":age_estimate,
        "created_at":datetime.now(timezone.utc),"last_seen":datetime.now(timezone.utc)})
    state=_get_faiss()
    state["index"].add(np.array(embedding,dtype="float32").reshape(1,-1))
    state["ids"].append(did)
    print(f"[main] Saved {name}")
    return {"person":True,"need_name":False,"person_name":name,"glasses":glasses,
            "emotion":emotion,"age_estimate":age_estimate,
            "greeting_message":_greet(name,emotion,age_estimate,glasses,True)}
