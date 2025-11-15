import os
import re
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnimateRequest(BaseModel):
    text: str = Field(..., description="Full story or text to animate")
    style: Optional[str] = Field("storybook", description="Visual style hint (e.g., storybook, noir, sci-fi, watercolor)")
    pacing: Optional[str] = Field("normal", description="slow | normal | fast")


class CharacterInstance(BaseModel):
    id: str
    emotion: Optional[str] = None
    action: Optional[str] = None
    dialogue: Optional[str] = None


class Scene(BaseModel):
    id: str
    title: str
    description: str
    environmentId: Optional[str] = None
    characters: List[CharacterInstance] = []
    transition: Dict[str, Any] = {"type": "crossfade", "duration": 0.8}


class AnimateResponse(BaseModel):
    style: str
    characters: List[Dict[str, Any]]
    environments: List[Dict[str, Any]]
    scenes: List[Scene]


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        from database import db  # type: ignore
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = getattr(db, 'name', "✅ Connected")
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:  # pragma: no cover
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


# ---------- Simple story-to-scenes engine ----------

COMMON_WORDS = set(
    "the a an and or but if then when while for with without into onto from by of in on at to as is are was were be been being he she they them him her it its his their our your my we you I".split()
)

def sentence_split(text: str) -> List[str]:
    # Split by sentence terminators while preserving meaningful chunks
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_characters(text: str) -> List[str]:
    # Naive proper-noun extractor: capitalized words not at start of sentence punctuation
    tokens = re.findall(r"[A-Z][a-zA-Z'-]+", text)
    names = []
    for t in tokens:
        lower = t.lower()
        if lower not in COMMON_WORDS and len(t) > 1:
            names.append(t)
    # Deduplicate preserving order
    seen = set()
    ordered = []
    for n in names:
        if n not in seen:
            ordered.append(n)
            seen.add(n)
    return ordered[:8] or ["Protagonist"]


def extract_environments(text: str) -> List[str]:
    # Very lightweight environment hinting by keywords
    env_map = {
        "forest": ["forest", "woods", "grove", "trees"],
        "castle": ["castle", "fortress", "keep"],
        "city": ["city", "street", "market", "alley"],
        "sea": ["sea", "ocean", "shore", "beach"],
        "mountain": ["mountain", "peak", "cliff", "ridge"],
        "desert": ["desert", "sand", "dune", "oasis"],
        "home": ["home", "house", "cottage", "room", "kitchen"],
        "space": ["space", "planet", "star", "station"],
    }
    found = set()
    lower = text.lower()
    for env, kws in env_map.items():
        if any(k in lower for k in kws):
            found.add(env)
    return list(found) or ["generic"]


def infer_emotion(chunk: str) -> str:
    emo_map = {
        "joy": ["happy", "joy", "smile", "laugh", "delight"],
        "fear": ["fear", "afraid", "scared", "tremble", "panic"],
        "anger": ["anger", "angry", "furious", "rage"],
        "sad": ["sad", "sorrow", "tears", "cry"],
        "surprise": ["surprise", "astonish", "sudden", "shock"],
        "calm": ["calm", "quiet", "peace", "still"],
    }
    l = chunk.lower()
    for emo, kws in emo_map.items():
        if any(k in l for k in kws):
            return emo
    return "neutral"


def choose_transition(pacing: str, idx: int) -> Dict[str, Any]:
    base = 1.0 if pacing == "normal" else (1.5 if pacing == "slow" else 0.6)
    types = ["crossfade", "pan", "dolly", "wipe", "fade-through-black"]
    return {"type": types[idx % len(types)], "duration": round(base, 2)}


def map_characters_to_ids(names: List[str]) -> Dict[str, str]:
    return {name: f"char_{i+1}" for i, name in enumerate(names)}


def build_response(req: AnimateRequest) -> AnimateResponse:
    # Prepare global cast and environments
    names = extract_characters(req.text)
    char_id_map = map_characters_to_ids(names)
    environments = extract_environments(req.text)
    env_id_map = {name: f"env_{i+1}" for i, name in enumerate(environments)}

    # Split into scene chunks
    chunks = sentence_split(req.text)
    scenes: List[Scene] = []
    for i, chunk in enumerate(chunks):
        # Assign environment heuristically by keyword presence
        env_for_scene = None
        for env_name, env_id in env_id_map.items():
            if env_name in chunk.lower():
                env_for_scene = env_id
                break
        if env_for_scene is None:
            env_for_scene = list(env_id_map.values())[i % len(env_id_map)]

        # Determine present characters (those whose names appear in chunk)
        present: List[CharacterInstance] = []
        for name, cid in char_id_map.items():
            if re.search(rf"\b{name}\b", chunk):
                present.append(CharacterInstance(id=cid, emotion=infer_emotion(chunk)))
        if not present:
            # Ensure at least the protagonist is present
            first_char = list(char_id_map.values())[0]
            present.append(CharacterInstance(id=first_char, emotion=infer_emotion(chunk)))

        # Try to extract dialogue in quotes
        dialogues = re.findall(r'"([^"]+)"|\'([^\']+)\'', chunk)
        dialogue_texts = [d[0] or d[1] for d in dialogues] if dialogues else []
        if dialogue_texts:
            # Attach first dialogue to first present character
            present[0].dialogue = dialogue_texts[0]

        scene = Scene(
            id=f"scene_{i+1}",
            title=f"Scene {i+1}",
            description=chunk,
            environmentId=env_for_scene,
            characters=present,
            transition=choose_transition(req.pacing or "normal", i),
        )
        scenes.append(scene)

    return AnimateResponse(
        style=req.style or "storybook",
        characters=[{"id": cid, "name": name, "color": i} for i, (name, cid) in enumerate(char_id_map.items(), start=1)],
        environments=[{"id": eid, "name": name} for name, eid in env_id_map.items()],
        scenes=scenes,
    )


@app.post("/api/animate", response_model=AnimateResponse)
def animate(req: AnimateRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    try:
        return build_response(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)[:200]}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
