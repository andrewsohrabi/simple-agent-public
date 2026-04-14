from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.chat_service import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_FACTS_EXTRACTOR,
    DEMO_USER_DEFAULT,
    build_system_prompt_preview,
    delete_memory_snapshot,
    finalize_chat_session,
    get_backend_health,
    list_demo_users,
    load_demo_user,
    read_facts_events_snapshot,
    read_memory_snapshot,
    run_chat_turn,
)

load_dotenv()

MEMORY_ROOT = Path("memory_store")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    memoryType: str = "none"
    userId: str = "demo_user"
    model: str | None = None
    factsExtractor: str = DEFAULT_FACTS_EXTRACTOR
    extractorModel: str | None = None


class DemoUserLoadRequest(BaseModel):
    demoUser: str = DEMO_USER_DEFAULT


class FinalizeSessionRequest(BaseModel):
    messages: list[Message]
    memoryType: str = "none"
    userId: str = "demo_user"
    factsExtractor: str = DEFAULT_FACTS_EXTRACTOR
    extractorModel: str | None = None


def build_memory_response(
    memory_type: str,
    user_id: str,
    *,
    snapshot: tuple[object, Path | None] | None = None,
) -> dict[str, object]:
    """Build the shared memory payload returned by inspection and mutation routes."""
    # GET, DELETE, and finalize all reuse this builder so the frontend sees one
    # stable response shape no matter how the memory snapshot was produced.
    memory, memory_path = snapshot or read_memory_snapshot(memory_type, MEMORY_ROOT, user_id)
    facts_events, facts_events_path = read_facts_events_snapshot(
        memory_type,
        MEMORY_ROOT,
        user_id,
    )
    return {
        "memory": memory,
        "memoryPath": str(memory_path) if memory_path else None,
        "factsEvents": facts_events,
        "factsEventsPath": str(facts_events_path) if facts_events_path else None,
        "promptPreview": build_system_prompt_preview(
            memory_type=memory_type,
            memory_root=MEMORY_ROOT,
            user_id=user_id,
        ),
    }


@app.get("/health")
def health():
    return get_backend_health(MEMORY_ROOT)


@app.get("/demo-users")
def demo_users():
    return {
        "defaultDemoUser": DEMO_USER_DEFAULT,
        "demoUsers": list_demo_users(),
    }


@app.post("/demo-users/load")
def load_demo_user_endpoint(req: DemoUserLoadRequest):
    return load_demo_user(MEMORY_ROOT, req.demoUser)


@app.post("/chat")
def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    result = run_chat_turn(
        messages=messages,
        memory_type=req.memoryType,
        user_id=req.userId,
        memory_root=MEMORY_ROOT,
        model=req.model or DEFAULT_CHAT_MODEL,
        facts_extractor=req.factsExtractor,
        extractor_model=req.extractorModel or DEFAULT_EXTRACTOR_MODEL,
    )
    facts_events, facts_events_path = read_facts_events_snapshot(
        req.memoryType,
        MEMORY_ROOT,
        req.userId,
    )
    return {
        "reply": result["reply"],
        "memory": result["memory"],
        "memoryPath": result["memory_path"],
        "factsEvents": facts_events,
        "factsEventsPath": str(facts_events_path) if facts_events_path else None,
        "modelUsed": result["model_used"],
        "promptPreview": result["system_prompt"],
    }


@app.post("/session/finalize")
def finalize_session(req: FinalizeSessionRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    finalized_memory = finalize_chat_session(
        messages=messages,
        memory_type=req.memoryType,
        user_id=req.userId,
        memory_root=MEMORY_ROOT,
        facts_extractor=req.factsExtractor,
        extractor_model=req.extractorModel or DEFAULT_EXTRACTOR_MODEL,
    )
    _, memory_path = read_memory_snapshot(req.memoryType, MEMORY_ROOT, req.userId)
    snapshot = None
    if memory_path is not None:
        snapshot = (finalized_memory, memory_path)
    return build_memory_response(req.memoryType, req.userId, snapshot=snapshot)


@app.get("/memory")
def get_memory(
    memoryType: str = Query(...),
    userId: str = Query(...),
):
    return build_memory_response(memoryType, userId)


@app.delete("/memory")
def clear_memory(
    memoryType: str = Query(...),
    userId: str = Query(...),
):
    snapshot = delete_memory_snapshot(memoryType, MEMORY_ROOT, userId)
    return build_memory_response(memoryType, userId, snapshot=snapshot)


def main():
    import uvicorn

    uvicorn.run("agent.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
