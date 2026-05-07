from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.config import load_config
from agent.core import make_agent
from agent.search.service import QmsSearchService
from agent.search.stats import collect_stats

load_dotenv()
config = load_config()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent once at startup with configured model.
agent = make_agent(config.agent_model)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class SearchRequest(BaseModel):
    query: str
    mode: str = "auto"
    limit: int = 16


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        result = agent.invoke({"messages": messages})
        ai_msg = result["messages"][-1]
        return {"reply": ai_msg.content}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    try:
        service = QmsSearchService(config)
        return service.search(req.query, mode=req.mode, limit=req.limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/stats")
def stats():
    return collect_stats(config)


@app.get("/health")
def health():
    return {"ok": True, "model_config": config.model_config()}


def main():
    import uvicorn

    uvicorn.run("agent.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
