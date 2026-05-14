# Context Package for Senior Architecture Review

Repository: /Users/andrewsohrabi/projects/codex-sandbox/valkai-onsite/simple-agent-public
Branch: codex/mvp
Git state: clean tracked tree with untracked corpus zip `Example_QMS_-_MedAI.zip`.

## Goal
Exercise objective and constraints:
- Build an internal search demo for a customer's own document corpus through a chat assistant.
- Focus only on Internal Search over customer documents; regulatory search and web search are out of scope.
- Corpus is a fictional MedAI Quality Management System for the MX1 portable X-ray system, with 189 .docx documents.
- Query patterns to support: known-item retrieval, exploratory search, compliance cross-reference, content extraction/synthesis, revision/change tracking, cross-document analysis, enumeration/counting.
- Evaluation dimensions: search quality, citations, architecture decisions, evals, code quality.
- Deliverables: working repo branch, DESIGN.md, evals, live demo/walkthrough.
- Desired implementation direction already chosen with the user: hybrid local index, OpenAI available, strict citations, build index locally, latest active revision first, polished fullstack workbench plus CLI, retrieval + answer evals.


## Current Project Structure
```text
.
├── .claude/settings.local.json
├── .env                         # local secrets, intentionally not included
├── .env.example
├── .gitignore
├── .python-version
├── Example_QMS_-_MedAI.zip       # untracked corpus artifact
├── README.md
├── docs/
│   ├── cli.md
│   └── fullstack.md
├── evals/
│   ├── __init__.py
│   └── test_agent.py
├── frontend/
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       └── main.jsx
├── pyproject.toml
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── cli.py
│       ├── core.py
│       └── server.py
└── uv.lock                       # gitignored by current .gitignore but present locally
```

## Current Tech Stack and Architecture
- Python 3.13 project managed with uv/hatchling.
- Backend/agent stack: LangChain, LangChain Deep Agents, provider adapters for OpenAI/Anthropic/Google, python-dotenv, FastAPI, uvicorn.
- Frontend: Vite + React 18, currently no CSS framework and no component library.
- Current architecture is intentionally minimal:
  - `agent.core.make_agent()` wraps `init_chat_model()` and `create_deep_agent()`.
  - CLI uses an in-memory message list and calls `agent.invoke()`.
  - FastAPI initializes one default agent at import/startup and exposes `POST /chat`.
  - React UI stores local chat state and posts full message history to `http://localhost:8000/chat`.
  - Existing evals are integration-style LLM tests and do not yet test search.

## Key File Purposes
- `README.md`: overview, setup, provider support, eval command, project structure.
- `docs/cli.md`: CLI usage and behavior.
- `docs/fullstack.md`: FastAPI + React startup guide and `/chat` contract.
- `pyproject.toml`: Python project metadata, dependencies, `chat` and `serve` console scripts, pytest config.
- `src/agent/core.py`: shared agent factory.
- `src/agent/cli.py`: terminal chat loop.
- `src/agent/server.py`: FastAPI app, CORS, `/chat` endpoint.
- `frontend/src/App.jsx`: entire current browser chat UI with inline styles.
- `evals/test_agent.py`: two live-LLM smoke tests for simple response and multi-turn memory.

## Corpus Summary
Corpus artifact:
- Local file: Example_QMS_-_MedAI.zip
- Real corpus documents after ignoring __MACOSX/resource fork entries: 189
- Extensions: .docx only
- Document prefix counts:
- 3P: 2
- BOM: 6
- DHF: 1
- DMR: 1
- DR: 1
- ECR: 3
- ESF: 3
- IFU: 2
- MEMO: 40
- PLN: 21
- QSR: 1
- RSK: 10
- TRA: 3
- VVAM: 2
- VVPR: 93
- Example filenames:
- VVPR-P01-179 - MX1 Software System Radiographic and Radioscopic Acquisition v3.0.0 Protocol and Report_B.docx
- MEMO-P01-660 Gamma Curve Consistency Assessment_A-signed.docx
- VVPR-P01-181 - MX1 Software System Critical Faults v3.1.0 Protocol and Report_B.docx
- BOM-055 - MX1 Top-level assembly_G.docx
- MEMO-P01-638 - MX1 Software Development Configuration Management and Maintenance Practices_A-signed.docx
- MEMO-P01-655 - P01-MAI User Interface Specification_C.docx
- RSK-P01-016 - MX1 MedAI PFMEA_A-Signed.docx
- VVPR-P01-230 - MX1 Software System Fuzz Testing v3.3.0 Protocol and Report_B.docx
- VVPR-P01-152- Pediatric Filtration Verification Protocol and Report_B-signed.docx
- VVPR-P01-159 - MX1 Weight Verification Protocol and Report_B-signed.docx
- MEMO-P01-642 - MX1 Use Specification_D.docx
- VVPR-SWV-011 - MX1 MedAI Rest Server Verification and Validation v1.0.0 Protocol and Report_B.docx
- VVPR-P01-214 - MX1 Software System Beam Angle Accuracy v3.2.1 Protocol and Report_C-Obsolete.docx
- PLN-P01-062 - MX1 Project Quality Plan_D-signed.docx
- ECR-593 - MX1 BOM-055 Rev G_A-signed.docx
- MEMO-P01-654 - P01-MAI Usability Engineering File_C.docx
- VVPR-SWV-026 Monoblock Test Fixture Firmware Verification and Validation Protocol and Report_B.docx
- VVPR-P01-176 - MX1 Software System Power OnOff and Power States v3.0.0 Protocol and Report_B.docx
- VVPR-P01-186 - MX1 Software System MedAI Device App v3.0.0 Protocol and Report_C.docx
- VVPR-P01-240 - WS-002 Workstation Installation Qualification_A.docx
- MEMO-P01-672 Dosimetric Indications_A-signed.docx
- BOM-055 - MX1 Top-level assembly_F.docx
- VVPR-P01-190 - MX1 Software System v3.1.0 Protocol and Report_C.docx
- VVPR-P01-267 - WS-010 Workstation Operational Qualification_B-Signed.docx
- MEMO-P01-658 - MX1 System Architecture Diagram_A-signed.docx
- VVPR-P01-148 - WTX Foreign Object Detection and Resonance Lock Verification Protocol and Report_C-signed.docx
- MEMO-P01-677 - MX1 Software System Verification via Code Review v3.0.0_A.docx
- VVPR-P01-189 - Collimation Accuracy Protocol and Report_B.docx
- VVPR-P01-172 Attenuation Equivalent Detector Verification Protocol  Report_B-Signed.docx
- VVPR-P01-206- MX1 MedAI Diagnostic Tool TWS-001 WS-002 WS-004 WS-005 and WS-006 v3.0.0 Verification Protocol and Report_B.docx
- VVPR-P01-168 - SSD and SID Accuracy Under Challenge Conditions Report_B-Signed.docx
- VVPR-P01-269 - WS-013 Operational Qualification Report_B-signed.docx
- VVPR-SWV-020 - IRay License Checker Verification and Validation v1.0.0 Protocol and Report_B.docx
- PLN-P01-059 - MX1 Assembly Workstations Verification and Validation Plan_E-signed.docx
- MEMO-P01-636 - MX1 Software System Unresolved Anomalies v3.0.0_A_Obsolete.docx
- VVPR-P01-151 - Half Value Layer Verification Protocol  Report_B-signed.docx
- MEMO-P01-631 - MX1 Software Design Specifications_F.docx
- VVPR-P01-229 - MX1 Software System v3.3.0 Protocol and Report_B.docx
- VVPR-P01-081 - Usability Summative Evaluation Protocol and Report_B - Signed.docx
- VVPR-P01-239 - WS-001 Workstation Installation Qualification_A.docx
- RSK-P01-011 - MX1 Security Risk Assessment_A-Obsolete.docx
- VVPR-P01-236 - MX1 Software System v3.4.0 Protocol and Report_B.docx
- PLN-P01-024 - MX1 Software Development Plan_D-Obsolete.docx
- VVAM-P01-004 - MX1 Verification  Validation Trace Matrix_D.docx
- VVPR-P01-253 - WS-008 Workstation PMUX PCBA Programming Operational Qualification_B-signed.docx
- Important observed traits: revisions are encoded in filenames (for example BOM-055 Rev E/F/G; VVAM-P01-004 Rev A/D), obsolete/signed status is encoded in filename suffixes, and some .docx files have little/no extractable Word body text and should be treated as metadata-only or skipped for synthesis.

## Important Current File Contents

### README.md
```md
# simple-agent

A minimal LLM agent built on [LangChain Deep Agents](https://github.com/langchain-ai/deepagents). Supports OpenAI, Anthropic, and Google models out of the box. Two ways to run it — pick one:

- [CLI guide](docs/cli.md) — interactive terminal chat
- [Fullstack guide](docs/fullstack.md) — FastAPI server + React frontend

---

## Core agent

The agent lives in `src/agent/core.py` and exposes a single factory:

```python
from agent.core import make_agent

agent = make_agent(
    model_str="anthropic:claude-haiku-4-5-20251001",  # provider:model
    system_prompt=None,                                # optional override
)
```

It wraps LangChain's `init_chat_model` + `create_deep_agent` and returns a compiled LangGraph agent that supports `.invoke()`, `.stream()`, and `.astream()`.

## Supported providers

| Provider  | Model string example                            | Required env var    |
|-----------|-------------------------------------------------|---------------------|
| Anthropic | `anthropic:claude-haiku-4-5-20251001` (default) | `ANTHROPIC_API_KEY` |
| OpenAI    | `openai:gpt-4o`                                 | `OPENAI_API_KEY`    |
| Google    | `google_genai:gemini-2.5-flash`                 | `GOOGLE_API_KEY`    |

Any model supported by LangChain's [`init_chat_model`](https://python.langchain.com/docs/how_to/chat_models_universal_init/) works — just pass the `provider:model` string.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- At least one LLM provider API key

## Initial setup

```bash
git clone https://github.com/valkai-tech/simple-agent-public.git
cd simple-agent-public
uv sync
cp .env.example .env
# Fill in your API key(s) in .env
```

## Running evals

```bash
uv run pytest evals/ -v
```

Evals make real LLM calls (not mocked) to verify provider integration end-to-end.

## Project structure

```
simple-agent/
├── README.md               # this file — core concepts
├── docs/
│   ├── cli.md              # CLI usage guide
│   └── fullstack.md        # server + frontend guide
├── pyproject.toml          # uv project config and dependencies
├── .env.example            # API key template
├── src/
│   └── agent/
│       ├── core.py         # agent factory (shared by both approaches)
│       ├── cli.py          # CLI entry point
│       └── server.py       # FastAPI server entry point
├── frontend/               # React chat UI
└── evals/
    └── test_agent.py       # pytest evals
```

```

### docs/cli.md
```md
# CLI guide

Run the agent as an interactive terminal chat. See the [core README](../README.md) for initial setup.

## Start

```bash
# Default model (Anthropic Claude Haiku)
uv run chat

# OpenAI
uv run chat --model openai:gpt-4o

# Google
uv run chat --model google_genai:gemini-2.5-flash

# Custom system prompt
uv run chat --system "You are a helpful coding assistant."
```

Type `quit` or `exit` to end the session.

## How it works

`src/agent/cli.py` keeps a running `messages` list in memory for the duration of the session, appending each user/assistant turn before passing the full history to `agent.invoke()`.

## Relevant files

```
src/agent/
├── core.py     # agent factory (shared)
└── cli.py      # REPL loop, argument parsing
```

```

### docs/fullstack.md
```md
# Fullstack guide

Run the agent as a FastAPI server with a React chat frontend. See the [core README](../README.md) for initial setup.

## Prerequisites

- Everything in the core README
- Node.js 18+

## Start

Run both processes in separate terminals.

**Terminal 1 — backend:**

```bash
uv run serve
```

Server starts at `http://localhost:8000`.

**Terminal 2 — frontend:**

```bash
cd frontend
npm install   # first time only
npm run dev
```

UI opens at `http://localhost:3000`.

## API

```
POST /chat
Content-Type: application/json

{
  "messages": [
    { "role": "user", "content": "Hello!" }
  ]
}
```

```json
{
  "reply": "Hi! How can I help you?"
}
```

The frontend sends the full conversation history on each request. The server is stateless — no session storage.

## How it works

`src/agent/server.py` initializes the agent once at startup using the same `make_agent()` factory from `core.py`. The React frontend (`frontend/src/App.jsx`) manages conversation state locally and posts the full message list on every send.

## Relevant files

```
src/agent/
├── core.py       # agent factory (shared)
└── server.py     # FastAPI app, POST /chat endpoint

frontend/
├── src/
│   ├── App.jsx   # chat UI component
│   └── main.jsx  # React entry point
├── index.html
├── vite.config.js
└── package.json
```

```

### pyproject.toml
```toml
[project]
name = "take-home"
version = "0.1.0"
description = "Barebones CLI chat agent built on LangChain Deep Agents"
requires-python = ">=3.13"
dependencies = [
    "deepagents",
    "langchain",
    "langchain-openai",
    "langchain-anthropic",
    "langchain-google-genai",
    "python-dotenv",
    "fastapi",
    "uvicorn[standard]",
]

[project.scripts]
chat = "agent.cli:main"
serve = "agent.server:main"

[dependency-groups]
dev = [
    "pytest",
    "pytest-asyncio",
]

[tool.pytest.ini_options]
testpaths = ["evals"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/agent"]

```

### .env.example
```
# Fill in the API key(s) for the provider(s) you want to use.
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

```

### .gitignore
```
# Environment
.env
.venv/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/

# OS
.DS_Store

# uv
uv.lock

# Node
node_modules/

```

### .python-version
```
3.13

```

### src/agent/core.py
```py
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent


def make_agent(
    model_str: str = "anthropic:claude-haiku-4-5-20251001",
    system_prompt: str | None = None,
):
    """Create a deep agent with the specified model provider.

    Args:
        model_str: Provider and model in "provider:model" format.
                   Examples: "openai:gpt-4o", "anthropic:claude-haiku-4-5-20251001",
                   "google_genai:gemini-2.5-flash"
        system_prompt: Optional system prompt override.

    Returns:
        A compiled LangGraph agent supporting .invoke(), .stream(), .astream().
    """
    model = init_chat_model(model_str)
    kwargs = {}
    if system_prompt:
        kwargs["system_prompt"] = system_prompt
    return create_deep_agent(model=model, **kwargs)

```

### src/agent/cli.py
```py
import argparse
import sys

from dotenv import load_dotenv

from agent.core import make_agent


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="CLI Chat Agent")
    parser.add_argument(
        "--model",
        default="anthropic:claude-haiku-4-5-20251001",
        help="Model string, e.g. openai:gpt-4o, anthropic:claude-haiku-4-5-20251001, google_genai:gemini-2.5-flash",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Custom system prompt",
    )
    args = parser.parse_args()

    agent = make_agent(args.model, args.system)
    messages = []

    print("Chat started. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})
        ai_msg = result["messages"][-1]
        print(f"\nAssistant: {ai_msg.content}\n")
        messages = result["messages"]


if __name__ == "__main__":
    main()

```

### src/agent/server.py
```py
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.core import make_agent

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent once at startup with default model
agent = make_agent()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


@app.post("/chat")
def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    result = agent.invoke({"messages": messages})
    ai_msg = result["messages"][-1]
    return {"reply": ai_msg.content}


def main():
    import uvicorn

    uvicorn.run("agent.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()

```

### evals/test_agent.py
```py
import pytest
from dotenv import load_dotenv

from agent.core import make_agent

load_dotenv()


@pytest.fixture
def agent():
    return make_agent()


def test_agent_responds(agent):
    """Agent should return a non-empty response to a simple question."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is 2 + 2?"}]}
    )
    assert len(result["messages"]) > 1
    ai_msg = result["messages"][-1]
    assert ai_msg.content
    assert "4" in ai_msg.content


def test_agent_multi_turn(agent):
    """Agent should handle multi-turn conversation."""
    r1 = agent.invoke(
        {"messages": [{"role": "user", "content": "My name is Alice."}]}
    )
    msgs = r1["messages"]
    msgs.append({"role": "user", "content": "What is my name?"})
    r2 = agent.invoke({"messages": msgs})
    ai_msg = r2["messages"][-1]
    assert "Alice" in ai_msg.content

```

### frontend/package.json
```json
{
  "name": "simple-agent-frontend",
  "private": true,
  "version": "0.0.1",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.0.0",
    "vite": "^5.0.0"
  }
}

```

### frontend/vite.config.js
```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
  },
})

```

### frontend/index.html
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Simple Agent Chat</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>

```

### frontend/src/main.jsx
```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'

ReactDOM.createRoot(document.getElementById('root')).render(<App />)

```

### frontend/src/App.jsx
```jsx
import { useEffect, useRef, useState } from 'react'

const API_URL = 'http://localhost:8000'

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function sendMessage() {
    if (!input.trim() || loading) return

    const newMessages = [...messages, { role: 'user', content: input.trim() }]
    setMessages(newMessages)
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: newMessages }),
      })
      const data = await res.json()
      setMessages([...newMessages, { role: 'assistant', content: data.reply }])
    } catch (err) {
      setMessages([...newMessages, { role: 'assistant', content: `Error: ${err.message}` }])
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Simple Agent Chat</h2>

      <div style={styles.messageList}>
        {messages.length === 0 && (
          <p style={styles.empty}>No messages yet. Start chatting!</p>
        )}
        {messages.map((m, i) => (
          <div key={i} style={m.role === 'user' ? styles.userMsg : styles.assistantMsg}>
            <strong>{m.role === 'user' ? 'You' : 'Assistant'}</strong>
            <p style={styles.msgContent}>{m.content}</p>
          </div>
        ))}
        {loading && (
          <div style={styles.assistantMsg}>
            <strong>Assistant</strong>
            <p style={{ ...styles.msgContent, color: '#888' }}>Thinking…</p>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div style={styles.inputRow}>
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
          style={styles.textarea}
          rows={3}
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          style={styles.button}
        >
          Send
        </button>
      </div>
    </div>
  )
}

const styles = {
  container: {
    maxWidth: 700,
    margin: '40px auto',
    fontFamily: 'monospace',
    padding: '0 16px',
  },
  title: {
    marginBottom: 16,
  },
  messageList: {
    border: '1px solid #ccc',
    borderRadius: 4,
    height: 450,
    overflowY: 'auto',
    padding: '12px 16px',
    marginBottom: 12,
    background: '#fafafa',
  },
  empty: {
    color: '#888',
    textAlign: 'center',
    marginTop: 180,
  },
  userMsg: {
    marginBottom: 16,
    textAlign: 'right',
  },
  assistantMsg: {
    marginBottom: 16,
    textAlign: 'left',
  },
  msgContent: {
    margin: '4px 0 0',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
  inputRow: {
    display: 'flex',
    gap: 8,
  },
  textarea: {
    flex: 1,
    padding: 8,
    fontFamily: 'monospace',
    fontSize: 14,
    resize: 'vertical',
    border: '1px solid #ccc',
    borderRadius: 4,
  },
  button: {
    padding: '0 20px',
    fontFamily: 'monospace',
    fontSize: 14,
    cursor: 'pointer',
    border: '1px solid #ccc',
    borderRadius: 4,
    background: '#fff',
  },
}

```


You are now acting as the senior architect and principal engineer for this entire project. Using your full capabilities and maximum reasoning depth on GPT-5.5 Pro, create an extremely robust, detailed, and professional implementation plan for the following objective:

Your goal is to build an internal search demo of Customer's own documents (SharePoint, Google Drive, Documentum, etc.) through a chat assistant.

Structure the plan with:
- Clear high-level phases
- Detailed step-by-step tasks with exact code/file locations where possible
- Key architectural decisions and trade-offs
- Potential risks, edge cases, and mitigations
- Testing strategy and validation steps
- Recommended execution order and dependencies between tasks

Be as thorough, thoughtful, and production-ready as possible. Think step-by-step with maximum depth.
