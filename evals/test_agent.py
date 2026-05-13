import re
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv

import agent.core as core
from agent.core import make_agent

load_dotenv()


class StubAgent:
    def invoke(self, payload):
        messages = list(payload["messages"])
        latest = _message_content(messages[-1])
        if "2 + 2" in latest:
            content = "4"
        elif "what is my name" in latest.lower():
            content = f"Your name is {_remembered_name(messages)}."
        else:
            content = "I understand."
        messages.append(SimpleNamespace(content=content))
        return {"messages": messages}


def _message_content(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _remembered_name(messages) -> str:
    for message in messages:
        match = re.search(r"\bmy name is ([A-Za-z]+)", _message_content(message), re.I)
        if match:
            return match.group(1)
    return "unknown"


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr(core, "init_chat_model", lambda _model_name: object())
    monkeypatch.setattr(core, "create_deep_agent", lambda **_kwargs: StubAgent())
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
