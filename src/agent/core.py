import os

from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

from agent.config import SearchConfig


def _langchain_model_name(model_str: str) -> str:
    """LangChain provider inference is inconsistent for new OpenAI aliases."""
    if ":" in model_str:
        return model_str
    if model_str.startswith(("gpt-", "o1", "o3", "o4")):
        return f"openai:{model_str}"
    return model_str


def _default_agent_model() -> str:
    return os.getenv("AGENT_MODEL") or SearchConfig.agent_model


def make_agent(
    model_str: str | None = None,
    system_prompt: str | None = None,
):
    """Create a deep agent with the specified model provider.

    Args:
        model_str: Provider and model in "provider:model" format, or an OpenAI
                   model id such as "gpt-5.5". Defaults to AGENT_MODEL.
                   Examples: "openai:gpt-4o", "anthropic:claude-haiku-4-5-20251001",
                   "google_genai:gemini-2.5-flash"
        system_prompt: Optional system prompt override.

    Returns:
        A compiled LangGraph agent supporting .invoke(), .stream(), .astream().
    """
    model_name = _langchain_model_name(model_str or _default_agent_model())
    model = init_chat_model(model_name)
    kwargs = {}
    if system_prompt:
        kwargs["system_prompt"] = system_prompt
    return create_deep_agent(model=model, **kwargs)
