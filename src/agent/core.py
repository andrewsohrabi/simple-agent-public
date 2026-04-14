from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

DEFAULT_MODEL_STR = "anthropic:claude-opus-4-6"


def make_agent(
    model_str: str = DEFAULT_MODEL_STR,
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
