"""
Model registry and API client configuration for evaluation runs.

Reads provider API keys from environment variables (OPENAI_API_KEY, GEMINI_API_KEY,
OPENROUTER_API_KEY) and instantiates one client per provider - OpenAI, Gemini, and
OpenRouter (via the OpenAI-compatible API).

MODELS maps each model id to its metadata: display name and version (used
in result CSVs and analysis grouping), the client to call it through, and
whether to request chain-of-thought. NO_TEMPERATURE_MODELS lists models
that reject an explicit temperature parameter (run at their default).

Requires the relevant environment variables to be set; clients whose key
is missing are None, and calling a model routed through a None client will
fail at call time.
"""


from types import SimpleNamespace
from openai import OpenAI
from google import genai
import os


NO_TEMPERATURE_MODELS = { # models with default temperature 1 in our experiments
    "gpt-5-2025-08-07",
    "gpt-5-nano-2025-08-07",
}


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
router_client = OpenAI(api_key=ROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")


MODELS = {
    # ChatGPT
    "gpt-5-2025-08-07": SimpleNamespace(model="ChatGPT", version="gpt-5", client=openai_client, cot=False),
    "gpt-5-nano-2025-08-07": SimpleNamespace(model="ChatGPT", version="gpt-5-nano", client=openai_client, cot=False),

    # Qwen
    "qwen/qwen3-235b-a22b-thinking-2507": SimpleNamespace(model="Qwen", version="3-235b-a22b-thinking", client=router_client, cot=True),

    # Gemini
    "gemini-2.5-pro": SimpleNamespace(model="Gemini", version="2.5-pro", client=gemini_client, cot=True),

    # Claude
    "anthropic/claude-opus-4.6": SimpleNamespace(model="Claude", version="opus-4.6", client=router_client, cot=True),
}
