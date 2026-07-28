"""
LiteLLM Router setup.

A single logical model name ("salon-llm") is backed by three real
deployments (Groq, Gemini Flash, Mistral). LiteLLM Router transparently:

- Picks the fastest/cheapest healthy deployment first (Groq, since it is
  both the cheapest and lowest-latency of the three), via
  routing_strategy="latency-based-routing".
- Retries a failed request against the next deployment in the group
  (num_retries + the built-in per-deployment failover).
- Temporarily "cools down" (health-checks out) any deployment that keeps
  failing, via allowed_fails/cooldown_time, so a broken provider stops
  being tried until it recovers.

No provider is ever hardcoded into the RAG pipeline: rag.py only ever
calls router.acompletion(model="salon-llm", ...).
"""

from functools import lru_cache

from litellm import Router

from config import get_settings

# Priority order = cheapest & fastest first. Groq's LPU inference is both
# free-tier friendly and extremely low latency, Gemini Flash is the next
# cheapest/fastest option, Mistral is the final fallback.
_MODEL_NAME = "salon-llm"


@lru_cache
def get_router() -> Router:
    settings = get_settings()

    model_list = [
        {
            "model_name": _MODEL_NAME,
            "litellm_params": {
                "model": "groq/llama-3.1-8b-instant",
                "api_key": settings.groq_api_key,
            },
            "model_info": {"id": "groq-primary"},
        },
        {
            "model_name": _MODEL_NAME,
            "litellm_params": {
                "model": "gemini/gemini-1.5-flash",
                "api_key": settings.gemini_api_key,
            },
            "model_info": {"id": "gemini-secondary"},
        },
        {
            "model_name": _MODEL_NAME,
            "litellm_params": {
                "model": "mistral/mistral-small-latest",
                "api_key": settings.mistral_api_key,
            },
            "model_info": {"id": "mistral-tertiary"},
        },
    ]

    return Router(
        model_list=model_list,
        routing_strategy="latency-based-routing",  # latency-aware + naturally cost-aware (Groq wins)
        num_retries=2,  # retry a failed request against the next deployment
        timeout=15,
        cooldown_time=30,  # health checking: temporarily benches a failing deployment
        allowed_fails=2,  # how many failures before a deployment is benched
        retry_after=1,
    )


def get_model_name() -> str:
    return _MODEL_NAME
