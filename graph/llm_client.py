from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from config import get_settings

log = structlog.get_logger()
settings = get_settings()

# OpenRouter exposes an OpenAI-compatible API — just swap base_url + api_key
_client = AsyncOpenAI(
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
    default_headers={
        "HTTP-Referer": "https://github.com/memgraph",
        "X-Title": "MemGraph",
    },
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def chat(
    prompt: str,
    system: str = "You are a helpful AI assistant.",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    model = model or settings.qa_model
    log.debug("llm.request", model=model, prompt_len=len(prompt))

    response = await _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content or ""
    log.debug("llm.response", tokens=response.usage.total_tokens if response.usage else None)
    return text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def chat_json(
    prompt: str,
    system: str = "You are a helpful AI assistant. Always respond with valid JSON only.",
    model: str | None = None,
) -> str:
    """Like chat() but nudges the model to return JSON. Parse the result yourself."""
    model = model or settings.extraction_model
    return await chat(prompt=prompt, system=system, model=model, temperature=0.0)
