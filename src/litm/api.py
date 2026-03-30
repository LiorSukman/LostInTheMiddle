import os
import time

import litm.env  # noqa: F401 — loads .env before anything else
from openai import OpenAI


def get_client(model: str = "gpt-4o-mini") -> tuple[OpenAI, str]:
    """Return an OpenAI client and model name.

    Reads OPENAI_API_KEY from .env at the project root (or environment).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Create a .env file in the project root "
            "or set the environment variable."
        )
    client = OpenAI(api_key=api_key)
    return client, model


def query_model(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 50,
    max_retries: int = 3,
) -> str:
    """Send a prompt and return the model's response text."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"API error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
