import os
import json
import time
import logging
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
logger = logging.getLogger(__name__)

MAX_RETRIES = 8
BASE_DELAY = 2  # seconds


def _retry_api_call(func):
    """Decorator: exponential backoff on rate-limit / transient errors."""
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                delay = BASE_DELAY * (2 ** attempt)  # 2, 4, 8, 16, 32, 64, 128, 256
                logger.warning(f"Rate limit hit (attempt {attempt+1}/{MAX_RETRIES}), "
                               f"retrying in {delay}s...")
                time.sleep(delay)
            except (APITimeoutError, APIConnectionError) as e:
                delay = BASE_DELAY * (2 ** attempt)
                logger.warning(f"API error: {e} (attempt {attempt+1}/{MAX_RETRIES}), "
                               f"retrying in {delay}s...")
                time.sleep(delay)
        # Final attempt — let exceptions propagate
        return func(*args, **kwargs)
    return wrapper


@_retry_api_call
def call_llm(prompt, temperature=0):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


@_retry_api_call
def call_llm_with_logprobs(prompt, temperature=0.2):
    """Call LLM and return (response_text, list_of_token_logprobs).
    Used for computing local uncertainty (predictive entropy) per the paper."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        logprobs=True,
        top_logprobs=1,
    )
    content = response.choices[0].message.content.strip()
    logprobs_content = response.choices[0].logprobs.content or []
    token_logprobs = [t.logprob for t in logprobs_content]
    return content, token_logprobs


@_retry_api_call
def call_llm_json(prompt, temperature=0):
    """Call LLM with JSON output mode. Returns parsed dict."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


@_retry_api_call
def call_llm_json_with_logprobs(prompt, temperature=0.2):
    """Call LLM with JSON output mode AND logprobs.
    Returns (parsed_dict, token_logprobs)."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        logprobs=True,
        top_logprobs=1,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content.strip()
    logprobs_content = response.choices[0].logprobs.content or []
    token_logprobs = [t.logprob for t in logprobs_content]
    try:
        return json.loads(content), token_logprobs
    except json.JSONDecodeError:
        return {}, token_logprobs
