import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def call_llm(prompt, temperature=0):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


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
