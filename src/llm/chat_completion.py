from openai import OpenAI

from utils import console, env

from .count_tokens import count_tokens
from .model_specs import get_model_spec, ModelType

# Initialize OpenAI client
_client = None

def get_client():
    global _client
    if _client is None:
        if openai_api_key := env["OPENAI_API_KEY"]:
            _client = OpenAI(api_key=openai_api_key)
        else:
            raise ValueError("Put your OpenAI API key in the OPENAI_API_KEY environment variable.")
    return _client

def compose_system(system):
    return [{
        "role": "system",
        "content": system
    }]

def compose_examples(examples):
    # TODO experiment with 'name' field (https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb)
    out = [] # TODO use list comprehension
    for example in examples:
        out.extend(
            (
                {"role": "user", "content": example[0]},
                {"role": "assistant", "content": example[1]},
            )
        )
    return out

def compose_user(user):
    return [{
        "role": "user",
        "content": user
    }]

def compose_messages(system, examples, user):
    return [
        *compose_system(system),
        *compose_examples(examples),
        *compose_user(user),
    ]

def chat_completion_token_counts(system, examples, user, model: ModelType):
    return {
        "system": count_tokens(compose_system(system), model, count_priming_tokens=False),
        "examples": count_tokens(compose_examples(examples), model, count_priming_tokens=False),
        "user": count_tokens(compose_user(user), model, count_priming_tokens=False),
        "total_prompt": count_tokens(compose_messages(system, examples, user), model),
        "model_max": get_model_spec(model)["max_tokens"]
    }

def chat_completion(system, examples, user, model=ModelType.GPT_4O):
    try:
        client = get_client()
        model_spec = get_model_spec(model)

        messages = compose_messages(system, examples, user)

        response = client.chat.completions.create(
            model=model_spec["id"],
            messages=messages,
            temperature=0,  # based on HuggingGPT
        )
        
        # Convert response to dict-like format for backward compatibility
        return {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                    },
                    "finish_reason": choice.finish_reason,
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
    except Exception as e:
        # New error handling for OpenAI v1.x
        error_type = type(e).__name__
        console.error(f"(llm) {error_type}: {e}")
        return None
