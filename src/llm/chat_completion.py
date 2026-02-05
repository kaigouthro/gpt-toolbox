from openai import OpenAI, APIError, APIConnectionError, RateLimitError
import threading

from utils import console, env

from .count_tokens import count_tokens
from .model_specs import get_model_spec, ModelType

# Initialize OpenAI client with thread-safe pattern
_client = None
_client_lock = threading.Lock()

def get_client():
    global _client
    if _client is None:
        with _client_lock:
            # Double-check locking pattern
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
        
        # Convert response to dict-like format for backward compatibility,
        # while preserving optional fields when present.
        choices = []
        for choice in response.choices:
            message_dict = {
                "role": choice.message.role,
                "content": choice.message.content,
            }

            # Optional message-level fields (may not be present on all responses)
            function_call = getattr(choice.message, "function_call", None)
            if function_call is not None:
                message_dict["function_call"] = function_call

            tool_calls = getattr(choice.message, "tool_calls", None)
            if tool_calls is not None:
                message_dict["tool_calls"] = tool_calls

            choice_dict = {
                "index": choice.index,
                "message": message_dict,
                "finish_reason": choice.finish_reason,
            }

            # Optional choice-level fields
            logprobs = getattr(choice, "logprobs", None)
            if logprobs is not None:
                choice_dict["logprobs"] = logprobs

            choices.append(choice_dict)

        result = {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": choices,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

        # Optional top-level fields
        system_fingerprint = getattr(response, "system_fingerprint", None)
        if system_fingerprint is not None:
            result["system_fingerprint"] = system_fingerprint

        return result
    except APIError as e:
        console.error(f"(llm) OpenAI API returned an API Error: {e}")
        return None
    except APIConnectionError as e:
        console.error(f"(llm) Failed to connect to OpenAI API: {e}")
        return None
    except RateLimitError as e:
        console.error(f"(llm) OpenAI API request exceeded rate limit: {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        error_type = type(e).__name__
        console.error(f"(llm) {error_type}: {e}")
        return None
