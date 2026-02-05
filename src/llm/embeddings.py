from openai import OpenAI
import threading
from utils import env

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

def generate_embedding(text, model="text-embedding-ada-002"):
    """
    Generate an embedding for the given text using the specified model.

    Args:
        text (str): The input text for which to generate the embedding.
        model (str): The model to use for generating the embedding. Default is "text-embedding-ada-002".

    Returns:
        list: The generated embedding as a list of floating-point numbers.
    """
    client = get_client()
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding
