from enum import Enum

# model info: https://platform.openai.com/docs/models
# pricing: https://openai.com/pricing
# token counting: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

class ModelType(Enum):
    # Modern models (recommended)
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    
    # Current GPT-3.5 (still available but less recommended)
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    
    # Legacy GPT-3.5 models (deprecated or being phased out)
    GPT_3_5_TURBO_0301 = "gpt-3.5-turbo-0301"
    GPT_3_5_TURBO_0613 = "gpt-3.5-turbo-0613"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_3_5_TURBO_16K_0613 = "gpt-3.5-turbo-16k-0613"

    # Legacy GPT-4 models (older versions)
    GPT_4 = "gpt-4"
    GPT_4_0314 = "gpt-4-0314"
    GPT_4_0613 = "gpt-4-0613"
    GPT_4_32K = "gpt-4-32k"
    GPT_4_32K_0314 = "gpt-4-32k-0314"
    GPT_4_32K_0613 = "gpt-4-32k-0613"

    # Embeddings
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"

def create_completion_model_spec(name, openai_id, usd_per_prompt_token, usd_per_completion_token, max_tokens, tokens_per_message, tokens_per_name, deprecation_date):
    return {
        "name": name,
        "id": openai_id,
        "usd_per_prompt_token": usd_per_prompt_token,
        "usd_per_completion_token": usd_per_completion_token,
        "max_tokens": max_tokens,
        "tokens_per_message": tokens_per_message,
        "tokens_per_name": tokens_per_name,
        "deprecation_date": deprecation_date,
    }

# Modern models (2024-2026)
gpt_4o_model_spec = create_completion_model_spec(
    name="GPT-4o",
    openai_id="gpt-4o",
    usd_per_prompt_token=2.50 / 1000 / 1000,  # $2.50 per 1M tokens
    usd_per_completion_token=10.00 / 1000 / 1000,  # $10.00 per 1M tokens
    max_tokens=128000,
    tokens_per_message=3,
    tokens_per_name=1,
    deprecation_date=None,
)

gpt_4o_mini_model_spec = create_completion_model_spec(
    name="GPT-4o Mini",
    openai_id="gpt-4o-mini",
    usd_per_prompt_token=0.150 / 1000 / 1000,  # $0.15 per 1M tokens
    usd_per_completion_token=0.600 / 1000 / 1000,  # $0.60 per 1M tokens
    max_tokens=128000,
    tokens_per_message=3,
    tokens_per_name=1,
    deprecation_date=None,
)

gpt_4_turbo_model_spec = create_completion_model_spec(
    name="GPT-4 Turbo",
    openai_id="gpt-4-turbo",
    usd_per_prompt_token=10.00 / 1000 / 1000,  # $10.00 per 1M tokens
    usd_per_completion_token=30.00 / 1000 / 1000,  # $30.00 per 1M tokens
    max_tokens=128000,
    tokens_per_message=3,
    tokens_per_name=1,
    deprecation_date=None,
)

gpt_4_turbo_preview_model_spec = create_completion_model_spec(
    name="GPT-4 Turbo Preview",
    openai_id="gpt-4-turbo-preview",
    usd_per_prompt_token=10.00 / 1000 / 1000,  # $10.00 per 1M tokens
    usd_per_completion_token=30.00 / 1000 / 1000,  # $30.00 per 1M tokens
    max_tokens=128000,
    tokens_per_message=3,
    tokens_per_name=1,
    deprecation_date=None,
)

# Legacy models (kept for compatibility)
gpt_3_5_turbo_0301_model_spec = create_completion_model_spec(
    name="GPT-3.5 Turbo (0301)",
    openai_id="gpt-3.5-turbo-0301",
    usd_per_prompt_token=0.0015 / 1000,
    usd_per_completion_token=0.002 / 1000,
    max_tokens=4096,
    tokens_per_message=4,
    tokens_per_name=-1,
    deprecation_date="2024-09-13", # Deprecated
)

gpt_3_5_turbo_0613_model_spec = create_completion_model_spec(
    name="GPT-3.5 Turbo (0613)",
    openai_id="gpt-3.5-turbo-0613",
    usd_per_prompt_token=0.0015 / 1000,
    usd_per_completion_token=0.002 / 1000,
    max_tokens=4096,
    tokens_per_message=3,
    tokens_per_name=1,
    deprecation_date="2024-09-13", # Deprecated
)

gpt_3_5_turbo_16k_0613_model_spec = create_completion_model_spec(
    name="GPT-3.5 Turbo 16k (0613)",
    openai_id="gpt-3.5-turbo-16k-0613",
    usd_per_prompt_token=0.003 / 1000,
    usd_per_completion_token=0.004 / 1000,
    max_tokens=16384,
    tokens_per_message=3,
    tokens_per_name=1,
    deprecation_date="2024-09-13", # Deprecated
)

gpt_4_0314_model_spec = create_completion_model_spec(
    name="GPT-4 (0314)",
    openai_id="gpt-4-0314",
    usd_per_prompt_token=0.03 / 1000,
    usd_per_completion_token=0.06 / 1000,
    max_tokens=8192,
    tokens_per_message=3,
    tokens_per_name=1,
    deprecation_date="2024-06-13", # Deprecated
)

gpt_4_0613_model_spec = create_completion_model_spec(
    name="GPT-4 (0613)",
    openai_id="gpt-4-0613",
    usd_per_prompt_token=0.03 / 1000,
    usd_per_completion_token=0.06 / 1000,
    max_tokens=8192,
    tokens_per_message=3,
    tokens_per_name=1,
    deprecation_date="2025-06-13", # Deprecated
)

gpt_4_32k_0314_model_spec = create_completion_model_spec(
    name="GPT-4 32k (0314)",
    openai_id="gpt-4-32k-0314",
    usd_per_prompt_token=0.06 / 1000,
    usd_per_completion_token=0.12 / 1000,
    max_tokens=32768,
    tokens_per_message=3,
    tokens_per_name=1,
    deprecation_date="2024-06-13", # Deprecated
)

gpt_4_32k_0613_model_spec = create_completion_model_spec(
    name="GPT-4 32k (0613)",
    openai_id="gpt-4-32k-0613",
    usd_per_prompt_token=0.06 / 1000,
    usd_per_completion_token=0.12 / 1000,
    max_tokens=32768,
    tokens_per_message=3,
    tokens_per_name=1,
    deprecation_date="2025-06-13", # Deprecated
)

text_embedding_ada_002_model_spec = {
    "name": "Text Embedding Ada (002)",
    "id": "text-embedding-ada-002",
    "usd_per_input_token": 0.1 / 1000 / 1000,  # $0.10 per 1M tokens
    "max_tokens": 8191,
}

text_embedding_3_small_model_spec = {
    "name": "Text Embedding 3 Small",
    "id": "text-embedding-3-small",
    "usd_per_input_token": 0.02 / 1000 / 1000,  # $0.02 per 1M tokens
    "max_tokens": 8191,
}

text_embedding_3_large_model_spec = {
    "name": "Text Embedding 3 Large",
    "id": "text-embedding-3-large",
    "usd_per_input_token": 0.13 / 1000 / 1000,  # $0.13 per 1M tokens
    "max_tokens": 8191,
}

# Using the fixed model versions for general reproducibility, and because the continuously updated models dont have guaranteed token counts
model_specs = {
    # Modern models (recommended)
    ModelType.GPT_4O: gpt_4o_model_spec,
    ModelType.GPT_4O_MINI: gpt_4o_mini_model_spec,
    ModelType.GPT_4_TURBO: gpt_4_turbo_model_spec,
    ModelType.GPT_4_TURBO_PREVIEW: gpt_4_turbo_preview_model_spec,
    
    # Current GPT-3.5
    ModelType.GPT_3_5_TURBO:      gpt_3_5_turbo_0613_model_spec,
    ModelType.GPT_3_5_TURBO_0301: gpt_3_5_turbo_0301_model_spec,
    ModelType.GPT_3_5_TURBO_0613: gpt_3_5_turbo_0613_model_spec,
    ModelType.GPT_3_5_TURBO_16K:      gpt_3_5_turbo_16k_0613_model_spec,
    ModelType.GPT_3_5_TURBO_16K_0613: gpt_3_5_turbo_16k_0613_model_spec,

    # Legacy GPT-4 models
    ModelType.GPT_4:      gpt_4_0613_model_spec,
    ModelType.GPT_4_0314: gpt_4_0314_model_spec,
    ModelType.GPT_4_0613: gpt_4_0613_model_spec,
    ModelType.GPT_4_32K:      gpt_4_32k_0613_model_spec,
    ModelType.GPT_4_32K_0314: gpt_4_32k_0314_model_spec,
    ModelType.GPT_4_32K_0613: gpt_4_32k_0613_model_spec,

    # Embeddings
    ModelType.TEXT_EMBEDDING_ADA_002: text_embedding_ada_002_model_spec,
    ModelType.TEXT_EMBEDDING_3_SMALL: text_embedding_3_small_model_spec,
    ModelType.TEXT_EMBEDDING_3_LARGE: text_embedding_3_large_model_spec,
}

def get_model_spec(model_type: ModelType):
    return model_specs[model_type]
