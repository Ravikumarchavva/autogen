from typing import Dict

from autogen_core.models import ModelInfo, ModelFamily

_MODEL_INFO: Dict[str, ModelInfo] = {
    "gemini-1.5-flash": {
        "vision": True,
        'function_calling': True,
        'json_output': True,
        'family': ModelFamily.GEMINI_1_5_FLASH,
        'structured_output': True,
    },
    "gemini-1.5-pro": {
        "vision": True,
        'function_calling': True,
        'json_output': True,
        'family': ModelFamily.GEMINI_1_5_PRO,
        'structured_output': True,
    },
    "gemini-2.0-flash": {
        "vision": True,
        'function_calling': True,
        'json_output': True,
        'family': ModelFamily.GEMINI_2_0_FLASH,
        'structured_output': True,
    },
    "gemini-2.5-pro": {
        "vision": True,
        'function_calling': True,
        'json_output': True,
        'family': ModelFamily.GEMINI_2_5_PRO,
        'structured_output': True,
    },
    "gemini-2.5-flash": {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": ModelFamily.GEMINI_2_5_FLASH,
        "structured_output": True,
    }
}

_MODEL_TOKEN_LIMITS: Dict[str, int] = {
    "gemini-1.5-flash": 1048576,  # 1 million tokens
    "gemini-1.5-pro": 2097152,  # 2 million tokens
    "gemini-2.0-flash": 1048576,
    "gemini-2.5-pro": 1048576,
    "gemini-2.5-flash": 1048576,
}

def get_info(model_name: str) -> ModelInfo:
    return _MODEL_INFO[model_name]

def get_token_limit(model_name: str) -> int:
    if model_name in _MODEL_TOKEN_LIMITS:
        return _MODEL_TOKEN_LIMITS[model_name]
    return 0