from typing import Dict, Optional, Literal, Union, TypedDict, Type, Any, Required

from pydantic import BaseModel, SecretStr

from google.genai import types, client
from google.auth.credentials import Credentials


class ResponseFormatConfig(TypedDict, total=False):
    type: Literal["text", "json_object", "pydantic"]
    schema: Optional[Union[Dict[str, Any], Type[BaseModel]]]


class GeminiClientConfig(TypedDict, total=False):
    """Configuration for Google AI Studio Gemini client."""
    model: Required[str]
    api_key: Required[SecretStr]
    config: types.GenerateContentConfigOrDict
    debug_config: Optional[client.DebugConfig]
    http_options: types.HttpOptionsOrDict
    # Vertex AI specific parameters
    vertexai: Optional[bool]
    credentials: Optional[Credentials]
    location: Optional[str]
    project: Optional[str]