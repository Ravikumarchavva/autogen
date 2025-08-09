from typing import Dict, Optional, Literal, Union, TypedDict, Type, Any, Required

from pydantic import BaseModel, SecretStr

from google.genai import types, client
from google.auth.credentials import Credentials

class ResponseFormatConfig(TypedDict, total=False):
    type: Literal["text", "json_object", "pydantic"]
    schema: Optional[Union[Dict[str, Any], Type[BaseModel]]]

class GeminiChatClientArguments(TypedDict, total=False):
    vertexai: Optional[bool]
    credentials: Optional[Credentials]
    location: Optional[str]
    project: Optional[str]
    debug_config: Optional[client.DebugConfig]
    http_options: types.HttpOptionsOrDict

class GeminiClientConfig(GeminiChatClientArguments):
    api_key: Required[SecretStr]

class GeminiContentConfig(TypedDict, total=False):
    config: types.GenerateContentConfigOrDict