import base64
import json
import logging
import warnings
import tiktoken
from typing import Any, AsyncGenerator, Dict, List, Literal, Mapping, Optional, Sequence, Set, Union
from pydantic import BaseModel, SecretStr
from typing_extensions import Self, Unpack

from google import genai
from google.genai import types
from google.auth.credentials import Credentials

from autogen_core.tools import Tool, ToolSchema
from autogen_core import EVENT_LOGGER_NAME, TRACE_LOGGER_NAME, CancellationToken, Component, FunctionCall, Image
from autogen_core.logging import LLMCallEvent, LLMStreamEndEvent, LLMStreamStartEvent
from autogen_core.models import AssistantMessage, ChatCompletionClient, CreateResult, FinishReasons, FunctionExecutionResultMessage, LLMMessage, ModelCapabilities, ModelInfo, RequestUsage, SystemMessage, UserMessage, validate_model_info

from . import _model_info
from .config import GeminiClientConfig, ResponseFormatConfig

logger = logging.getLogger(EVENT_LOGGER_NAME) 
trace_logger = logging.getLogger(TRACE_LOGGER_NAME)

# Common parameters for content generation
gemini_content_params = {
    "model",
    "contents",
    "config",
    "system_instruction",
}

disallowed_create_args = {"contents"}
required_create_args: Set[str] = {"model"}

gemini_init_kwargs = {
    "api_key",
    "vertexai",
    "credentials", 
    "location",
    "project",
    "debug_config",
    "http_options"
}


def _gemini_client_from_config(config: Mapping[str, Any]) -> genai.Client:
    """Create a Gemini client from configuration."""
    client_config = {k: v for k, v in config.items() if k in gemini_init_kwargs}
    
    # Determine if this is Vertex AI or Google AI Studio
    has_api_key = "api_key" in client_config
    has_project = "project" in client_config
    explicit_vertexai = client_config.get("vertexai", False)
    
    # Validation logic
    if explicit_vertexai or has_project:
        # Vertex AI mode
        if has_api_key:
            raise ValueError("Cannot use api_key with Vertex AI. Use Google Cloud credentials instead.")
        if not has_project:
            raise ValueError("project is required for Vertex AI")
        client_config["vertexai"] = True
    else:
        # Google AI Studio mode
        if not has_api_key:
            raise ValueError("api_key is required for Google AI Studio")
        client_config["vertexai"] = False
    
    # Handle API key
    if "api_key" in client_config:
        api_key = client_config.pop("api_key")
        if isinstance(api_key, SecretStr):
            api_key = api_key.get_secret_value()
        client_config["api_key"] = api_key
    
    return genai.Client(**client_config)


def _create_args_from_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract create arguments from configuration."""
    create_args = {k: v for k, v in config.items() if k in gemini_content_params or k == "model"}
    create_args_keys = set(create_args.keys())

    if not required_create_args.issubset(create_args_keys):
        raise ValueError(f"Required create args are missing: {required_create_args - create_args_keys}")

    if disallowed_create_args.intersection(create_args_keys):
        raise ValueError(f"Disallowed create args are present: {disallowed_create_args.intersection(create_args_keys)}")

    return create_args


def type_to_role(message: LLMMessage) -> str:
    """Convert LLMMessage type to Gemini role."""
    if isinstance(message, SystemMessage):
        return "system"
    elif isinstance(message, UserMessage):
        return "user"
    elif isinstance(message, AssistantMessage):
        return "model"
    else:
        return "user"  # Tool results go as user messages in Gemini


def get_mime_type_from_image(image: Image) -> str:
    """Get MIME type from Image object."""
    base64_data = image.to_base64()
    image_data = base64.b64decode(base64_data)

    if image_data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    elif image_data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif image_data.startswith(b"GIF87a") or image_data.startswith(b"GIF89a"):
        return "image/gif"
    elif image_data.startswith(b"RIFF") and image_data[8:12] == b"WEBP":
        return "image/webp"
    else:
        return "image/jpeg"  # Default fallback


def convert_tool_choice_gemini(tool_choice: Tool | Literal["auto", "required", "none"]) -> Optional[types.ToolConfig]:
    """Convert tool_choice to Gemini ToolConfig format."""
    if tool_choice == "none":
        return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE"))
    
    if tool_choice == "auto":
        return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="AUTO"))
    
    if tool_choice == "required":
        return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY"))
    
    # Must be a Tool object
    if isinstance(tool_choice, Tool):
        return types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=[tool_choice.schema["name"]]
            )
        )
    else:
        raise ValueError(f"tool_choice must be a Tool object, 'auto', 'required', or 'none', got {type(tool_choice)}")


def user_message_to_gemini(message: UserMessage) -> types.Content:
    """Convert UserMessage to Gemini Content."""
    if isinstance(message.content, str):
        return types.Content(
            role="user",
            parts=[types.Part(text=message.content)]
        )
    else:
        parts: List[types.Part] = []
        
        for part in message.content:
            if isinstance(part, str):
                parts.append(types.Part(text=part))
            elif isinstance(part, Image):
                parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=get_mime_type_from_image(part),
                            data=base64.b64decode(part.to_base64())
                        )
                    )
                )
            else:
                raise ValueError(f"Unknown content type: {part}")

        return types.Content(role="user", parts=parts)


def system_message_to_gemini(message: SystemMessage) -> str:
    """Convert SystemMessage to system instruction string."""
    return message.content


def assistant_message_to_gemini(message: AssistantMessage) -> types.Content:
    """Convert AssistantMessage to Gemini Content."""
    if isinstance(message.content, list):
        # Function calls
        parts: List[types.Part] = []
        
        # Add thought if available
        if hasattr(message, "thought") and message.thought:
            parts.append(types.Part(text=message.thought))
        
        # Add function calls
        for func_call in message.content:
            # Parse arguments
            args = func_call.arguments
            if isinstance(args, str):
                try:
                    args_dict = json.loads(args)
                except json.JSONDecodeError:
                    args_dict = {"text": args}
            else:
                args_dict = args
            
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        name=func_call.name,
                        args=args_dict
                    )
                )
            )
        
        return types.Content(role="model", parts=parts)
    else:
        # Simple text content
        return types.Content(
            role="model",
            parts=[types.Part(text=message.content)]
        )


def tool_message_to_gemini(message: FunctionExecutionResultMessage) -> types.Content:
    """Convert FunctionExecutionResultMessage to Gemini Content."""
    parts: List[types.Part] = []
    
    for result in message.content:
        parts.append(
            types.Part(
                function_response=types.FunctionResponse(
                    name=result.call_id,  # Use call_id as name for response
                    response={"result": result.content}
                )
            )
        )
    
    return types.Content(role="user", parts=parts)


def to_gemini_type(message: LLMMessage) -> Union[str, types.Content]:
    """Convert LLMMessage to appropriate Gemini type."""
    if isinstance(message, SystemMessage):
        return system_message_to_gemini(message)
    elif isinstance(message, UserMessage):
        return user_message_to_gemini(message)
    elif isinstance(message, AssistantMessage):
        return assistant_message_to_gemini(message)
    else:
        return tool_message_to_gemini(message)


def convert_tools(tools: Sequence[Tool | ToolSchema]) -> List[types.Tool]:
    """Convert tools to Gemini Tool format."""
    result: List[types.Tool] = []
    
    for tool in tools:
        if isinstance(tool, Tool):
            tool_schema = tool.schema
        else:
            tool_schema = tool
        
        # Convert parameters schema
        parameters = tool_schema.get("parameters", {})
        
        # Create function declaration
        function_declaration = types.FunctionDeclaration(
            name=tool_schema["name"],
            description=tool_schema.get("description", ""),
            parameters=types.Schema(
                type=parameters.get("type", "object"),
                properties={
                    name: types.Schema(**prop) 
                    for name, prop in parameters.get("properties", {}).items()
                },
                required=parameters.get("required", [])
            )
        )
        
        result.append(types.Tool(function_declarations=[function_declaration]))
    
    return result


def normalize_stop_reason(finish_reason: str | None) -> FinishReasons:
    """Convert Gemini finish reason to standard format."""
    if finish_reason is None:
        return "unknown"
    
    # Map Gemini finish reasons
    KNOWN_STOP_MAPPINGS: Dict[str, FinishReasons] = {
        "STOP": "stop",
        "MAX_TOKENS": "length", 
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "OTHER": "unknown",
    }
    
    return KNOWN_STOP_MAPPINGS.get(finish_reason.upper(), "unknown")


def _add_usage(usage1: RequestUsage, usage2: RequestUsage) -> RequestUsage:
    """Add two RequestUsage objects."""
    return RequestUsage(
        prompt_tokens=usage1.prompt_tokens + usage2.prompt_tokens,
        completion_tokens=usage1.completion_tokens + usage2.completion_tokens,
    )


class BaseGeminiChatCompletionClient(ChatCompletionClient):
    """Base class for Gemini chat completion clients."""
    
    def __init__(
        self,
        client: genai.Client,
        *,
        create_args: Dict[str, Any],
        model_info: Optional[ModelInfo] = None,
    ):
        self._client = client
        
        if model_info is None:
            try:
                self._model_info = _model_info.get_info(create_args["model"])
            except ValueError as err:
                raise ValueError("model_info is required when model name is not recognized") from err
        else:
            self._model_info = model_info
        
        # Validate model_info
        validate_model_info(self._model_info)
        
        self._create_args = create_args
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        # Copy create args and update with extra args
        create_args = self._create_args.copy()
        create_args.update(extra_create_args)
        
        # Check for vision capability
        if self.model_info["vision"] is False:
            for message in messages:
                if isinstance(message, UserMessage):
                    if isinstance(message.content, list) and any(isinstance(x, Image) for x in message.content):
                        raise ValueError("Model does not support vision and image was provided")
        
        # Process messages
        system_instruction = None
        contents: List[types.Content] = []
        
        for message in messages:
            converted = to_gemini_type(message)
            if isinstance(converted, str):
                if system_instruction is None:
                    system_instruction = converted
                else:
                    raise ValueError("Multiple system messages are not supported")
            else:
                contents.append(converted)
        
        # Check for function calling support
        if self.model_info["function_calling"] is False and len(tools) > 0:
            raise ValueError("Model does not support function calling")
        
        # Setup generation config
        generation_config = types.GenerateContentConfig()
        
        # Handle JSON output
        if json_output is not None:
            if self.model_info["json_output"] is False:
                raise ValueError("Model does not support JSON output")
            
            if json_output is True:
                generation_config.response_mime_type = "application/json"
            elif isinstance(json_output, type) and issubclass(json_output, BaseModel):
                if self.model_info["structured_output"] is False:
                    raise ValueError("Model does not support structured output")
                generation_config.response_mime_type = "application/json"
                generation_config.response_schema = json_output.model_json_schema()
        
        # Setup tools
        gemini_tools = None
        tool_config = None
        
        if len(tools) > 0:
            gemini_tools = convert_tools(tools)
            tool_config = convert_tool_choice_gemini(tool_choice)
        
        # Build request
        request_args = {
            "model": create_args["model"],
            "contents": contents,
            "config": generation_config,
        }
        
        if system_instruction:
            request_args["system_instruction"] = system_instruction
        
        if gemini_tools:
            request_args["tools"] = gemini_tools
        
        if tool_config:
            request_args["tool_config"] = tool_config
        
        # Execute request
        try:
            response = await self._client.aio.models.generate_content(**request_args)
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}") from e
        
        # Process response
        if not response.candidates:
            raise RuntimeError("No candidates returned from Gemini")
        
        candidate = response.candidates[0]
        
        # Extract usage
        usage_metadata = response.usage_metadata
        usage = RequestUsage(
            prompt_tokens=usage_metadata.prompt_token_count if usage_metadata else 0,
            completion_tokens=usage_metadata.candidates_token_count if usage_metadata else 0,
        )
        
        # Process content
        content: Union[str, List[FunctionCall]]
        thought = None
        
        # Check for function calls
        function_calls = []
        text_parts = []
        
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append(
                    FunctionCall(
                        id=f"call_{len(function_calls)}",  # Generate ID
                        name=part.function_call.name,
                        arguments=json.dumps(dict(part.function_call.args))
                    )
                )
            elif part.text:
                text_parts.append(part.text)
        
        if function_calls:
            content = function_calls
            if text_parts:
                thought = "".join(text_parts)
        else:
            content = "".join(text_parts)
        
        # Create result
        result = CreateResult(
            finish_reason=normalize_stop_reason(candidate.finish_reason),
            content=content,
            usage=usage,
            cached=False,
            thought=thought,
        )
        
        # Update usage
        self._total_usage = _add_usage(self._total_usage, usage)
        self._actual_usage = _add_usage(self._actual_usage, usage)
        
        return result
    
    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
        max_consecutive_empty_chunk_tolerance: int = 0,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        """Create streaming response."""
        # Similar setup as create() but with streaming
        create_args = self._create_args.copy()
        create_args.update(extra_create_args)
        
        # Check for vision capability
        if self.model_info["vision"] is False:
            for message in messages:
                if isinstance(message, UserMessage):
                    if isinstance(message.content, list) and any(isinstance(x, Image) for x in message.content):
                        raise ValueError("Model does not support vision and image was provided")
        
        # Process messages
        system_instruction = None
        contents: List[types.Content] = []
        
        for message in messages:
            converted = to_gemini_type(message)
            if isinstance(converted, str):
                if system_instruction is None:
                    system_instruction = converted
                else:
                    raise ValueError("Multiple system messages are not supported")
            else:
                contents.append(converted)
        
        # Check for function calling support
        if self.model_info["function_calling"] is False and len(tools) > 0:
            raise ValueError("Model does not support function calling")
        
        # Setup generation config
        generation_config = types.GenerateContentConfig()
        
        # Handle JSON output
        if json_output is not None:
            if self.model_info["json_output"] is False:
                raise ValueError("Model does not support JSON output")
            
            if json_output is True:
                generation_config.response_mime_type = "application/json"
            elif isinstance(json_output, type) and issubclass(json_output, BaseModel):
                if self.model_info["structured_output"] is False:
                    raise ValueError("Model does not support structured output")
                generation_config.response_mime_type = "application/json"
                generation_config.response_schema = json_output.model_json_schema()
        
        # Setup tools
        gemini_tools = None
        tool_config = None
        
        if len(tools) > 0:
            gemini_tools = convert_tools(tools)
            tool_config = convert_tool_choice_gemini(tool_choice)
        
        # Build request
        request_args = {
            "model": create_args["model"],
            "contents": contents,
            "config": generation_config,
        }
        
        if system_instruction:
            request_args["system_instruction"] = system_instruction
        
        if gemini_tools:
            request_args["tools"] = gemini_tools
        
        if tool_config:
            request_args["tool_config"] = tool_config
        
        # Start streaming
        try:
            stream = await self._client.aio.models.generate_content_stream(**request_args)
        except Exception as e:
            raise RuntimeError(f"Gemini streaming API call failed: {e}") from e
        
        # Process stream
        text_content: List[str] = []
        function_calls: Dict[str, Dict[str, Any]] = {}
        usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        finish_reason = None
        
        first_chunk = True
        
        async for chunk in stream:
            if first_chunk:
                first_chunk = False
                logger.info(LLMStreamStartEvent(messages=[]))
            
            if chunk.candidates:
                candidate = chunk.candidates[0]
                
                for part in candidate.content.parts:
                    if part.text:
                        text_content.append(part.text)
                        yield part.text
                    elif part.function_call:
                        call_id = f"call_{len(function_calls)}"
                        function_calls[call_id] = {
                            "id": call_id,
                            "name": part.function_call.name,
                            "arguments": json.dumps(dict(part.function_call.args))
                        }
                
                if candidate.finish_reason:
                    finish_reason = candidate.finish_reason
            
            # Update usage if available
            if chunk.usage_metadata:
                usage = RequestUsage(
                    prompt_tokens=chunk.usage_metadata.prompt_token_count,
                    completion_tokens=chunk.usage_metadata.candidates_token_count,
                )
        
        # Create final result
        if function_calls:
            content = [
                FunctionCall(
                    id=call_data["id"],
                    name=call_data["name"], 
                    arguments=call_data["arguments"]
                )
                for call_data in function_calls.values()
            ]
            thought = "".join(text_content) if text_content else None
        else:
            content = "".join(text_content)
            thought = None
        
        result = CreateResult(
            finish_reason=normalize_stop_reason(finish_reason),
            content=content,
            usage=usage,
            cached=False,
            thought=thought,
        )
        
        logger.info(LLMStreamEndEvent(
            response=result.model_dump(),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        ))
        
        # Update usage
        self._total_usage = _add_usage(self._total_usage, usage)
        self._actual_usage = _add_usage(self._actual_usage, usage)
        
        yield result
    
    async def close(self) -> None:
        """Close the client."""
        if hasattr(self._client, 'close'):
            await self._client.close()
    
    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        """Estimate token count for messages and tools."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = tiktoken.get_encoding("gpt2")
        
        num_tokens = 0
        
        # Count message tokens
        for message in messages:
            num_tokens += 10  # Message overhead
            
            if isinstance(message, (UserMessage, AssistantMessage)):
                if isinstance(message.content, str):
                    num_tokens += len(encoding.encode(message.content))
                elif isinstance(message.content, list):
                    for part in message.content:
                        if isinstance(part, str):
                            num_tokens += len(encoding.encode(part))
                        elif isinstance(part, Image):
                            num_tokens += 512  # Image estimation
                        elif isinstance(part, FunctionCall):
                            num_tokens += len(encoding.encode(part.name))
                            num_tokens += len(encoding.encode(part.arguments))
            elif isinstance(message, SystemMessage):
                num_tokens += len(encoding.encode(message.content))
            elif isinstance(message, FunctionExecutionResultMessage):
                for result in message.content:
                    num_tokens += len(encoding.encode(result.content))
        
        # Count tool tokens
        for tool in tools:
            if isinstance(tool, Tool):
                tool_schema = tool.schema
            else:
                tool_schema = tool
            
            num_tokens += len(encoding.encode(tool_schema["name"]))
            if "description" in tool_schema:
                num_tokens += len(encoding.encode(tool_schema["description"]))
            
            # Parameters
            if "parameters" in tool_schema:
                params = tool_schema["parameters"]
                if "properties" in params:
                    for prop_name, prop_schema in params["properties"].items():
                        num_tokens += len(encoding.encode(prop_name))
                        if "description" in prop_schema:
                            num_tokens += len(encoding.encode(prop_schema["description"]))
            
            num_tokens += 20  # Tool overhead
        
        return num_tokens
    
    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        """Calculate remaining tokens based on model limit."""
        token_limit = _model_info.get_token_limit(self._create_args["model"])
        return token_limit - self.count_tokens(messages, tools=tools)
    
    def actual_usage(self) -> RequestUsage:
        return self._actual_usage
    
    def total_usage(self) -> RequestUsage:
        return self._total_usage
    
    @property
    def capabilities(self) -> ModelCapabilities:
        warnings.warn("capabilities is deprecated, use model_info instead", DeprecationWarning, stacklevel=2)
        return self._model_info
    
    @property
    def model_info(self) -> ModelInfo:
        return self._model_info


class GeminiChatCompletionClient(BaseGeminiChatCompletionClient):
    """
    Unified chat completion client for Google's Gemini models.
    
    Supports both Google AI Studio and Vertex AI based on provided configuration:
    
    **Google AI Studio Usage:**
    - Requires: `api_key`
    - Optional: All other parameters are ignored
    
    **Vertex AI Usage:**
    - Requires: `project`
    - Optional: `location` (defaults to "us-central1"), `credentials`
    - Note: `api_key` should NOT be provided for Vertex AI
    
    Args:
        model (str): The Gemini model to use (e.g., "gemini-1.5-pro", "gemini-2.0-flash")
        api_key (SecretStr, optional): Google AI Studio API key (for Google AI Studio)
        project (str, optional): Google Cloud project ID (for Vertex AI)
        location (str, optional): Google Cloud location (for Vertex AI, defaults to "us-central1")
        credentials (Credentials, optional): Google Cloud credentials (for Vertex AI)
        vertexai (bool, optional): Explicit flag to use Vertex AI (auto-detected if not provided)
        model_info (ModelInfo, optional): Model capabilities override
        debug_config (DebugConfig, optional): Debug configuration
        http_options (HttpOptionsOrDict, optional): HTTP client options
    
    To use this client, you must install the Gemini extension:

    .. code-block:: bash

        pip install "autogen-ext[gemini]"

    **Google AI Studio Example:**
    
    .. code-block:: python
    
        import asyncio
        from autogen_ext.models.gemini import GeminiChatCompletionClient
        from autogen_core.models import UserMessage
        from pydantic import SecretStr
        
        
        async def main():
            # Google AI Studio client
            client = GeminiChatCompletionClient(
                model="gemini-1.5-pro",
                api_key=SecretStr("your-google-ai-api-key")
            )
            
            result = await client.create([
                UserMessage(content="What is the capital of France?", source="user")
            ])
            print(result)
        
        
        if __name__ == "__main__":
            asyncio.run(main())
    
    **Vertex AI Example:**
    
    .. code-block:: python
    
        import asyncio
        from autogen_ext.models.gemini import GeminiChatCompletionClient
        from autogen_core.models import UserMessage
        
        
        async def main():
            # Vertex AI client
            client = GeminiChatCompletionClient(
                model="gemini-1.5-pro",
                project="your-gcp-project-id",
                location="us-central1"  # optional
            )
            
            result = await client.create([
                UserMessage(content="What is the capital of France?", source="user")
            ])
            print(result)
        
        
        if __name__ == "__main__":
            asyncio.run(main())
    """
    
    def __init__(self, **kwargs: Unpack[GeminiClientConfig]):
        if "model" not in kwargs:
            raise ValueError("model is required for GeminiChatCompletionClient")
        
        self._raw_config: Dict[str, Any] = dict(kwargs).copy()
        copied_args = dict(kwargs).copy()
        
        model_info: Optional[ModelInfo] = None
        if "model_info" in kwargs:
            model_info = kwargs["model_info"]
            del copied_args["model_info"]
        
        # The validation logic is now in _gemini_client_from_config
        client = _gemini_client_from_config(copied_args)
        create_args = _create_args_from_config(copied_args)
        
        super().__init__(
            client=client,
            create_args=create_args,
            model_info=model_info,
        )
    
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        return state
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = _gemini_client_from_config(state["_raw_config"])
