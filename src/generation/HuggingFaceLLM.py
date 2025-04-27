import logging
from threading import Lock
from typing import Dict, Any, Generator

from huggingface_hub import InferenceClient  # no HuggingFaceHubError
from config.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _SingletonMeta(type):
    """
    Thread-safe Singleton metaclass.
    Ensures one instance per model_name.
    """
    _instances: Dict[str, Any] = {}
    _lock: Lock = Lock()

    def __call__(cls, model_name: str, *args, **kwargs):
        if model_name not in cls._instances:
            with cls._lock:
                if model_name not in cls._instances:
                    logger.info(f"Creating new {cls.__name__} for model: {model_name}")
                    cls._instances[model_name] = super().__call__(model_name, *args, **kwargs)
        return cls._instances[model_name]


class HuggingFaceLLM(metaclass=_SingletonMeta):
    """
    Singleton wrapper around HF InferenceClient for chat-completions,
    with error handling and logging.
    """

    def __init__(self, model_name: str):
        """
        Initialize the HF Inference client for the given model.
        """
        self.model_name = model_name
        try:
            self.client = InferenceClient(
                provider="hf-inference",
                api_key=settings.HF_TOKEN.get_secret_value(),
            )
            logger.info(f"InferenceClient initialized for model: {model_name}")
        except Exception as e:
            logger.exception("Failed to initialize InferenceClient")
            raise RuntimeError(f"Could not initialize HF client: {e}") from e

    def _call_api(self, **kwargs) -> Any:
        """
        Internal helper to call HF API with centralized error handling.
        """
        try:
            response = self.client.chat.completions.create(**kwargs)
            logger.info(
                f"API call successful: model={kwargs.get('model')} "
                f"messages={len(kwargs.get('messages', []))}"
            )
            return response
        except Exception as e:
            logger.exception("Error during HF API call")
            raise RuntimeError(f"HF API call failed: {e}") from e

    def get_answer(self, sys_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Generate a chat completion and return the assistant’s answer.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            **kwargs,
        }
        logger.info("get_answer called")
        response = self._call_api(**payload)
        try:
            answer = response.choices[0].message.content
            logger.info(f"get_answer returning {len(answer)} characters")
            return answer
        except (AttributeError, IndexError) as e:
            logger.exception("Malformed response in get_answer")
            raise RuntimeError("HF response missing expected content") from e

    def get_response(self, sys_prompt: str, user_prompt: str, **kwargs) -> Any:
        """
        Generate a chat completion and return the full raw API response.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            **kwargs,
        }
        logger.info("get_response called")
        return self._call_api(**payload)

    def stream_answer(self, sys_prompt: str, user_prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Stream a chat completion, yielding partial content chunks.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": True,
            **kwargs,
        }
        logger.info("stream_answer called")
        try:
            stream = self._call_api(**payload)
            for chunk in stream:
                delta = chunk.choices[0].delta.get("content", "")
                logger.debug(f"stream_answer chunk: {delta!r}")
                yield delta
        except Exception as e:
            logger.exception("Error during stream_answer")
            raise
