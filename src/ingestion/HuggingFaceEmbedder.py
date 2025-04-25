from typing import Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.metrics import track_metrics

class _EmbedderSingletonMeta(type):
    """
    Metaclass that ensures only one instance per (class, model_name) is ever created.
    """
    _instances: dict[tuple[type, str], object] = {}

    def __call__(cls, model_name: str, *args, **kwargs):
        key = (cls, model_name)
        if key not in cls._instances:
            # first time we see this model_name → create & cache
            cls._instances[key] = super().__call__(model_name, *args, **kwargs)
        return cls._instances[key]


class HuggingFaceEmbedder(metaclass=_EmbedderSingletonMeta):
    """
    Singleton-backed wrapper around HuggingFaceEmbeddings.
    
    Usage:
        embedder = HuggingFaceEmbedder("sentence-transformers/all-mpnet-base-v2")
        vec, dim = embedder.embed_query("hello world")
        
        # Later on, reusing the same model string:
        again = HuggingFaceEmbedder("sentence-transformers/all-mpnet-base-v2")
        # again is the same object as embedder
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        # instantiate the heavy HF object only once per model_name
        self._client = HuggingFaceEmbeddings(model_name=model_name)
        
        
    @track_metrics(lambda txt, vec: print("embedding dimension",vec))
    def embed_query(self, text: str) -> Tuple[list[float], int]:
        """
        Embed a single piece of text and return (vector, dimension).
        """
        vector = self._client.embed_query(text)
        return vector, len(vector)
    
         
        
