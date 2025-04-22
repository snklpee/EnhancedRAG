"""
Generation pipeline package:
- GenerateAnswer: LLM‑based answer generator
- merge_fusion: combines multiple docs into prompt
- prompt_augmentation: enriches the prompt
- query_embedding: obtains embeddings for queries
- vector_search: retrieves nearest neighbors from vector DB
"""

from .generate_answer import GenerateAnswer
from .merge_fusion import generate_and_fuse
from .prompt_augmentation import augment_prompt
from .query_embedding import get_query_embedding
from .vector_search import vector_search

__all__ = [
    "GenerateAnswer",
    "generate_and_fuse",
    "augment_prompt",
    "get_query_embedding",
    "vector_search",
]
