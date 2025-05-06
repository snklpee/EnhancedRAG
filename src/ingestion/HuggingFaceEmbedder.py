# src/ingestion/HuggingFaceEmbedder.py

import logging
from typing import List, Sequence

from tqdm.auto import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings.embeddings import Embeddings

from src.utils.metrics import track_metrics
from src.ingestion.DocumentChunker import DocumentChunker

# configure module‐level logger
logger = logging.getLogger(__name__)


class HuggingFaceEmbedder(Embeddings):
    """
    Composition‐based wrapper around HuggingFaceEmbeddings.

    - Caches one HuggingFaceEmbeddings client per model_name.
    - Implements `embed_query` & `embed_documents` to conform with LangChain’s `Embeddings` interface.
    - Tracks token‐count metrics on inputs and embedding dimensions on outputs.
    - Adds logging, error handling, and progress bars for batch embedding.
    """

    # cache of clients to ensure one instance per model name
    _clients: dict[str, HuggingFaceEmbeddings] = {}

    def __init__(self, model_name: str):
        """
        Initialize the embedder.

        Args:
            model_name: HF model identifier for sentence embeddings.
        """
        logger.info(f"Initializing HuggingFaceEmbedder(model_name='{model_name}')")
        self.model_name = model_name
        try:
            if model_name not in self._clients:
                self._clients[model_name] = HuggingFaceEmbeddings(model_name=model_name)
            self._client = self._clients[model_name]
        except Exception as e:
            logger.exception(f"Failed to instantiate HuggingFaceEmbeddings for '{model_name}'")
            raise

    def _token_count(self, texts: Sequence[str] | str) -> int:
        """
        Compute total token count for input text(s) using DocumentChunker.

        Args:
            texts: A single string or sequence of strings to count tokens for.

        Returns:
            Sum of tokens across all provided texts.

        Raises:
            Exception if token counting fails.
        """
        try:
            chunker = DocumentChunker(hf_embedding_model=self.model_name)
            if isinstance(texts, str):
                return chunker.get_token_count(text=texts)
            return sum(chunker.get_token_count(text=t) for t in texts)
        except Exception:
            logger.exception("Error computing token count")
            raise

    @track_metrics(_token_count, target="inputs")
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.

        Args:
            text: Input text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            Exception if embedding fails.
        """
        logger.debug(f"Embedding query (length={len(text)} chars)")
        try:
            vector = self._client.embed_query(text)
            logger.debug(f"Embedding dimension: {len(vector)}")
            return vector
        except Exception:
            logger.exception("Error in embed_query")
            raise

    @track_metrics(_token_count, target="inputs")
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of documnets, displaying progress.

        Args:
            texts: List of strings to embed.

        Returns:
            A list of embedding vectors (one per input string).

        Raises:
            Exception if any document embedding fails.
        """
        n = len(texts)
        logger.info(f"Embedding {n} documents with model '{self.model_name}'")
        embeddings: List[List[float]] = []

        # tqdm progress bar over individual texts
        for idx, txt in enumerate(tqdm(texts, desc="Embedding documents", unit="doc")):
            try:
                # call single-doc embedding for progress tracking;
                # HuggingFaceEmbeddings.embed_documents returns a list
                vec = self._client.embed_documents([txt])[0]
                embeddings.append(vec)
            except Exception:
                logger.exception(f"Failed to embed document at index {idx}")
                raise

        logger.info(f"Successfully embedded {len(embeddings)}/{n} documents")
        return embeddings
