# src/ingestion/HuggingFaceEmbedder.py

from typing import List, Sequence
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings.embeddings import Embeddings
from src.utils.metrics import track_metrics
from src.ingestion.DocumentChunker import DocumentChunker


class HuggingFaceEmbedder(Embeddings):
    """
    Composition-based wrapper around HuggingFaceEmbeddings.

    - Contains a single HuggingFaceEmbeddings instance per model_name.
    - Implements embed_documents & embed_query as required by LangChain.
    - Tracks metrics on inputs (token count) and outputs (vector dimension).
    """

    # cache of clients to ensure one-per-model
    _clients: dict[str, HuggingFaceEmbeddings] = {}

    def __init__(self, model_name: str):
        self.model_name = model_name
        if model_name not in self._clients:
            self._clients[model_name] = HuggingFaceEmbeddings(model_name=model_name)
        self._client = self._clients[model_name]

    def _token_count(self, texts):
        chunker = DocumentChunker(hf_embedding_model=self.model_name)
        if isinstance(texts, str):
            return chunker.get_token_count(text=texts)
        return sum(chunker.get_token_count(text=t) for t in texts)

    @track_metrics(_token_count, target="inputs")
    def embed_query(self, text: str) -> List[float]:
        vec = self._client.embed_query(text)
        return vec

    @track_metrics(_token_count, target="inputs")
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        vecs = self._client.embed_documents(list(texts))
        return vecs
