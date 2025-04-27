# src/utils/vector_store_manager.py

import os
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from uuid import uuid4

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from src.utils.metrics import track_metrics
from config.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VectorStoreManager:
    """
    High-level manager for a FAISS vector store (langchain_community.vectorstores.FAISS).

    Args:
        embedding_function: Embeddings or Callable[[str], List[float]]
        index_name:         Name under which to save/load on disk.
        distance_strategy:  Distance metric enum (EUCLIDEAN_DISTANCE or COSINE).
    """

    def __init__(
        self,
        embedding_function: Union[Callable[[str], List[float]], Any],
        index_name: str,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
    ):
        self.embedding_function = embedding_function
        self.index_name = index_name
        self.distance_strategy = distance_strategy
        self.folder_path = settings.FAISS_INDEXES
        self.store: Optional[FAISS] = None

    def _ensure_store(self):
        if self.store is None:
            raise RuntimeError("Vector store not initialized; call create_index/from_texts/load_local first.")

    def create_index(self) -> None:
        """
        Create a fresh FAISS index in memory.
        """
        try:
            dim = len(self.embedding_function.embed_query("hello world"))
            idx = faiss.IndexFlatL2(dim)
            self.store = FAISS(
                embedding_function=self.embedding_function,
                index=idx,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                normalize_L2=False,
                distance_strategy=self.distance_strategy,
            )
            logger.info(f"Created in‐memory FAISS index '{self.index_name}' (dim={dim})")
        except Exception as e:
            logger.exception("Error creating FAISS index")
            raise

    @track_metrics(lambda self, docs: len(docs), target="inputs")
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """
        Add or update a batch of Documents.

        Args:
            documents: List of Document
            ids:       Optional list of IDs; will be generated if None.

        Returns:
            List of assigned document IDs.
        """
        self._ensure_store()
        try:
            if ids and len(ids) != len(documents):
                raise ValueError("Length of ids must match documents")
            ids = ids or [str(uuid4()) for _ in documents]
            result = self.store.add_documents(documents=documents, ids=ids)
            logger.info(f"Added {len(documents)} docs into '{self.index_name}'")
            return result
        except Exception:
            logger.exception("Error adding documents")
            raise

    @track_metrics(lambda ids: len(ids), target="inputs")
    def delete(self, ids: Optional[List[str]] = None) -> bool:
        """
        Delete documents by IDs (or all if ids=None).

        Args:
            ids: List of document IDs or None to delete all.

        Returns:
            True if deletion succeeded.
        """
        self._ensure_store()
        try:
            result = self.store.delete(ids=ids)
            logger.info(f"Deleted docs {ids or 'ALL'} from '{self.index_name}'")
            return result
        except Exception:
            logger.exception("Error deleting documents")
            raise

    def drop_index(self) -> None:
        """
        Remove local files and clear in-memory store.
        """
        try:
            for ext in (".faiss", ".pkl"):
                path = os.path.join(self.folder_path, f"{self.index_name}{ext}")
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Removed file {path}")
            self.store = None
            logger.info(f"Dropped index '{self.index_name}'")
        except Exception:
            logger.exception("Error dropping index")
            raise

    def save_local(self) -> None:
        """
        Persist current FAISS store to disk.
        """
        self._ensure_store()
        try:
            self.store.save_local(folder_path=self.folder_path, index_name=self.index_name)
            logger.info(f"Saved index '{self.index_name}' to {self.folder_path}")
        except Exception:
            logger.exception("Error saving index")
            raise

    def load_local(self, allow_pickle: bool = False) -> None:
        """
        Load a saved FAISS store from disk.

        Args:
            allow_pickle: allow pickle‐based docstore deserialization.
        """
        try:
            self.store = FAISS.load_local(
                folder_path=self.folder_path,
                index_name=self.index_name,
                embeddings=self.embedding_function,
                allow_dangerous_deserialization=allow_pickle,
            )
            logger.info(f"Loaded index '{self.index_name}' from disk")
        except Exception:
            logger.exception("Error loading index")
            raise

    def from_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Synchronous factory: build store straight from raw texts.

        Args:
            texts, metadatas, ids, plus any FAISS.from_texts kwargs.
        """
        try:
            self.store = FAISS.from_texts(
                texts=texts,
                embedding=self.embedding_function,
                metadatas=metadatas,
                ids=ids,
                **kwargs,
            )
            logger.info(f"Initialized '{self.index_name}' via from_texts")
        except Exception:
            logger.exception("Error in from_texts")
            raise

    def from_documents(self, documents: List[Document], **kwargs: Any) -> None:
        """
        Synchronous factory: build store from Document objects.

        Args:
            documents: List[Document], plus FAISS.from_documents kwargs.
        """
        try:
            self.store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                **kwargs,
            )
            logger.info(f"Initialized '{self.index_name}' via from_documents")
        except Exception:
            logger.exception("Error in from_documents")
            raise

    def retriever(self, **kwargs: Any) -> Any:
        """
        Build a Retriever wrapper around this store.

        kwargs may include:
          - search_type: "similarity", "mmr", "similarity_score_threshold"
          - search_kwargs: dict with k, fetch_k, lambda_mult, filter, score_threshold
        """
        self._ensure_store()
        try:
            rt = self.store.as_retriever(**kwargs)
            logger.info(f"Created retriever for '{self.index_name}' with {kwargs}")
            return rt
        except Exception:
            logger.exception("Error creating retriever")
            raise

    @track_metrics(lambda docs: len(docs), target="outputs")
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[Dict[str, Any], Callable]] = None,
        fetch_k: int = 20,
    ) -> List[Document]:
        """
        Basic similarity search.

        Args:
            query: Text query
            k:     # results
            filter: metadata filter dict or callable
            fetch_k: passed to FAISS under the hood
        """
        self._ensure_store()
        try:
            return self.store.similarity_search(query=query, k=k, filter=filter, fetch_k=fetch_k)
        except Exception:
            logger.exception("Error in similarity_search")
            raise

    @track_metrics(lambda docs: len(docs), target="outputs")
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[Dict[str, Any], Callable]] = None,
        fetch_k: int = 20,
    ) -> List[Tuple[Document, float]]:
        """
        Return (Document, score) pairs.
        """
        self._ensure_store()
        try:
            return self.store.similarity_search_with_score(
                query=query, k=k, filter=filter, fetch_k=fetch_k
            )
        except Exception:
            logger.exception("Error in similarity_search_with_score")
            raise

    # def max_marginal_relevance_search(
    #     self,
    #     query: str,
    #     k: int = 4,
    #     fetch_k: int = 20,
    #     lambda_mult: float = 0.5,
    #     filter: Optional[Union[Dict[str, Any], Callable]] = None,
    # ) -> List[Document]:
    #     """
    #     Max-Marginal-Relevance search to balance relevance vs diversity.
    #     """
    #     self._ensure_store()
    #     try:
    #         return self.store.max_marginal_relevance_search(
    #             query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
    #         )
    #     except Exception:
    #         logger.exception("Error in MMR search")
    #         raise

    # def merge_from(self, other: "VectorStoreManager") -> None:
    #     """
    #     Merge another FAISS store into this one.
    #     """
    #     self._ensure_store()
    #     if other.store is None:
    #         raise ValueError("Other store not initialized")
    #     try:
    #         self.store.merge_from(other.store)
    #         logger.info(f"Merged index '{other.index_name}' into '{self.index_name}'")
    #     except Exception:
    #         logger.exception("Error merging stores")
    #         raise

    # def serialize_to_bytes(self) -> bytes:
    #     """
    #     Return serialized bytes (index + docstore).
    #     """
    #     self._ensure_store()
    #     try:
    #         return self.store.serialize_to_bytes()
    #     except Exception:
    #         logger.exception("Error serializing to bytes")
    #         raise

    # @classmethod
    # async def afrom_texts(
    #     cls,
    #     texts: List[str],
    #     embedding: Any,
    #     metadatas: Optional[List[dict]] = None,
    #     ids: Optional[List[str]] = None,
    #     **kwargs: Any,
    # ) -> "VectorStoreManager":
    #     """
    #     Async constructor from raw texts.
    #     """
    #     inst = cls(embedding_function=embedding, index_name=kwargs.get("index_name", "index"))
    #     inst.store = await FAISS.afrom_texts(
    #         texts=texts, embedding=embedding, metadatas=metadatas, ids=ids, **kwargs
    #     )
    #     logger.info(f"Async initialized '{inst.index_name}' via afrom_texts")
    #     return inst

    # @classmethod
    # async def afrom_documents(
    #     cls, documents: List[Document], embedding: Any, **kwargs: Any
    # ) -> "VectorStoreManager":
    #     """
    #     Async constructor from Documents.
    #     """
    #     inst = cls(embedding_function=embedding, index_name=kwargs.get("index_name", "index"))
    #     inst.store = await FAISS.afrom_documents(
    #         documents=documents, embedding=embedding, **kwargs
    #     )
    #     logger.info(f"Async initialized '{inst.index_name}' via afrom_documents")
    #     return inst

    # @classmethod
    # async def afrom_embeddings(
    #     cls,
    #     text_embeddings: Iterable[Tuple[str, List[float]]],
    #     embedding: Any,
    #     metadatas: Optional[Iterable[dict]] = None,
    #     ids: Optional[List[str]] = None,
    #     **kwargs: Any,
    # ) -> "VectorStoreManager":
    #     """
    #     Async constructor from (text,embedding) pairs.
    #     """
    #     inst = cls(embedding_function=embedding, index_name=kwargs.get("index_name", "index"))
    #     inst.store = await FAISS.afrom_embeddings(
    #         text_embeddings=text_embeddings,
    #         embedding=embedding,
    #         metadatas=metadatas,
    #         ids=ids,
    #         **kwargs,
    #     )
    #     logger.info(f"Async initialized '{inst.index_name}' via afrom_embeddings")
    #     return inst

    # async def asimilarity_search(
    #     self, query: str, k: int = 4, filter: Any = None, fetch_k: int = 20, **kwargs: Any
    # ) -> List[Document]:
    #     """
    #     Async similarity_search.
    #     """
    #     self._ensure_store()
    #     return await self.store.asimilarity_search(query=query, k=k, filter=filter, fetch_k=fetch_k, **kwargs)

    # async def asimilarity_search_with_score(
    #     self, query: str, k: int = 4, filter: Any = None, fetch_k: int = 20, **kwargs: Any
    # ) -> List[Tuple[Document, float]]:
    #     """
    #     Async similarity_search_with_score.
    #     """
    #     self._ensure_store()
    #     return await self.store.asimilarity_search_with_score(query=query, k=k, filter=filter, fetch_k=fetch_k, **kwargs)

    # async def adelete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
    #     """
    #     Async delete.
    #     """
    #     self._ensure_store()
    #     return await self.store.adelete(ids=ids, **kwargs)

    # async def aget_by_ids(self, ids: List[str]) -> List[Document]:
    #     """
    #     Async get_by_ids.
    #     """
    #     self._ensure_store()
    #     return await self.store.aget_by_ids(ids)

    # async def amax_marginal_relevance_search(
    #     self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, filter: Any = None, **kwargs: Any
    # ) -> List[Document]:
    #     """
    #     Async MMR search.
    #     """
    #     self._ensure_store()
    #     return await self.store.amax_marginal_relevance_search(
    #         query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter, **kwargs
    #     )
