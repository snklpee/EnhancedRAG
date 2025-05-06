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
from tqdm.auto import tqdm
from src.utils.metrics import track_metrics
from config.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VectorStoreManager:
    """
    High-level manager for a FAISS vector store.

    Provides synchronous and asynchronous methods for index management,
    document CRUD, searching (including MMR), serialization, and retrieval.

    Attributes:
        embedding_function: Callable[[str], List[float]] or Embeddings
            Function or object to compute embeddings.
        index_name: str
            Identifier used when saving/loading index files.
        distance_strategy: DistanceStrategy
            Metric for FAISS (e.g., EUCLIDEAN_DISTANCE or COSINE).
        folder_path: str
            Base directory for local index files (from settings.FAISS_INDEXES).
        store: FAISS | None
            Underlying LangChain FAISS store instance.
    """

    def __init__(
        self,
        embedding_function: Union[Callable[[str], List[float]], Any],
        index_name: str,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
    ):
        """
        Initialize the manager without loading or creating an index.

        Args:
            embedding_function: Embeddings object or callable to generate vectors.
            index_name: Name to assign to this vector index on disk.
            distance_strategy: FAISS distance metric.
        """
        self.embedding_function = embedding_function
        self.index_name = index_name
        self.distance_strategy = distance_strategy
        self.folder_path = settings.FAISS_INDEXES
        self.store: Optional[FAISS] = None
        logger.info(f"VectorStoreManager initialized for index '{index_name}'")

    def _ensure_store(self) -> None:
        """
        Internal: ensure a FAISS store is present.
        Raises:
            RuntimeError: If store is not initialized.
        """
        if self.store is None:
            raise RuntimeError(
                "Vector store not initialized; call create_index, from_texts, or load_local first."
            )

    def create_index(self) -> None:
        """
        Create a fresh FAISS index in memory using the embedding dimension.

        Raises:
            Exception: On failure to create index.
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
            logger.info(f"Created FAISS index '{self.index_name}' with dim={dim}")
        except Exception as e:
            logger.exception("Failed to create FAISS index")
            raise

    @track_metrics(lambda self, docs: len(docs), target="inputs")
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add or update documents in the vector store with progress bar.

        Args:
            documents: Documents to index.
            ids: Optional IDs; generated if omitted.

        Returns:
            List of assigned document IDs.

        Raises:
            RuntimeError: If store not initialized.
            ValueError: If ids length mismatches documents.
        """
        self._ensure_store()
        try:
            if ids and len(ids) != len(documents):
                raise ValueError("Length of ids must match documents")
            ids = ids or [str(uuid4()) for _ in documents]
            for doc, uid in tqdm(zip(documents, ids), total=len(documents), desc="Adding documents"):
                self.store.add_documents(documents=[doc], ids=[uid])
            logger.info(f"Added {len(documents)} documents to '{self.index_name}'")
            return ids
        except Exception as e:
            logger.exception("Error adding documents to store")
            raise

    @track_metrics(lambda self, ids: len(ids or []), target="inputs")
    def delete(self, ids: Optional[List[str]] = None) -> bool:
        """
        Delete documents by their IDs (or all if ids is None).

        Args:
            ids: IDs to remove; None to delete all.

        Returns:
            True if deletion succeeded.

        Raises:
            RuntimeError: If store not initialized.
        """
        self._ensure_store()
        try:
            result = self.store.delete(ids=ids)
            logger.info(f"Deleted {len(ids) if ids else 'all'} docs from '{self.index_name}'")
            return result
        except Exception:
            logger.exception("Error deleting documents from store")
            raise

    def drop_index(self) -> None:
        """
        Remove index files from disk and clear in-memory store.

        Raises:
            Exception: On filesystem errors.
        """
        try:
            for ext in (".faiss", ".pkl"):
                path = os.path.join(self.folder_path, f"{self.index_name}{ext}")
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Removed file {path}")
            self.store = None
            logger.info(f"Dropped index '{self.index_name}' completely")
        except Exception:
            logger.exception("Error dropping index files")
            raise

    def save_local(self) -> None:
        """
        Persist the FAISS index and metadata to disk.

        Raises:
            RuntimeError: If store not initialized.
        """
        self._ensure_store()
        try:
            self.store.save_local(folder_path=self.folder_path, index_name=self.index_name)
            logger.info(f"Saved index '{self.index_name}' to '{self.folder_path}'")
        except Exception:
            logger.exception("Error saving index to disk")
            raise

    def load_local(self, allow_pickle: bool = False) -> None:
        """
        Load an existing FAISS index from disk.

        Args:
            allow_pickle: Permit pickle-based deserialization.

        Raises:
            Exception: If loading fails.
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
            logger.exception("Error loading index from disk")
            raise

    def from_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Build the store directly from raw texts.

        Args:
            texts: Input strings.
            metadatas: Optional metadata per text.
            ids: Optional IDs.
            kwargs: Forwarded to FAISS.from_texts.

        Raises:
            Exception: On construction error.
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
            logger.exception("Error in from_texts factory")
            raise

    def from_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> None:
        """
        Build the store from Document objects.

        Args:
            documents: List of Document instances.
            kwargs: Forwarded to FAISS.from_documents.

        Raises:
            Exception: On construction error.
        """
        try:
            self.store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                **kwargs,
            )
            logger.info(f"Initialized '{self.index_name}' via from_documents")
        except Exception:
            logger.exception("Error in from_documents factory")
            raise

    def retriever(self, **kwargs: Any) -> Any:
        """
        Construct a Retriever for advanced querying.

        Args:
            kwargs: Options like search_type, search_kwargs, etc.

        Returns:
            A VectorStoreRetriever instance.

        Raises:
            RuntimeError: If store not initialized.
        """
        self._ensure_store()
        try:
            rt = self.store.as_retriever(**kwargs)
            logger.info(f"Created retriever for '{self.index_name}' with params {kwargs}")
            return rt
        except Exception:
            logger.exception("Error creating retriever")
            raise

    @track_metrics(lambda self, docs: len(docs), target="outputs")
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[Dict[str, Any], Callable]] = None,
        fetch_k: int = 20,
    ) -> List[Document]:
        """
        Perform a basic similarity search.

        Args:
            query: Query text.
            k: Results count.
            filter: Metadata filter.
            fetch_k: Number fetched before filtering.

        Returns:
            List of Documents.

        Raises:
            RuntimeError: If store not initialized.
        """
        self._ensure_store()
        try:
            docs = self.store.similarity_search(query=query, k=k, filter=filter, fetch_k=fetch_k)
            logger.info(f"Retrieved {len(docs)} docs for query '{query}'")
            return docs
        except Exception:
            logger.exception("Error in similarity_search")
            raise

    @track_metrics(lambda self, docs: len(docs), target="outputs")
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[Dict[str, Any], Callable]] = None,
        fetch_k: int = 20,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve Documents along with similarity scores.

        Args:
            query: Text to search.
            k: Number of results.
            filter: Metadata filter.
            fetch_k: Pre-MMR fetch limit.

        Returns:
            List of (Document, score).

        Raises:
            RuntimeError: If store not initialized.
        """
        self._ensure_store()
        try:
            results = self.store.similarity_search_with_score(
                query=query, k=k, filter=filter, fetch_k=fetch_k
            )
            logger.info(f"Retrieved {len(results)} scored docs for '{query}'")
            return results
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
