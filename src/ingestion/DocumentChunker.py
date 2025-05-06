#src/ingestion/DocumentChunker.py

import logging
from typing import List, Tuple, Union

from transformers import AutoTokenizer, PreTrainedTokenizer
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.documents import Document

from src.utils.metrics import track_metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class _TokenizerSingletonMeta(type):
    """
    Metaclass that ensures only one DocumentChunker instance per HF embedding model.
    Prevents repeated loading of heavy tokenizer+splitter objects.
    """
    _instances: dict[Tuple[type, str], object] = {}

    def __call__(cls, hf_embedding_model: str, *args, **kwargs):
        key = (cls, hf_embedding_model)
        if key not in cls._instances:
            logger.info(f"Creating new DocumentChunker for model '{hf_embedding_model}'")
            cls._instances[key] = super().__call__(hf_embedding_model, *args, **kwargs)
        else:
            logger.debug(f"Reusing existing DocumentChunker for model '{hf_embedding_model}'")
        return cls._instances[key]


class DocumentChunker(metaclass=_TokenizerSingletonMeta):
    """
    Singleton class responsible for splitting a list of LangChain Document objects
    into token-based chunks, using a HuggingFace tokenizer and the
    SentenceTransformersTokenTextSplitter under the hood.

    Attributes:
        tokenizer_name (str): HF model name used to load the tokenizer.
        tokenizer (PreTrainedTokenizer): Loaded HuggingFace tokenizer.
        splitter (SentenceTransformersTokenTextSplitter): Configured splitter.
    """

    def __init__(
        self,
        hf_embedding_model: str,
        chunk_size: int = 300,
        chunk_overlap: int = 80,
    ):
        """
        Initialize the DocumentChunker.

        Args:
            hf_embedding_model (str):
                Name or path of the HuggingFace embeddings model to load tokenizer from.
            chunk_size (int):
                Maximum number of tokens in each chunk.
            chunk_overlap (int):
                Number of overlapping tokens between consecutive chunks.

        Raises:
            ValueError: If tokenizer or splitter fail to initialize.
        """
        self.tokenizer_name = hf_embedding_model
        try:
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                hf_embedding_model
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer '{hf_embedding_model}': {e}")
            raise ValueError(f"Could not load tokenizer '{hf_embedding_model}'") from e

        try:
            self.splitter = SentenceTransformersTokenTextSplitter.from_huggingface_tokenizer(
                tokenizer=self.tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as e:
            logger.error(
                f"Failed to create splitter for '{hf_embedding_model}': {e}"
            )
            raise ValueError(
                f"Could not initialize splitter for '{hf_embedding_model}'"
            ) from e

        logger.info(
            f"DocumentChunker initialized with model='{hf_embedding_model}', "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

    @property
    def tokenizer_instance(self) -> PreTrainedTokenizer:
        """Access the underlying HuggingFace tokenizer."""
        return self.tokenizer

    @property
    def splitter_instance(self) -> SentenceTransformersTokenTextSplitter:
        """Access the underlying SentenceTransformersTokenTextSplitter."""
        return self.splitter

    def get_token_count(self, text: str) -> int:
        """
        Count tokens in a single text string via the splitter.

        Args:
            text (str): Input text.

        Returns:
            int: Number of tokens.
        """
        try:
            return self.splitter.count_tokens(text=text)
        except Exception as e:
            logger.error(f"Token count failed for text (len={len(text)}): {e}")
            # Fallback: approximate by whitespace count
            return len(text.split())

    @track_metrics(lambda self, docs: len(docs), target="inputs")
    def get_docs_token_count(self, docs: List[Document]) -> int:
        """
        Compute the total token count across multiple Documents.

        Args:
            docs (List[Document]): List of Documents to measure.

        Returns:
            int: Combined token count.
        """
        total = 0
        for doc in docs:
            total += self.get_token_count(doc.page_content)
        logger.info(f"Total input tokens across {len(docs)} docs: {total}")
        return total

    @track_metrics(lambda chunks: len(chunks), target="outputs")
    def chunk_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Split each Document into smaller token-based chunks, preserving metadata.


        Args:
            documents (List[Document]):
                List of LangChain Document objects to chunk.

        Returns:
            List[Document]: Flattened list of chunked Document objects,
                with metadata fields:
                  - "chunk_index": index of the chunk within the source doc
                  - "tokenizer": the hf_embedding_model name
                  - "tokens": token count for the chunk
                  - "source": original metadata["source"] if present
        """
        all_chunks: List[Document] = []

        logger.info(f"Starting chunking of {len(documents)} documents")
        for doc in documents:
            try:
                texts = self.splitter.split_text(doc.page_content)
            except Exception as e:
                logger.error(f"Splitter failed on doc source={doc.metadata.get('source')}: {e}")
                continue

            for idx, chunk_text in enumerate(texts):
                token_count = self.get_token_count(chunk_text)
                md = dict(doc.metadata)  # copy original metadata
                md.update({
                    "chunk_index": idx,
                    "tokenizer": self.tokenizer_name,
                    "tokens": token_count,
                })
                md.setdefault("source", "")
                all_chunks.append(Document(page_content=chunk_text, metadata=md))

        logger.info(f"Finished chunking: produced {len(all_chunks)} chunks")
        return all_chunks
