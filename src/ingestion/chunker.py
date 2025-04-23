from typing import List, Optional
from pathlib import Path

from src.utils.metrics import track_metrics

from transformers import AutoTokenizer
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class SingletonMeta(type):
    """
    Metaclass that ensures only one instance of any class using it is ever created.
    """
    _instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in SingletonMeta._instances:
            SingletonMeta._instances[cls] = super().__call__(*args, **kwargs)
        return SingletonMeta._instances[cls]


class Chunker(metaclass=SingletonMeta):
    """
    Singleton class responsible for splitting Documents into token-based chunks
    using a HuggingFace tokenizer under the hood.
    """

    def __init__(
        self,
        hf_tokenizer_name: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the Chunker with a single HF tokenizer and a text splitter.
        
        Args:
            hf_tokenizer_name (str): Name or path of the HuggingFace tokenizer.
            chunk_size (int): Maximum number of tokens per chunk.
            chunk_overlap (int): Number of overlapping tokens between chunks.
        """
        # Load tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
        # Create a single splitter instance
        self.splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
    def count_processed(result):
        return len(result)
    
    @track_metrics(count_processed)
    def chunk_documents(self, documents: List[Document], return_token_count: bool = True) -> List[Document]:
        """
        Split a list of Documents into smaller Documents (chunks), preserving metadata.

        Args:
            documents (List[Document]): Original LangChain Documents to chunk.

        Returns:
            List[Document]: A flattened list of chunked Documents.
        """
        all_chunks: List[Document] = []
        for doc in documents:
            # Split into raw text chunks
            texts = self.splitter.split_text(doc.page_content)
            # Wrap each chunk back in a Document, copying metadata
            for i, chunk_text in enumerate(texts):
                metadata = dict(doc.metadata)  # shallow copy
                # Optionally you can add chunk-specific metadata, e.g. index
                metadata["chunk_index"] = i
                metadata["source"] = metadata.get("source", "")  # ensure source exists
                all_chunks.append(Document(page_content=chunk_text, metadata=metadata))
        return all_chunks
