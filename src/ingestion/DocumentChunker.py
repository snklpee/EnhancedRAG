from typing import List, Tuple, Union

from transformers import AutoTokenizer
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.documents import Document

from src.utils.metrics import track_metrics

class _TokenizerSingletonMeta(type):
    """
    Metaclass that ensures only one instance per (class, hf_embedding_model) is ever created.
    """
    _instances: dict[tuple[type, str], object] = {}

    def __call__(cls, hf_embedding_model: str, *args, **kwargs):
        key = (cls, hf_embedding_model)
        if key not in cls._instances:
            # first time we see this hf_embedding_model → create & cache
            cls._instances[key] = super().__call__(hf_embedding_model, *args, **kwargs)
        return cls._instances[key]


class DocumentChunker(metaclass=_TokenizerSingletonMeta):
    """
    Singleton class responsible for splitting Documents into token-based chunks
    using a HuggingFace tokenizer under the hood.
    """

    def __init__(
        self,
        hf_embedding_model: str,
        chunk_size: int = 300,
        chunk_overlap: int = 80,
    ):
        """
        Initialize the Chunker with a single HF tokenizer and a token-based text splitter.
        
        Args:
            hf_embedding_model (str): Name or path of the HuggingFace Model to get the tokenizer for.
            chunk_size (int): Maximum number of tokens per chunk.
            chunk_overlap (int): Number of overlapping tokens between chunks.
        """
        self.tokenizer_name = hf_embedding_model
        # Load and keep one tokenizer instance
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        # Create one splitter instance
        self.splitter = SentenceTransformersTokenTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
    @property
    def tokenizer_instance(self)->AutoTokenizer:
        return self.tokenizer
    
    @property
    def splitter_instance(self)-> SentenceTransformersTokenTextSplitter:
        return self.splitter
                     
    def get_token_count(self,text:str)->int:
        token_count = self.splitter.count_tokens(text=text)
        return token_count
    
    @track_metrics(lambda self, docs: len(docs), target="inputs")   
    def get_docs_token_count(self, docs: List[Document])->int:
        doc_token_count = 0
        for doc in docs:
            doc_token_count += self.get_token_count(doc.page_content)
        return doc_token_count    
        
    @track_metrics(lambda chunks: len(chunks), target="outputs")
    def chunk_documents(
        self,
        documents: List[Document]
        ) -> List[Document]:
        """
        Split a list of Documents into smaller Documents (chunks), preserving metadata.

        Args:
            documents (List[Document]): Original LangChain Documents to chunk.
            return_token_count (bool): If True, also return total tokens seen.

        Returns:
            If return_token_count:
                (chunks: List[Document], token_count: int)
            else:
                chunks: List[Document]
        """
        all_chunks: List[Document] = []
        for doc in documents:
            # split into text chunks
            texts = self.splitter.split_text(doc.page_content)
            for idx, chunk_text in enumerate(texts):
                chunk_text_token_count = self.get_token_count(chunk_text)
                md = dict(doc.metadata)  
                md["chunk_index"],md["tokenizer"],md["tokens"] = idx, self.tokenizer_name, chunk_text_token_count
                md.setdefault("source", "")
                all_chunks.append(Document(page_content=chunk_text, metadata=md))
        
        return all_chunks
   
