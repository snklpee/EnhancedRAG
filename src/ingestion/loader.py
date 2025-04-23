# src/ingestion/loader.py
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFDirectoryLoader,
    TextLoader,
    PythonLoader,
    JSONLoader,
)
from config.settings import settings
from src.utils.metrics import track_metrics

class DocumentLoader:
    def __init__(self):
        self.base_context_dir = Path(settings.CONTEXT_DIR)
        

    @track_metrics(lambda docs: len(docs))
    def load_documents(
        self,
        subdir: str,
        patterns: Optional[List[str]] = None,
        recursive: bool = True,
        silent_errors: bool = True,
    ) -> List[Document]:
        target_dir = self.base_context_dir / subdir

        if patterns is None:
            patterns = [
                "**/*.txt",
                "**/*.md",
                "**/*.py",
                "**/*.rs",
                "**/*.cpp",
                "**/*.json",
                "**/*.pdf",
            ]

        all_docs: List[Document] = []
        for pattern in patterns:
            ext = Path(pattern).suffix.lower()

            if ext == ".pdf":
                # Use PyPDFDirectoryLoader for entire dir of PDFs
                loader = PyPDFDirectoryLoader(
                    path=str(target_dir),
                    glob=pattern,
                    silent_errors=silent_errors,
                    recursive=recursive,
                )
            else:
                # Pick the single‑file loader class + kwargs
                if ext == ".py":
                    loader_cls, loader_kwargs = PythonLoader, {}
                elif ext == ".json":
                    loader_cls, loader_kwargs = JSONLoader, {
                        "jq_schema": ".",
                        "text_content": True,
                        "json_lines": False,
                    }
                else:
                    loader_cls, loader_kwargs = TextLoader, {
                        "encoding": "utf-8",
                        "autodetect_encoding": True,
                    }

                # Wrap it in a DirectoryLoader for pattern matching
                loader = DirectoryLoader(
                    path=str(target_dir),
                    glob=pattern,
                    recursive=recursive,
                    loader_cls=loader_cls,
                    loader_kwargs=loader_kwargs,
                    silent_errors=silent_errors,
                )

            all_docs.extend(loader.load())
            
        return all_docs
