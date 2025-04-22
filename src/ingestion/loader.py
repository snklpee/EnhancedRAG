import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PythonLoader,
    JSONLoader,
    UnstructuredPDFLoader,
)

from utils.metrics import track_metrics
from config.settings import settings


class DocumentLoader:
    """
    Encapsulates loading of all text‑containing files from a directory,
    with execution‑time and record‑count metrics automatically tracked.
    """

    def __init__(self):
        self.base_context_dir = Path(settings.CONTEXT_DIR)

    @track_metrics(component="ingestion.document_loader")
    def load_documents(
        self,
        subdir: str,
        patterns: Optional[List[str]] = None,
        recursive: bool = True,
        silent_errors: bool = True,
    ) -> List[Document]:
        """
        Load all text‑containing files under `base_context_dir/subdir`.

        Args:
            subdir: Name of the folder under context/ to load from.
            patterns: Glob patterns (default to txt, md, py, rs, cpp, json, pdf).
            recursive: Recurse into subdirectories?
            silent_errors: Skip files that fail?

        Returns:
            List[Document]: all successfully loaded documents.
        """
        target_dir = self.base_context_dir / subdir

        # fallback defaults
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

            # pick loader class + kwargs based on extension
            if ext == ".py":
                loader_cls = PythonLoader
                loader_kwargs = {}
            elif ext == ".json":
                loader_cls = JSONLoader
                loader_kwargs = {
                    "jq_schema": ".",
                    "text_content": True,
                    "json_lines": False,
                }
            elif ext == ".pdf":
                loader_cls = UnstructuredPDFLoader
                loader_kwargs = {}
            else:
                loader_cls = TextLoader
                loader_kwargs = {
                    "encoding": "utf-8",
                    "autodetect_encoding": True,
                }

            loader = DirectoryLoader(
                path=target_dir,
                glob=pattern,
                recursive=recursive,
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs,
                silent_errors=silent_errors,
                show_progress=False,
                use_multithreading=True,
            )

            docs = loader.load()
            all_docs.extend(docs)

        return all_docs
