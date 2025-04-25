# src/ingestion/loader.py
import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFDirectoryLoader,
    PyPDFLoader,
    TextLoader,
    PythonLoader,
    JSONLoader,
)
from config.settings import settings
from src.utils.metrics import track_metrics

class DocumentLoader:
    def __init__(self):
        self.base_context_dir = Path(settings.CONTEXT_DIR)
        
    def list_filenames(
        self,
        base_dir: str,
        recursive: bool = True
    ) -> List[str]:
        """
        Returns a list of filenames (excluding the base directory) under base_dir.

        Args:
            base_dir (str): Relative Path to the directory to scan.
            recursive (bool): If True, walk subdirectories recursively. 
                            If False, only list files in the top-level directory.

        Returns:
            List[str]: List of file paths relative to base_dir.
        """
        filenames: List[str] = []
        context_dir = self.base_context_dir / base_dir

        if recursive:
            # Walk through context_dir and all subdirectories
            for root, _, files in os.walk(context_dir):
                for fname in files:
                    # Compute relative path
                    rel_dir = os.path.relpath(root, context_dir)
                    # If in the base directory itself, rel_dir == "."
                    if rel_dir == ".":
                        filenames.append(fname)
                    else:
                        filenames.append(os.path.join(rel_dir, fname))
        else:
            # Only list files in context_dir (non-recursive)
            for fname in os.listdir(context_dir):
                full_path = os.path.join(context_dir, fname)
                if os.path.isfile(full_path):
                    filenames.append(fname)

        return filenames

        
    @track_metrics(lambda docs: len(docs))
    def load_directory(
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
    
        
    def load_documents(
        self,
        subdir: str,
        file_names: List[str],
        silent_errors: bool = True,
    ) -> List[Document]:
        """
        Load exactly the files in `file_names` under `base_context_dir/subdir`.
        """
        all_docs: List[Document] = []
        parent = self.base_context_dir / subdir

        for fname in file_names:
            file_path = parent / fname
            ext = file_path.suffix.lower()

            try:
                if ext == ".pdf":
                    # single-PDF loader
                    loader = PyPDFLoader(str(file_path))

                elif ext == ".py":
                    loader = PythonLoader(str(file_path))

                elif ext == ".json":
                    loader = JSONLoader(
                        str(file_path),
                        jq_schema=".",
                        text_content=True,
                        json_lines=False,
                    )

                else:
                    loader = TextLoader(
                        str(file_path),
                        encoding="utf-8",
                        autodetect_encoding=True,
                    )

                docs = loader.load()
                all_docs.extend(docs)

            except Exception as e:
                if silent_errors:
                    # you may want to log the error here
                    continue
                else:
                    raise

        return all_docs
