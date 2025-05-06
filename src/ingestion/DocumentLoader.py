# src/ingestion/loader.py

import os
import logging
from pathlib import Path
from typing import List, Optional

from tqdm.auto import tqdm
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DocumentLoader:
    """
    Handles discovery and loading of documents from a centralized context directory.

    Uses langchain_community loaders under the hood and tracks:
      - Number of inputs/outputs via metrics decorator.
      - Loading progress via tqdm.
      - Errors via structured logging.

    Attributes:
        base_context_dir (Path): Root directory under which all subdirs reside.
    """

    def __init__(self):
        """
        Initialize the loader, setting base_context_dir from settings.CONTEXT_DIR.
        """
        self.base_context_dir = Path(settings.CONTEXT_DIR)

    @track_metrics(lambda filenames: len(filenames), target="outputs")
    def list_filenames(
        self,
        base_dir: str,
        recursive: bool = True
    ) -> List[str]:
        """
        List all filenames under a given subdirectory.

        Args:
            base_dir (str): Relative subdirectory under base_context_dir.
            recursive (bool): If True, walk recursively; else only top-level.

        Returns:
            List[str]: Relative file paths under base_dir.
        """
        context_dir = self.base_context_dir / base_dir
        if not context_dir.exists():
            logger.error(f"Directory not found: {context_dir}")
            return []

        filenames: List[str] = []
        if recursive:
            for root, _, files in os.walk(context_dir):
                rel_root = os.path.relpath(root, context_dir)
                for fname in files:
                    rel_path = fname if rel_root == "." else os.path.join(rel_root, fname)
                    filenames.append(rel_path)
        else:
            for fname in os.listdir(context_dir):
                full_path = context_dir / fname
                if full_path.is_file():
                    filenames.append(fname)

        logger.info(f"Found {len(filenames)} files in '{base_dir}' (recursive={recursive})")
        return filenames

    @track_metrics(lambda docs: len(docs), target="outputs")
    def load_directory(
        self,
        subdir: str,
        patterns: Optional[List[str]] = None,
        recursive: bool = True,
        silent_errors: bool = True,
    ) -> List[Document]:
        """
        Load documents matching glob patterns under a subdirectory.

        Args:
            subdir (str): Subfolder under base_context_dir to scan.
            patterns (List[str], optional): Glob patterns (e.g., ["**/*.txt"]). 
                Defaults to common extensions txt, md, py, rs, cpp, json, pdf.
            recursive (bool): Whether loaders should recurse into subdirs.
            silent_errors (bool): If True, skip files on error; else raise.

        Returns:
            List[Document]: All successfully loaded Document objects.
        """
        target_dir = self.base_context_dir / subdir
        if not target_dir.exists():
            msg = f"Target directory does not exist: {target_dir}"
            logger.error(msg)
            if silent_errors:
                return []
            else:
                raise FileNotFoundError(msg)

        patterns = patterns or [
            "**/*.txt", "**/*.md", "**/*.py", "**/*.rs",
            "**/*.cpp", "**/*.json", "**/*.pdf",
        ]

        all_docs: List[Document] = []
        # Iterate patterns with a progress bar
        for pattern in tqdm(patterns, desc="Patterns", unit="pattern"):
            ext = Path(pattern).suffix.lower()

            try:
                if ext == ".pdf":
                    loader = PyPDFDirectoryLoader(
                        path=str(target_dir),
                        glob=pattern,
                        recursive=recursive,
                        silent_errors=silent_errors,
                    )
                else:
                    # Choose loader class and kwargs
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

                    loader = DirectoryLoader(
                        path=str(target_dir),
                        glob=pattern,
                        recursive=recursive,
                        loader_cls=loader_cls,
                        loader_kwargs=loader_kwargs,
                        silent_errors=silent_errors,
                    )

                docs = loader.load()
                logger.info(f"Loaded {len(docs)} docs from pattern '{pattern}'")
                all_docs.extend(docs)

            except Exception as e:
                logger.exception(f"Failed loading pattern '{pattern}': {e}")
                if not silent_errors:
                    raise

        logger.info(f"Total documents loaded from directory '{subdir}': {len(all_docs)}")
        return all_docs

    @track_metrics(lambda docs: len(docs), target="outputs")
    def load_documents(
        self,
        subdir: str,
        file_names: List[str],
        silent_errors: bool = True,
    ) -> List[Document]:
        """
        Load specific files by name from a subdirectory.

        Args:
            subdir (str): Subfolder under base_context_dir.
            file_names (List[str]): List of filenames (relative to subdir) to load.
            silent_errors (bool): If True, skip on load error; else raise.

        Returns:
            List[Document]: All successfully loaded Document objects.
        """
        parent_dir = self.base_context_dir / subdir
        if not parent_dir.exists():
            msg = f"Subdirectory not found: {parent_dir}"
            logger.error(msg)
            if silent_errors:
                return []
            else:
                raise FileNotFoundError(msg)

        loaded_docs: List[Document] = []
        for fname in tqdm(file_names, desc="Files", unit="file"):
            file_path = parent_dir / fname
            ext = file_path.suffix.lower()

            try:
                if ext == ".pdf":
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
                logger.info(f"Loaded {len(docs)} docs from file '{fname}'")
                loaded_docs.extend(docs)

            except Exception as e:
                logger.exception(f"Error loading file '{fname}': {e}")
                if not silent_errors:
                    raise
                # else skip

        logger.info(f"Total documents loaded by name from '{subdir}': {len(loaded_docs)}")
        return loaded_docs
