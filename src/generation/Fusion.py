import logging
from typing import List, Tuple

from langchain_core.documents import Document
from src.generation.HuggingFaceLLM import HuggingFaceLLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FusionSummarizer:
    """
    Class for summarizing prompt chunks using a provided fusion LLM.

    Attributes:
        fusion_llm (HuggingFaceLLM): Initialized LLM for generating summaries.
        sys_prompt (str): System prompt guiding the fusion LLM.
        temperature (float): Sampling temperature for LLM calls.
        max_tokens (int): Maximum tokens to generate per summary.
    """

    def __init__(
        self,
        fusion_llm: HuggingFaceLLM,
        sys_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 300
    ):
        """
        Initializes the FusionSummarizer.

        Args:
            fusion_llm (HuggingFaceLLM): Pre-initialized LLM client.
            sys_prompt (str): System prompt for context.
            temperature (float, optional): Sampling temperature. Defaults to 0.3.
            max_tokens (int, optional): Max tokens per summary. Defaults to 300.
        """
        self.fusion_llm = fusion_llm
        self.sys_prompt = sys_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    def summarize(
        self,
        prompt_chunks: List[Tuple[str, List[Document]]]
    ) -> List[str]:
        """
        Summarize each prompt and its associated document chunks.

        For each (prompt_text, chunks) tuple, calls the fusion LLM to generate
        a summary and formats it as markdown.

        Args:
            prompt_chunks (List[Tuple[str, List[Document]]]):
                List of tuples containing user prompts and document chunks.

        Returns:
            List[str]: Markdown-formatted summary strings.

        Raises:
            ValueError: If prompt_chunks is empty.
            RuntimeError: If LLM call fails for any entry.
        """
        if not prompt_chunks:
            logger.error("No prompt chunks provided for summarization.")
            raise ValueError("prompt_chunks must contain at least one item.")

        summaries: List[str] = []
        for idx, (prompt_text, chunks) in enumerate(prompt_chunks, start=1):
            try:
                # Build context from chunks
                context_lines = [f"Chunk {i}: {doc.page_content}" for i, doc in enumerate(chunks, start=1)]
                user_prompt = f"Query: {prompt_text}\n\n" + "\n".join(context_lines)

                logger.info(f"Generating summary for prompt {idx}: '{prompt_text[:50]}...'")
                summary = self.fusion_llm.get_answer(
                    sys_prompt=self.sys_prompt,
                    user_prompt=user_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                formatted = f"**Prompt {idx} Summary:** {summary}"
                summaries.append(formatted)
                logger.debug(f"Summary {idx} generated successfully.")

            except Exception as e:
                logger.exception(f"Failed to generate summary for prompt {idx}.")
                raise RuntimeError(f"Error summarizing prompt {idx}: {e}")

        return summaries
