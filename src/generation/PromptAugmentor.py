import logging
from typing import List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PromptAugmentor:
    """
    A helper class to generate a sequence of unique, concise prompts
    based on an original query, using a provided LLM client.

    Attributes:
        client: An instantiated LLM client.
        temperature: Sampling temperature for the LLM.
    """

    def __init__(
        self,
        client,
        temperature: float = 0.2,
    ):
        """
        Initializes the PromptAugmentor.

        Args:
            client: An LLM client instance with a get_answer(...) method.
            temperature: Sampling temperature between 0 and 1.
        """
        if not hasattr(client, 'get_answer'):
            raise ValueError("Client must implement a get_answer method")
        if not (0 <= temperature <= 1):
            raise ValueError("Temperature must be between 0 and 1")

        self.client = client
        self.temperature = temperature

    def _build_system_message(self, history: List[str]) -> str:
        """
        Constructs the system-role message, including any history of generated prompts.

        Args:
            history: List of prior generated prompts.

        Returns:
            A string for the system message.
        """
        if history:
            prompt_hist = "".join(p + "\n" for p in history)
            return (
                "You are a Prompt Generator. Your task is to produce a concise, unique prompt "
                "designed to augment the given query. "
                f"Previously generated prompts are as follows: {prompt_hist}. "
                "Generate a new prompt that is distinct and does not duplicate any of the existing prompts. "
                "Ensure that you do not include extraneous information or introduce data beyond the scope "
                "of the given query. Return only the new, concise, and unique prompt without any additional "
                "explanations. Keep a check that output remains concise and approximately the same length "
                "as the original query."
            )
        else:
            return (
                "You are a Prompt Generator. Your task is to refine and enhance "
                "the given query while preserving its original intent. Only return"
                "the augmented prompt. Do not introduce "
                "external information—strictly adhere to the content of the query. Ensure "
                "the output remains concise and approximately the same length as the original query."
            )

    def generate(self, query: str, synthetic_count: int = 3, max_tokens: int = 200) -> List[str]:
        """
        Runs the augmentation loop to produce a list of unique prompts.

        Args:
            query: The original query string to augment.
            synthetic_count: Number of unique prompts to generate.
            max_tokens: Maximum tokens per generated prompt.

        Returns:
            A list of strings, where the last element is the original query,
            preceded by synthetic_count generated prompts.

        Raises:
            ValueError: If query is empty or parameters are out of range.
            RuntimeError: If the LLM client fails to generate a prompt.
        """
        if not query or not isinstance(query, str):
            logger.error("Invalid query provided: %r", query)
            raise ValueError("Query must be a non-empty string")
        if synthetic_count < 1:
            logger.error("Invalid synthetic_count: %s", synthetic_count)
            raise ValueError("synthetic_count must be at least 1")
        if max_tokens < 1:
            logger.error("Invalid max_tokens: %s", max_tokens)
            raise ValueError("max_tokens must be at least 1")

        prompts: List[str] = []

        try:
            for i in range(synthetic_count):
                system_content = self._build_system_message(prompts)
                logger.info("Requesting synthetic prompt %d/%d", i + 1, synthetic_count)

                response = self.client.get_answer(
                    sys_prompt=system_content,
                    user_prompt=query,
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )

                if not response or not isinstance(response, str):
                    logger.error("Empty or invalid response from LLM: %r", response)
                    raise RuntimeError("Received invalid prompt from LLM")

                new_prompt = response.strip()
                prompts.append(new_prompt)
                logger.info("Generated prompt #%d: %r", i + 1, new_prompt)

        except Exception as e:
            logger.exception("Error during prompt generation")
            raise

        # append original query at end
        prompts.append(query)
        logger.info("Prompt generation completed with %d prompts", len(prompts))
        return prompts
