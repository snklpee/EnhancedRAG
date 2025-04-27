import logging
from typing import List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PromptAugmentor:
    """
    A helper class to generate a sequence of unique, concise prompts
    based on an original query, using a provided LLM client.

    Attributes:
        client: An instantiated LLM client .
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
            client: An LLM client instance.
            temperature: Sampling temperature.
        """
        self.client = client
        self.temperature = temperature

    def _build_system_message(self, history: List[str]) -> str:
        """
        Constructs the system‐role message, including any history of generated prompts.

        Args:
            history: List of prior generated prompts.

        Returns:
            A string for the system message.
        """
        if history:
            prompt_hist = ""
            for prompt in history:
                prompt_hist += prompt+"\n"
                
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

    def generate(self, query:str, synthetic_count: int = 3, max_tokens: int = 200) -> List[str]:
        """
        Runs the augmentation loop to produce a list of unique prompts.

        Returns:
            A list of strings, where the last element is the original query,
            followed by synthetic_count generated prompts.
        """
        prompts = []

        try:
            for i in range(synthetic_count):
                system_content = self._build_system_message(prompts)
                
                logger.info(f"Requesting synthetic prompt {i+1}/{synthetic_count}")
                
                response = self.client.get_answer(
                    sys_prompt=system_content,
                    user_prompt = query,
                    temperature=self.temperature,
                    max_tokens = max_tokens
                )

                new_prompt = response
                if not new_prompt:
                    raise ValueError("Received empty prompt from LLM")

                prompts.append(new_prompt)
                logger.info(f"Generated prompt #{i+1}: {new_prompt!r}")
        
        except Exception as e:
            logger.error("Error during prompt generation", exc_info=True)
            # Depending on your use case you might re-raise or return what you have so far
            raise
        prompts.append(query)
        
        return prompts
