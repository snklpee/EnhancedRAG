import logging
from typing import List, Optional

from huggingface_hub import HfApi, login
from huggingface_hub.utils import RepositoryNotFoundError

from config.settings import settings
from src.utils.metrics import track_metrics 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HuggingFaceModelLister:
    """
    A utility class to list Hugging Face models filtered by task, custom filters, and sort order.

    This class handles authentication, interacts with the Hugging Face Hub API, and
    provides robust error handling, logging, and metrics tracking.

    Attributes:
        hf_api (HfApi): An authenticated Hugging Face Hub API client.
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize the HuggingFaceModelLister and authenticate to the Hub.

        Args:
            token (Optional[str]): A Hugging Face access token. If not provided,
                                   the token is read from settings.HF_TOKEN .
        Raises:
            ValueError: If no token is provided or found in settings.
            Exception: For any unexpected errors during authentication.
        """
        try:
            _token = token or settings.HF_TOKEN.get_secret_value()
            if not _token:
                raise ValueError("No Hugging Face token provided.")
            login(token=_token)
            self.hf_api = HfApi()
            logger.info("Authenticated to Hugging Face Hub successfully.")
        except ValueError as ve:
            logger.error("Authentication failed: %s", ve)
            raise
        except Exception as e:
            logger.error("Unexpected error during Hugging Face login: %s", e)
            raise

    @track_metrics(lambda model_ids: len(model_ids))
    def list_models(
        self,
        task: str,
        filter: str,
        sort: str = "likes",
        gated: bool = False,
        inference: str = None,
        get_top: int = 10,
    ) -> List[str]:
        """
        List model IDs from the Hugging Face Hub matching the given criteria.

        Args:
            task (str): The task tag to filter models by (e.g., "text-classification").
            filter (str): A substring filter that model IDs must contain.
            sort (str, optional): The sort order (e.g., "downloads", "likes", "trending_score"). Defaults to "likes".
            gated (bool, optional): Whether to include gated (private/restricted) models. Defaults to False.
            inference : Literal["cold", "frozen", "warm"], optional A string to filter models on the Hub by their state on the Inference API. Warm models are available for immediate use. Cold models will be loaded on first inference call.
            get_top (int, optional): Maximum number of models to return. Defaults to 10.

        Returns:
            List[str]: A list of model IDs matching the criteria.

        Raises:
            RepositoryNotFoundError: If the specified task or filter yields no results.
            Exception: For any other errors during the API call.
        """
        try:
            logger.info(
                "Listing models: task=%s, filter=%s, sort=%s, gated=%s, limit=%d",
                task, filter, sort, gated, get_top
            )
            models = self.hf_api.list_models(
                sort=sort,
                inference=inference,
                task=task,
                filter=filter,
                gated=gated,
                limit=get_top,
            )
            model_ids = [model.modelId for model in models]
            logger.info("Found %d models", len(model_ids))
            return model_ids
        except RepositoryNotFoundError as rnfe:
            logger.error("No models found for task='%s' with filter='%s': %s", task, filter, rnfe)
            raise
        except Exception as e:
            logger.error("Error listing models: %s", e)
            raise