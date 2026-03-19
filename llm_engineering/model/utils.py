from loguru import logger
from azure.ai.ml import MLClient
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential

from llm_engineering.settings import settings


def _build_ml_client() -> MLClient:
    required_settings = {
        "AZURE_SUBSCRIPTION_ID": settings.AZURE_SUBSCRIPTION_ID,
        "AZURE_RESOURCE_GROUP": settings.AZURE_RESOURCE_GROUP,
        "AZURE_ML_WORKSPACE": settings.AZURE_ML_WORKSPACE,
    }
    missing = [key for key, value in required_settings.items() if not value]
    if missing:
        raise ValueError(f"Missing Azure ML settings: {', '.join(missing)}")

    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=settings.AZURE_SUBSCRIPTION_ID,  # type: ignore[arg-type]
        resource_group_name=settings.AZURE_RESOURCE_GROUP,  # type: ignore[arg-type]
        workspace_name=settings.AZURE_ML_WORKSPACE,  # type: ignore[arg-type]
    )


class ResourceManager:
    def __init__(self) -> None:
        self.ml_client = _build_ml_client()

    def endpoint_exists(self, endpoint_name: str) -> bool:
        """Check if the Azure ML endpoint exists."""
        try:
            self.ml_client.online_endpoints.get(name=endpoint_name)
            logger.info(f"Endpoint '{endpoint_name}' exists.")
            return True
        except HttpResponseError:
            logger.info(f"Endpoint '{endpoint_name}' does not exist.")
            return False
