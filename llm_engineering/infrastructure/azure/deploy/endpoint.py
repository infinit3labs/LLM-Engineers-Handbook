from __future__ import annotations

from loguru import logger
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint, Model
from azure.identity import DefaultAzureCredential

from llm_engineering.model.utils import ResourceManager
from llm_engineering.settings import settings


def _ml_client() -> MLClient:
    required_settings = {
        "AZURE_SUBSCRIPTION_ID": settings.AZURE_SUBSCRIPTION_ID,
        "AZURE_RESOURCE_GROUP": settings.AZURE_RESOURCE_GROUP,
        "AZURE_ML_WORKSPACE": settings.AZURE_ML_WORKSPACE,
    }
    missing = [key for key, value in required_settings.items() if not value]
    if missing:
        raise ValueError(f"Missing Azure ML settings: {', '.join(missing)}")

    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=settings.AZURE_SUBSCRIPTION_ID,  # type: ignore[arg-type]
        resource_group_name=settings.AZURE_RESOURCE_GROUP,  # type: ignore[arg-type]
        workspace_name=settings.AZURE_ML_WORKSPACE,  # type: ignore[arg-type]
    )


def create_endpoint() -> None:
    """
    Create or update an Azure ML managed online endpoint for LLM inference.
    """

    ml_client = _ml_client()
    endpoint_name = settings.AZURE_ENDPOINT_NAME
    deployment_name = settings.AZURE_DEPLOYMENT_NAME

    logger.info(f"Creating or updating Azure ML endpoint '{endpoint_name}' with deployment '{deployment_name}'.")

    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        auth_mode="key",
        description="LLM Twin managed online endpoint",
    )
    ml_client.begin_create_or_update(endpoint).result()

    model = Model(
        name=f"{endpoint_name}-model",
        path=settings.HF_MODEL_ID,
        type="huggingface",
        description="Hugging Face model pulled directly from the Hub.",
    )

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        instance_type=settings.AZURE_INSTANCE_TYPE,
        instance_count=settings.AZURE_INSTANCE_COUNT,
        environment_variables={
            "HF_MODEL_ID": settings.HF_MODEL_ID,
            "MAX_INPUT_LENGTH": str(settings.MAX_INPUT_LENGTH),
            "MAX_TOTAL_TOKENS": str(settings.MAX_TOTAL_TOKENS),
            "MAX_BATCH_TOTAL_TOKENS": str(settings.MAX_BATCH_TOTAL_TOKENS),
        },
    )

    ml_client.begin_create_or_update(deployment).result()
    logger.success(f"Endpoint '{endpoint_name}' is ready with deployment '{deployment_name}'.")


def delete_endpoint() -> None:
    """
    Delete the Azure ML managed online endpoint.
    """

    endpoint_name = settings.AZURE_ENDPOINT_NAME
    ml_client = _ml_client()
    manager = ResourceManager()

    if not manager.endpoint_exists(endpoint_name):
        logger.info(f"Endpoint '{endpoint_name}' does not exist. Skipping deletion.")
        return

    logger.info(f"Deleting Azure ML endpoint '{endpoint_name}'.")
    ml_client.online_endpoints.begin_delete(name=endpoint_name).result()
    logger.success(f"Endpoint '{endpoint_name}' deleted.")
