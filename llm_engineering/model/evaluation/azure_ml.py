from __future__ import annotations

from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import CommandJob, Environment
from azure.identity import DefaultAzureCredential
from loguru import logger

from llm_engineering.settings import settings

evaluation_dir = Path(__file__).resolve().parent


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


def run_evaluation_on_azure_ml(is_dummy: bool = True) -> None:
    if not settings.HUGGINGFACE_ACCESS_TOKEN:
        raise ValueError("Hugging Face access token is required.")
    if not settings.OPENAI_API_KEY:
        raise ValueError("OpenAI API key is required.")

    if not evaluation_dir.exists():
        raise FileNotFoundError(f"The directory {evaluation_dir} does not exist.")

    ml_client = _ml_client()
    logger.info(f"Submitting Azure ML evaluation job on compute '{settings.AZURE_COMPUTE_TARGET}'.")

    environment = Environment(
        name="hf-evaluation",
        image="mcr.microsoft.com/azureml/curated/minimal-ubuntu22.04-py39-cpu-inference:latest",
    )

    job = CommandJob(
        code=str(evaluation_dir),
        command="python evaluate.py --is-dummy ${inputs.is_dummy}",
        environment=environment,
        compute=settings.AZURE_COMPUTE_TARGET,
        experiment_name="evaluate-llm-twin",
        inputs={"is_dummy": str(is_dummy)},
        environment_variables={
            "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
            "OPENAI_API_KEY": settings.OPENAI_API_KEY,
        },
        display_name="evaluate-llm-twin",
    )

    submitted_job = ml_client.jobs.create_or_update(job)
    logger.success(f"Evaluation job submitted. Monitor at: {submitted_job.services.get('Studio')}")
