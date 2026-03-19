from __future__ import annotations

from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import CommandJob, Environment
from azure.identity import DefaultAzureCredential
from loguru import logger

from llm_engineering.settings import settings

finetuning_dir = Path(__file__).resolve().parent


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


def run_finetuning_on_azure_ml(
    finetuning_type: str = "sft",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 3e-4,
    dataset_huggingface_workspace: str = "mlabonne",
    is_dummy: bool = False,
) -> None:
    if not settings.HUGGINGFACE_ACCESS_TOKEN:
        raise ValueError("Hugging Face access token is required.")

    if not finetuning_dir.exists():
        raise FileNotFoundError(f"The directory {finetuning_dir} does not exist.")

    ml_client = _ml_client()
    logger.info(
        f"Submitting Azure ML finetuning job on compute '{settings.AZURE_COMPUTE_TARGET}' "
        f"with dataset workspace '{dataset_huggingface_workspace}'."
    )

    environment = Environment(
        name="hf-finetune",
        image="mcr.microsoft.com/azureml/curated/minimal-ubuntu22.04-py39-cpu-inference:latest",
    )

    job = CommandJob(
        code=str(finetuning_dir),
        command=(
            "python finetune.py "
            "--finetuning-type ${inputs.finetuning_type} "
            "--num-train-epochs ${inputs.num_train_epochs} "
            "--per-device-train-batch-size ${inputs.per_device_train_batch_size} "
            "--learning-rate ${inputs.learning_rate} "
            "--dataset-huggingface-workspace ${inputs.dataset_huggingface_workspace} "
            "--is-dummy ${inputs.is_dummy}"
        ),
        environment=environment,
        compute=settings.AZURE_COMPUTE_TARGET,
        experiment_name="finetune-llm-twin",
        inputs={
            "finetuning_type": finetuning_type,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "learning_rate": learning_rate,
            "dataset_huggingface_workspace": dataset_huggingface_workspace,
            "is_dummy": str(is_dummy),
        },
        environment_variables={
            "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
            "COMET_API_KEY": settings.COMET_API_KEY or "",
            "COMET_PROJECT": settings.COMET_PROJECT,
        },
        display_name=f"finetune-{finetuning_type}",
    )

    submitted_job = ml_client.jobs.create_or_update(job)
    logger.success(f"Finetuning job submitted. Monitor at: {submitted_job.services.get('Studio')}")
