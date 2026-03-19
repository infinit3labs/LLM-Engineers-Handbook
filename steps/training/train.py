from zenml import step

from llm_engineering.model.finetuning.azure_ml import run_finetuning_on_azure_ml


@step
def train(
    finetuning_type: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    learning_rate: float,
    dataset_huggingface_workspace: str = "mlabonne",
    is_dummy: bool = False,
) -> None:
    run_finetuning_on_azure_ml(
        finetuning_type=finetuning_type,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        dataset_huggingface_workspace=dataset_huggingface_workspace,
        is_dummy=is_dummy,
    )
