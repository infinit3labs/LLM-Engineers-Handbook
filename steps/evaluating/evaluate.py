from zenml import step

from llm_engineering.model.evaluation.azure_ml import run_evaluation_on_azure_ml


@step
def evaluate(
    is_dummy: bool = False,
) -> None:
    run_evaluation_on_azure_ml(
        is_dummy=is_dummy,
    )
