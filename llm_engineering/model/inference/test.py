from loguru import logger

from llm_engineering.model.inference.inference import LLMInferenceAzureEndpoint
from llm_engineering.model.inference.run import InferenceExecutor
from llm_engineering.settings import settings

if __name__ == "__main__":
    text = "Write me a post about deploying LLM endpoints with Azure ML."
    logger.info(f"Running inference for text: '{text}'")
    llm = LLMInferenceAzureEndpoint(endpoint_name=settings.AZURE_ENDPOINT_NAME, deployment_name=None)
    answer = InferenceExecutor(llm, text).execute()

    logger.info(f"Answer: '{answer}'")
