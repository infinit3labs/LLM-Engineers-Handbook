import json
from typing import Any, Dict, Optional

import requests
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from loguru import logger

from llm_engineering.domain.inference import Inference
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


class LLMInferenceAzureEndpoint(Inference):
    """
    Perform inference against an Azure ML managed online endpoint.
    """

    def __init__(
        self,
        endpoint_name: str,
        default_payload: Optional[Dict[str, Any]] = None,
        deployment_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.endpoint_name = endpoint_name
        self.deployment_name = deployment_name or settings.AZURE_DEPLOYMENT_NAME
        self.payload = default_payload if default_payload else self._default_payload()

        self.ml_client = _build_ml_client()
        endpoint = self.ml_client.online_endpoints.get(name=self.endpoint_name)
        keys = self.ml_client.online_endpoints.get_keys(name=self.endpoint_name)

        self.scoring_uri = endpoint.scoring_uri
        self.api_key = keys.primary_key

        if not self.scoring_uri or not self.api_key:
            raise ValueError(
                "Azure ML endpoint is missing scoring URI or access key. Ensure the endpoint is deployed."
            )

    def _default_payload(self) -> Dict[str, Any]:
        """
        Generates the default payload for the inference request.

        Returns:
            dict: The default payload.
        """

        return {
            "input_data": {
                "input_string": ["How is the weather?"],
                "parameters": {
                    "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                    "top_p": settings.TOP_P_INFERENCE,
                    "temperature": settings.TEMPERATURE_INFERENCE,
                },
            }
        }

    def set_payload(self, inputs: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Sets the payload for the inference request.

        Args:
            inputs (str): The input text for the inference.
            parameters (dict, optional): Additional parameters for the inference. Defaults to None.
        """

        self.payload["input_data"]["input_string"] = [inputs]
        if parameters:
            self.payload["input_data"]["parameters"].update(parameters)

    def inference(self) -> Dict[str, Any]:
        """
        Performs the inference request using the Azure ML endpoint.

        Returns:
            dict: The response from the inference request.
        Raises:
            Exception: If an error occurs during the inference request.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.deployment_name:
            headers["azureml-model-deployment"] = self.deployment_name

        try:
            logger.info("Sending inference request to Azure ML endpoint.")
            response = requests.post(
                self.scoring_uri,
                headers=headers,
                data=json.dumps(self.payload),
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            logger.exception("Azure ML inference failed.")
            raise
