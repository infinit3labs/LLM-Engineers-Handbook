# Azure ML Deployment and Inference

This repository contains scripts for configuring Azure ML, deploying a Hugging Face model as a managed online endpoint, and testing inference end-to-end.

## Contents

1. [Azure Configuration](#azure-configuration)
2. [Deploying a Hugging Face Endpoint](#deploying-a-hugging-face-endpoint)
3. [Testing Inference on the Deployed Endpoint](#testing-inference-on-the-deployed-endpoint)
4. [Helper Tasks](#helper-tasks)

## Azure Configuration

Before using the deployment scripts, make sure you have:

- An Azure subscription with access to an Azure ML workspace.
- Azure CLI installed and logged in: `az login`.
- The following environment variables set (for example in `.env`):

```bash
AZURE_SUBSCRIPTION_ID=<subscription_id>
AZURE_RESOURCE_GROUP=<resource_group>
AZURE_ML_WORKSPACE=<workspace_name>
AZURE_ENDPOINT_NAME=twin
AZURE_DEPLOYMENT_NAME=blue
AZURE_INSTANCE_TYPE=Standard_DS3_v2  # choose a GPU SKU for heavy models
AZURE_INSTANCE_COUNT=1
```

## Deploying a Hugging Face Endpoint

Use the provided Poe task to create or update an Azure ML managed online endpoint:

```bash
poetry poe deploy-inference-endpoint
```

The script in `llm_engineering.infrastructure.azure.deploy.endpoint` will create the endpoint and deployment using the Hugging Face model ID from `llm_engineering.settings`.

## Testing Inference on the Deployed Endpoint

After the endpoint is live, run a test inference:

```bash
poetry poe test-inference-endpoint
```

The test script will fetch the endpoint keys via the Azure ML SDK and send a sample prompt to verify the deployment.

## Helper Tasks

- Delete the managed endpoint when you are done to avoid charges:

```bash
poetry poe delete-inference-endpoint
```

- Submit training or evaluation jobs to Azure ML directly:

```bash
poetry run python -m llm_engineering.model.finetuning.azure_ml
poetry run python -m llm_engineering.model.evaluation.azure_ml
```

Keep the Azure resources stopped or deleted when not in use to control costs.
