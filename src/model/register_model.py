
import json
import mlflow
import logging
import os

# Set up DagsHub credentials
dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    raise EnvironmentError(
        "DAGSHUB_PAT environment variable is not set"
    )

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Endtroducing1993"
repo_name = "mlops-project-2"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(
    f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
)

# Logging configuration
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    "model_registration_errors.log"
)
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load model information from a JSON file."""
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)

        logger.debug("Model info loaded from %s", file_path)
        return model_info

    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise

    except Exception:
        logger.exception(
            "Unexpected error occurred while loading model info"
        )
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""

    try:
        model_uri = (
            f"runs:/{model_info['run_id']}/"
            f"{model_info['model_path']}"
        )

        logger.info("Registering model from URI: %s", model_uri)

        # Register the model
        model_version = mlflow.register_model(
            model_uri,
            model_name
        )

        logger.info(
            "Model registered successfully. Version: %s",
            model_version.version
        )

        # Transition model to Staging
        client = mlflow.tracking.MlflowClient()

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        # Verify model registration and stage
        versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        staged_versions = [
            version for version in versions
            if version.current_stage == "Staging"
        ]

        if not staged_versions:
            raise RuntimeError(
                f"Model {model_name} was not found in Staging"
            )

        logger.info(
            "Model %s version %s successfully moved to Staging",
            model_name,
            model_version.version
        )

    except Exception:
        logger.exception("Error during model registration")
        raise


def main():
    try:
        model_info_path = "reports/experiment_info.json"

        model_info = load_model_info(model_info_path)

        model_name = "my_model"

        register_model(model_name, model_info)

        logger.info("Model registration completed successfully")

    except Exception:
        logger.exception(
            "Failed to complete model registration process"
        )
        raise


if __name__ == "__main__":
    main()
