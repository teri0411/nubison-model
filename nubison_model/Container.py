import importlib.resources
import shutil
from os import getenv
from typing import Optional

from mlflow.artifacts import download_artifacts, list_artifacts

from nubison_model.Model import (
    DEAFULT_MLFLOW_URI,
    ENV_VAR_MLFLOW_MODEL_URI,
    ENV_VAR_MLFLOW_TRACKING_URI,
)


def _load_dockerfile(mlflow_tracking_uri, mlflow_model_uri, dst_path):
    """
    Load the Dockerfile from the MLflow model or use the default Dockerfile
    """

    artifacts_dir_uri = mlflow_model_uri + "artifacts/"
    artifacts = list_artifacts(artifacts_dir_uri)
    exists_in_mlflow = any(artifact.path == "Dockerfile" for artifact in artifacts)
    if exists_in_mlflow:
        download_artifacts(
            artifact_uri=artifacts_dir_uri + "Dockerfile",
            tracking_uri=mlflow_tracking_uri,
            dst_path=dst_path,
        )
    else:
        with importlib.resources.path("nubison_model", "Dockerfile") as path:
            dockerfile_path = path
            shutil.copy(dockerfile_path, dst_path)


def load_mlflow_artifacts_for_container(
    mlflow_model_uri: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
):
    """
    Load MLflow artifacts for the container
    """

    mlflow_tracking_uri = (
        mlflow_tracking_uri or getenv(ENV_VAR_MLFLOW_TRACKING_URI) or DEAFULT_MLFLOW_URI
    )
    mlflow_model_uri = mlflow_model_uri or getenv(ENV_VAR_MLFLOW_MODEL_URI) or ""

    # Load conda.yaml and Dockerfile
    download_artifacts(
        artifact_uri=mlflow_model_uri + "conda.yaml",
        tracking_uri=mlflow_tracking_uri,
        dst_path=".",
    )
    _load_dockerfile(mlflow_tracking_uri, mlflow_model_uri, ".")
