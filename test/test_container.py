import importlib
from os.path import exists

from nubison_model import NubisonModel, register
from nubison_model.Container import load_mlflow_artifacts_for_container
from nubison_model.utils import temporary_cwd
from test.utils import temporary_dirs, temporary_file_with_content


def test_load_mlflow_artifacts_without_dockerfile(mlflow_server):
    class DummyModel(NubisonModel):
        pass

    with temporary_cwd("test/fixtures"):
        model_uri = register(DummyModel())

    with temporary_dirs(["infer"]), temporary_cwd("infer"):
        load_mlflow_artifacts_for_container(model_uri)

        assert exists("conda.yaml"), "conda.yaml file not found"
        assert exists("Dockerfile"), "Dockerfile not found"

        # Check if the default Dockerfile content is the same as the loaded Dockerfile content
        with importlib.resources.path(
            "nubison_model", "Dockerfile"
        ) as dockerfile_path, open("Dockerfile") as file:
            default_dockerfile_content = dockerfile_path.read_text()
            loaded_dockerfile_content = file.read()
            assert (
                default_dockerfile_content == loaded_dockerfile_content
            ), "Dockerfile content does not match"


def test_load_mlflow_artifacts_with_dockerfile(mlflow_server):
    class DummyModel(NubisonModel):
        pass

    custom_dockerfile_content = "# Test docker file content"

    # Register the model with a custom Dockerfile
    with temporary_cwd("test/fixtures"), temporary_file_with_content(
        "Dockerfile", custom_dockerfile_content
    ):
        model_uri = register(DummyModel())

    with temporary_dirs(["infer"]), temporary_cwd("infer"):
        load_mlflow_artifacts_for_container(model_uri)

        assert exists("conda.yaml"), "conda.yaml file not found"
        assert exists("Dockerfile"), "Dockerfile not found"

        # Check if the custom Dockerfile content is the same as the loaded Dockerfile content
        with open("Dockerfile") as file:
            loaded_dockerfile_content = file.read()
            assert (
                custom_dockerfile_content == loaded_dockerfile_content
            ), "Dockerfile content does not match"
