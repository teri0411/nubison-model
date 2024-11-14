from contextlib import contextmanager
from os import environ, getcwd, makedirs, path, remove
from shutil import rmtree
from typing import List

from nubison_model.utils import temporary_cwd


@contextmanager
def temporary_dirs(dirs: List[str]):
    dirs = [path.join(getcwd(), src_dir) for src_dir in dirs]

    try:
        for dir in dirs:
            makedirs(dir, exist_ok=True)

        yield dirs

    finally:
        for dir in dirs:
            if path.exists(dir):
                rmtree(dir)


@contextmanager
def temporary_env(env: dict):
    original_env = environ.copy()
    for key, value in env.items():
        environ[key] = value

    yield

    environ.clear()
    environ.update(original_env)


@contextmanager
def temporary_file_with_content(filename, content=""):
    try:
        with open(filename, "w") as temp_file:
            temp_file.write(content)
        yield temp_file
    finally:
        if path.exists(filename):
            remove(filename)


__all__ = [
    "temporary_cwd",
    "temporary_dirs",
    "temporary_env",
    "temporary_file_with_content",
]
