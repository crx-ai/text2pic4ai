
import importlib.resources as pkg_resources
from pathlib import Path


def get_pkg_data_path() -> Path:
    base_path = pkg_resources.files("text2pic4ai").joinpath("_pkg_data")
    return Path(base_path)
