from fastcore.xtras import Path


def get_project_root() -> Path:
    """Returns the root folder of the whole project."""
    return Path(__file__).parent.parent.parent


def get_package_root() -> Path:
    """Returns the root folder of the tsp-cls package."""
    package = Path(f"{get_project_root()}/tsp_cls")
    return package


def get_data_root() -> Path:
    """Returns the data folder."""
    data = Path(f"{get_project_root()}/data/ai-crowd-snake-dataset")
    return data


if __name__ == "__main__":
    print("[PROJECT ROOT]:", get_project_root())
    print("[PACKAGE ROOT]:", get_package_root())
    print("[DATA ROOT]:", get_data_root())
