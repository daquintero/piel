import pathlib
import os


def configure_development_environment():
    """Configures the development environment for local testing."""
    os.environ["DESIGNS"] = str(
        pathlib.Path(os.environ.get("PIEL_PACKAGE_DIRECTORY"))
        / "docs"
        / "examples"
        / "designs"
    )
    return 0
