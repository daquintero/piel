"""Top-level package for piel."""
import os
import pathlib

from .defaults import *  # NOQA: F403
from .file_system import *  # NOQA: F403
from .openlane import *  # NOQA: F403
from .parametric import *  # NOQA: F403
from .cocotb import *  # NOQA: F403

os.environ["PIEL_PACKAGE_DIRECTORY"] = str(
    pathlib.Path(__file__).parent.parent.resolve()
)

__author__ = """Dario Quintero"""
__email__ = "darioaquintero@gmail.com"
__version__ = "0.0.29"
