"""Console script for piel."""

import sys
from .cli.main import main
from .file_system import create_piel_home_directory


if __name__ == "__main__":
    create_piel_home_directory()
    sys.exit(main())  # pragma: no cover
