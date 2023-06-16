import os
import shutil

def copy_source_folder(source_directory: str, target_directory: str):
    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)
    shutil.copytree(
        source_directory,
        target_directory,
        symlinks=False,
        ignore=None,
        copy_function=shutil.copy2,
        ignore_dangling_symlinks=False,
        dirs_exist_ok=False,
    )

__all__ = [
    "copy_source_folder",
]
