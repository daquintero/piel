import matplotlib as mpl
import pathlib

__all__ = [
    "activate_piel_styles",
]


def activate_piel_styles():
    """
    Activates the piel fast rc params.

    Returns:
        None
    """
    mpl.style.use(pathlib.Path(__file__).parent / pathlib.Path("piel_fast.rcParams"))
