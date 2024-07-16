import matplotlib as mpl
import pathlib


def activate_piel_styles():
    """
    Activates the piel fast rc params.

    Returns:
        None
    """
    mpl.style.use(pathlib.Path(__file__).parent / pathlib.Path("piel_fast.rcParams"))
