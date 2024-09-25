import pathlib
from cycler import cycler


def activate_piel_styles():
    """
    Activates the piel fast rc params.

    Returns:
        None
    """
    import matplotlib.pyplot as mpl

    mpl.style.use(pathlib.Path(__file__).parent / pathlib.Path("piel_fast.rcParams"))


# Define the primary and secondary color palettes with proper hex formatting
primary_color_palette = [
    "#1982C4",
    "#B79174",
    "#6B4C93",
    "#C6B7DA",
    "#A17500",
    "#8C564B",
    "#32490E",
    "#6A5541",
    "#87C7F0",
    "#43A8E7",
]
secondary_color_palette = [
    "#145B93",
    "#8D7059",
    "#50356A",
    "#978BAA",
    "#754F00",
    "#693E36",
    "#24360A",
    "#4D3D31",
    "#6594BA",
    "#327FAF",
]

# Define a custom color cycle
primary_cycler = cycler(color=primary_color_palette)
secondary_cycler = cycler(color=secondary_color_palette)
