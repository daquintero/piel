"""
One of the main complexities of creating multi-physical systems is translating the dimensionality between the physics
that connect them. For example, we know that a photonic pulse contains multiple frequencies. If dispersion is present,
the propagation time of the pulse could account for all the decompositions of signals. However, in the context of a
bucket photodetector, those time-dimensions don't matter for conversion. However, they might matter for nonlinear optical interactions.
As such, it is essential, computationally, to reduce the dimensional size of certain properties according to the type of analysis that is being
performed.
"""

from piel.types import c


def v_g_from_n_g(n_g: float) -> float:
    """
    Calculate the Group Velocity (v₉) from the Group Index (n_g).

    The relationship between group velocity and group index is given by the equation:

    \[
    v_g = \frac{c}{n_g}
    \]

    where:
        - \( v_g \) is the **group velocity** (m/s),
        - \( n_g \) is the **group index** (dimensionless),
        - \( c \) is the **speed of light in vacuum** (m/s).

    **Parameters:**
        n_g (float):
            The group index, representing the factor by which the group velocity is reduced
            relative to the speed of light in vacuum.

    **Returns:**
        float:
            The calculated group velocity in meters per second (m/s).
    """
    return c.value / n_g


def n_g_from_v_g(v_g: float) -> float:
    """
    Calculate the Group Index (n_g) from the Group Velocity (v₉).

    The relationship between group velocity and group index is given by the equation:

    \[
    n_g = \frac{c}{v_g}
    \]

    where:
        - \( n_g \) is the **group index** (dimensionless),
        - \( v_g \) is the **group velocity** (m/s),
        - \( c \) is the **speed of light in vacuum** (m/s).

    **Parameters:**
        v_g (float):
            The group velocity in meters per second (m/s).

    **Returns:**
        float:
            The calculated group index (dimensionless).
    """
    return c.value / v_g
