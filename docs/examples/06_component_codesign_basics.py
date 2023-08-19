# # Component Codesign Basics

# When we have photonic components driven by electronic devices, there is a scope that we might want to optimize certain devices to be faster, smaller, or less power-consumptive. It can be complicated to do this just analytically, so we would like to have the capability of integrating our design software for each of our devices with the simulation software of our electronics. There might be multiple software tools to design different devices, and the benefit of integrating these tools via open-source is that co-design becomes much more feasible and meaningful.
#
# In this example, we will continue exploring the co-design of a thermo-optic phase shifter in continuation of all the previous examples.
#
# This example consists of a few things:
# * Connect a `gdsfactory` component to a `femwell` model, and relate the corresponding metrics.
# * Extract a `hdl21 SPICE`model from the generated component we can use in circuit simulators.
# * Simulate the mode profiles via `Tidy3D FDTD` and extract the `S-Parameters` from the layout.
# * Create a flow where variations on both electronic and photonic parameters can be quantified through this model.
#
# Good examples of sections of this flow are:
# * [gplugins-femwell HEAT](https://gdsfactory.github.io/gplugins/notebooks/femwell_02_heater.html)
# * [gplugins-tidy3d MODE](https://gdsfactory.github.io/gplugins/notebooks/tidy3d_01_tidy3d_modes.html)
# * [gplugins-tidy3d FDTD](https://gdsfactory.github.io/gplugins/notebooks/tidy3d_00_tidy3d.html)
# * Some of the previous examples in `piel`


# ## Starting from our heater geometry
#
# We will begin by extracting the electrical parameters of the basic `gf.components.straight_heater_metal_simple` which we have been using in our Mach-Zehnder MZI2x2 from the example provided in [`femwell`](https://helgegehring.github.io/femwell/photonics/examples/metal_heater_phase_shifter.html).
#
# We will create a function where we change the width of the heater, and we explore the change in resistance, but also in thermo-optic phase modulation efficiency. So, we want to have some easy functions that allow us to easily mesh a component because we plan to do a few variations.

# +
import gdsfactory as gf
import meshio
from gdsfactory.generic_tech import get_generic_pdk
from gdsfactory.technology import LayerStack
from femwell.thermal import solve_thermal
from skfem.io import from_meshio
from gplugins.gmsh.mesh import create_physical_mesh

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()
LAYER_STACK = PDK.layer_stack
LAYER_STACK.layers["heater"].thickness = 0.13
LAYER_STACK.layers["heater"].zmin = 2.2

heater = gf.components.straight_heater_metal_simple()
heater
# -

print(LAYER_STACK.layers.keys())

filtered_layerstack = LayerStack(
    layers={
        k: gf.pdk.get_layer_stack().layers[k]
        for k in ("slab90", "core", "via_contact", "heater")
    }
)

# +
filename = "mesh"


def mesh_with_physicals(mesh, filename):
    mesh_from_file = meshio.read(f"{filename}.msh")
    return create_physical_mesh(mesh_from_file, "triangle")


mesh = heater.to_gmsh(
    type="uz",
    xsection_bounds=[(4, -4), (4, 4)],
    layer_stack=LAYER_STACK,
    filename=f"{filename}.msh",
)
mesh = mesh_with_physicals(mesh, filename)
resolutions = dict(
    core={"resolution": 0.04, "distance": 1},
    clad={"resolution": 0.6, "distance": 1},
    box={"resolution": 0.6, "distance": 1},
    heater={"resolution": 0.1, "distance": 1},
)
mesh = from_meshio(mesh)
mesh.draw().plot()
# -

solve_thermal(
    mesh,
    thermal_conductivity={"heater": 28, "oxide": 1.38, "core": 148},
    specific_conductivity={"heater": 2.3e6},
    thermal_diffusivity={
        "heater": 28 / 598 / 5240,
        "oxide": 1.38 / 709 / 2203,
        "core": 148 / 711 / 2330,
    },
    # specific_heat={"(47, 0)_0": 598, 'oxide': 709, '(1, 0)': 711},
    # density={"(47, 0)_0": 5240, 'oxide': 2203, '(1, 0)': 2330},
    currents={"heater": 0.007},
)

# +
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict
from femwell.thermal import solve_thermal
from shapely.geometry import LineString, Polygon
from skfem import Basis, ElementTriP0
from skfem.io import from_meshio
from tqdm import tqdm

w_sim = 8 * 2
h_clad = 2.8
h_box = 2
w_core = 0.5
h_core = 0.22
h_heater = 0.14
w_heater = 2
offset_heater = 2 + (h_core + h_heater) / 2
h_silicon = 0.5

polygons = OrderedDict(
    bottom=LineString(
        [
            (-w_sim / 2, -h_core / 2 - h_box - h_silicon),
            (w_sim / 2, -h_core / 2 - h_box - h_silicon),
        ]
    ),
    core=Polygon(
        [
            (-w_core / 2, -h_core / 2),
            (-w_core / 2, h_core / 2),
            (w_core / 2, h_core / 2),
            (w_core / 2, -h_core / 2),
        ]
    ),
    heater=Polygon(
        [
            (-w_heater / 2, -h_heater / 2 + offset_heater),
            (-w_heater / 2, h_heater / 2 + offset_heater),
            (w_heater / 2, h_heater / 2 + offset_heater),
            (w_heater / 2, -h_heater / 2 + offset_heater),
        ]
    ),
    clad=Polygon(
        [
            (-w_sim / 2, -h_core / 2),
            (-w_sim / 2, -h_core / 2 + h_clad),
            (w_sim / 2, -h_core / 2 + h_clad),
            (w_sim / 2, -h_core / 2),
        ]
    ),
    box=Polygon(
        [
            (-w_sim / 2, -h_core / 2),
            (-w_sim / 2, -h_core / 2 - h_box),
            (w_sim / 2, -h_core / 2 - h_box),
            (w_sim / 2, -h_core / 2),
        ]
    ),
    wafer=Polygon(
        [
            (-w_sim / 2, -h_core / 2 - h_box - h_silicon),
            (-w_sim / 2, -h_core / 2 - h_box),
            (w_sim / 2, -h_core / 2 - h_box),
            (w_sim / 2, -h_core / 2 - h_box - h_silicon),
        ]
    ),
)

resolutions = dict(
    core={"resolution": 0.04, "distance": 1},
    clad={"resolution": 0.6, "distance": 1},
    box={"resolution": 0.6, "distance": 1},
    heater={"resolution": 0.1, "distance": 1},
)

mesh = from_meshio(
    mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=0.6)
)
mesh.draw().show()

# +
currents = np.linspace(0.0, 7.4e-3, 10)
current_densities = currents / polygons["heater"].area
neffs = []

for current_density in tqdm(current_densities):
    basis0 = Basis(mesh, ElementTriP0(), intorder=4)
    thermal_conductivity_p0 = basis0.zeros()
    for domain, value in {
        "core": 90,
        "box": 1.38,
        "clad": 1.38,
        "heater": 28,
        "wafer": 148,
    }.items():
        thermal_conductivity_p0[basis0.get_dofs(elements=domain)] = value
    thermal_conductivity_p0 *= 1e-12  # 1e-12 -> conversion from 1/m^2 -> 1/um^2

    basis, temperature = solve_thermal(
        basis0,
        thermal_conductivity_p0,
        specific_conductivity={"heater": 2.3e6},
        current_densities={"heater": current_density},
        fixed_boundaries={"bottom": 0},
    )

    if current_density == current_densities[-1]:
        basis.plot(temperature, shading="gouraud", colorbar=True)
        plt.show()

    temperature0 = basis0.project(basis.interpolate(temperature))
    epsilon = basis0.zeros() + (1.444 + 1.00e-5 * temperature0) ** 2
    epsilon[basis0.get_dofs(elements="core")] = (
        3.4777 + 1.86e-4 * temperature0[basis0.get_dofs(elements="core")]
    ) ** 2
    # basis0.plot(epsilon, colorbar=True).show()

    modes = compute_modes(basis0, epsilon, wavelength=1.55, num_modes=1, solver="scipy")

    if current_density == current_densities[-1]:
        modes[0].show(modes[0].E.real)

    neffs.append(np.real(modes[0].n_eff))

print(f"Phase shift: {2 * np.pi / 1.55 * (neffs[-1] - neffs[0]) * 320}")
plt.xlabel("Current / mA")
plt.ylabel("Effective refractive index $n_{eff}$")
plt.plot(currents * 1e3, neffs)
plt.show()
# -
