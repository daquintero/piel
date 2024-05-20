# +
from __future__ import annotations

import numpy as np

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.mzi import mzi2x2_2x2
from gdsfactory.components.straight import straight
from gdsfactory.port import select_ports_electrical

# from gdsfactory.routing import get_route
from gdsfactory.routing.route_single import route_single


def find_largest_component(
    component_list: list,
) -> Component:
    """
    This function finds the largest component in a list of components.

    Args:
        component_list (list): The list of components.

    Returns:
        Component: The largest component in the list.
    """
    # TODO optimise speed here
    largest_component = component_list[0]
    for component_i in component_list:
        try:
            assert component_i.xsize is not None
            pass
        except AttributeError as e:
            raise AttributeError(
                "Component does not have xsize or ysize attributes. "
                "Check the component you are using: "
                + str(component_i)
                + " "
                + str(type(component_i))
            ) from e

        # If element is larger than largest component
        if (component_i.xsize * component_i.ysize) > (
            largest_component.xsize * largest_component.ysize
        ):
            largest_component = component_i
    return largest_component


@cell
def component_lattice_generic(
    network: list[list] | None = None, gap_elements: list = None, **kwargs
) -> Component:
    """
    The shape of the `network` matrix determines the physical interconnection.
    Note that there should be at least S+1=N modes
    based on this formalism of interconnection,
    and the position of the component implements a connectivity in between the modes,
    and assumes a 2x2 network encoding.
    One nice functionality by this component is that it can generate a
    component lattice for generic variable components with different x and y pitches.
    Initially this will maximise the surface area required
    but different placement algorithms can compact the size.

    Args:
        network: A list of lists of components that are to be placed in the lattice.
        gap_elements: A list of elements that are to be considered as gaps in the lattice.
        **kwargs: Additional keyword arguments to be passed to the function.

    Returns:
        Component: A component lattice that implements the physical network.

    The placement matrix is in this form:
    .. math::

        M = X & 0 & X
            0 & P & 0
            X & 0 & X


    :include-source:
        import gdsfactory as gf
        from gdsfactory.components.mzi import mzi2x2_2x2

        example_component_lattice = [
            [mzi2x2_2x2(), 0, mzi2x2_2x2()],
            [0, mzi2x2_2x2(delta_length=30.0), 0],
            [mzi2x2_2x2(), 0, mzi2x2_2x2()],
        ]
        c = gf.components.component_lattice_generic(example_component_lattice)


    Another example that demonstrates the generic-nature of this component lattice
    algorithm can be with an mixed set of actively driven and passiver interferometers.
    The placement matrix is in this form:

    .. math::

        M = Y & 0 & A
            0 & B & 0
            C & 0 & Y

    :include-source:
        import gdsfactory as gf
        from gdsfactory.components import mzi2x2_2x2_phase_shifter, mzi2x2_2x2

        example_mixed_component_lattice = [
            [mzi2x2_2x2_phase_shifter(), 0, mzi2x2_2x2(delta_length=20.0)],
            [0, mzi2x2_2x2(delta_length=30.0), 0],
            [mzi2x2_2x2(delta_length=15.0), 0, mzi2x2_2x2_phase_shifter()],
        ]
        c = gf.components.component_lattice_generic(
            network=example_mixed_component_lattice
        )

    # TODO implement balanced waveguide paths function per stage
    # TODO automatic electrical fanout?
    # TODO multiple placement optimization algorithms.
    """
    # Temporary for compatibility, before extending functionality
    if gap_elements is None:
        gap_elements = [0]

    gap_elements = gap_elements[0]

    network = network or [
        [mzi2x2_2x2(), gap_elements, mzi2x2_2x2()],
        [gap_elements, mzi2x2_2x2(delta_length=30.0), gap_elements],
        [mzi2x2_2x2(), gap_elements, mzi2x2_2x2()],
    ]

    element_references = list()
    network = np.array(network)
    # Check number of dimensions is 2
    if network.ndim != 2:
        # Get the length and then width of the array
        raise AttributeError(
            "Physical network dimensions don't work."
            "Check the dimensional structure of your network matrix."
        )

    c = Component()
    # Estimate the size of the network fabric
    condition = network != gap_elements
    non_zero_indices = np.where(condition)
    elements_list = np.vstack([non_zero_indices, network[condition]])
    # elements_list = np.vstack([np.nonzero(network), network[np.nonzero(network)]])
    largest_component = find_largest_component(elements_list[2])  # List of elements
    component_column_amount = len(network[0])
    mode_amount = component_column_amount + 1
    inter_stage_clearance_x_offset = 40
    inter_stage_clearance_y_offset = 40
    x_length = (
        mode_amount * largest_component.xsize
        + mode_amount * inter_stage_clearance_x_offset
    )
    y_length = (
        mode_amount * largest_component.ysize
        + mode_amount * inter_stage_clearance_y_offset
    )
    x_component_pitch = x_length / mode_amount
    y_component_pitch = y_length / mode_amount
    x_mode_pitch = x_length / mode_amount
    y_mode_pitch = y_length / mode_amount
    # each distinct operation on the network is a separate iteration for-loop so new
    # functionality can be extended and easily identified.

    # Create all the waveguides inputs and outputs
    # Originally implemented in gdsfactory array but got netlist errors
    interconnection_ports_array = []
    for column_j in range(mode_amount):
        interconnection_ports_array.append([])
        for row_i in range(mode_amount):
            straight_i = c << straight(length=1, width=0.5)
            interconnection_ports_array[column_j].extend([straight_i])
            interconnection_ports_array[column_j][row_i].move(
                destination=(x_mode_pitch * column_j, -y_mode_pitch * row_i)
            )

            if column_j == 0:
                # Inputs
                c.add_port(
                    port=straight_i.ports["o1"],
                    name="in_o_" + str(row_i),
                )
            elif column_j == (mode_amount - 1):
                # Outputs
                c.add_port(
                    port=straight_i.ports["o2"],
                    name="out_o_" + str(row_i),
                )

    # Place the components in between the corresponding waveguide modes
    j = 0
    k = 0
    for column_j in network:
        i = 0
        for element_i in column_j:
            # Check if element is nonzero
            if element_i != gap_elements:
                element_references.append(c << element_i)
                element_references[k].center = (0, 0)
                element_references[k].move(
                    destination=(
                        x_component_pitch * j
                        + largest_component.xsize / 2
                        + inter_stage_clearance_x_offset / 2,
                        -y_component_pitch * i - inter_stage_clearance_y_offset,
                    )
                )
                k += 1
            i += 1
        j += 1

    # Go position by position to place and connect everything, column by column
    j = 0
    k = 0
    for column_j in network:
        i = 0
        # Connect the modes together
        for element_i in column_j:
            # Row in column
            if element_i != gap_elements:
                # Connect the adjacent input waveguide ports to the first element columns
                # if j == 0:
                route_single(
                    c,
                    interconnection_ports_array[j][i].ports["o2"],
                    element_references[k].ports["o2"],
                    radius=5,
                )
                route_single(
                    c,
                    interconnection_ports_array[j][i + 1].ports["o2"],
                    element_references[k].ports["o1"],
                    radius=5,
                )
                # Connect output of the component to the component
                route_single(
                    c,
                    interconnection_ports_array[j + 1][i].ports["o1"],
                    element_references[k].ports["o3"],
                    radius=5,
                )
                route_single(
                    c,
                    interconnection_ports_array[j + 1][i + 1].ports["o1"],
                    element_references[k].ports["o4"],
                    radius=5,
                )
                k += 1

            elif element_i == gap_elements:
                # When no element at junction, connect straight ahead between
                if i == 0:
                    # If at start top row then just connect top
                    route_single(
                        c,
                        interconnection_ports_array[j][i].ports["o2"],
                        interconnection_ports_array[j + 1][i].ports["o1"],
                        radius=5,
                    )
                elif i == (len(column_j) - 1):
                    # If at end then connect bottom
                    route_single(
                        c,
                        interconnection_ports_array[j][i + 1].ports["o2"],
                        interconnection_ports_array[j + 1][i + 1].ports["o1"],
                        radius=5,
                    )

                    if column_j[i - 1] != gap_elements:
                        # If previous element nonzero then pass
                        pass
                    elif column_j[i - 1] == gap_elements:
                        # If previous element is zero then connect top straight
                        route_single(
                            c,
                            interconnection_ports_array[j][i].ports["o2"],
                            interconnection_ports_array[j + 1][i].ports["o1"],
                            radius=5,
                        )

                elif column_j[i - 1] == gap_elements:
                    # If previous element is zero then connect top straight
                    route_single(
                        c,
                        interconnection_ports_array[j][i].ports["o2"],
                        interconnection_ports_array[j + 1][i].ports["o1"],
                        radius=5,
                    )

                elif column_j[i - 1] != gap_elements:
                    # If previous element nonzero then pass
                    pass
            i += 1
        j += 1

    # Append electrical ports to component to total connectivity can be constructed.
    j = 0
    k = 0
    for column_j in network:
        i = 0
        for element_i in column_j:
            # Check if element is nonzero
            if element_i != gap_elements:
                electrical_ports_list_i = select_ports_electrical(
                    element_references[k].ports
                ).items()
                if len(electrical_ports_list_i) > 0:
                    # Electrical ports exist in component
                    for electrical_port_i in electrical_ports_list_i:
                        c.add_port(
                            port=electrical_port_i[1],
                            name=electrical_port_i[0] + "_" + str(i) + "_" + str(j),
                        )
                        # Row column notation
                k += 1
            i += 1
        j += 1
    return c


if __name__ == "__main__":
    # from gdsfactory.components.mzi import mzi2x2_2x2
    # from gdsfactory.components.mzi_phase_shifter import mzi2x2_2x2_phase_shifter

    # example_component_lattice = [
    #     [mzi2x2_2x2(), 0, mzi2x2_2x2()],
    #     [0, mzi2x2_2x2(), 0],
    #     [mzi2x2_2x2(), 0, mzi2x2_2x2()],
    # ]
    # c = component_lattice_generic(example_component_lattice)
    # c.show(show_ports=True)

    # example_mixed_component_lattice = [
    #     [mzi2x2_2x2_phase_shifter(), 0, mzi2x2_2x2(delta_length=20.0)],
    #     [0, mzi2x2_2x2(delta_length=50.0), 0],
    #     [mzi2x2_2x2(delta_length=100.0), 0, mzi2x2_2x2_phase_shifter()],
    # ]
    # c_mixed = component_lattice_generic(example_mixed_component_lattice)
    # c_mixed.show(show_ports=True)
    c = component_lattice_generic()
    c.show()
# -
