from __future__ import annotations


import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.port import Port
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

__all__ = ["straight_heater_metal_simple"]


@cell
def simple_taper(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    port: Port | None = None,
    with_bbox: bool = True,
    with_two_ports: bool = True,
    cross_section: CrossSectionSpec = "strip",
    port_order_name: tuple | None = ("o1", "o2"),
    port_order_types: tuple | None = ("optical", "optical"),
    **kwargs,
) -> Component:
    """Linear taper.

    Deprecated, use gf.components.taper_cross_section instead

    Args:
        length: taper length.
        width1: width of the west port.
        width2: width of the east port.
        port: can taper from a port instead of defining width1.
        with_bbox: box in bbox_layers and bbox_offsets to avoid DRC sharp edges.
        with_two_ports: includes a second port.
            False for terminator and edge coupler fiber interface.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
         port_order_name(tuple): Ordered tuple of port names. First port is default taper port, second name only if with_two_ports flags used.
        port_order_types(tuple): Ordered tuple of port types. First port is default taper port, second name only if with_two_ports flags used.
        kwargs: cross_section settings.
    """
    x = gf.get_cross_section(cross_section, **kwargs)
    layer = x.layer

    if isinstance(port, gf.Port) and width1 is None:
        width1 = port.width
    if width2 is None:
        width2 = width1

    c = gf.Component()

    y1 = width1 / 2
    y2 = width2 / 2
    x1 = x.copy(width=width1)
    x2 = x.copy(width=width2)
    xpts = [0, length, length, 0]
    ypts = [y1, y2, -y2, -y1]
    c.add_polygon((xpts, ypts), layer=layer)

    for section in x.sections:
        layer = section.layer
        y1 = section.width / 2
        y2 = y1 + (width2 - width1)
        ypts = [y1, y2, -y2, -y1]
        c.add_polygon((xpts, ypts), layer=layer)

    if x.cladding_layers and x.cladding_offsets:
        for layer, offset in zip(x.cladding_layers, x.cladding_offsets, strict=True):
            y1 = width1 / 2 + offset
            y2 = width2 / 2 + offset
            ypts = [y1, y2, -y2, -y1]
            c.add_polygon((xpts, ypts), layer=gf.get_layer(layer))

    c.add_port(
        name=port_order_name[0],
        center=(0, 0),
        width=width1,
        orientation=180,
        layer=x.layer,
        cross_section=x1,
        port_type=port_order_types[0],
    )
    if with_two_ports:
        c.add_port(
            name=port_order_name[1],
            center=(length, 0),
            width=width2,
            orientation=0,
            layer=x.layer,
            cross_section=x2,
            port_type=port_order_types[1],
        )

    if with_bbox and length:
        padding = []
        for offset in x.bbox_offsets:
            points = get_padding_points(
                component=c,
                default=0,
                bottom=offset,
                top=offset,
            )
            padding.append(points)

        for layer, points in zip(x.bbox_layers, padding, strict=True):
            c.add_polygon(points, layer=layer)

    c.info["length"] = length
    c.info["width1"] = float(width1)
    c.info["width2"] = float(width2)

    if x.add_bbox:
        c = x.add_bbox(c)
    if x.add_pins:
        c = x.add_pins(c)
    return c


@cell
def straight_heater_metal_simple(
    length: float = 320.0,
    length_straight_input: float = 15.0,
    heater_width: float = 2.5,
    cross_section_heater: CrossSectionSpec = "heater_metal",
    cross_section_waveguide_heater: CrossSectionSpec = "strip_heater_metal",
    via_stack: ComponentSpec | None = "via_stack_heater_mtop",
    port_orientation1: int | None = None,
    port_orientation2: int | None = None,
    heater_taper_length: float | None = 5.0,
    ohms_per_square: float | None = None,
    **kwargs,
) -> Component:
    """Returns a thermal phase shifter that has properly fixed electrical connectivity to extract a suitable electrical netlist and models.
    dimensions from https://doi.org/10.1364/OE.27.010456
    Args:
        length: of the waveguide.
        length_undercut_spacing: from undercut regions.
        length_undercut: length of each undercut section.
        length_straight_input: from input port to where trenches start.
        heater_width: in um.
        cross_section_heater: for heated sections. heater metal only.
        cross_section_waveguide_heater: for heated sections.
        cross_section_heater_undercut: for heated sections with undercut.
        with_undercut: isolation trenches for higher efficiency.
        via_stack: via stack.
        port_orientation1: left via stack port orientation.
        port_orientation2: right via stack port orientation.
        heater_taper_length: minimizes current concentrations from heater to via_stack.
        ohms_per_square: to calculate resistance.
        cross_section: for waveguide ports.
        kwargs: cross_section common settings.
    """
    c = Component()
    straight_heater_section = gf.components.straight(
        cross_section=cross_section_waveguide_heater,
        length=length,
        heater_width=heater_width,
        # **kwargs # Note cross section is set from the provided, no more settings should be set
    )

    c.add_ref(straight_heater_section)
    c.add_ports(straight_heater_section.ports)

    if via_stack:
        via = via_stackw = via_stacke = gf.get_component(via_stack)
        via_stack_west_center = straight_heater_section.size_info.cw
        via_stack_east_center = straight_heater_section.size_info.ce
        dx = via_stackw.get_ports_xsize() / 2 + heater_taper_length or 0

        via_stack_west = c << via_stackw
        via_stack_east = c << via_stacke
        via_stack_west.move(via_stack_west_center - (dx, 0))
        via_stack_east.move(via_stack_east_center + (dx, 0))

        valid_orientations = {p.orientation for p in via.ports.values()}
        p1 = via_stack_west.get_ports_list(orientation=port_orientation1)
        p2 = via_stack_east.get_ports_list(orientation=port_orientation2)

        if not p1:
            raise ValueError(
                f"No ports for port_orientation1 {port_orientation1} in {valid_orientations}"
            )
        if not p2:
            raise ValueError(
                f"No ports for port_orientation2 {port_orientation2} in {valid_orientations}"
            )

        # c.add_ports(p1, prefix="l_")
        # c.add_ports(p2, prefix="r_")
        if heater_taper_length:
            x = gf.get_cross_section(cross_section_heater, width=heater_width)
            taper = simple_taper(
                width1=via_stackw.ports["e1"].width,
                width2=heater_width,
                length=heater_taper_length,
                cross_section=x,
                port_order_name=("e1", "e2"),
                port_order_types=("electrical", "electrical"),
            )
            taper1 = c << taper
            taper2 = c << taper
            taper1.connect("e1", via_stack_west.ports["e3"])
            taper2.connect("e1", via_stack_east.ports["e1"])

    c.info["resistance"] = (
        ohms_per_square * heater_width * length if ohms_per_square else None
    )
    return c
