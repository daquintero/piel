:py:mod:`piel.tools.gdsfactory.patch`
=====================================

.. py:module:: piel.tools.gdsfactory.patch


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.gdsfactory.patch.straight_heater_metal_simple



.. py:function:: straight_heater_metal_simple(length: float = 320.0, length_straight_input: float = 15.0, heater_width: float = 2.5, cross_section_heater: gdsfactory.typings.CrossSectionSpec = 'heater_metal', cross_section_waveguide_heater: gdsfactory.typings.CrossSectionSpec = 'strip_heater_metal', via_stack: gdsfactory.typings.ComponentSpec | None = 'via_stack_heater_mtop', port_orientation1: int | None = None, port_orientation2: int | None = None, heater_taper_length: float | None = 5.0, ohms_per_square: float | None = None, **kwargs) -> gdsfactory.component.Component

   Returns a thermal phase shifter that has properly fixed electrical connectivity to extract a suitable electrical netlist and models.
   dimensions from https://doi.org/10.1364/OE.27.010456
   :param length: of the waveguide.
   :param length_undercut_spacing: from undercut regions.
   :param length_undercut: length of each undercut section.
   :param length_straight_input: from input port to where trenches start.
   :param heater_width: in um.
   :param cross_section_heater: for heated sections. heater metal only.
   :param cross_section_waveguide_heater: for heated sections.
   :param cross_section_heater_undercut: for heated sections with undercut.
   :param with_undercut: isolation trenches for higher efficiency.
   :param via_stack: via stack.
   :param port_orientation1: left via stack port orientation.
   :param port_orientation2: right via stack port orientation.
   :param heater_taper_length: minimizes current concentrations from heater to via_stack.
   :param ohms_per_square: to calculate resistance.
   :param cross_section: for waveguide ports.
   :param kwargs: cross_section common settings.
