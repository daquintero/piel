:py:mod:`piel.tools.gdsfactory`
===============================

.. py:module:: piel.tools.gdsfactory


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   netlist/index.rst
   patch/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.gdsfactory.get_input_ports_index
   piel.tools.gdsfactory.get_matched_ports_tuple_index
   piel.tools.gdsfactory.straight_heater_metal_simple



.. py:function:: get_input_ports_index(ports_index: dict, sorting_algorithm: Literal[get_input_ports_index.prefix] = 'prefix', prefix: str = 'in') -> tuple

   This function returns the input ports of a component. However, input ports may have different sets of prefixes and suffixes. This function implements different sorting algorithms for different ports names. The default algorithm is `prefix`, which sorts the ports by their prefix. The Endianness implementation means that the tuple order is determined according to the last numerical index order of the port numbering.

   .. code-block:: python

       raw_ports_index = {
           "in_o_0": 0,
           "out_o_0": 1,
           "out_o_1": 2,
           "out_o_2": 3,
           "out_o_3": 4,
           "in_o_1": 5,
           "in_o_2": 6,
           "in_o_3": 7,
       }

       get_input_ports_index(ports_index=raw_ports_index)

       # Output
       ((0, "in_o_0"), (5, "in_o_1"), (6, "in_o_2"), (7, "in_o_3"))

   :param ports_index: The ports index dictionary.
   :type ports_index: dict
   :param sorting_algorithm: The sorting algorithm to use. Defaults to "prefix".
   :type sorting_algorithm: Literal["prefix"], optional
   :param prefix: The prefix to use for the sorting algorithm. Defaults to "in".
   :type prefix: str, optional

   :returns: The ordered input ports index tuple.
   :rtype: tuple


.. py:function:: get_matched_ports_tuple_index(ports_index: dict, selected_ports_tuple: Optional[tuple] = None, sorting_algorithm: Literal[get_matched_ports_tuple_index.prefix, selected_ports] = 'prefix', prefix: str = 'in') -> (tuple, tuple)

   This function returns the input ports of a component. However, input ports may have different sets of prefixes
   and suffixes. This function implements different sorting algorithms for different ports names. The default
   algorithm is `prefix`, which sorts the ports by their prefix. The Endianness implementation means that the tuple
   order is determined according to the last numerical index order of the port numbering. Returns just a tuple of
   the index.

   .. code-block:: python

       raw_ports_index = {
           "in_o_0": 0,
           "out_o_0": 1,
           "out_o_1": 2,
           "out_o_2": 3,
           "out_o_3": 4,
           "in_o_1": 5,
           "in_o_2": 6,
           "in_o_3": 7,
       }

       get_input_ports_tuple_index(ports_index=raw_ports_index)

       # Output
       (0, 5, 6, 7)

   :param ports_index: The ports index dictionary.
   :type ports_index: dict
   :param selected_ports_tuple: The selected ports tuple. Defaults to None.
   :type selected_ports_tuple: tuple, optional
   :param sorting_algorithm: The sorting algorithm to use. Defaults to "prefix".
   :type sorting_algorithm: Literal["prefix"], optional
   :param prefix: The prefix to use for the sorting algorithm. Defaults to "in".
   :type prefix: str, optional

   :returns: The ordered input ports index tuple.
             matched_ports_name_tuple_order(tuple): The ordered input ports name tuple.
   :rtype: matches_ports_index_tuple_order(tuple)


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


