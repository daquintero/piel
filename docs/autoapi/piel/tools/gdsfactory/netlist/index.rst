:py:mod:`piel.tools.gdsfactory.netlist`
=======================================

.. py:module:: piel.tools.gdsfactory.netlist


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.gdsfactory.netlist.get_matched_ports_tuple_index
   piel.tools.gdsfactory.netlist.get_input_ports_index



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
