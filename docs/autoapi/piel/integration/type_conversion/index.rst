:py:mod:`piel.integration.type_conversion`
==========================================

.. py:module:: piel.integration.type_conversion

.. autoapi-nested-parse::

   This file provides a set of utilities in converting between common data types to represent information between different toolsets.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.integration.type_conversion.convert_2d_array_to_string



.. py:function:: convert_2d_array_to_string(list_2D: list[list])

   This function is particularly useful to convert digital data when it is represented as a 2D array into a set of strings.

   :param list_2D: A 2D array of binary data.
   :type list_2D: list[list]

   :returns: A string of binary data.
   :rtype: binary_string (str)

   Usage:

       list_2D=[[0], [0], [0], [1]]
       convert_2d_array_to_string(list_2D)
       >>> "0001"
