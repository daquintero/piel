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

   piel.integration.type_conversion.convert_array_type
   piel.integration.type_conversion.convert_2d_array_to_string
   piel.integration.type_conversion.absolute_to_threshold



Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.integration.type_conversion.array_types
   piel.integration.type_conversion.tuple_int_type
   piel.integration.type_conversion.package_array_types


.. py:data:: array_types



.. py:data:: tuple_int_type



.. py:data:: package_array_types



.. py:function:: convert_array_type(array: array_types, output_type: package_array_types)


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


.. py:function:: absolute_to_threshold(array: array_types, threshold: float = 1e-06, dtype_output: int | float | bool = int, output_array_type: package_array_types = 'jax') -> package_array_types

   This function converts the computed optical transmission arrays to single bit digital signals.
   The function takes the absolute value of the array and compares it to a threshold to determine the digital signal.

   :param array: The optical transmission array of any dimension.
   :type array: array_types
   :param dtype_output: The output type. Defaults to int.
   :type dtype_output: int | float | bool, optional
   :param threshold: The threshold to compare the array to. Defaults to 1e-6.
   :type threshold: float, optional
   :param output_array_type: The output type. Defaults to "jax".
   :type output_array_type: array_types, optional

   Returns:
