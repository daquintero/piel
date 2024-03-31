:py:mod:`piel.types`
====================

.. py:module:: piel.types

.. autoapi-nested-parse::

   We create a set of parameters that can be used throughout the project for optimisation.

   The numerical solver is jax and is imported throughout the module.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   piel.types.PielBaseModel




Attributes
~~~~~~~~~~

.. autoapisummary::

   piel.types.PathTypes
   piel.types.ArrayTypes


.. py:data:: PathTypes

   

.. py:data:: ArrayTypes

   

.. py:class:: PielBaseModel


   Bases: :py:obj:`pydantic.BaseModel`

   .. py:class:: Config


      .. py:attribute:: arbitrary_types_allowed
         :value: True

         


   .. py:method:: supplied_parameters()



