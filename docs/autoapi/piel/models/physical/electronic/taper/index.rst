:py:mod:`piel.models.physical.electronic.taper`
===============================================

.. py:module:: piel.models.physical.electronic.taper


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   piel.models.physical.electronic.taper.TaperParameters



Functions
~~~~~~~~~

.. autoapisummary::

   piel.models.physical.electronic.taper.taper



.. py:class:: TaperParameters



.. py:function:: taper(params: TaperParameters) -> hdl21.Module

   Implements a `hdl21` taper resistor class. We need to include the mapping ports as we expect our gdsfactory component to be with the instance of the model.
