:py:mod:`piel.models.physical.electronic.types`
===============================================

.. py:module:: piel.models.physical.electronic.types


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   piel.models.physical.electronic.types.LNAMetricsType




.. py:class:: LNAMetricsType


   Bases: :py:obj:`piel.types.PielBaseModel`

   Low-noise amplifier metrics.

   .. py:attribute:: footprint_mm2
      :type: Optional[float]



   .. py:attribute:: bandwidth_Hz
      :type: Optional[MinimumMaximumType]



   .. py:attribute:: noise_figure
      :type: Optional[MinimumMaximumType]



   .. py:attribute:: power_consumption_mW
      :type: Optional[MinimumMaximumType]



   .. py:attribute:: power_gain_dB
      :type: Optional[MinimumMaximumType]



   .. py:attribute:: supply_voltage_V
      :type: Optional[float]



   .. py:attribute:: technology_nm
      :type: Optional[float]



   .. py:attribute:: technology_material
      :type: Optional[str]
