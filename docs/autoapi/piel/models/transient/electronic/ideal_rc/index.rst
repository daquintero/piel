:py:mod:`piel.models.transient.electronic.ideal_rc`
===================================================

.. py:module:: piel.models.transient.electronic.ideal_rc


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.models.transient.electronic.ideal_rc.calculate_multistage_rc_performance



.. py:function:: calculate_multistage_rc_performance(multistage_configuration: Optional[piel.models.transient.electronic.types.RCMultiStageConfigurationType] = None, switching_frequency_Hz: Optional[float] = 100000.0)

   Calculates the total energy and power consumption for charging and discharging
   in a multistage RC configuration, as well as the transition energy and power consumption.

   :param multistage_configuration: A list of dictionaries containing the voltage and capacitance for each stage.
   :type multistage_configuration: Optional[RCMultiStageConfigurationType]
   :param switching_frequency_Hz: The switching frequency of the RC stages.
   :type switching_frequency_Hz: Optional[float]

   :returns:

             - Total charge and discharge energy.
             - Total charge and discharge power consumption.
             - Transition energy.
             - Transition power consumption.
   :rtype: A tuple containing


