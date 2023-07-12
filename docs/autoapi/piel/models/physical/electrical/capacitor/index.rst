:py:mod:`piel.models.physical.electrical.capacitor`
===================================================

.. py:module:: piel.models.physical.electrical.capacitor

.. autoapi-nested-parse::

   SPICE capacitor model:

   .. code-block::
       CXXXXXXX N+ N- VALUE <IC=INCOND>

   Where the parameters are:

   .. code-block::
       N+ = the positive terminal
       N- = the negative terminal
       VALUE = capacitance in farads
       <IC=INCOND> = starting voltage in a simulation
