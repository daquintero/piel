:py:mod:`piel.models.frequency.photonic`
========================================

.. py:module:: piel.models.frequency.photonic


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   coupler_simple/index.rst
   directional_coupler_length/index.rst
   directional_coupler_real/index.rst
   directional_coupler_simple/index.rst
   grating_coupler/index.rst
   mmi1x2/index.rst
   mmi2x2/index.rst
   straight_waveguide/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.models.frequency.photonic.coupler
   piel.models.frequency.photonic.directional_coupler_with_length
   piel.models.frequency.photonic.directional_coupler
   piel.models.frequency.photonic.grating_coupler_simple
   piel.models.frequency.photonic.mmi1x2_50_50
   piel.models.frequency.photonic.mmi2x2_50_50
   piel.models.frequency.photonic.ideal_active_waveguide
   piel.models.frequency.photonic.waveguide
   piel.models.frequency.photonic.simple_straight



.. py:function:: coupler(coupling=0.5)


.. py:function:: directional_coupler_with_length(length=1e-05, coupling=0.5, loss=0, neff=2.34, wl0=1.55e-06, ng=3.4, phase=0)


.. py:function:: directional_coupler(coupling=0.5)


.. py:function:: grating_coupler_simple(R=0.0, R_in=0.0, Tmax=1.0, bandwidth=6e-08, wl0=1.55e-06)


.. py:function:: mmi1x2_50_50()


.. py:function:: mmi2x2_50_50()


.. py:function:: ideal_active_waveguide(wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss=0.0, active_phase_rad=0.0)


.. py:function:: waveguide(wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss=0.0)


.. py:function:: simple_straight(length=10.0, width=0.5)
