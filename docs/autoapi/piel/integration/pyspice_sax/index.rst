:py:mod:`piel.integration.pyspice_sax`
======================================

.. py:module:: piel.integration.pyspice_sax

.. autoapi-nested-parse::

   This function implements a SAX backend to compile SPICE based circuit from a GDSFactory-based netlist. We provide
   SPICE model functions that return a configured string of SPICE directives, that come from our component models. Each
   of our component models can be configured with a set of instances that may have parameter functions directly from the
   component geometry and layer information.
