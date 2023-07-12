:py:mod:`piel.integration.pyspice_gdsfactory`
=============================================

.. py:module:: piel.integration.pyspice_gdsfactory


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   circuit/index.rst
   instance_model/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.integration.pyspice_gdsfactory.gdsfactory_netlist_to_pyspice



.. py:function:: gdsfactory_netlist_to_pyspice(gdsfactory_netlist: dict, return_raw_spice: bool = False)

   This function converts a GDSFactory electrical netlist into a standard PySpice configuration. It follows the same
   principle as the `sax` circuit composition. It returns a PySpice circuit and can return it in raw_spice form if
   necessary.

   Each GDSFactory netlist has a set of instances, each with a corresponding model, and each instance with a given
   set of geometrical settings that can be applied to each particular model. We know the type of SPICE model from
   the instance model we provides.

   We know that the gdsfactory has a set of instances, and we can map unique models via sax through our own composition circuit.
