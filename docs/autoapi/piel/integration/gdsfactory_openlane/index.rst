:py:mod:`piel.integration.gdsfactory_openlane`
==============================================

.. py:module:: piel.integration.gdsfactory_openlane

.. autoapi-nested-parse::

   There are a number of ways to generate gdsfactory integration.

   It is worth noting that GDSFactory has already the following PDKs installed:
   * SKY130nm https://gdsfactory.github.io/skywater130/
   * GF180nm https://gdsfactory.github.io/gf180/



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.integration.gdsfactory_openlane.create_gdsfactory_component_from_openlane



.. py:function:: create_gdsfactory_component_from_openlane(design_name_v1: str | None = None, design_directory: piel.config.piel_path_types | None = None, run_name: str | None = None, v1: bool = True) -> gdsfactory.Component

   This function cretes a gdsfactory layout component that can be included in the network codesign of the device, or that can be used for interconnection codesign.

   It will look into the latest design run and extract the final OpenLane-generated GDS. You do not have to have run this with OpenLane2 as it just looks at the latest run.

   :param design_name_v1: Design name of the v1 design that can be found within `$OPENLANE_ROOT/"<latest>"/designs`.
   :type design_name_v1: str
   :param design_directory: Design directory PATH.
   :type design_directory: piel_path_types
   :param run_name: Name of the run to extract the GDS from. If None, it will look at the latest run.
   :type run_name: str
   :param v1: If True, it will import the design from the OpenLane v1 configuration.
   :type v1: bool

   :returns: GDSFactory component.
   :rtype: component(gf.Component)
