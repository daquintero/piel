:py:mod:`piel.gdsfactory`
=========================

.. py:module:: piel.gdsfactory

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

   piel.gdsfactory.create_gdsfactory_component_from_openlane



.. py:function:: create_gdsfactory_component_from_openlane(design_directory: str | pathlib.Path, run_name: str | None = None) -> gdsfactory.Component

   This function cretes a gdsfactory layout component that can be included in the network codesign of the device, or that can be used for interconnection codesign.

   It will look into the latest design run and extract the final OpenLane-generated GDS. You do not have to have run this with OpenLane2 as it just looks at the latest run.

   :param design_directory: Design directory PATH.
   :type design_directory: str
   :param run_name: Name of the run to extract the GDS from. If None, it will look at the latest run.
   :type run_name: str

   :returns: GDSFactory component.
   :rtype: component(gf.Component)
