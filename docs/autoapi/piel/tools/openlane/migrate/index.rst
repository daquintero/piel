:py:mod:`piel.tools.openlane.migrate`
=====================================

.. py:module:: piel.tools.openlane.migrate

.. autoapi-nested-parse::

   These functions provide easy tools for easily migrating between OpenLane v1 and v2 based designs.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.openlane.migrate.get_design_from_openlane_migration



.. py:function:: get_design_from_openlane_migration(v1: bool = True, design_name_v1: str | None = None, design_directory: str | pathlib.Path | None = None, root_directory_v1: str | pathlib.Path | None = None) -> (str, pathlib.Path)

   This function provides the integration mechanism for easily migrating the interconnection with other toolsets from an OpenLane v1 design to an OpenLane v2 design.

   This function checks if the inputs are to be treated as v1 inputs. If so, and a `design_name` is provided then it will set the `design_directory` to the corresponding `design_name` directory in the corresponding `root_directory_v1 / designs`. If no `root_directory` is provided then it returns `$OPENLANE_ROOT/"<latest>"/. If a `design_directory` is provided then this will always take precedence even with a `v1` flag.

   :param v1: If True, it will migrate from v1 to v2.
   :type v1: bool
   :param design_name_v1: Design name of the v1 design that can be found within `$OPENLANE_ROOT/"<latest>"/designs`.
   :type design_name_v1: str
   :param design_directory: Design directory PATH. Optional path for v2-based designs.
   :type design_directory: str
   :param root_directory_v1: Root directory of OpenLane v1. If set to None it will return `$OPENLANE_ROOT/"<latest>"`
   :type root_directory_v1: str

   :returns: None
