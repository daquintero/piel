:py:mod:`piel.tools.openlane.utils`
===================================

.. py:module:: piel.tools.openlane.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.tools.openlane.utils.find_design_run



.. py:function:: find_design_run(design_directory: piel.config.piel_path_types, run_name: str | None = None) -> pathlib.Path

   For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory.

   They get sorted based on a reverse `list.sort()` method.

   # TODO docs
