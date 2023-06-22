:py:mod:`piel.openlane.utils`
=============================

.. py:module:: piel.openlane.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.openlane.utils.find_design_run



.. py:function:: find_design_run(design_directory: str | pathlib.Path, run_name: str | None = None) -> pathlib.Path

   For a given `design_directory`, the `openlane` output can be found in the `runs` subdirectory.

   They get sorted based on a reverse `list.sort()` method.
