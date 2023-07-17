:py:mod:`piel.config`
=====================

.. py:module:: piel.config

.. autoapi-nested-parse::

   We create a set of parameters that can be used throughout the project for optimisation.

   The numerical solver is normally delegated for as `numpy` but there are cases where a much faster solver is desired, and where different functioanlity is required. For example, `sax` uses `JAX` for its numerical solver. In this case, we will create a global numerical solver that we can use throughout the project, and that can be extended and solved accordingly for the particular project requirements.



Module Contents
---------------

.. py:data:: numerical_solver



.. py:data:: nso



.. py:data:: piel_path_types
