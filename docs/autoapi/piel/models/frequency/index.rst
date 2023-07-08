:py:mod:`piel.models.frequency`
===============================

.. py:module:: piel.models.frequency


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   electrical/index.rst
   electro_optic/index.rst
   electronic/index.rst
   opto_electronic/index.rst
   photonic/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   all/index.rst
   defaults/index.rst
   utils/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.models.frequency.compose_custom_model_library_from_defaults
   piel.models.frequency.get_all_models
   piel.models.frequency.get_default_models



.. py:function:: compose_custom_model_library_from_defaults(custom_models: dict) -> dict

   Compose the default models with the custom models.

   :param custom_models: Custom models dictionary.
   :type custom_models: dict

   :returns: Composed models dictionary.
   :rtype: dict


.. py:function:: get_all_models(custom_library: dict | None = None) -> dict

   Returns the default models dictionary.

   :param custom_library: Custom defaults dictionary.
   :type custom_library: dict

   :returns: Default models dictionary.
   :rtype: dict


.. py:function:: get_default_models(custom_defaults: dict | None = None) -> dict

   Returns the default models dictionary.

   :param custom_defaults: Custom defaults dictionary.
   :type custom_defaults: dict

   :returns: Default models dictionary.
   :rtype: dict
