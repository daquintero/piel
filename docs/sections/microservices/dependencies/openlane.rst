``openlane``
============

v1/v2 Migration
---------------

Currently, OpenLane v1 is integrated into the project whilst Openlane v2
is still in development.

``OpenLane v2`` designs do not have to be stored in a particular
``$OPENLANE_ROOT/design`` directory, and are a lot more flexible in
terms of the design implementation and hardening. When performing
parametric analysis, copying folder and generating multiple
configurations may still be desired. However, because of the mess it can
become to maintain multiple functions throughout the migration, some
functionality is repeated for the sake of mantainability except when
explicitly functions can be shared. This is in the case of internal
functionality.

.. toctree::

    ../../../autoapi/piel/tools/openlane/index.rst
