**************************
Data Structures
**************************

Types
=====

``piel`` is built on top of the popular python package `pydantic <https://docs.pydantic.dev/latest/>`_.
All ``piel.types`` are defined as static ``pydantic.BaseClass`` subclasses. The goal is to enable purely functional
data containers and operations. This enables extensibility when using common data types between multiple tools or workflows.
By translating the data from multiple tools into standard types, we guarantee that the functionality developed on top
of this data type can be validated, verified and extensibly used at a higher-level without worrying of the lower-level implementation.


Note that there are both standard types and experimental types according to the functionality required:

.. toctree::
    :maxdepth: 2

    ../../autoapi/piel/types/index.rst


Experimental
-------------

There are some `experimental-specific` data types, where functionality integrating common devices into the standard ``piel`` data types is described. See the related examples too. TODO link.

.. toctree::
    :maxdepth: 2

    ../../autoapi/piel/experimental/types/index.rst


Models
======

`piel` provides a component model library to aid co-simulation. However, to extract a range of system performance parameters, simulations often tend to have to be done in different domains. This is why we need to have models that represent the same component, in different domains.

.. include:: domains.rst
.. include:: time_domain.rst
.. include:: spice_integration.rst

.. toctree::
    :maxdepth: 2
    ../../autoapi/piel/models/index.rst
