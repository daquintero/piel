**************************
Tools & Structure
**************************

``piel`` aims to provide an integrated workflow to co-design photonics and electronics. It does not aim to replace the individual functionality of each design tool, but rather provide a glue to easily connect them all together and extract the system performance.


This package provides interconnection functions to easily co-design
microelectronics through the functionality of multiple electronic and photonic design tools.

.. figure:: ../../_static/img/piel_microservice_structure.png
   :alt: `piel` microservices structure.


Project Structure
==================

A general overview of the ``piel`` python package structure:

.. raw::

    piel/
        cli/ # command-line scripts
        experimental/ # selected set of functions useful when interacting with relevant equipment
        flows/ # High-level user-specific functionality to automate common design tasks
        integration/ # Deterministic explicit functionality between toolsets
        materials/ # Self-contained or referenced material models
        models/ # Specific instances, or target application usages, operate on other directories
        tools/ # Relevant extended functionality onto supported tools
        visual/ # Utilities to plot and visualise relationships



.. toctree::
    :caption: Contents:

    dependencies/index
    integration/index
