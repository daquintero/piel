*************
Environment
*************

.. include:: tools_environment.rst

``apptainer`` Configuration (In Active Development)
====================================================

``apptainer`` is a good open-source container management system that aims to be optimised for high performance computation. We want to have a distributed container environment where all the open-source ``piel`` toolchain is pre-installed and ready for custom design. What this distribution aims to provide is both an easy installation script for Ubuntu environments which are common in open-source development and a specific environment configuration that resolves the particular supported versions of the toolchains.

.. include:: apptainer/apptainer_install.rst


``docker`` Configuration (In Passive Development)
=====================================================

.. include:: docker/docker_setup.rst
.. include:: docker/docker_environment_configuration.rst
.. include:: docker/relevant_docker_commands.rst
.. include:: docker/developer_docker_configuration.rst


.. include:: installation.rst
.. include:: project_structure.rst
.. include:: design_files_interaction.rst
.. include:: relevant_python_commands.rst
