*************
Environment
*************

.. include:: tools_environment.rst

``piel CLI`` (Recommended - In Active Development)
========================================================================

``piel`` is a command line interface (CLI) that is designed to be a simple and easy to use tool for managing the ``piel`` toolchain. It is necessary to first install ``piel`` in a given python environment and then the CLI tool should automatically be active in that envrionment. You then just need to run ``$ piel`` in the terminal to see the available commands.

.. include:: piel_cli/common_useful_commands.rst

``nix`` Configuration (In Passive Development)
========================================================================

`OpenLane 2 via nix <https://openlane2.readthedocs.io/en/latest/getting_started/index.html#nix-recommended>`__ have recently released another way to package their `python`-driven ``Openlane 2`` digital chip layout flow. We have previously had issues reproducibly building the `docker` configuration, and because most users are likely to use these tools for developing their chips rather than distributing software, `nix <https://nixos.org/>`__ might be well suited for these applications.

.. include:: nix/development_installation.rst
.. include:: nix/custom_nix_installation.rst
.. include:: nix/relevant_nix_commands.rst


``apptainer`` Configuration (In Passive Development)
====================================================

``apptainer`` is a good open-source container management system that aims to be optimised for high performance computation. We want to have a distributed container environment where all the open-source ``piel`` toolchain is pre-installed and ready for custom design. What this distribution aims to provide is both an easy installation script for Ubuntu environments which are common in open-source development and a specific environment configuration that resolves the particular supported versions of the toolchains.

.. include:: apptainer/apptainer_install.rst
.. include:: apptainer/relevant_apptainer_commands.rst


``docker`` Configuration (In Passive Development)
=====================================================

.. include:: docker/docker_setup.rst
.. include:: docker/docker_environment_configuration.rst
.. include:: docker/relevant_docker_commands.rst
.. include:: docker/developer_docker_configuration.rst


``mamba`` Recipie Configuration (In Passive Development)
=================================================

.. include:: mamba/relevant_mamba_commands.rst


.. include:: installation.rst
.. include:: project_structure.rst
.. include:: design_files_interaction.rst
.. include:: relevant_python_commands.rst
