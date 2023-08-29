Environment
===========

.. include:: tools_environment.rst

``nix`` Configuration (In Active Development)
==============================================

`OpenLane 2 via nix <https://openlane2.readthedocs.io/en/latest/getting_started/index.html#nix-recommended>`__ have recently released another way to package their `python`-driven ``Openlane 2`` digital chip layout flow. We have previously had issues reproducibly building the `docker` configuration, and because most users are likely to use these tools for developing their chips rather than distributing software, `nix <https://nixos.org/>`__ might be well suited for these applications.

.. include:: nix/nix_install.rst
.. include:: nix/custom_nix_installation.rst


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
