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

If you want to enter the corresponding `nix-shell` environment, you can run the following command which will print the updated command you just need to copy paste into your terminal to activate the `nix-shell` environment.

.. code-block:: bash

    $ piel activate

It will print:

.. code-block::

    # Please run this in your shell:
    cd ~/<path_to_piel>
    nix shell . github:efabless/nix-eda#{ngspice,xschem,verilator,yosys} github:efabless/openlane2 nixpkgs#verilog nixpkgs#gtkwave

This is because, I believe, for security reasons it is very difficult to automatically enter a nix shell directly from python or a subprocess.

`OpenLane 2 via nix <https://openlane2.readthedocs.io/en/latest/getting_started/index.html#nix-recommended>`__ have recently released another way to package their `python`-driven ``Openlane 2`` digital chip layout flow. We have previously had issues reproducibly building the `docker` configuration, and because most users are likely to use these tools for developing their chips rather than distributing software, `nix <https://nixos.org/>`__ might be well suited for these applications.

.. include:: nix/development_installation.rst
.. include:: nix/custom_nix_installation.rst
.. include:: nix/relevant_nix_commands.rst
