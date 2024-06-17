``piel`` Useful Commands
------------------------------------------

All commands are echoed to the terminal before they are executed. This is to help you understand what is happening under the hood.

.. list-table:: Useful ``piel`` commands
   :header-rows: 1

   * - Description
     - Command
    * - Prints nix environment command to run:
     - ``piel environment activate-nix-shell``
   * - Builds the piel project documentation. Assumes correct documentation environment requirements.
     - ``piel develop build-docs``
   * - Gets the install directory for the piel project.
     - ``piel get-install-directory``
   * - Installs ``nix`` on your system per the openlane instructions.
     - ``piel environment install-nix``
   * - List all the piel CLI functions relating to environment configuration.
     - ``piel environment``
   * - List all the piel CLI functions relating to configuring a piel project development environment.
     - ``piel develop``
