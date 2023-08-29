``nix`` Useful Commands
---------------------

.. list-table:: Useful ``nix`` commands
   :header-rows: 1

   * - Description
     - Command
   * - Build production shell with all available CPU cores
     - ``nix-build --cores 0``
   * - Build development shell with all available CPU cores
     - ``nix-shell --cores 0``
   * - Install a nix environment package eg. `mach-nix`
     - ``nix-env -if https://github.com/DavHau/mach-nix/tarball/3.5.0 -A mach-nix``
   * - List all available packages
     - ``nix-env -qaP --description``
   * - Search for a specific <package>
     - ``nix-env -qaP <package>``
