``nix`` Installation
--------------------------

The ``piel-nix`` configuration files aim to install and manage all the relevant `piel-integrated` tools as described accordingly.

This script is derived from the `openlane 2 installation instructions <https://openlane2.readthedocs.io/en/latest/getting_started/nix_installation/installation_win.html>`__


On Linux you can run, in summary:

.. code-block::

    sudo apt-get install -y curl
    sh <(curl -L https://nixos.org/nix/install) --no-daemon --yes
    # Restart the terminal or run ``. /home/<yourusername>/.nix-profile/etc/profile.d/nix.sh``
    nix-env -f "<nixpkgs>" -iA cachix
    cachix use openlane


Now, we install the ``piel`` specific configuration and you can follow the instructions in the developer section for now.

If this does not take less than a few seconds to build, then make sure you have the ``cachix`` ``openlane`` installation set up.
