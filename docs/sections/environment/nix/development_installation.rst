``nix`` Development Installation
--------------------------------------

This process sets up a ``nix`` development environment, good for when
developing examples. Make sure to follow the ``nix`` installation
instructions for your platform.

**The Fast Lane**

Assuming you already have ``piel`` installed in a local environment, you can simply run the commands to perform the installation.

.. code:: bash

    piel environment # To see all commands
    piel environment install-nix # To install nix
    piel environment install-openlane # To install openlane

To enter the nix environment, run:

.. code:: bash

    piel environment activate-piel-nix
    # piel environment activate-openlane-nix # if you want to enter the openlane one instead


System requirements
^^^^^^^^^^^^^^^^^^^^^^

Before starting, make sure you system has:

-  ``curl``
-  ``git``
-  ``bash``

``nix`` installation
^^^^^^^^^^^^^^^^^^^^^^

**The Fast Lane**

Assuming you already have ``piel`` installed in a local environment, you can simply run:

.. code:: bash

    piel environment install-nix


**The Detailed Lane**

First, `install nix <https://nixos.org/download>`__. The approach below
is not the *recommended* install method, as it creates a ‘single-user’
installation, with the ``/nix`` cache owned by the invoking user, rather
than shared between all users. The recommended ‘multi-user’ method,
however, doesn’t work well on systems with SELinux, i.e. Fedora, so
we’ll live with this alternative for now.

.. code:: bash

   sh <(curl -L https://nixos.org/nix/install) --no-daemon

Provide your user password when prompted.

The following line should have been added to both your
``~/.bash_profile`` and ``~/.bashrc``:

.. code:: bash

   if [ -e /users/<youruser>/.nix-profile/etc/profile.d/nix.sh ]; then . /users/<youruser>/.nix-profile/etc/profile.d/nix.sh; fi

This ensures that ``nix-shell`` will be available in ``$PATH`` whether
you’re starting a shell in “login” mode or in “non-login” mode. More
info `can be found
here. <https://askubuntu.com/questions/121073/why-bash-profile-is-not-getting-sourced-when-opening-a-terminal>`__

After verifying this, to update the changes to ``$PATH`` either close
and reopen your terminal, or run:

.. code:: bash

   . ~/.nix-profile/etc/profile.d/nix.sh

OpenLane2 installation
^^^^^^^^^^^^^^^^^^^^^^

**The Fast Lane**

Assuming you already have ``piel`` installed in a local environment, you can simply run:

.. code:: bash

    piel environment install-openlane


**The Detailed Lane**

Before installing ``piel``, let’s first get ``OpenLane2``, as it will
automatically also give us ``OpenROAD``, ``Yosys``, ``Magic``,
``KLayout``, and ``Verilator``. The instructions below are `copied from
here. <https://openlane2.readthedocs.io/en/latest/getting_started/nix_installation/installation_linux.html>`__

``Cachix`` allows the reproducible Nix builds to be stored on a cloud
server so you do not have to build OpenLane’s dependencies from scratch
on every computer, which will take a long time.

First, you want to install Cachix by running the following in your
terminal:

.. code:: bash

   nix-env -f "<nixpkgs>" -iA cachix

Then set up the OpenLane binary cache as follows:

.. code:: bash

   cachix use openlane

``cd`` to a working directory of choice, and clone down ``openlane2``:

.. code:: bash

   git clone https://github.com/efabless/openlane2

OpenLane Nix environment
'''''''''''''''''''''''''

**The Fast Lane**

Assuming you already have ``piel`` installed in a local environment, you can simply run:

.. code:: bash

    piel environment activate-openlane-nix




**The Detailed Lane - (Depreciated) Pre-Flakes Migration **


Now move inside the folder:

.. code:: bash

   cd openlane2

And build the environment of dependencies. Wait for it to fetch and
cache the dependencies.

.. code:: bash

   nix-shell


Piel Nix environment
''''''''''''''''''''

**The Fast Lane**

Assuming you already have ``piel`` installed in a local environment and have followed the previous installation process, you can simply run:

.. code:: bash

    piel environment activate-piel-nix

**The Detailed Lane**

We are now using ``nix-flakes`` to manage the nix environment.
This is an experimental nix feature, but far more powerful than the previous ``nix-shell`` approach.
To learn more about ``nix-flakes``, see `here <https://nixos.wiki/wiki/Flakes>`__.

To run our ``nix`` flakes environment run the following:

.. code::

    cd environment/nix
    nix develop --extra-experimental-features nix-command --extra-experimental-features flakes

This will take some time as it is both installing the openlane2 nix dependencies and the piel ones,
and building them into a specific environment.
The total installation size is approximately 4 Gb.
All the python packages that are dependencies of pip are installed from the wheels in PyPi from the versions defined by the ``poetry.lock`` file.

In my computer, running this command for the first time took about 20 minutes. Eventually we will distribute this in a binary.

**The Detailed Lane - (Depreciated) Pre-Flakes Migration **

We’ll similarly use nix to grab all the compiled dependencies for
``piel``, including:

-  ``ngspice``: 41 (latest)
-  ``gtkwave``: 3.3.117, from Aug 2023 (latest)
-  ``Xyce``: 7.6, from Nov 2022 (7.7 is latest)
-  ``verilog``: 12.0, from Jun 2023 (latest)

Do do this, simply move inside of the ``piel`` directory:

.. code:: bash

   cd piel/environment/nix

And run the command below, which implicitly reads in the local
``shell.nix`` file:

.. code:: bash

   nix-shell

Piel Python environment
'''''''''''''''''''''''

**The Fast Lane**

.. code:: bash

    piel environment # TODO


**The Detailed Lane - Depreciated**

For the time being, PyPI and pip isn’t easily compatible with Nix. See
the `complexity here. <https://nixos.wiki/wiki/Python>`__. ``machnix``
used to `solve this problem <https://github.com/DavHau/mach-nix>`__, but
it’s unmaintained/deprecated `in favor of the migration
to <https://github.com/nix-community/dream2nix>`__ ``dream2nix``. So
hopefully in a couple months, this whole section below will be rolled
into ``shell.nix`` file mentioned above.

Anyways, in the interim, we’ll use the ``requirements_dev.txt`` file to
fetch all the Python dependencies for ``piel``.

While still inside of the nix-shell, check you’re using
``python 3.10.9``:

.. code:: bash

   python --version

This is essential because we want all of our ``piel`` Python virtual
environment to be on the same version as Openlane. Create and activate
this ``venv`` on the top level of the ``.piel`` directory:

.. code:: bash

   python -m venv ~/.piel/.venv

.. code:: bash

   source ~/.piel/.venv/bin/activate

And fetch the dependencies via:

.. code:: bash

   pip install -e .[develop]

The ``.venv`` folder should be created inside the top level of the
``.piel`` in your home directory.

You’re now done!

Subsequent usage - Depreciated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   cd /path/to/piel

And implicitly load the ``piel`` ``shell.nix`` environment via:

.. code:: bash

   nix-shell

This will automatically add to ``$PATH`` the packages , but it will also
add to path the packages managed the ``pip`` venv by the OpenLane2
``nix`` environment. This is accomplished via a the shell hook in
``shell.nix``:

.. code:: nix

      shellHook = ''
        export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc
        ]}
          # Reinstalling the pacakges should guarantee a reproducible build every time
          set -e
          echo "Running: micromamba shell hook --shell=bash"
          eval "$(micromamba shell hook --shell=bash)"
          echo "Running: micromamba shell init --shell=bash --prefix=~/micromamba"
          micromamba shell init --shell=bash --prefix=~/micromamba
          echo "Running: micromamba create --yes -q -n pielenv -c conda-forge python=3.10"
          micromamba create --yes -q -n pielenv -c conda-forge python=3.10
          echo "Running: micromamba activate pielenv --yes"
          micromamba activate pielenv --yes
          set +e
          export PATH="$PATH:$HOME/.local/bin/"
          echo "Running: micromamba run -n pielenv pip install -r $HOME/.piel/openlane2/requirements_dev.txt --user --break-system-packages;"
          micromamba run -n pielenv pip install -r $HOME/.piel/openlane2/requirements_dev.txt --user --break-system-packages;
          echo "Running: micromamba run -n pielenv pip install ../../[develop] --user --break-system-packages;"
          micromamba run -n pielenv pip install -e "../../[develop]" --user --break-system-packages;
          source $HOME/.piel/.venv/bin/activate;
        fi
        nix-shell ~/.piel/openlane2/shell.nix
      '';
      LOCALE_ARCHIVE="/usr/lib/locale/locale-archive";  # let's nix read the LOCALE, to silence warning messages
    }

VSCode support for nix
^^^^^^^^^^^^^^^^^^^^^^

There is a `useful
plugin <https://marketplace.visualstudio.com/items?itemName=arrterian.nix-env-selector>`__
for automatically running the top-level ``shell.nix`` file after setup.
More info can be found `in this
blog. <https://matthewrhone.dev/nixos-vscode-environment>`__
