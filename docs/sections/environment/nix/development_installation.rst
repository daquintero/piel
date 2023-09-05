``nix`` Development Installation
--------------------------------------

This process sets up a ``nix`` development environment, good for when
developing examples. Make sure to follow the ``nix`` installation
instructions for your platform. TODO ADD LINK. TODO ADD LINK TO THE
UBUNTU INSTALL SCRIPT WE PROVIDE.

System requirements
^^^^^^^^^^^^^^^^^^^^^^

Before starting, make sure you system has:

-  ``curl``
-  ``git``
-  ``bash``

``nix`` installation
^^^^^^^^^^^^^^^^^^^^^^

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

   if [ -e /users/kcaisley/.nix-profile/etc/profile.d/nix.sh ]; then . /users/kcaisley/.nix-profile/etc/profile.d/nix.sh; fi

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
''''''''''''''''''''''''''''''''''''

Now move inside the folder:

.. code:: bash

   cd openlane2

And build the environment of dependencies. Wait for it to fetch and
cache the dependencies.

.. code:: bash

   nix-shell

Piel installation
^^^^^^^^^^^^^^^^^^^^^^

Next, ``cd ..`` back up one level, and clone ``piel`` itself, next to
the ``openlane2`` directory:

.. code:: bash

   git@github.com:daquintero/piel.git

Piel Nix environment
''''''''''''''''''''

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
this ``venv``:

.. code:: bash

   python -m venv .venv

.. code:: bash

   source .venv/bin/activate

And fetch the dependencies via:

.. code:: bash

   pip install -r requirements_dev.txt

The ``.venv`` folder should be created inside the top level of the
``piel`` directory. The ``.gitignore`` file will prevent it from being
committed to the remote repo.

You’re now done!

Subsequent usage
^^^^^^^^^^^^^^^^^^^^^^

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
       nix-shell ../openlane2/shell.nix
       if [ -e .venv/bin/activate ]; then source .venv/bin/activate; fi
       '';

VSCode support for nix
^^^^^^^^^^^^^^^^^^^^^^

There is a `useful
plugin <https://marketplace.visualstudio.com/items?itemName=arrterian.nix-env-selector>`__
for automatically running the top-level ``shell.nix`` file after setup.
More info can be found `in this
blog. <https://matthewrhone.dev/nixos-vscode-environment>`__
