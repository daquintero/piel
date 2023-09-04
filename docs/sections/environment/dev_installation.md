# Summary

This process sets up a Nix development environment,good for when developing examples.

# System requirements

- `curl`
- `git`
- `bash `

# Nix installation

First, [install nix](https://nixos.org/download). The approach below is not the *recommended* install method, as it creates a 'single' user installation, with the `/nix` cache owned by the invoking user, rather than shared between all users. But it doesn't work well on Fedora, so we'll live with it for now.

```
sh <(curl -L https://nixos.org/nix/install) --no-daemon
```

Provide your user password when prompted.

The following line should have been added to both your `~/.bash_profile` and `~/.bashrc`:

```
if [ -e /users/kcaisley/.nix-profile/etc/profile.d/nix.sh ]; then . /users/kcaisley/.nix-profile/etc/profile.d/nix.sh; fi # added by Nix installer
```

This ensures that `nix-shell` will be available in `$PATH` whether you're starting a shell in "login" mode or in "non-login" mode. More info [can be found here.](https://askubuntu.com/questions/121073/why-bash-profile-is-not-getting-sourced-when-opening-a-terminal)

After verifying this, to update the changes to `$PATH` either close and reopen your terminal, or run:

```
. ~/.nix-profile/etc/profile.d/nix.sh
```

# OpenLane2 installation

Before installing `piel`, let's first get Openlane2, as it will automatically also give us OpenROAD, Yosys, Magic, KLayout, and Verilator. The instructions below are [copied from here.](https://openlane2.readthedocs.io/en/latest/getting_started/nix_installation/installation_linux.html)

Cachix allows the reproducible Nix builds to be stored on a cloud server so you do not have to build OpenLane’s dependencies from scratch on every computer, which will take a long time.

First, you want to install Cachix by running the following in your terminal:

```
nix-env -f "<nixpkgs>" -iA cachix
```

Then set up the OpenLane binary cache as follows:

```
cachix use openlane
```

`cd` to a working directory of choice, and clone down `openlane2`:

```
git clone https://github.com/efabless/openlane2
```

### Nix Environment

Now move inside the folder:

```
cd openlane2
```

And build the environment of dependencies. Wait for it to fetch and cache the dependencies.

```
nix-shell
```

# Piel Installation

Next, `cd ..` back up one level, and clone `piel` itself, next to the `openlane2` directory:

```
git@github.com:daquintero/piel.git
```

### Nix Environment
We'll similarly use nix to grab all the compiled dependencies for `piel`, including:

- `ngspice`: 41 (latest)
- `gtkwave`: 3.3.117, from Aug 2023 (latest)
- `Xyce`: 7.6, from Nov 2022 (7.7 is latest)
- `verilog`: 12.0, from Jun 2023 (latest)

Move inside of the `piel` directory:

```
cd piel
```

And run:

```
nix-shell
```

### Python Environment

For the time being, PyPI and pip isn't easily compatible with Nix. See the [complexity here.](https://nixos.wiki/wiki/Python). `machnix` used to [solve this problem](https://github.com/DavHau/mach-nix), but it's unmaintained/deprecated [in favor of migrating to](https://github.com/nix-community/dream2nix) of `dream2nix`. So hopefully in a couple months, this whole section below will be part of the `shell.nix` process above.

Anyways, in the interim, we'll use the `requirements_dev.txt` file to fetch all the Python dependencies for `piel`.

While still inside of the nix-shell from before, check you're still using `python 3.10.9` from nix:

```
python --version
```

This is essential because we want all of our own `piel` venv to be on the same version of Python. Now create and activate this venv:

```
python -m venv .venv
```

```
source .venv/bin/activate
```

And fetch these dependencies via:

```
pip install -r requirements_dev.txt
```

The `.venv` folder should be created inside the top level of the `piel` directory. The `.gitignore` file will prevent it from being commited to the remote repo.

You're now done!

# Subsequent usage:

```
cd /path/to/piel
```

And implicitly load the `piel` `shell.nix` environment via:

```
nix-shell
```

This will automatically add the `piel` package managed with nix, but it will also add to path the packges managed the `pip` venv by the `openlane2` `nix` environment. This is accomplished via a the shell hook in `shell.nix`:

```nix
    shellHook = ''
    nix-shell ../openlane2/shell.nix
    if [ -e .venv/bin/activate ]; then source .venv/bin/activate; fi
    '';
```

# Vscode support for nix:

Useful for automatically running the top-level `shell.nix` file after setup.

https://matthewrhone.dev/nixos-vscode-environment

https://marketplace.visualstudio.com/items?itemName=arrterian.nix-env-selector