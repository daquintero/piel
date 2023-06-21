# Extension of IIC-OSIC-TOOLS

Existing Toolset:

## Jupyter Notebook

From the IIC-OSIC-TOOLS directory run:

```shell
./start_jupyter.sh
```

## Jupyter Lab

Now, there is no direct `jupyterlab` running script as of June 2023:

Equivalently, what you can do is simple:

```shell
cd icc-osic-tools
source ./start_x.sh

# In the docker environment terminal
python -m jupyterlab
```

And you will get a Jupyter Lab firefox instance.

However, to view our `Jupytext` notebooks you need to run:

```shell
pip install jupytext
```

## Required Toolset

To install `jupyter lab` and related plugins.

Run in the docker environment:

```shell
cd /foss/designs
pip install virtualenv
python -m venv pielenv
source pielenv/bin/activate
pip install jupyterlab
```
