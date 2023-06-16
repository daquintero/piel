# Environment Setup Instructions

Install the docker environment provided by [IIC-OSIC-TOOLS](https://github.com/iic-jku/iic-osic-tools).

Start your docker environment accorignly running, for example:
```shell
cd iic-osic-tools
./start_vnc.sh
```

Find out which docker instance is running:
```shell
docker ps
```

You can start a bash terminal in the correct docker environment by running:
```shell
docker exec -it <yourdockercontainername> bash
```

You can explore the environment a little bit just between running `ls` and `cd` commands. If it has started from the default installation you should have started under the `foss/designs` folder. It is here that we will load our initial design for evaluation and interconnectivity.

Now, we can begin and follow the other tutorials.

## For Developers

Note that in the docker container, it is just an environment, it is not an independent file system. It appears like the only folder that can be edited from `iic-osic-tools` is the `designs` folder, maybe I will see how to disable this in another set of instructions. This means that, say, you want to install `piel` in the default docker environment you might have to run:

```shell
cd $HOME/eda/designs # The default $DESIGN environment
git clone https://github.com/daquintero/piel.git
```

