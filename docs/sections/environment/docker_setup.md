## Linux - Docker Setup Instructions

Install the docker environment provided by [IIC-OSIC-TOOLS](https://github.com/iic-jku/iic-osic-tools).

Start your docker environment accordingly running, for example:
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

Then go into your [localhost:8888](http://localhost:8888) to access Jupyter Lab directly from your Chrome notebook.
