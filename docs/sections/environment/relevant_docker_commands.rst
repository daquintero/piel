Docker Useful Commands
======================

You might want to interact a lot with the filesystem when running OSIC
EDA tools, so here is a set of useful Docker commands that might make
the experience easier.

Useful Commands Table
---------------------

+--------------------------------------------+-----------------------------------------+
| Description                                | Command                                 |
+============================================+=========================================+
| Build a docker image in CD                 | ``docker build .``                      |
+--------------------------------------------+-----------------------------------------+
| Build a docker image in to a local tag     | ``docker build . -t user/app`` --load   |
+--------------------------------------------+-----------------------------------------+
| List available docker image                | ``docker images``                       |
+--------------------------------------------+-----------------------------------------+
| Remove  docker images                      | ``docker rmi <your-image-id>``          |
+--------------------------------------------+-----------------------------------------+
| List running docker containers             | ``docker ps``                           |
+--------------------------------------------+-----------------------------------------+
| Start bash terminal on running             | ``docker exec -it <containername> bash``|
+--------------------------------------------+-----------------------------------------+
| Stop docker container                      | ``docker stop <containerid>``           |
+--------------------------------------------+-----------------------------------------+
| IIC-OSIC-TOOLS sudo shell control          | ``./start_shell.sh``                    |
+--------------------------------------------+-----------------------------------------+
