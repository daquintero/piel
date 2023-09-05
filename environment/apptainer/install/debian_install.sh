# Taken from https://apptainer.org/docs/admin/main/installation.html#install-debian-packages
sudo apt update
sudo apt install -y wget
cd /tmp
wget https://github.com/apptainer/apptainer/releases/download/v1.2.2/apptainer_1.2.2_amd64.deb
sudo apt install -y ./apptainer_1.2.2_amd64.deb
wget https://github.com/apptainer/apptainer/releases/download/v1.2.2/apptainer-suid_1.2.2_amd64.deb
sudo dpkg -i ./apptainer-suid_1.2.2_amd64.deb
