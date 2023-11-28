# Instructions taken from https://nixos.wiki/wiki/Nix_Installation_Guide
cd ~
cd piel/environment/nix/install
sudo apt-get install -y curl
sudo install -d -m755 -o $(id -u) -g $(id -g) /nix
. $HOME/.nix-profile/etc/profile.d/nix.sh
nix-env -f "<nixpkgs>" -iA cachix
cachix use openlane
git clone https://github.com/efabless/openlane2.git ../../../../openlane2
