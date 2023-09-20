cd ~
cd piel/environment/nix/install
sudo apt-get install -y curl
sh <(curl -L https://nixos.org/nix/install) --no-daemon --yes
. $HOME/.nix-profile/etc/profile.d/nix.sh
nix-env -f "<nixpkgs>" -iA cachix
cachix use openlane
git clone https://github.com/efabless/openlane.git -o ../../../../
nix-shell ..
