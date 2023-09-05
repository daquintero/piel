sudo apt-get install -y curl
sh <(curl -L https://nixos.org/nix/install) --no-daemon --yes
nix-env -f "<nixpkgs>" -iA cachix
