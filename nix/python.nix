
with (import ./inputs.nix);
mach-nix.mkPython {
  requirements = builtins.readFile ./requirements.txt;

  # providers = {
  #   # disallow wheels by default
  #   _default = "nixpkgs,sdist,wheel";
  # };
}
