
with (import ./inputs.nix);
pkgs.mkShell {
  buildInputs = [
    (import ./python.nix)
    mach-nix.mach-nix
  ];
}
