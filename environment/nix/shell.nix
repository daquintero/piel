{ pkgs ? import <nixpkgs> {}
}:

pkgs.mkShell {
  buildInputs = [
    pkgs.which
    pkgs.htop
    pkgs.zlib
    pkgs.pandoc
    pkgs.ngspice # 41 (latest)
    pkgs.gtkwave # 3.3.117, from Aug 2023 (latest)
    pkgs.xyce # 7.6, from Nov 2022 (7.7 is latest)
    pkgs.verilog # 12.0, from Jun 2023 (latest)
  ];


  shellHook = ''
    nix-shell ../../../openlane2/shell.nix
    if [ -e .venv/bin/activate ]; then source .venv/bin/activate; fi
  '';

  MY_ENVIRONMENT_VARIABLE = "test";
}
