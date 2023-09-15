{ pkgs ? import <nixpkgs> {}
}:

pkgs.mkShell {
  buildInputs = [
    pkgs.stdenv.cc.cc.lib
    pkgs.jupyter
    pkgs.which
    pkgs.htop
    pkgs.zlib
    pkgs.pandoc
    pkgs.ngspice # 41 (latest)
    pkgs.gtkwave # 3.3.117, from Aug 2023 (latest)
    pkgs.xyce # 7.6, from Nov 2022 (7.7 is latest)
    pkgs.verilog # 12.0, from Jun 2023 (latest)
    pkgs.python3Packages.virtualenv # run virtualenv .
    pkgs.python3Packages.numpy
  ];

  nativeBuildInputs = [
    pkgs.autoPatchelfHook
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/ # fixes libstdc++ issues and libgl.so issues
    if [ -e ../../.venv/bin/activate ];
     then source ../../.venv/bin/activate;
    else
      pip install --upgrade pip;
      python -m venv ../../.venv;
      source ../../.venv/bin/activate;
      pip install -r ../../requirements_dev.txt;
      pip install -r ../../../openlane2/requirements_dev.txt;
      pip install -e ../../;
      source ../../.venv/bin/activate;
    fi
    nix-shell ../../../openlane2/shell.nix
  '';
  LOCALE_ARCHIVE="/usr/lib/locale/locale-archive";  # let's nix read the LOCALE, to silence warning messages
}
