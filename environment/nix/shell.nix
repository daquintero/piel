{ pkgs ? import <nixpkgs> {}
}:

pkgs.mkShell {
  buildInputs = [
    pkgs.stdenv.cc.cc.lib
    pkgs.which
    pkgs.htop
    pkgs.zlib
    pkgs.pandoc
    pkgs.ngspice # 41 (latest)
    pkgs.gtkwave # 3.3.117, from Aug 2023 (latest)
    pkgs.xyce # 7.6, from Nov 2022 (7.7 is latest)
    pkgs.verilog # 12.0, from Jun 2023 (latest)
    pkgs.nodejs
    pkgs.micromamba
  ];

  nativeBuildInputs = [
    pkgs.autoPatchelfHook
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc
    ]}
    if [ -e ../../.venv/bin/activate ];
     then 
      # source ../../.venv/bin/activate;
      echo "Environment configured"; 
    else
      # pip install --upgrade pip;
      set -e
      eval "$(micromamba shell hook --shell=bash)"
      micromamba shell init --shell=bash --prefix=~/micromamba
      micromamba create --yes -q -n pielenv -c conda-forge python=3.10
      # micromamba install --yes --file ../../requirements_dev.txt -c conda-forge;
      micromamba activate pielenv --yes
      set +e
      export PATH="$PATH:$HOME/.local/bin/"
      micromamba run -n pielenv pip install -r ../../requirements_dev.txt --user --break-system-packages;
      micromamba run -n pielenv pip install -r ../../../openlane2/requirements_dev.txt --user --break-system-packages;
      micromamba run -n pielenv pip install -e ../../ --user --break-system-packages;
      # source ../../.venv/bin/activate;
    fi
    nix-shell ../../../openlane2/shell.nix
  '';
  LOCALE_ARCHIVE="/usr/lib/locale/locale-archive";  # let's nix read the LOCALE, to silence warning messages
}
