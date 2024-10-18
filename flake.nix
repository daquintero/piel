{
  description = "Python application with standard virtual environment and pip";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nix-eda.url = "github:efabless/nix-eda";
    openlane2.url = "github:efabless/openlane2";
  };

  outputs = { self, nixpkgs, nix-eda, openlane2 }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      edaPkgs = import nix-eda { inherit pkgs; };
      openlane = import openlane2 { inherit pkgs; };

      # Use Python 3.11 from nixpkgs
      python = pkgs.python311;

      # Path to the pyproject.toml and requirements file
      pyproject = ./pyproject.toml;

      # Define a virtual environment that installs dependencies via pip
      pythonEnv = pkgs.python311.withPackages (ps: [
        ps.virtualenv
        ps.pip
        ps.setuptools
      ]);

      # Create a script to install dependencies in the virtual environment
      installDepsScript = pkgs.writeShellScriptBin "install-deps" ''
        set -e
        # Create virtual environment
        python -m venv venv
        source venv/bin/activate
        # Install dependencies from pyproject.toml using pip
        pip install --upgrade pip
        pip install .  # This will install the project in the virtual environment, resolving dependencies from pyproject.toml
      '';

    in
    {
      # Packages for different systems
      packages.${system} = {
        default = pkgs.buildEnv {
          name = "piel";
          paths = [
            pythonEnv
            installDepsScript
            edaPkgs.ngspice
            edaPkgs.xschem
            edaPkgs.verilator
            edaPkgs.yosys
            openlane.openlane2
            pkgs.verilog
            pkgs.gtkwave
          ];
        };

        python = python;
      };

      # Shell definition for development
      shell.${system}.default = pkgs.mkShell {
        buildInputs = [
          pythonEnv
          edaPkgs.ngspice
          edaPkgs.xschem
          edaPkgs.verilator
          edaPkgs.yosys
          openlane.openlane2
          pkgs.verilog
          pkgs.gtkwave
        ];

        # Run the install script to set up the virtual environment and install dependencies
        shellHook = ''
          echo "Setting up Python virtual environment with pip"
          ${installDepsScript}
          source venv/bin/activate
          export PATH=$PWD/venv/bin:$PATH
        '';
      };
    };
}
