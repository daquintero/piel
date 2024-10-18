# file: flake.nix
{
  description = "Python application packaged using a standard virtual environment and additional EDA tools";

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

      # Define the Python version to use
      python = pkgs.python311Full;

      # List of Python dependencies
      pythonDeps = with python.pkgs; [
        setuptools
        flit-core
        hatchling
        scikit-build-core
        scikit-learn
        rectpack
        vlsir
        vlsirtools
        hdl21
        sky130-hdl21
        gdsfactory
        sky130
        thewalrus
        # Add other dependencies as needed
      ];

      # Create a Python environment with the specified packages
      pythonEnv = python.withPackages (ps: pythonDeps);

      # Define the Python application using buildPythonApplication
      app = pkgs.python311Full.buildPythonApplication {
        pname = "piel";
        version = "1.0.0"; # Replace with your actual version

        src = ./.;

        # Specify build inputs (dependencies)
        buildInputs = pythonDeps;

        # If using a setup.py or pyproject.toml, ensure it's properly configured
        # For example, if using setuptools:
        # setupBuildInputs = [ setuptools ];

        # If your project specifies entry points in setup.py or pyproject.toml,
        # they will be automatically handled. Otherwise, you can define scripts here.
        # Example:
        # scripts = [ "bin/piel" ];

        # Propagate build inputs to ensure dependencies are available to consumers
        propagatedBuildInputs = pythonDeps;

        # Optionally, define build flags or phases if needed
        # For example, to enable tests:
        # doCheck = true;
        # checkPhase = ''
        #   pytest tests
        # '';
      };

      # Override the application if additional customizations are needed
      overriddenApp = app.overrideAttrs (old: {
        name = "${old.pname}-overridden-${old.version}";
        nativeBuildInputs = old.nativeBuildInputs or [] ++ [
          pkgs.python3Packages.hatchling
          pkgs.python3Packages.scikit-build-core
          pkgs.python3Packages.scikit-learn
        ];
        propagatedBuildInputs = old.propagatedBuildInputs or [] ++ [
          pkgs.python3Packages.hatchling
          pkgs.python3Packages.scikit-build-core
          pkgs.python3Packages.scikit-learn
        ];
      });

      # Define a dependency environment if needed
      depEnv = app.dependencyEnv.override {
        app = overriddenApp;
      };

      # Include any legacy packages if necessary
      packagesLegacy = pkgs.legacyPackages.${system};
      packagesLegacy.piel = pkgs.legacyPackages.${system}.piel;

      # Define the development shell environment
      shellEnv = pkgs.mkShell {
        buildInputs = [
          app
          edaPkgs.ngspice
          edaPkgs.xschem
          edaPkgs.verilator
          edaPkgs.yosys
          openlane.openlane2
          pkgs.verilog
          pkgs.gtkwave
        ];
        shellHook = ''
          echo "Setting Up Piel-Nix Environment"
          export PATH=${app}/bin:$PATH
          export PYTHONPATH=${app}/lib/python${python.version}/site-packages:$PYTHONPATH
        '';
      };

    in
    {
      # Define the package outputs
      packages.${system} = {
        default = app;
        python = python;
      };

      # Define application entries
      apps.${system}.default = {
        type = "app";
        program = "${app}/bin/piel"; # Replace with your actual entry point
      };

      # Define the default shell environment
      shell.${system}.default = shellEnv;

      # Define the development shell environment
      devShells.${system}.default = shellEnv;
    };
}
