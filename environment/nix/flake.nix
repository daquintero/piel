{
  description = "Application packaged using poetry2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    openlane.url = "github:efabless/openlane2";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix, openlane }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Access OpenLane packages
        openlanePackages = openlane.packages.${system};

        pielAppOverlay = final: prev: {
          pielApp = prev.pielApp.overrideAttrs (oldAttrs: {
            buildInputs = oldAttrs.buildInputs ++ (with final;
            [
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
                pkgs.vtk
                pkgs.tk
                pkgs.tcl
                pkgs.libX11
            ]);
          });
        };

        # Merge OpenLane packages with your pkgs
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            pielAppOverlay
            (self: super: openlanePackages)
          ];
        };

        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication defaultPoetryOverrides;

      in
      {
        packages = {
          pielApp = mkPoetryApplication {
            projectDir = builtins.path {
                path = ../../.;
            };
            python = pkgs.python311;
            preferWheels = true;
            overrides = defaultPoetryOverrides.extend
              (self: super: {
                pandoc = super.pandoc.overridePythonAttrs
                (
                  old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ];
                  }
                );
                rectpack = super.rectpack.overridePythonAttrs
                (
                  old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ];
                  }
                );
                sphinx-jsonschema = super.sphinx-jsonschema.overridePythonAttrs
                (
                  old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ];
                  }
                );
                vlsir = super.vlsir.overridePythonAttrs
                (
                  old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ];
                  }
                );
                vlsirtools = super.vlsirtools.overridePythonAttrs
                (
                  old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ];
                  }
                );
                hdl21 = super.hdl21.overridePythonAttrs
                (
                  old: {
                    buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ];
                  }
                );
            });
           };
          default = self.packages.${system}.pielApp;
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ self.packages.${system}.pielApp ];
        };
      });
}
