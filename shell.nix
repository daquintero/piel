# Copyright 2023 Efabless Corporation
# Modified by Dario Quintero 2023 for the piel project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
{
    pkgs ? import ./nix/pkgs.nix,
    piel-app ? import ./. {}
}:

let
  mach-nix = import (builtins.fetchGit {
    url = "https://github.com/DavHau/mach-nix/";
    ref = "refs/tags/3.5.0";
  }) {};
in
mach-nix.mkPythonShell {
  requirements = builtins.readFile ./requirements_dev.txt;
}

with pkgs; mkShell {
  name = "piel";

  propagatedBuildInputs = [
    piel-app

    # Conveniences
    git
    neovim
    delta
    zsh

    # Docs + Testing
    enchant
    jupyter
    graphviz
  ];
}
