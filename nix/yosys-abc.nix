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
  pkgs ? import ./pkgs.nix
}:

with pkgs; fetchFromGitHub {
  owner = "YosysHQ";
  repo = "abc";
  rev = "bb64142b07794ee685494564471e67365a093710";
  sha256 = "sha256-Qkk61Lh84ervtehWskSB9GKh+JPB7mI1IuG32OSZMdg=";
}