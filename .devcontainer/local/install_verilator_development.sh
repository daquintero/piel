#!/bin/bash

# Update the system
sudo apt update

# Verilator installation script
# Required
sudo apt-get install git help2man perl python3 make
sudo apt-get install g++  # Alternatively, clang
sudo apt-get install libgz  # Non-Ubuntu (ignore if gives error)
sudo apt-get install libfl2  # Ubuntu only (ignore if gives error)
sudo apt-get install libfl-dev  # Ubuntu only (ignore if gives error)
sudo apt-get install zlibc zlib1g zlib1g-dev  # Ubuntu only (ignore if gives error)

# Performance
sudo apt-get install ccache  # If present at build, needed for run
sudo apt-get install libgoogle-perftools-dev numactl
sudo apt-get install perl-doc

sudo apt-get install git autoconf flex bison
sudo apt-get install gdb graphviz cmake clang clang-format-14 gprof lcov
sudo apt-get install libclang-dev yapf3
sudo pip3 install clang sphinx sphinx_rtd_theme sphinxcontrib-spelling breathe
cpan install Pod::Perldoc
cpan install Parallel::Forker

git clone https://github.com/verilator/verilator   # Only first time
## Note the URL above is not a page you can see with a browser; it's for git only

# Local Install
cd verilator
git pull        # Make sure we're up-to-date
git tag         # See what versions exist
#git checkout master      # Use development branch (e.g. recent bug fix)
git checkout stable      # Use most recent release
#git checkout v{version}  # Switch to specified release version
autoconf        # Create ./configure script

export VERILATOR_ROOT=`pwd`   # if your shell is bash
#setenv VERILATOR_ROOT `pwd`   # if your shell is csh
./configure
# Running will use files from $VERILATOR_ROOT, so no install needed

make -j$(nproc)

# Install Verilator
sudo make install

# Verify installation
verilator --version

# Clean up
cd ..
rm -rf verilator

echo "Verilator installation complete."
# TODO rest
