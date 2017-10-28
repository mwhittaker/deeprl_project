#! /usr/bin/env bash

set -euo pipefail

main() {
  pip install -r requirements.txt
  pip install tensorflow

  sudo apt-get install mpich
  git clone https://github.com/openai/baselines.git
  cd baselines
  git checkout 4993286230ac92ead39a66005b7042b56b8598b0
  env MPICC=/usr/bin/mpicc pip install mpi4py
  pip install -e .
  cd ..
}

main
