#! /usr/bin/env bash

set -euo pipefail

main() {
  PYTHONPATH=src pylint --disable=locally-disabled,fixme src
}

main
