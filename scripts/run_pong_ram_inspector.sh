#! /usr/bin/env bash

set -euo pipefail

main() {
    time python src/pong_ram_inspector.py --compute_change_frequencies
}

main
