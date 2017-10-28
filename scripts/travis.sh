#! /usr/bin/env bash

set -euo pipefail

main() {
  bash scripts/lint.sh
  bash scripts/run_all_environments.sh
  bash scripts/run_pong_ram_inspector.sh
  bash scripts/run_atari_ppo.sh
}

main
