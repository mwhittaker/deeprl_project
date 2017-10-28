#! /usr/bin/env bash

set -euo pipefail

main() {
  python src/run_atari_ppo.py --max_timesteps=1
}

main
