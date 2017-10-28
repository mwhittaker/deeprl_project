#! /usr/bin/env bash

set -euo pipefail

underline() {
    local readonly s="$1"
    echo "$s"
    echo "$(echo $s | sed 's/./=/g')"
}

main() {
    for type in video ram cherry_picked; do
        for n in 1 2; do
            underline "Running $type on $n cores."
            time python src/capture_pong_play.py \
                --pong_type "$type" \
                --maxproc 1 \
                --maxsteps 1 \
                --savefile /tmp/temp.h5
            echo
        done
    done
}

main
