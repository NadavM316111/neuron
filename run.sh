#!/bin/bash
# Run any experiment with the library and worlds importable.
#
# The experiments use flat imports (from world import ..., from stability
# import ...), so the directories holding those modules go on the path
# rather than every file being rewritten.
#
#   ./run.sh experiments/grid/grow.py
#   ./run.sh experiments/inheritance/merge.py
#
# Results are written to the directory you run from, so run from the repo
# root and they land in results/ if the script writes there.

ROOT="$(cd "$(dirname "$0")" && pwd)/neuron"
export PYTHONPATH="$ROOT/core:$ROOT/worlds:$PYTHONPATH"
cd "$ROOT/results" && python -u "$ROOT/${1#neuron/}" "${@:2}"
