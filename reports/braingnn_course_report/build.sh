#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TECTONIC="$PROJECT_ROOT/.conda/latex/bin/tectonic"
CACHE_DIR="$PROJECT_ROOT/.tmp/tectonic-cache"

mkdir -p "$SCRIPT_DIR/build" "$CACHE_DIR"
cd "$SCRIPT_DIR"

XDG_CACHE_HOME="$CACHE_DIR" "$TECTONIC" main.tex --outdir build
