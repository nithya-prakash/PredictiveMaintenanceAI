#!/usr/bin/env sh
# Regenerates the hash-pinned lockfiles in locks/ from the requirements files:
# one per image type (runtime = API, dashboard, dev) and CPU architecture
# (x86_64: CI and servers, aarch64: Apple-silicon Docker). Run after changing a
# pin in requirements*.txt, then rebuild the images and run the tests.
# Extra arguments are passed to `uv pip compile` (e.g. -c constraints.txt).
set -e
cd "$(dirname "$0")/.."
docker run --rm -v "$PWD:/src" -w /src python:3.11-slim sh -c '
  pip install -q uv
  for arch in x86_64 aarch64; do
    for kind in runtime dashboard dev; do
      case $kind in runtime) req=requirements.txt;; dashboard) req=requirements-dashboard.txt;; dev) req=requirements-dev.txt;; esac
      uv pip compile "$req" --python-platform "${arch}-manylinux_2_28" --python-version 3.11 \
        --generate-hashes --no-header -q -o "locks/${kind}-${arch}.txt" '"$*"'
    done
  done'
