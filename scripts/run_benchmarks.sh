#!/usr/bin/env bash
# Capability suite for every model in the comparison, same tasks and limits.
set -u
cd "$(dirname "$0")/.."
BENCH="PYTHONPATH=. .venv/bin/python scripts/05_benchmark.py"
PYTHONPATH=. .venv/bin/python scripts/05_benchmark.py --name base --path models/Qwen2.5-1.5B-Instruct
PYTHONPATH=. .venv/bin/python scripts/05_benchmark.py --name textbook_mid --path out/export/textbook_mid
PYTHONPATH=. .venv/bin/python scripts/05_benchmark.py --name tap_top14 --path out/export/tap_top14
PYTHONPATH=. .venv/bin/python scripts/05_benchmark.py --name tap_uniform --path out/export/tap_uniform
PYTHONPATH=. .venv/bin/python scripts/05_benchmark.py --name heretic_mpoa --path out/heretic_mpoa/model
PYTHONPATH=. .venv/bin/python scripts/05_benchmark.py --name heretic_ara --path out/heretic_ara/model
