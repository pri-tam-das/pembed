#!/bin/bash
# Run the pembed test suite in parallel. Usage: [PYTHON=/path/to/python] [NPROC=16] ./run_all_tests.sh [pytest args]
set -e
PY=${PYTHON:-python}
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
PYTHONPATH=$PWD:$PWD/tests exec $PY -m pytest -q -n ${NPROC:-16} tests "$@"
