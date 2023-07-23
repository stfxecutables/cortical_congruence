#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT_90
#SBATCH --profile=all
#SBATCH --job-name=adhd_select
#SBATCH --array=0-2
#SBATCH --output=adhd_select_%A_%a_%j.out
#SBATCH --time=00-24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80

SCRATCH="$(readlink -f "$SCRATCH")"
PROJECT="$SCRATCH/cortical_congruence"
PYTHON="$PROJECT/.venv/bin/python"

PY_SCRIPTS="$PROJECT/scripts"
PY_SCRIPT="$(readlink -f "$PY_SCRIPTS/adhd200_select.py")"

"$PYTHON" "$PY_SCRIPT"