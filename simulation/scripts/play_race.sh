#!/usr/bin/env bash
# Play a checkpoint for AS-Tello-Race-v0.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AS_ROOT="${ACTION_SPACE_ROOT:-$HOME/action-space}"
ISAAC_ENV="${ISAAC_ENV:-$HOME/isaac/env_isaaclab}"

# shellcheck disable=SC1091
source "$ISAAC_ENV/bin/activate"
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export OMNI_KIT_ACCEPT_EULA=YES
export TELLO_USD_PATH="${TELLO_USD_PATH:-$HOME/isaac/assets/tello.usd}"

pip install -e "$ROOT/simulation/source/drone_race_kit" -q
python -c "import drone_race_kit"

cd "$AS_ROOT/simulation_infra"
exec python -u scripts/sb3/play.py --task AS-Tello-Race-v0 "$@"
