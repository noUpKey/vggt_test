#!/usr/bin/env bash
# Full evaluation on GPU 5. Run it inside tmux so it survives a dropped SSH connection:
#   tmux new -s vggt
#   bash ~/shashank/vggt_eval/run_vggt_a40.sh                     # all 4 scenes, 30k iterations
#   bash ~/shashank/vggt_eval/run_vggt_a40.sh --iterations 7000   # quick pass
#   bash ~/shashank/vggt_eval/run_vggt_a40.sh --pixel_aspect none # ablation: VGGT's own intrinsics
# Detach with Ctrl-b d, come back with: tmux attach -t vggt
# Scenes that already have results are skipped, so re-running after an interruption resumes.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=vggt_env.sh
source "$HERE/vggt_env.sh"
activate_vggt_env
check_gpu 16000
mkdir -p "$SH/output_vggt"
LOG="$SH/output_vggt/run_$(date +%Y%m%d_%H%M%S).log"
echo "log: $LOG"
python -u "$HERE/run_vggt_eval.py" --vggt_repo "$VGGT_HOME/vggt" "$@" 2>&1 | tee "$LOG"
