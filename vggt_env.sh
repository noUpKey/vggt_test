# vggt_env.sh - sourced by setup_vggt_a40.sh and run_vggt_a40.sh (not run on its own).
# Everything this pipeline writes stays inside ~/shashank, and jobs only see GPU 5.

SH="${SH:-$HOME/shashank}"
RUS_ENV="${RUS_ENV:-$SH/envs/rus_shashank}"  # existing conda env (py3.12, torch 2.5.1+cu124, nvcc 12.4), used read-only
VENV="${VENV:-$SH/envs/vggt_venv}"           # thin venv layered on RUS_ENV; every new package goes here
VGGT_HOME="${VGGT_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"   # this folder
GPU_ID=5

case "$VGGT_HOME/" in
    "$SH"/*) ;;
    *) echo "Put this folder inside $SH (it is at $VGGT_HOME)" >&2; return 1 ;;
esac

export SH
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCH_CUDA_ARCH_LIST="8.6"            # A40
export XDG_CACHE_HOME="$SH/.cache"
export HF_HOME="$SH/.cache/huggingface"
export TORCH_HOME="$SH/.cache/torch"
export PIP_CACHE_DIR="$SH/.cache/pip"
export MPLCONFIGDIR="$SH/.cache/matplotlib"
export TORCH_EXTENSIONS_DIR="$SH/.cache/torch_extensions"
export CUDA_CACHE_PATH="$SH/.cache/nv"
export TRITON_CACHE_DIR="$SH/.cache/triton"
export TMPDIR="$SH/tmp"
export PYTHONNOUSERSITE=1
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$TORCH_EXTENSIONS_DIR" \
    "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TMPDIR"

activate_rus_env() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is not on PATH: log in with a normal shell (or source conda.sh) and try again" >&2
        return 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "$RUS_ENV"
}

activate_vggt_env() {
    activate_rus_env || return 1
    if [ ! -x "$VENV/bin/python" ]; then
        echo "No venv at $VENV: run setup_vggt_a40.sh first" >&2
        return 1
    fi
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
}

# check_gpu MIN_FREE_MIB: print what is running on GPU 5 and stop if there is not enough free memory.
check_gpu() {
    local need="${1:-16000}" line used total util
    line=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits) || return 1
    used=$(echo "$line" | cut -d, -f1 | tr -d ' ')
    total=$(echo "$line" | cut -d, -f2 | tr -d ' ')
    util=$(echo "$line" | cut -d, -f3 | tr -d ' ')
    echo "GPU $GPU_ID: ${used} of ${total} MiB in use, ${util}% busy"
    nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
        | sed 's/^/  already running: pid, memory = /'
    if [ $((total - used)) -lt "$need" ]; then
        echo "Only $((total - used)) MiB free on GPU $GPU_ID; this step needs about $need MiB. Try again later." >&2
        return 1
    fi
}
