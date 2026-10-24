#!/usr/bin/env bash
# One-off setup on the A40 server, about 10-15 minutes (mostly compiling gsplat). Safe to re-run.
#   bash ~/shashank/vggt_eval/setup_vggt_a40.sh
# Creates a venv on top of rus_shashank (which is not modified), clones VGGT at a fixed commit,
# compiles gsplat 1.5.3 for the A40, checks it on GPU 5 and downloads the VGGT-1B and LPIPS weights.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=vggt_env.sh
source "$HERE/vggt_env.sh"
VGGT_COMMIT="a288dd0f14786c93483e45524328726ab7b1b4ce"   # facebookresearch/vggt, 18 May 2026
GSPLAT_VERSION="1.5.3"
shopt -s nullglob

echo "== 1/5 base environment $RUS_ENV (read-only)"
activate_rus_env
python -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| CUDA', torch.version.cuda)"

echo "== 2/5 venv $VENV"
if [ ! -x "$VENV/bin/python" ]; then
    python -m venv --system-site-packages "$VENV" 2>/dev/null \
        || { rm -rf "$VENV"; python -m venv --system-site-packages --without-pip "$VENV"; }
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q einops safetensors huggingface_hub ninja jaxtyping rich
python -c "import pandas, matplotlib, scipy" 2>/dev/null || python -m pip install -q pandas matplotlib scipy
# own build tools in the venv, so the gsplat build never depends on the base env's setuptools
python -m pip install -q --ignore-installed "setuptools>=69,<81" wheel
python -c "import torch, torchvision; print('venv sees torch', torch.__version__, 'and torchvision', torchvision.__version__)"

echo "== 3/5 VGGT at ${VGGT_COMMIT:0:7}"
if [ ! -d "$VGGT_HOME/vggt/.git" ]; then
    git clone -q https://github.com/facebookresearch/vggt.git "$VGGT_HOME/vggt"
fi
git -C "$VGGT_HOME/vggt" checkout -q "$VGGT_COMMIT" 2>/dev/null \
    || { git -C "$VGGT_HOME/vggt" fetch -q origin && git -C "$VGGT_HOME/vggt" checkout -q "$VGGT_COMMIT"; }

echo "== 4/5 gsplat $GSPLAT_VERSION compiled for the A40 (sm_86)"
if python -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('gsplat') and u.find_spec('gsplat.csrc') else 1)"; then
    echo "gsplat with compiled CUDA kernels is already installed"
else
    wheels=("$VGGT_HOME"/wheels/gsplat-"$GSPLAT_VERSION"-*.whl)
    if [ ${#wheels[@]} -eq 0 ]; then
        want=$(python -c "import torch; print(torch.version.cuda.split('.')[0])")
        nvcc_on_path=$(command -v nvcc || true)
        cuda_home=""
        for c in "${CUDA_HOME:-}" "${CONDA_PREFIX:-}" "${nvcc_on_path%/bin/nvcc}"; do
            [ -n "$c" ] && [ -x "$c/bin/nvcc" ] || continue
            v=$("$c/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
            if [ "$v" = "$want" ]; then cuda_home="$c"; break; fi
        done
        if [ -z "$cuda_home" ]; then
            echo "No nvcc for CUDA $want found (checked CUDA_HOME, $RUS_ENV and PATH)" >&2
            exit 1
        fi
        jobs=$(nproc)
        if [ "$jobs" -gt 8 ]; then jobs=8; fi      # shared machine: leave cores for others
        echo "compiling with $cuda_home/bin/nvcc, $jobs parallel jobs ..."
        mkdir -p "$VGGT_HOME/wheels"
        # the venv's python explicitly: $cuda_home/bin also holds the base env's python
        CUDA_HOME="$cuda_home" MAX_JOBS="$jobs" \
            "$VENV/bin/python" -m pip wheel --no-build-isolation --no-deps --no-binary gsplat \
            "gsplat==$GSPLAT_VERSION" -w "$VGGT_HOME/wheels"
        wheels=("$VGGT_HOME"/wheels/gsplat-"$GSPLAT_VERSION"-*.whl)
    fi
    "$VENV/bin/python" -m pip install -q --no-deps --force-reinstall "${wheels[0]}"
fi

echo "== 5/5 GPU check and weights"
check_gpu 4000
python - "$VGGT_HOME/vggt" <<'PY'
import sys
import torch
sys.path.insert(0, sys.argv[1])
import vggt.models.vggt  # noqa: F401  (import check only)
from gsplat import rasterization
assert torch.cuda.is_available(), "no GPU visible"
n, d = 200, "cuda"
img, _, _ = rasterization(
    torch.randn(n, 3, device=d) * 0.5 + torch.tensor([0.0, 0.0, 4.0], device=d),
    torch.nn.functional.normalize(torch.randn(n, 4, device=d), dim=-1),
    torch.full((n, 3), 0.05, device=d), torch.full((n,), 0.8, device=d), torch.rand(n, 3, device=d),
    torch.eye(4, device=d)[None], torch.tensor([[[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]]], device=d),
    320, 240)
print("gsplat renders on", torch.cuda.get_device_name(0), tuple(img.shape))
PY
python -c "from huggingface_hub import snapshot_download as s; print('VGGT-1B:', s('facebook/VGGT-1B', allow_patterns=['config.json', 'model.safetensors']))"
python - <<'PY'
import torch, torchvision
torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
torch.hub.load_state_dict_from_url(
    "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth",
    map_location="cpu", progress=False)
print("LPIPS (VGG16) weights cached")
PY
echo
echo "Setup done. Start a run inside tmux:"
echo "  tmux new -s vggt"
echo "  bash $HERE/run_vggt_a40.sh"
