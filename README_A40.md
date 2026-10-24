# VGGT → 3DGS on Submerged3D, A40 version

Same pipeline as the Colab notebook (`vggt_nvs.py` is identical), as scripts for the university A40.

What these scripts do on the server:
- **Files:** they only write inside `~/shashank`, including the pip, Hugging Face, torch and matplotlib caches and `TMPDIR`.
- **GPU:** they only use GPU 5 (`CUDA_VISIBLE_DEVICES=5`). Before starting, they show what is already running on GPU 5 and stop if there isn't enough free memory.
- **Your RUSplatting environment:** `rus_shashank` is only read. New packages go into a separate small venv, `~/shashank/envs/vggt_venv`, that sits on top of it.
- **Data:** they read the original frames from `~/shashank/data/Submerged3D/<Scene>/input/` and never modify them.

## 1. Copy the folder to the server

From your laptop:

```bash
unzip vggt_eval_a40.zip                      # gives a vggt_eval/ folder
scp -r vggt_eval <you>@<a40-server>:~/shashank/
```

## 2. Set up (once, 10–15 minutes)

```bash
bash ~/shashank/vggt_eval/setup_vggt_a40.sh
```

This takes 10–15 minutes and does the following:
- builds the venv;
- clones VGGT at commit `a288dd0`;
- compiles gsplat 1.5.3 for the A40 (sm_86) with the nvcc 12.4 in `rus_shashank`, using at most 8 cores;
- renders a test image on GPU 5;
- downloads the VGGT-1B weights (5 GB) and the LPIPS weights.

You can safely run it again; anything that is already done is skipped.

## 3. Run (inside tmux)

```bash
tmux new -s vggt
bash ~/shashank/vggt_eval/run_vggt_a40.sh                       # 4 scenes, 30k iterations
# other options:
bash ~/shashank/vggt_eval/run_vggt_a40.sh --iterations 7000     # quick pass
bash ~/shashank/vggt_eval/run_vggt_a40.sh --pixel_aspect none   # ablation: VGGT's own intrinsics
bash ~/shashank/vggt_eval/run_vggt_a40.sh --export_colmap       # also write VGGT cameras as COLMAP models
```

To leave the run going, detach with `Ctrl-b d`; to come back, use `tmux attach -t vggt`. If the run is interrupted, start it again with the same command: scenes that already have results are skipped. The log is saved as `~/shashank/output_vggt/run_<date>.log`.

## 4. Results

Results go to `~/shashank/output_vggt/it30000_aspect0.84375/` (the folder name changes with the settings). It contains:
- `results.md`: per-scene table, plus the 4-scene mean next to the RUSplatting paper, GbtW and your RUSplatting reproduction.
- `per_view.csv`, `per_scene.csv` and `comparison.csv`.
- Figures: `comparison.png`, `per_scene.png` and `<Scene>/test_views.png`.
- Renders: `<Scene>/test/{gt,raw,aligned}/*.png`.
- `vggt_diag.csv`: the fy/fx VGGT predicts, its field of view and the camera path length.

To copy everything back to your laptop:

```bash
scp -r <you>@<a40-server>:~/shashank/output_vggt/it30000_aspect0.84375 .
```

## If the settings differ on your server

`vggt_env.sh` assumes the base environment is `~/shashank/envs/rus_shashank`. If yours is somewhere else, override it when you call the scripts, for example `RUS_ENV=~/shashank/envs/other bash setup_vggt_a40.sh`. The data folder is set with `--data_root`.
