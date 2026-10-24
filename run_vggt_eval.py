#!/usr/bin/env python
"""VGGT -> 3D Gaussian Splatting on Submerged3D, scored like RUSplatting (command-line version of the
Colab notebook). Writes tables, figures and renders to <out_root>/<run tag>/.

    python run_vggt_eval.py                                   # all 4 scenes, 30k iterations
    python run_vggt_eval.py --iterations 7000                 # quick pass
    python run_vggt_eval.py --pixel_aspect none               # trust VGGT's own intrinsics (ablation)
    python run_vggt_eval.py --scenes Tokai --export_colmap    # one scene + COLMAP export
"""
import argparse
import json
import os
import shutil
import sys
import time

SH = os.path.expanduser(os.environ.get("SH", "~/shashank"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default=os.path.join(SH, "data", "Submerged3D"),
                   help="folder with <Scene>/input/*.jpg (default: %(default)s)")
    p.add_argument("--out_root", default=os.path.join(SH, "output_vggt"), help="default: %(default)s")
    p.add_argument("--vggt_repo", default=os.path.join(SH, "vggt_eval", "vggt"), help="default: %(default)s")
    p.add_argument("--scenes", default="Cormoran,Isro,Kwaj,Tokai")
    p.add_argument("--iterations", type=int, default=30_000)
    p.add_argument("--pixel_aspect", default="0.84375",
                   help="camera fy/fx used to un-squeeze frames for VGGT; 'none' trusts VGGT (default: %(default)s)")
    p.add_argument("--no_align", action="store_true", help="skip the aligned-test-pose scores")
    p.add_argument("--export_colmap", action="store_true",
                   help="also write VGGT cameras + points as a COLMAP text model per scene")
    p.add_argument("--force", action="store_true", help="retrain scenes that already have results")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    adv = p.add_argument_group("advanced")
    adv.add_argument("--max_init_points", type=int, default=1_000_000)
    adv.add_argument("--align_iters", type=int, default=400)
    adv.add_argument("--render_downscale", type=int, default=1,
                     help="1 = full 1280x720 (required for comparison with the papers)")
    adv.add_argument("--vggt_long_side", type=int, default=518)
    return p.parse_args(argv)


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else "", flush=True)


def markdown_table(df, floatfmt="{:.4f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def main(argv=None):
    a = parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (a.vggt_repo, here):
        if p not in sys.path:
            sys.path.insert(0, p)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch

    import vggt_nvs as vn

    scenes = tuple(s for s in a.scenes.split(",") if s)
    aspect = None if a.pixel_aspect.lower() in ("none", "vggt") else float(a.pixel_aspect)
    tag = f"it{a.iterations}_aspect{'vggt' if aspect is None else aspect}"
    if a.render_downscale != 1:
        tag += f"_ds{a.render_downscale}"
    if a.no_align:
        tag += "_noalign"
    cfg = vn.Config(data_root=a.data_root, out_root=os.path.join(a.out_root, tag), scenes=scenes,
                    iterations=a.iterations, pixel_aspect=aspect, align_test_poses=not a.no_align,
                    render_downscale=a.render_downscale, max_init_points=a.max_init_points,
                    align_iters=a.align_iters, vggt_long_side=a.vggt_long_side, seed=a.seed)
    if a.device.startswith("cuda"):
        if not torch.cuda.is_available():
            sys.exit("CUDA is not available (is CUDA_VISIBLE_DEVICES set to the right GPU?)")
        log(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {torch.cuda.get_device_name(0)}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    os.makedirs(cfg.out_root, exist_ok=True)
    log(f"results -> {cfg.out_root}")
    with open(os.path.join(cfg.out_root, "run_args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)

    log("data and split (every 8th sorted frame from index 0 is a test view)")
    for s in scenes:
        frames = vn.list_frames(os.path.join(a.data_root, s))
        tr, te = vn.split_indices(len(frames), cfg.test_every)
        warn = "" if len(frames) == 20 else f"   <-- expected 20 frames, found {len(frames)}"
        log(f"  {s:9s} {len(frames)} frames, {len(tr)} train / {len(te)} test: "
            f"{[vn.frame_name(frames[i]) for i in te]}{warn}")

    log("stage 1/3: VGGT")
    info = vn.run_vggt_stage(cfg, device=a.device, log=log)
    pd.DataFrame(info).T.to_csv(os.path.join(cfg.out_root, "vggt_diag.csv"))

    log("stage 2/3: 3DGS training and test-view scoring")
    lpips_model = vn.LPIPSVGG().to(a.device)
    rows = []
    for s in scenes:
        t0 = time.time()
        rows += vn.run_scene(cfg, s, lpips_model, device=a.device, log=log, force=a.force)
        log(f"  {s} finished in {(time.time() - t0) / 60:.1f} min")

    log("stage 3/3: tables and figures")
    per_view = pd.DataFrame(rows)
    per_scene = vn.per_scene_table(rows)
    comp = vn.comparison_table(rows)
    per_view.to_csv(os.path.join(cfg.out_root, "per_view.csv"), index=False)
    per_scene.to_csv(os.path.join(cfg.out_root, "per_scene.csv"), index=False)
    comp.to_csv(os.path.join(cfg.out_root, "comparison.csv"), index=False)
    vn.plot_comparison(rows, os.path.join(cfg.out_root, "comparison.png"))
    vn.plot_per_scene(rows, os.path.join(cfg.out_root, "per_scene.png"))
    plt.close("all")
    for s in scenes:
        vn.show_test_views(cfg, s, rows, os.path.join(cfg.out_root, s, "test_views.png"))
        plt.close("all")
    show = comp[["source", "method", "PSNR", "SSIM", "LPIPS", "dPSNR vs RUSplatting"]]
    with open(os.path.join(cfg.out_root, "results.md"), "w") as f:
        f.write(f"# VGGT -> 3DGS on Submerged3D ({tag})\n\n")
        f.write("Every-8th split (3 test / 17 train per scene). 'raw' = VGGT test pose, "
                "'aligned' = 6-DoF test-pose alignment with the scene frozen. LPIPS = VGG, RUSplatting/3DGS "
                "convention.\n\n## Per scene (mean of 3 test views)\n\n")
        f.write(markdown_table(per_scene) + "\n\n## 4-scene mean vs published Submerged3D numbers\n\n")
        f.write(markdown_table(show) + "\n")

    if a.export_colmap:
        for s in scenes:
            sparse = vn.export_scene_colmap(cfg, s)
            root = os.path.dirname(os.path.dirname(sparse))
            os.makedirs(os.path.join(root, "images"), exist_ok=True)
            for fr in vn.list_frames(os.path.join(a.data_root, s)):
                shutil.copy(fr, os.path.join(root, "images"))
            log(f"  COLMAP model for {s}: {root}")

    pd.set_option("display.width", 200)
    log("per scene (mean of the 3 test views):")
    print(per_scene.round(4).to_string(index=False), flush=True)
    log("4-scene mean vs published numbers:")
    print(show.round(4).to_string(index=False), flush=True)
    log(f"done. Tables: {cfg.out_root}/results.md, figures: comparison.png, per_scene.png, <scene>/test_views.png")
    return rows


if __name__ == "__main__":
    main()
