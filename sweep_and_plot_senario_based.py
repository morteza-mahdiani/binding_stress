#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sweep tasks × k for Qwen2.5-VL on the Shapes dataset, log results, and plot accuracy.

Requires:
- icl_episodes.py  (load_meta, make_episode)
- qwen_client.py   (qwen_infer)
- Your dataset CSV with 'file' column pointing to 'shapes_ds/images/...'

Usage (defaults should work):
  python sweep_and_plot.py \
    --meta_csv shapes_ds/metadata.csv \
    --episodes 200 \
    --ks 0 1 2 4 8 \
    --tasks color_of_shape shape_of_color shape_only color_only \
    --outdir results
"""

import argparse, os, json, csv, re, time, random
import numpy as np
from pathlib import Path
from typing import List
from PIL import Image
from typing import Dict, Any
from collections import defaultdict

import matplotlib.pyplot as plt  # no seaborn; 1 chart per fig; use default colors

from icl_episodes_senario_based import load_meta, make_episode
from qwen_client import qwen_infer, build_prompt_and_image, get_tokenizer, get_model, get_processor

from binding_probe import run_binding_probe

VALID_COLORS = {"red","green","blue","yellow"}
VALID_SHAPES = {"circle","square","triangle","star"}

def normalize(s: str) -> str:
    return "".join(ch.lower() for ch in s.strip() if ch.isalnum())

def postprocess(raw_text: str, task: str) -> str:
    txt = normalize(raw_text)
    # Tasks where the gold label is a COLOR
    if task in ("color_of_shape", "color_only"):
        for c in VALID_COLORS:
            if c in txt:
                return c

    # Tasks where the gold label is a SHAPE
    if task in ("shape_of_color", "shape_only"):
        for sh in VALID_SHAPES:
            if sh in txt:
                return sh
    # fallback: return the cleaned token (won’t match gold but logs are useful)
    return txt


def _extract_query_file_from_meta_or_msgs(qmeta: Dict[str, Any], messages: list) -> str | None:
    # Prefer what icl_episodes.make_episode returns
    if qmeta and "file" in qmeta:
        return qmeta["file"]
    # Fallback: scan last user message for an image path
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", [])
            # content may be a list of dicts with {"type":"image","path":...} or {"image": "..."}
            for part in content:
                if isinstance(part, dict):
                    if "image" in part:
                        return part["image"]
                    if "path" in part:
                        return part["path"]
                    if "url" in part:
                        return part["url"]
    return None

def run_eval_core(messages: list, gold: str, task: str, qmeta: Dict[str, Any] | None = None):
    """
    1) Calls qwen_infer to get a raw answer
    2) Postprocesses to a one-word canonical token
    3) Compares to gold (case-insensitive, normalized)
    4) Returns (ok, gold, pred, file)
    """
    raw = qwen_infer(messages, max_new_tokens=4, temperature=0.0, do_sample=False)
    # print(raw)
    pred = postprocess(raw, task)
    # print(f"Task: {task} | Gold: {gold} | Pred: {pred}")
    ok = normalize(pred) == normalize(gold)
    file = _extract_query_file_from_meta_or_msgs(qmeta or {}, messages)
    return ok, gold, pred, file

def run_eval_once(
    samples,
    k: int,
    task: str,
    outdir: Path,
    episode_idx: int,
    probe: bool,
    icl_mode: str,
):
    # NEW: make_episode now returns shots as well
    messages, gold, qmeta, shots_meta = make_episode(
        samples, k=k, task=task, icl_mode=icl_mode
    )

    ok, pred, file = None, None, None

    # Regular inference (existing):
    ok, gold, pred, file = run_eval_core(messages, gold, task, qmeta)

    # --- Binding probe (NEW) ---
    bind = None
    if probe:
        prompt, pil_img = build_prompt_and_image(messages)
        # pick target words (color_word, shape_word) from the gold and meta
        # For color_of_shape: color_word = gold, shape_word = qmeta["shape"]
        # For shape_of_color: shape_word = gold, color_word = qmeta["color"]
        if task == "color_of_shape":
            color_word = gold
            shape_word = qmeta["shape"]
        elif task == "shape_of_color":
            shape_word = gold
            color_word = qmeta["color"]
        else:
            # 'only' tasks: fall back to generic targets (won't be perfect but still runs)
            color_word = qmeta.get("color", "red")
            shape_word = qmeta.get("shape", "circle")

        bind = run_binding_probe(
            model=get_model(),
            processor=get_processor(),
            tokenizer=get_tokenizer(),
            prompt=prompt,
            pil_image=pil_img,
            color_word=color_word,
            shape_word=shape_word,
            return_attn=False,
            saliency_frac=0.10,  # top 10% pixels for causal ablation
        )

        # Save heatmaps as PNGs
        imH, imW = pil_img.size[1], pil_img.size[0]
        def save_heat(name, arr):
            if arr is None: return None
            import cv2
            up = cv2.resize(arr, (pil_img.size[0], pil_img.size[1]), interpolation=cv2.INTER_CUBIC)
            up = (255 * up).astype(np.uint8)
            up = Image.fromarray(up)
            path = outdir / f"ep{episode_idx:04d}_{task}_k{k}_{name}.png"
            up.save(path); return str(path)

        color_png = save_heat("colorheat", bind.color_heatmap)
        shape_png = save_heat("shapeheat", bind.shape_heatmap)

        # Append per-episode binding metrics to CSV
        bcsv = outdir / "binding_metrics.csv"
        new_file = not bcsv.exists()
        with open(bcsv, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow([
                    "episode","task","k","icl_mode",
                    "ok","gold","pred",
                    "query_color","query_shape",
                    "color_logit","shape_logit",
                    "drop_color_after_colorMask","drop_color_after_shapeMask",
                    "drop_shape_after_colorMask","drop_shape_after_shapeMask",
                    "color_heatmap_png","shape_heatmap_png","query_file",
                ])
            w.writerow([
                episode_idx, task, k, icl_mode,
                int(ok), gold, pred,
                qmeta.get("color"), qmeta.get("shape"),
                bind.color_logit, bind.shape_logit,
                bind.color_drop_after_color_mask, bind.color_drop_after_shape_mask,
                bind.shape_drop_after_color_mask, bind.shape_drop_after_shape_mask,
                color_png, shape_png, file,
            ])

    return ok, gold, pred, file, qmeta, shots_meta

from collections import defaultdict

def run_sweep(
    samples,
    tasks: List[str],
    ks: List[int],
    episodes: int,
    outdir: Path,
    probe: bool,
    icl_mode: str,
):
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []                 # rows for CSV
    acc_map: dict[str, list] = defaultdict(list)  # task -> [(k, acc)]

    for task in tasks:
        for k in ks:
            n = 0
            accs: list[int] = []
            run_name = f"{task}_k{k}_{icl_mode}"
            log_path = outdir / f"log_{task}_k{k}_{icl_mode}.jsonl"

            with open(log_path, "w") as wlog:
                for ep in range(episodes):
                    n += 1
                    ok, gold, pred, file, qmeta, shots_meta = run_eval_once(
                        samples, k, task, outdir, ep, probe, icl_mode
                    )
                    accs.append(int(ok))

                    log_row = {
                        "episode": ep,
                        "task": task,
                        "k": k,
                        "icl_mode": icl_mode,
                        "ok": bool(ok),
                        "gold": gold,
                        "pred": pred,
                        "file": file,
                        "query": {
                            "file": qmeta.get("file"),
                            "color": qmeta.get("color"),
                            "shape": qmeta.get("shape"),
                            "condition": qmeta.get("condition"),
                            "split": qmeta.get("split"),
                        },
                        "shots": [
                            {
                                "file": s.get("file"),
                                "color": s.get("color"),
                                "shape": s.get("shape"),
                                "condition": s.get("condition"),
                                "split": s.get("split"),
                            }
                            for s in shots_meta
                        ],
                    }
                    wlog.write(json.dumps(log_row) + "\n")

            if n == 0:
                # No episodes actually ran for this (task, k) – skip
                print(f"[WARN] No episodes for {run_name}, skipping.")
                continue

            accuracy = float(np.mean(accs))
            summary.append({
                "task": task,
                "k": k,
                "icl_mode": icl_mode,
                "episodes": n,
                "accuracy": round(accuracy, 4),
            })
            acc_map[task].append((k, accuracy))
            print(f"[{run_name}] accuracy={accuracy:.3f} episodes={n} -> {log_path}")

    # --- write summary CSV ---
    summary_csv = os.path.join(outdir, "summary_accuracy.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["task", "k", "icl_mode", "episodes", "accuracy"]
        )
        w.writeheader()
        w.writerows(summary)
    print("Wrote:", summary_csv)

    # --- plotting ---

    # tasks that actually have data
    tasks_with_data = [t for t in tasks if acc_map.get(t)]

    if not tasks_with_data:
        print("[WARN] No accuracy data to plot; skipping plots.")
        return summary_csv

    # 1) Combined plot
    plt.figure(figsize=(7, 5))
    for task in tasks_with_data:
        ks_sorted, accs_sorted = zip(*sorted(acc_map[task], key=lambda x: x[0]))
        plt.plot(ks_sorted, accs_sorted, marker="o", label=task)
    plt.xlabel("k (number of few-shot demonstrations)")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs k (icl_mode={icl_mode})")
    plt.xticks(ks)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    out_combined = os.path.join(outdir, f"acc_vs_k_combined_{icl_mode}.png")
    plt.savefig(out_combined, bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved:", out_combined)

    # 2) Per-task plots
    for task in tasks_with_data:
        plt.figure(figsize=(6, 4))
        ks_sorted, accs_sorted = zip(*sorted(acc_map[task], key=lambda x: x[0]))
        plt.plot(ks_sorted, accs_sorted, marker="o")
        plt.xlabel("k")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs k — {task} (icl_mode={icl_mode})")
        plt.xticks(ks)
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle="--", alpha=0.4)
        out_task = os.path.join(outdir, f"acc_vs_k_{task}_{icl_mode}.png")
        plt.savefig(out_task, bbox_inches="tight", dpi=150)
        plt.close()
        print("Saved:", out_task)

    return summary_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--ks", nargs="+", type=int, default=[0,1,2,4,8])
    ap.add_argument("--tasks", nargs="+", default=["color_of_shape","shape_of_color","shape_only","color_only"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--probe_binding", action="store_true", help="collect heatmaps/causal ablations per episode")
    ap.add_argument(
        "--icl_mode",
        choices=["random", "distinct", "binding_stress"],
        default="random",
        help="How to construct BIND in-context shots.",
    )
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    # -------------------------------
    # Make randomness reproducible
    # -------------------------------
    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            print(f"Set torch and torch.cuda seeds to {args.seed}")
    except ImportError:
        pass
    # -------------------------------

    t0 = time.time()
    samples = load_meta(args.meta_csv)
    run_sweep(
        samples,
        args.tasks,
        args.ks,
        args.episodes,
        Path(args.outdir),
        args.probe_binding,
        args.icl_mode,
    )
    print(f"Done in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
# python sweep_and_plot_senario_based.py --meta_csv shapes_ds/metadata.csv --episodes 20  --ks 0 2  --tasks color_of_shape shape_of_color --outdir results_distinct  --probe_binding --icl_mode distinct  --seed 0

