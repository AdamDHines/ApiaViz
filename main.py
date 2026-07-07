#!/usr/bin/env python3
"""ApiaViz entry point: train the vision backbone or evaluate a downstream task.

    pixi run train      # self-supervised backbone training
    pixi run flowers    # flower identification eval (chromatic KC code vs CLAHE)
    pixi run nav        # navigation eval (retino_kc vs CLAHE)

See README.md for the full task list.
"""

from __future__ import annotations

import argparse

from apiaviz.src.logger import model_logger
from apiaviz.eval import EvalVision, NavEval

def apiaviz_eval(args, logger, output_folder):
    if args.eval_dataset == "nav":
        NavEval(args, logger, output_folder).run()
    else:
        EvalVision(args, logger, output_folder).eval()


def parse_args():
    parser = argparse.ArgumentParser(description="ApiaViz — insect-inspired vision backbone")

    # --- Evaluation (shared) ---
    parser.add_argument("-d", "--eval_dataset", default="nav", choices=["17flowers", "nav"],
                        help="downstream evaluation task")
    parser.add_argument("--code_dim", type=int, default=4000, help="Kenyon-cell code dimensionality")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ablate", default="none",
                        choices=["none", "hex", "adapt", "opponency", "dog"],
                        help="remove one VisionBackbone stage to measure its contribution to the task "
                             "(hex sampling / light adaptation / spectral opponency / DoG center-surround)")

    # --- Evaluation: flowers (object identification) ---
    parser.add_argument("--per_class", type=int, default=40, help="images sampled per flower class")
    parser.add_argument("--ks", type=int, nargs="*", default=[1, 8], help="scan lengths (fixations) to report")
    parser.add_argument("--patch_size", type=int, default=75, help="flower object size (px) pasted on the canvas")
    parser.add_argument("--canvas_size", type=int, default=135, help="neutral canvas size (px) the object is centred on")
    parser.add_argument("--cv_folds", type=int, default=5, help="stratified CV folds for flower accuracy CIs")
    parser.add_argument("--cv_repeats", type=int, default=1, help="repeat CV with reshuffled folds for tighter CIs")
    parser.add_argument("--depression", type=float, default=0.5, help="anti-Hebbian MBON depression for the flower readout")
    parser.add_argument("--mbon_sweep", action="store_true", help="also report an MBON depression/graded robustness sweep")
    parser.add_argument("--task", default="reward_gonogo", choices=["reward_gonogo", "classify"],
                        help="flower task: reward-gated go/no-go foraging (default) or 17-way novelty classification")
    parser.add_argument("--n_rewarded", type=int, default=5, help="number of rewarded flower classes for the go/no-go task")
    parser.add_argument("--reward_lr", type=float, default=1.0, help="reward-MBON learning rate (approach-synapse gain)")
    parser.add_argument("--corruption", default="none", choices=["none", "luminance", "noise"],
                        help="per-fixation test-time corruption; 'luminance' is isoluminant (spares opponent colour)")
    parser.add_argument("--severities", type=float, nargs="*", default=[],
                        help="corruption severities to sweep (0 = clean is always included)")
    parser.add_argument("--ablation", action="store_true",
                        help="channel-ablation panel: full vs achromatic-only vs opponent-only vs chroma-ablated")

    # --- Evaluation: nav (route following) ---
    parser.add_argument("--ant", type=int, default=0, help="ant index for navigation routes (<=0 = all ants, the default)")
    parser.add_argument("--routes", type=int, nargs="*", default=[],
                        help="route indices to navigate; empty (the default) or any value <=0 = all available routes "
                             "for the ant. Selections absent from the data are skipped with a warning.")
    parser.add_argument("--segments", type=int, default=80, help="MBON population size (route segments)")
    parser.add_argument("--landmark_fraction", type=float, default=0.5, help="fraction of world triangles that are colour landmarks")
    parser.add_argument("--chroma", type=float, default=60.0, help="landmark opponent-colour strength")
    parser.add_argument("--freenav", action="store_true",
                        help="open-loop free navigation (no snap-back to the route) instead of the corrected loop")
    parser.add_argument("--viewpoints", type=int, default=9,
                        help="number of laterally-shifted corridor viewpoints stored per route index in "
                             "freenav memory (1 = classic single centreline viewpoint)")
    parser.add_argument("--corridor_width", type=float, default=0.2,
                        help="half-width (m) of the memory corridor spanned by the lateral viewpoints")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="maximum number of steps for the ant to take during navigation evaluation")

    # --- Directories ---
    parser.add_argument("--dataset_dir", default="./apiaviz/dataset/")
    parser.add_argument("--output_dir", default="./apiaviz/output/")

    args = parser.parse_args()
    logger, output_dir = model_logger(args)

    apiaviz_eval(args, logger, output_dir)

if __name__ == "__main__":
    parse_args()
