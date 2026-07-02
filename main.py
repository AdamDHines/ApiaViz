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

    # --- Evaluation: flowers (object identification) ---
    parser.add_argument("--per_class", type=int, default=40, help="images sampled per flower class")
    parser.add_argument("--ks", type=int, nargs="*", default=[1, 8], help="scan lengths (fixations) to report")

    # --- Evaluation: nav (route following) ---
    parser.add_argument("--ant", type=int, default=1, help="ant index for navigation routes")
    parser.add_argument("--routes", type=int, nargs="*", default=[1, 2, 3], help="route indices to navigate")
    parser.add_argument("--segments", type=int, default=16, help="MBON population size (route segments)")
    parser.add_argument("--severity", type=float, default=1.0, help="luminance corruption severity")
    parser.add_argument("--landmark_fraction", type=float, default=0.5, help="fraction of world triangles that are colour landmarks")
    parser.add_argument("--chroma", type=float, default=60.0, help="landmark opponent-colour strength")
    parser.add_argument("--freenav", action="store_true",
                        help="open-loop free navigation (no snap-back to the route) instead of the corrected loop")

    # --- Directories ---
    parser.add_argument("--dataset_dir", default="./apiaviz/dataset/")
    parser.add_argument("--output_dir", default="./apiaviz/output/")

    args = parser.parse_args()
    logger, output_dir = model_logger(args)

    apiaviz_eval(args, logger, output_dir)

if __name__ == "__main__":
    parse_args()
