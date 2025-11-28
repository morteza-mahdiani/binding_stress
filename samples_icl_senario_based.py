import csv
import random
import numpy as np
from pathlib import Path

from icl_episodes_senario_based import load_meta, make_episode


def save_icl_pairs(
    meta_csv: str,
    out_csv: str,
    tasks: list[str],
    ks: list[int],
    episodes: int,
    icl_mode: str,
    seed: int = 0,
):
    """
    Generate ICL episodes using make_episode and save all (color, shape)
    pairs used in the shots for later analysis.

    This does NOT run the model; it only uses the episode constructor.
    """

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)

    samples = load_meta(meta_csv)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "task",
            "k",
            "icl_mode",
            "episode",
            "shot_idx",
            "shot_color",
            "shot_shape",
            "shot_file",
            "query_color",
            "query_shape",
            "query_file",
            "query_condition",
            "query_split",
        ])

        for task in tasks:
            for k in ks:
                for ep in range(episodes):
                    # make_episode must be your updated one:
                    #   returns (messages, answer, qmeta, shots_meta)
                    messages, gold, qmeta, shots_meta = make_episode(
                        samples,
                        k=k,
                        task=task,
                        icl_mode=icl_mode,
                    )

                    q_color = qmeta.get("color")
                    q_shape = qmeta.get("shape")
                    q_file = qmeta.get("file")
                    q_cond = qmeta.get("condition")
                    q_split = qmeta.get("split")

                    for shot_idx, s in enumerate(shots_meta):
                        w.writerow([
                            task,
                            k,
                            icl_mode,
                            ep,
                            shot_idx,
                            s.get("color"),
                            s.get("shape"),
                            s.get("file"),
                            q_color,
                            q_shape,
                            q_file,
                            q_cond,
                            q_split,
                        ])

    print(f"Saved ICL pairs to {out_path}")


if __name__ == "__main__":
    save_icl_pairs(
        meta_csv="shapes_ds/metadata.csv",
        out_csv="analysis/icl_pairs_binding_stress.csv",
        tasks=["color_of_shape", "shape_of_color",
        #  "shape_only", "color_only"
         ],
        ks=[0, 2, 4, 8],
        episodes=200,
        icl_mode="binding_stress",
        seed=0,
    )
    save_icl_pairs(
        meta_csv="shapes_ds/metadata.csv",
        out_csv="analysis/icl_pairs_distinct.csv",
        tasks=["color_of_shape", "shape_of_color",
        #  "shape_only", "color_only"
         ],
        ks=[0, 2, 4, 8],
        episodes=200,
        icl_mode="distinct",
        seed=0,
    )
# "color_of_shape","shape_of_color","shape_only","color_only"