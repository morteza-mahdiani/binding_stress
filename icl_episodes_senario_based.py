# icl_episodes_senario_based.py
import csv, random, json, os
from collections import defaultdict

COLORS = ["red", "green", "blue", "yellow"]
SHAPES = ["circle", "square", "triangle", "star"]

def load_meta(path):
    with open(path) as f: 
        rows = list(csv.DictReader(f))
    # store dataset_root to join later
    for r in rows:
        r["_dataset_root"] = os.path.dirname(os.path.dirname(path))  # e.g., shapes_ds
    return rows

def _img_path(row):
    # images live in <dataset_root>/images/<file>
    return os.path.join(row["_dataset_root"], "images", row["file"])

def _filter_split(samples, condition, split=None):
    """Filter rows by condition and optional split."""
    rows = [r for r in samples if r["condition"] == condition]
    if split is not None:
        rows = [r for r in rows if r.get("split") == split]
    return rows

def _choose_bind_shots_random(shots_pool, k):
    """Original behavior: uniform random sample from BIND/train."""
    return random.sample(shots_pool, min(k, len(shots_pool)))

def _choose_bind_shots_distinct(shots_pool, k):
    """
    Try to pick shots so that colors and shapes do not repeat across the
    in-context examples. Fall back to filling with random if needed.
    """
    # Deduplicate by (color, shape) so we don't pick many copies of same combo
    by_pair = {}
    for r in shots_pool:
        key = (r["color"], r["shape"])
        if key not in by_pair:
            by_pair[key] = r

    # Greedy: avoid reusing colors or shapes
    chosen = []
    used_colors, used_shapes = set(), set()
    candidates = list(by_pair.values())
    random.shuffle(candidates)
    for r in candidates:
        c, s = r["color"], r["shape"]
        if c in used_colors or s in used_shapes:
            continue
        chosen.append(r)
        used_colors.add(c)
        used_shapes.add(s)
        if len(chosen) == k:
            break

    # If we couldn't get k fully distinct items, top up randomly
    if len(chosen) < k:
        remaining = [r for r in shots_pool if r not in chosen]
        if remaining:
            extra = random.sample(remaining, min(k - len(chosen), len(remaining)))
            chosen.extend(extra)

    return chosen

def choose_distinct_shots(shots_pool, k):
    """
    Pick k shots with unique shapes and unique colors.
    """
    # Deduplicate by full pair
    unique = {(r["color"], r["shape"]): r for r in shots_pool}.values()

    chosen = []
    used_colors = set()
    used_shapes = set()

    for r in unique:
        c, s = r["color"], r["shape"]
        if c not in used_colors and s not in used_shapes:
            chosen.append(r)
            used_colors.add(c)
            used_shapes.add(s)
            if len(chosen) == k:
                break

    return chosen  # guaranteed distinct if enough combos exist

def choose_binding_stress(shots_pool, k, mode="swap"):
    """
    Binding-stress ICL selector.

    Goals:
      - k = 0: no shots
      - k = 1: just one random shot
      - k = 2: two shots that share a feature (color or shape)
      - k >= 4: build contexts from 2x2 swap blocks:
          (c1,s1), (c2,s2), (c1,s2), (c2,s1)
        and tile them until we reach k (or run out of combos).

    'mode' is kept only for backward compatibility and is ignored.
    """

    shots_pool = list(shots_pool)
    if k <= 0 or not shots_pool:
        return []

    if k == 1:
        return [random.choice(shots_pool)]

    # ------------------------------------------------------------
    # Build indices
    # ------------------------------------------------------------
    by_pair = {}
    by_color = defaultdict(list)
    by_shape = defaultdict(list)

    for r in shots_pool:
        c, s = r["color"], r["shape"]
        by_pair[(c, s)] = r          # one representative per (color, shape)
        by_color[c].append(r)
        by_shape[s].append(r)

    colors = list(by_color.keys())
    shapes = list(by_shape.keys())

    # ------------------------------------------------------------
    # Helper 1: pick TWO shots that share a feature (color or shape)
    # Used for k = 2.
    # ------------------------------------------------------------
    def sample_shared_feature_pair():
        # colors with at least 2 distinct shapes
        color_candidates = []
        for c, rows in by_color.items():
            shapes_for_c = {r["shape"] for r in rows}
            if len(shapes_for_c) >= 2:
                color_candidates.append(c)

        # shapes with at least 2 distinct colors
        shape_candidates = []
        for s, rows in by_shape.items():
            colors_for_s = {r["color"] for r in rows}
            if len(colors_for_s) >= 2:
                shape_candidates.append(s)

        axes = []
        if color_candidates:
            axes.append("color")
        if shape_candidates:
            axes.append("shape")

        if not axes:
            # Degenerate case: no structure possible, just return 2 random shots
            return random.sample(shots_pool, min(2, len(shots_pool)))

        axis = random.choice(axes)

        if axis == "color":
            c = random.choice(color_candidates)
            shapes_for_c = list({r["shape"] for r in by_color[c]})
            s1, s2 = random.sample(shapes_for_c, 2)
            return [by_pair[(c, s1)], by_pair[(c, s2)]]
        else:
            s = random.choice(shape_candidates)
            colors_for_s = list({r["color"] for r in by_shape[s]})
            c1, c2 = random.sample(colors_for_s, 2)
            return [by_pair[(c1, s)], by_pair[(c2, s)]]

    # ------------------------------------------------------------
    # Helper 2: sample a full 2x2 swap block:
    #    (c1,s1), (c2,s2), (c1,s2), (c2,s1)
    # Returns (list_of_rows, list_of_pairs) or (None, None) on failure.
    # ------------------------------------------------------------
    def sample_swap_block(used_pairs=None):
        if len(colors) < 2 or len(shapes) < 2:
            return None, None

        for _ in range(100):
            c1, c2 = random.sample(colors, 2)
            s1, s2 = random.sample(shapes, 2)
            candidate_pairs = [(c1, s1), (c2, s2), (c1, s2), (c2, s1)]

            # All combos must exist
            if any(p not in by_pair for p in candidate_pairs):
                continue

            # Optionally avoid blocks that are 100% already used
            if used_pairs is not None and all(p in used_pairs for p in candidate_pairs):
                continue

            rows = [by_pair[p] for p in candidate_pairs]
            return rows, candidate_pairs

        return None, None

    # ------------------------------------------------------------
    # Case k == 2: guarantee at least one shared feature
    # ------------------------------------------------------------
    if k == 2:
        return sample_shared_feature_pair()

    # ------------------------------------------------------------
    # Case k == 3: base pair with shared feature + one more shot
    # (still at least one overlapping pair; simple and good enough)
    # ------------------------------------------------------------
    if k == 3:
        base_pair = sample_shared_feature_pair()
        chosen = list(base_pair)
        used_pairs = {(r["color"], r["shape"]) for r in chosen}

        remaining = [r for r in shots_pool if (r["color"], r["shape"]) not in used_pairs]
        random.shuffle(remaining)
        if remaining:
            chosen.append(remaining[0])
        return chosen[:3]

    # ------------------------------------------------------------
    # Case k >= 4: build from swap blocks
    # ------------------------------------------------------------
    chosen = []
    used_pairs = set()

    while len(chosen) < k:
        block, block_pairs = sample_swap_block(used_pairs if chosen else None)
        if block is None:
            break

        for r, p in zip(block, block_pairs):
            if len(chosen) >= k:
                break
            if p in used_pairs:
                continue
            chosen.append(r)
            used_pairs.add(p)

    # If we still haven't reached k, fill with any remaining unique pairs
    if len(chosen) < k:
        remaining = [
            r for r in shots_pool
            if (r["color"], r["shape"]) not in used_pairs
        ]
        random.shuffle(remaining)
        for r in remaining:
            if len(chosen) >= k:
                break
            chosen.append(r)
            used_pairs.add((r["color"], r["shape"]))

    return chosen[:k]

def make_episode(samples, k=4, task="color_of_shape", icl_mode="random"):
    """
    samples: list of dicts with file, shape, color, condition, split
    Returns:
      - messages: chat-style messages (system + shots + query)
      - answer: gold label for the query (color or shape)
      - q:      query row (metadata dict)
      - shots:  list of shot rows (metadata dicts)
    """
    # ---- Choose query + pool ----
    if task in ("color_of_shape", "shape_of_color"):
        # BIND queries from test split, shots from train split
        tests = _filter_split(samples, condition="BIND", split="test")
        if tests:
            q = random.choice(tests)
        else:
            testt = _filter_split(samples, condition="BIND", split=None)
            print(testt)
            q = random.choice(testt)
        shots_pool = _filter_split(samples, condition="BIND", split="train")

    elif task == "shape_only":
        # grayscale / shape-only rows, no train/test split
        q = random.choice([r for r in samples if r["condition"] == "SHAPE_ONLY"])
        shots_pool = [r for r in samples if r["condition"] == "SHAPE_ONLY"]

    elif task == "color_only":
        q = random.choice([r for r in samples if r["condition"] == "COLOR_ONLY"])
        shots_pool = [r for r in samples if r["condition"] == "COLOR_ONLY"]

    else:
        raise ValueError(f"Unknown task: {task}")

    # ---- Build question/answer for the query ----
    if task == "color_of_shape":
        question = f"What color is the {q['shape']}?"
        answer = q["color"]
    elif task == "shape_of_color":
        question = f"Which shape is {q['color']}?"
        answer = q["shape"]
    elif task == "shape_only":
        question = "What shape is shown?"
        answer = q["shape"]
    elif task == "color_only":
        question = "What color is the shape?"
        answer = q["color"]

    # ---- Select few-shot demonstrations (ICL regime) ----
    if k <= 0 or len(shots_pool) == 0:
        shots = []
    else:
        if task in ("color_of_shape", "shape_of_color"):
            if icl_mode == "distinct":
                shots = choose_distinct_shots(shots_pool, k)

            elif icl_mode == "binding_stress":
                shots = choose_binding_stress(shots_pool, k, mode="swap")  
                
            else:
                shots = random.sample(shots_pool, min(k, len(shots_pool)))
        else:
            # For shape_only / color_only we just use random for now
            shots = random.sample(shots_pool, min(k, len(shots_pool)))

    # ---- Build messages (unchanged) ----
    messages = []
    system = {
        "role": "system",
        "content": [{
            "type": "text",
            "text": "Answer concisely using a single word from the allowed set."
        }],
    }
    messages.append(system)

    for s in shots:
        if task == "color_of_shape":
            qtxt = f"What color is the {s['shape']}?"
            atxt = s["color"]
        elif task == "shape_of_color":
            qtxt = f"Which shape is {s['color']}?"
            atxt = s["shape"]
        elif task == "shape_only":
            qtxt = "What shape is shown?"
            atxt = s["shape"]
        else:  # color_only
            qtxt = "What color is the shape?"
            atxt = s["color"]

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image_url": s["file"]},
                {"type": "text", "text": qtxt},
            ],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": atxt}],
        })

    # Query message
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image_url": q["file"]},
            {"type": "text", "text": question},
        ],
    })

    return messages, answer, q, shots
