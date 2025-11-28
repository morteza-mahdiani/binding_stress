import pandas as pd
from collections import defaultdict

distinct_df = pd.read_csv("analysis/icl_pairs_distinct.csv")
binding_df = pd.read_csv("analysis/icl_pairs_binding_stress.csv")

violations = []

for (task, k, ep, mode), g in distinct_df.groupby(
    ["task", "k", "episode", "icl_mode"]
):
    pairs = list(zip(g["shot_color"], g["shot_shape"]))
    colors = [c for c, _ in pairs]
    shapes = [s for _, s in pairs]
    n = len(pairs)
    if n == 0:
        continue

    # Distinct regime requirement:
    #   - all pairs unique
    #   - all colors unique
    #   - all shapes unique
    if len(set(pairs)) != n or len(set(colors)) != n or len(set(shapes)) != n:
        violations.append(((task, k, ep, mode), pairs))

print(f"[DISTINCT] number of violating episodes: {len(violations)}")
if violations:
    print("First few violations:")
    for meta, pairs in violations[:5]:
        print("  meta:", meta)
        print("  pairs:", pairs)


def shares_feature(p1, p2):
    (c1, s1), (c2, s2) = p1, p2
    return (c1 == c2) or (s1 == s2)

k2_violations = []
k3_violations = []
swap_violations = []   # for k >= 4

for (task, k, ep, mode), g in binding_df.groupby(
    ["task", "k", "episode", "icl_mode"]
):
    pairs = list(zip(g["shot_color"], g["shot_shape"]))
    n = len(pairs)
    if n == 0:
        continue

    # --- k = 2: two shots MUST share a feature ---
    if k == 2:
        if n != 2:
            k2_violations.append((task, k, ep, pairs, "wrong n_shots"))
        else:
            if not shares_feature(pairs[0], pairs[1]):
                k2_violations.append((task, k, ep, pairs, "no shared feature"))

    # --- k = 3: at least one pair of shots shares a feature ---
    elif k == 3:
        ok = False
        for i in range(n):
            for j in range(i+1, n):
                if shares_feature(pairs[i], pairs[j]):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            k3_violations.append((task, k, ep, pairs))

    # --- k >= 4: first 4 shots should form a full 2x2 swap block ---
    elif k >= 4 and n >= 4:
        block = pairs[:4]
        colors = {c for c, _ in block}
        shapes = {s for _, s in block}
        # requirement: 2 distinct colors, 2 distinct shapes, 4 distinct pairs
        if not (len(colors) == 2 and len(shapes) == 2 and len(set(block)) == 4):
            swap_violations.append((task, k, ep, block))

print(f"[BINDING] k=2 violations: {len(k2_violations)}")
if k2_violations:
    print("Example k=2 violation:", k2_violations[0])

print(f"[BINDING] k=3 violations: {len(k3_violations)}")
if k3_violations:
    print("Example k=3 violation:", k3_violations[0])

print(f"[BINDING] k>=4 swap-block violations: {len(swap_violations)}")
if swap_violations:
    print("Example swap violation:", swap_violations[0])

def binding_load_for_pairs(pairs):
    # pairs = list of (color, shape)
    from collections import Counter
    colors = [c for c, _ in pairs]
    shapes = [s for _, s in pairs]
    load = 0
    for cnt in Counter(colors).values():
        load += (cnt - 1)
    for cnt in Counter(shapes).values():
        load += (cnt - 1)
    return load

def summarize_binding_load(df, label):
    rows = []
    for (mode, k, ep), g in df.groupby(["icl_mode", "k", "episode"]):
        pairs = list(zip(g["shot_color"], g["shot_shape"]))
        if not pairs:
            continue
        load = binding_load_for_pairs(pairs)
        rows.append({"icl_mode": mode, "k": k, "episode": ep, "binding_load": load})
    rep = pd.DataFrame(rows)
    print(f"\n=== Binding load summary: {label} ===")
    print(rep.groupby(["icl_mode", "k"])["binding_load"].mean())

summarize_binding_load(distinct_df, "DISTINCT")
summarize_binding_load(binding_df, "BINDING_STRESS")
