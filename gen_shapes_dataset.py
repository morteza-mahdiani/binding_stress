# gen_shapes_dataset.py
import os, json, csv, math, random
from PIL import Image, ImageDraw

SHAPES = ["circle", "square", "triangle", "star"]
COLORS = {
    "red":    (255, 0, 0),
    "green":  (0, 170, 0),
    "blue":   (0, 120, 255),
    "yellow": (235, 200, 0),
}
GRAY = (160, 160, 160)

def draw_shape(draw: ImageDraw.ImageDraw, shape: str, bbox, fill):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0+x1)/2, (y0+y1)/2
    w, h = (x1-x0), (y1-y0)
    if shape == "circle":
        draw.ellipse(bbox, fill=fill)
    elif shape == "square":
        draw.rectangle(bbox, fill=fill)
    elif shape == "triangle":
        # upward triangle
        pts = [(cx, y0), (x0, y1), (x1, y1)]
        draw.polygon(pts, fill=fill)
    elif shape == "star":
        # 5-point star
        R, r = w/2, w/4
        pts = []
        for i in range(10):
            ang = math.pi/2 + i*math.pi/5
            rad = R if i%2==0 else r
            pts.append((cx + rad*math.cos(ang), cy - rad*math.sin(ang)))
        draw.polygon(pts, fill=fill)
    else:
        raise ValueError(f"Unknown shape {shape}")

def save_img(path, shape, color_rgb, size=256, jitter=0.06):
    img = Image.new("RGB", (size, size), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    margin = int(size*0.18)
    # size jitter
    j = random.uniform(-jitter, jitter)
    pad = int(margin*(1+j))
    bbox = (pad, pad, size-pad, size-pad)
    draw_shape(draw, shape, bbox, color_rgb)
    img.save(path)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main(
    outdir="shapes_ds",
    n_per_combo=20,
    seed=42,
    debug=False,
    color_only_shape="circle",
    shape_only_color="red",
):
    random.seed(seed)
    ensure_dir(outdir)
    imgdir = os.path.join(outdir, "images")
    ensure_dir(imgdir)
    meta_path = os.path.join(outdir, "metadata.csv")
    splits_path = os.path.join(outdir, "splits.json")

    rows = []
    uid = 0

    def add_row(fname, cond, shape, color, is_gray, split):
        nonlocal uid
        rows.append({
            "id": uid, "file": fname, "condition": cond,
            "shape": shape, "color": color,
            "is_grayscale": int(is_gray),
            "split": split
        })
        uid += 1

    # -------- Binding (all combos) --------
    combos = [(s, c) for s in SHAPES for c in COLORS.keys()]
    # Optionally hold out a few combos for test (novel binding)
    heldout = {("triangle","yellow"), ("star","green")}  # edit as you like

    for shape, color in combos:
        for i in range(1 if debug else n_per_combo):
            split = "test" if (shape, color) in heldout else "train"
            fname = f"BIND_{shape}_{color}_{i:03d}.png"
            save_img(os.path.join(imgdir, fname), shape, COLORS[color])
            add_row(outdir + "/images/" + fname, "BIND", shape, color, False, split)

    # -------- Grayscale (shape only) --------
    for shape in SHAPES:
        for i in range(1 if debug else n_per_combo):
            fname = f"SHAPE_ONLY_{shape}_gray_{i:03d}.png"
            save_img(os.path.join(imgdir, fname), shape, GRAY)
            add_row(outdir + "/images/" + fname, "SHAPE_ONLY", shape, "gray", True, "train")

    # -------- Same-shape / different colors (color only) --------
    sfix = color_only_shape
    for color in COLORS.keys():
        for i in range(1 if debug else n_per_combo):
            fname = f"COLOR_ONLY_{sfix}_{color}_{i:03d}.png"
            save_img(os.path.join(imgdir, fname), sfix, COLORS[color])
            add_row(outdir + "/images/" + fname, "COLOR_ONLY", sfix, color, False, "train")

    # -------- Same-color / different shapes (shape only with fixed color) --------
    cfix = shape_only_color
    for shape in SHAPES:
        for i in range(1 if debug else n_per_combo):
            fname = f"SHAPE_ONLY_FIXEDCOLOR_{shape}_{cfix}_{i:03d}.png"
            save_img(os.path.join(imgdir, fname), shape, COLORS[cfix])
            add_row(outdir + "/images/" + fname, "SHAPE_ONLY_FIXEDCOLOR", shape, cfix, False, "train")

    # write metadata
    with open(meta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # write a JSON split summary
    with open(splits_path, "w") as f:
        json.dump({
            "heldout_pairs_for_compositional_test": list(map(list, heldout)),
            "note": "Evaluate on BIND rows with split=='test' for novel bindings."
        }, f, indent=2)

if __name__ == "__main__":
    # Quick defaults; set debug=True to generate ~10 images total
    main(outdir="shapes_ds", n_per_combo=20, seed=42, debug=False)
