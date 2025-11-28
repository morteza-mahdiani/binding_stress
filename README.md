````markdown
# Shape–Color Binding ICL

This repo runs small, controlled **in-context learning (ICL)** experiments on synthetic shape–color data.  
You can:

- Generate a toy **shapes–colors dataset**
- Run **scenario-based ICL experiments** for:
  - `color_of_shape` (“What is the color of this shape?”)
  - `shape_of_color` (“What is the shape with this color?”)
- Inspect and sanity-check generated episodes

---

## 1. Environment Setup (Compute Canada)

```bash
module load StdEnv/2023 gcc cuda/12.2 cmake protobuf cudnn python/3.11 abseil cusparselt opencv/4.8.1 arrow/21.0.0
source /home/mahdiani/scratch/envs/icl/bin/activate
````

Adjust the `source` path if your virtualenv lives elsewhere.

---

## 2. Generate the Dataset

From the project root:

```bash
python binding_stress/gen_shapes_dataset.py
```

This will create a shapes dataset and a metadata file, e.g.:

* `shapes_ds/metadata.csv`

---

## 3. Run ICL Experiments

Example command for **distinct** ICL mode with multiple shots:

```bash
python sweep_and_plot_senario_based.py \
  --meta_csv shapes_ds/metadata.csv \
  --episodes 200 \
  --ks 0 2 4 8 \
  --tasks color_of_shape shape_of_color \
  --outdir results_distinct \
  --probe_binding \
  --icl_mode distinct \
  --seed 0
```

Key flags:

* `--episodes` – number of episodes to evaluate
* `--ks` – number of in-context examples (0-shot, 2-shot, 4-shot, 8-shot)
* `--tasks` – which tasks to run (`color_of_shape`, `shape_of_color`, …)
* `--outdir` – where to store results and plots
* `--icl_mode` – e.g. `distinct` for distinct binding scenarios
* `--seed` – random seed for reproducibility

---

## 4. Inspect & Sanity-Check Episodes

You can visually check that episodes look correct and bindings are consistent.

**Sample episodes:**

```bash
python binding_stress/samples_icl_senario_based.py
```

**Sanity check episodes:**

```bash
python binding_stress/sanity_check_episodes.py
```

> In both scripts, update any **file paths inside the code** (e.g. to the generated episodes or metadata) if your directory layout differs.

---

## 5. Dependencies

This project was developed in a Python 3.11 env on Compute Canada with packages such as:

* `torch`, `torchvision`, `accelerate`, `transformers`, `datasets`
* `numpy`, `pandas`, `scipy`
* `matplotlib`, `tqdm`
* `opencv-python` (and related OpenCV wheels)
* `timm`

You can recreate the exact environment by saving your installed packages as:

```bash
pip freeze > requirements.txt
```

and then on a new machine:

```bash
pip install -r requirements.txt
```

---

## 6. Minimal Quickstart

1. Load modules + activate env (Section 1)
2. Run `python binding_stress/gen_shapes_dataset.py`
3. Run the `sweep_and_plot_senario_based.py` command from Section 3
4. Optionally inspect episodes with the scripts in `binding_stress/`

That’s all you need to reproduce the core experiments.
