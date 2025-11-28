# Shape–Color Binding ICL Experiments

This repository contains scripts to generate synthetic shape–color datasets and evaluate in-context learning (ICL) behavior on **binding** tasks such as:

- **`color_of_shape`** – “What is the color of a given shape?”
- **`shape_of_color`** – “What is the shape with a given color?”

The pipeline includes:
1. Environment setup (tested on **Compute Canada**).
2. Dataset generation.
3. Running scenario-based ICL experiments.
4. Sanity-checking the generated episodes.

---

## 1. Environment Setup

### 1.1. Compute Canada (recommended)

Load the required modules and activate your virtual environment:

```bash
module load StdEnv/2023 gcc cuda/12.2 cmake protobuf cudnn python/3.11 abseil cusparselt opencv/4.8.1 arrow/21.0.0
source /home/mahdiani/scratch/envs/icl/bin/activate
