# human_psychophysics_experiments (Human Object Priming, HOP)

This directory contains the **human psychophysics data and code** for the **Human Object Priming (HOP)** experiment.

The goal of the experiment is to measure **where humans expect a given object to appear in a natural scene**, without showing that object and **without any feedback**, by asking people to click on likely locations for that object.

If you want to create your own OCD dataset using virtualhome, please follow the drive [link](https://drive.google.com/drive/folders/1WGanqI4UC1_70kUhSqgeFVfKoNrYr6MM?usp=sharing) and the GitHub [link](https://github.com/kreimanlab/WhenPigsFlyContext?tab=readme-ov-file#datasets)

---

## 1. Folder contents

- `requirements.txt` ← Python environment for running the Psiturk experiment
- `Code.zip` ← Psiturk experiment for collecting human object priming data
- `Data.zip` ← Processed data and human priming maps used in the paper

You can download `Code.zip`/`Data.zip` directly:

```bash
./data_download.sh
```
Or you can directly download from [here](https://drive.google.com/drive/folders/1HCJwgkhSN9SGCdZfrBK9pmmTMpjvHSWv?usp=sharing)

---

## 2. Overview of the experiment

### 2.1 Task summary

On each trial, a participant sees:

- A **natural scene image** (800×800 pixels, drawn from MS-COCO),
- The **name of a target object** (e.g., *“wine glass”*, *“remote”*, *“microwave”*),

and is instructed to:

> “Place the object in appropriate locations by clicking where you think this object could plausibly appear in this scene.”

Key properties:

- Participants make **10 non-overlapping mouse clicks** per trial (enforced in code by rejecting clicks within a small radius of previous clicks).
- There is **no time limit** for each trial.
- **No feedback** is given about whether their clicks are “correct” (this is a generative expectation task, not a detection/recognition task).
- The underlying object is **not visible** in the image; participants are imagining plausible positions.

These clicks are aggregated to build **human “priming maps”** that describe spatial expectations for each (image, object) pair.

---

### 2.2 Stimuli

- Images are drawn from MS-COCO and resized/cropped to **800×800 pixels**.
- The HOP dataset covers **15 household object categories**:

  - `apple`, `book`, `bowl`, `cake`, `cell phone`,  
    `computer mouse`, `cup`, `keyboard`, `knife`,  
    `microwave`, `pillow`, `remote`, `toothbrush`,  
    `towel`, `wine glass`.

- There are **198 unique images** in the final dataset.
- Each image is paired with one or more target objects from this set.
- The final HOP dataset contains **864 unique image–object pairs**.

> Initial pool of **206 candidate images**. After applying semantic-meaningfulness filters and quality control, **198 images** remain in the released HOP dataset and are used to form the 864 image–object pairs.

---

### 2.3 Participants

- Data were collected on **Amazon Mechanical Turk (AMT)** using **Psiturk 2.3.12**.
- We recruited **437 participants** in total.

**Trial assignment and completion**

- Each participant was **assigned up to 20 image–object pairs**.
- In practice, participants completed **between 10 and 20 trials**, depending on early quitting and quality control.
- This yielded:
  - **5,960 raw trials** (`data/all_trials.json`),
  - **3,078 trials** remaining after quality control (`data/filtered_all_trials.json`).

**Quality control**

- Trial-level QC was applied (e.g., to remove obviously corrupt or invalid trials).
- After QC:
  - **434 participants** still had ≥1 valid trial,
  - **3 participants** had all their trials excluded (they remain in `filtered_all_trials.json` with an empty trial list).

---

## 3. From clicks to human priming maps

The HOP dataset you will typically use (for model evaluation) is built in three steps:

### 3.1 Raw trial logs → cleaned trials

- **`data/all_trials.json`**:  
  Raw per-participant logs. For each participant (`Participant_i`) we store a list of trials. Each trial includes:
  - `imageID`: path to the image file used in the experiment (e.g. `/static/images/imgset/0177.jpg`),
  - `prime_obj`: target object name,
  - `click_history`: list of 10 (x, y) pixel coordinates on the 800×800 image,
  - `click_rt_history`: 10 reaction times (ms) for each click,
  - `rt`: total reaction time (ms) for the trial,
  - `hit`: a boolean used internally (not a “correctness” measure in the sense of feedback to participants),
  - additional metadata (`trial`, `counterbalance`, `phase`, etc.).

- **`data/filtered_all_trials.json`**:  
  Same structure as `all_trials.json`, but with **problematic trials removed**. This is the basis for all subsequent aggregation.

### 3.2 Cleaned trials → per–image–object click sets

From `filtered_all_trials.json` we build, for each (image, object) pair:

- A set of **participants who completed that pair**.
- The number of participants per pair ranges from **3 to 7**:

  - 476 pairs have 3 participants,
  - 306 pairs have 4 participants,
  - 68 pairs have 5 participants,
  - 12 pairs have 6 participants,
  - 2 pairs have 7 participants.

For the **final HOP dataset**, we retain exactly **3 participants per image–object pair**:

- If more than 3 participants completed a pair, a subset of 3 is selected (according to a fixed internal rule in the preprocessing code).
- Each of these 3 participants contributes **10 clicks**, so each (image, object) pair is represented by **30 clicks** in the final dataset.

This curated set is stored in:

- **`data/main_data.json`**  

  Structure (conceptually):

  ```json
  {
    "0001.jpg": {
      "apple": [
        [[x1, y1], ..., [x10, y10]],   // subject 1, 10 clicks
        [[x1, y1], ..., [x10, y10]],   // subject 2, 10 clicks
        [[x1, y1], ..., [x10, y10]]    // subject 3, 10 clicks
      ],
      "cell phone": [...],
      "cake": [...],
      ...
    },
    "0002.jpg": { ... },
    ...
  }
  ```

- There are **198 image keys** (e.g., `"0001.jpg"`, …, `"0199.jpg"`) and **864 total image–object pairs**.

### 3.3 Click sets → human priming maps

For each of the 864 image–object pairs:

1. We start from the **30 2D click coordinates** in `main_data.json` on an 800×800 pixel grid.
2. We discretize the image into a coarse **32×32 grid** and count how many clicks fall into each cell.
3. We obtain a **2D attention map** by accumulating these counts.
4. We apply **Gaussian smoothing** using an **11×11 kernel** to produce a smooth density-like map.
5. We **upsample** the smoothed map to **224×224** pixels.
6. We apply **min–max rescaling** so that map values lie in a fixed numeric range.

The resulting priming maps are stored as:

- **`data/human_priming_maps_np_arrays/*.npy`**

  - One file per image–object pair, e.g. `0044_apple.npy`, `0177_wine glass.npy`.
  - Each file is a **224×224 `float64` NumPy array**.
  - Values range approximately from **0.0 to 255.0** and are **linearly related to click density** (higher values indicate more clicks in that region after smoothing).
  - These maps are suitable for **comparisons with model-generated 224×224 attention or saliency maps**.

 
> We also refer to these as “probabilistic priming maps”. They represent a smoothed click distribution. The released `.npy` arrays are **min–max–rescaled intensity maps**, not strictly normalized probability distributions (they do not sum to 1). If you need probability maps, you can renormalize each array by dividing by its sum.

---

## 4. Dataset statistics (summary)

From `Data` folder:

- **Participants**
  - 437 total AMT participants.
  - 434 participants with ≥1 valid trial after QC.
- **Trials**
  - 5,960 raw trials (`all_trials.json`).
  - 3,078 trials after QC (`filtered_all_trials.json`).
- **Images and objects**
  - 15 object categories.
  - 198 unique images.
  - 864 unique image–object pairs.
- **Per-participant trials**
  - Raw: between 10 and 20 trials per participant (mean ≈ 13.6, median = 10).
  - After QC: many participants have fewer retained trials (mean ≈ 7.0, median = 5).
- **Per–image–object participants (after QC)**
  - 3–7 participants per pair (mean ≈ 3.56, median = 3).
  - Final HOP dataset uses 3 participants × 10 clicks = 30 clicks per pair as stored in `main_data.json` and in the human priming maps.

These are the exact statistics used to build the HOP maps that are evaluated against models in the paper (e.g., via normalized RMSE alignment scores).

---

## 5. Data layout and file descriptions

In the `Data` folder, you should see:

```text
data/
  main_data.json
  all_trials.json
  filtered_all_trials.json
  human_priming_maps_np_arrays/
  click_viz/
  final_avg_attn_maps/
  final_avg_attn_maps_colormap/
  final_avg_attn_maps_overlay/
  final_avg_attn_maps_overlay_2/
  final_avg_attn_maps_overlay_3/
  final_avg_attn_maps_overlay_4/
```

### 5.1 JSON files

- **`data/all_trials.json`**  
  Raw logs from the Psiturk experiment (per participant).

- **`data/filtered_all_trials.json`**  
  Same structure as `all_trials.json`, but with trial-level QC applied (invalid trials removed).

- **`data/main_data.json`**  
  Curated click data for the final HOP dataset:
  - 198 images,
  - 864 image–object pairs,
  - 3 participants × 10 clicks per pair.

### 5.2 Priming maps and visualizations

- **`data/human_priming_maps_np_arrays/`**  
  - 864 `.npy` maps, one per image–object pair.
  - Each is a 224×224 float64 array (values ≈ 0–255).

- **`data/click_viz/`**  
  - 864 `.jpg` files visualizing the 30 clicks for each image–object pair directly on the 800×800 image.

- **`data/final_avg_attn_maps/`**  
  - 864 `.jpg` files showing the 32×32 → 224×224 smoothed maps.

- **`data/final_avg_attn_maps_colormap/`**  
  - 864 `.jpg` files visualizing priming maps with a colormap (no underlying image).

- **`data/final_avg_attn_maps_overlay/`, `_2`, `_3`, `_4`**  
  - Variants of overlays combining the priming maps with the original images (different visual styles / color maps), used for qualitative visualization in the paper.

---

## 6. Using the data

### 6.1 Loading priming maps in Python: Unzip the Data.zip folder

Example:

```python
import json
import numpy as np
import zipfile

# Load curated click data
with z.open("data/main_data.json") as f:
    main_data = json.load(f)

# Load one human priming map
with z.open("data/human_priming_maps_np_arrays/0001_apple.npy") as f:
    prim_map = np.load(f)

print(prim_map.shape)  # (224, 224)
print(prim_map.min(), prim_map.max())
```

In the `Data` folder, you can just use:

```python
prim_map = np.load("data/human_priming_maps_np_arrays/0001_apple.npy")
```

### 6.2 Matching maps to images

- The filename pattern is `{imageID}_{object}.npy` / `.jpg`.
  - `0001_apple.npy` corresponds to image `0001.jpg` and object `apple`.
- The mapping of images to objects is fully encoded in `main_data.json`.

---

## 7. Running the experiment locally with Psiturk

To reproduce or inspect the experiment interface:

### 7.1 Install dependencies

We recommend creating a dedicated environment (Psiturk 2.3.12 expects Python 3.8):

```bash
conda create -n mturkenv python=3.8
conda activate mturkenv
pip install --upgrade psiturk==2.3.12
pip install -r requirements.txt  
```

### 7.2 Run the experiment in debug mode

1. Unzip `Code.zip`:

   ```bash
   unzip Code.zip
   ```

   This will create:

   ```text
   objprime_latest/
   ```

2. Start Psiturk inside that folder:

   ```bash
   cd objprime_latest
   psiturk
   ```

3. In the Psiturk shell:

   ```text
   server on
   debug
   ```

4. Open your browser at:

   - `http://localhost:22380`

   and follow the AMT-like flow locally in debug mode.

This will let you see the **exact instructions, trial flow, and click interface** that AMT participants saw when generating this dataset.

---

## 8. Experiment Front End

![frontend](images/FigS1.png)
