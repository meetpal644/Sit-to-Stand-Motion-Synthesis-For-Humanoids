# Natural Sit-to-Stand Motion Synthesis for Humanoids

Official evaluation code and pretrained policy for the paper:

> **Natural Sit-to-Stand Motion Synthesis for Humanoids via Guided Assistance Curricula and Staged Rewards**
> Meet Pal Singh, Vyankatesh Ashtekar, Ashish Dutta — Department of Mechanical Engineering, IIT Kanpur.

A single reference-free Proximal Policy Optimization (PPO) policy learns smooth, human-like
sit-to-stand (STS) motion for the 29-DOF **Unitree G1** across **eight chair heights**, driven by a
coupled assist-force / chair-height curriculum, an IK-generated pose library, and biomechanics-inspired
staged rewards. On a deterministic, force-free evaluator the released policy reaches **97.8% balanced-standing
success** across all eight chair heights.

**Demo video:** https://youtu.be/SgSlgJRrlcE

<p align="center">
  <img src="sts_stop_motion.png" width="72%" alt="Stop-motion frames of a sit-to-stand rollout">
</p>
<p align="center"><em>Stop-motion frames of a single sit-to-stand rollout: the robot rises from the chair, loads through the feet, and settles into a balanced stand. Direct output of the released policy (see <a href="#usage">Usage</a>).</em></p>

> This repository is a **validation / reproducibility package**. It ships the pretrained policy, its full
> checkpoint trajectory, training curves, and a fully self-contained evaluation + visualization stack so
> that anyone can independently confirm the paper's claims. The training environment and learning code are
> not included.

## Table of Contents
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Pretrained Model](#pretrained-model)
  - [Two released policies (and why the paper reports `gen_best`)](#two-released-policies-and-why-the-paper-reports-gen_best)
- [Usage](#usage)
  - [1. Headless evaluation across all chair heights](#1-headless-evaluation-across-all-chair-heights)
  - [2. Watch a live rollout](#2-watch-a-live-rollout)
  - [3. Reproduce the COP-locus figure](#3-reproduce-the-cop-locus-figure)
  - [4. Inspect training curves (TensorBoard)](#4-inspect-training-curves-tensorboard)
- [Expected Results](#expected-results)
- [Metric Definitions](#metric-definitions)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Installation

The evaluation runs on CPU (headless, force-free); no GPU is required.

```bash
git clone https://github.com/meetpal644/Sit-to-Stand-Motion-Synthesis-For-Humanoids.git
cd Sit-to-Stand-Motion-Synthesis-For-Humanoids

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The pinned versions in [`requirements.txt`](requirements.txt) match the exact stack recorded inside the
released checkpoint (`Stable-Baselines3 2.7.1`, `PyTorch 2.10.0`, `NumPy 2.2.6`, `Gymnasium 1.2.3`,
`MuJoCo 3.4.0`), so the policy de-serialises and evaluates identically.

> **macOS note:** the interactive viewer requires an OpenGL context launched via Apple's `mjpython`
> wrapper (shipped with the `mujoco` package). The headless evaluator uses plain `python`.

## Repository Structure

```
.
├── g1_sit_eval.py                 # headless deterministic evaluator (8 chair lanes)
├── g1_sit_viz_v3.py               # interactive MuJoCo viewer for a single rollout
├── g1_sit_cop_locus.py            # regenerates the COP-transfer (balance) figure
├── g1_smooth.xml                  # MuJoCo model of the Unitree G1 + 8 seats
├── assets/                        # G1 STL meshes referenced by g1_smooth.xml
├── keyframes/                     # 8 IK-generated seated-pose pools, one CSV per chair height
│   └── chair1_pose_{0,0.01,...,0.1}z.csv
├── models_g1_sit_gen_best/        # policy reported in the paper (full checkpoint trajectory)
│   ├── best_model.zip             # the evaluated policy
│   ├── vec_normalize_best.pkl     # matching observation/return normaliser
│   ├── run_meta.json              # curriculum state of this checkpoint (beta=0.6, force=0)
│   ├── eval_full.json             # reference eval used in the paper (97.8%, 50 resets/lane)
│   └── sit_v3_ckpt_*_steps.zip    # intermediate checkpoints every 5M steps
├── models_g1_sit_genbest_v2/      # alternate seed: higher raw success, less smooth (see below)
│   ├── best_model.zip
│   ├── vec_normalize_best.pkl
│   ├── run_meta.json
│   ├── eval_full_n40.json         # eval at 40 resets/lane
│   └── sit_v3_ckpt_*_steps.zip
├── logs_g1_sit_gen_best/          # TensorBoard training curves for the paper policy
└── media/                         # teaser figures for this README
```

## Pretrained Model

`models_g1_sit_gen_best/best_model.zip` is the policy reported in the paper.

| Property | Value |
|---|---|
| Observation | 97-D proprioceptive (gravity, IMU, joint pos/vel, previous action, `beta`) |
| Action | 29-D incremental joint command, scaled by `beta` |
| Curriculum state at checkpoint | `beta_max = 0.6`, assist `force_max = 0.0` N (fully decayed) |
| Evaluation | deterministic (mean action), **force-free** (pelvis assist zeroed) |

> **Important:** evaluation must use `--beta 0.6`. `beta` is both fed into the observation and scales the
> action, so a mismatched value produces both out-of-distribution observations and wrong physics. The
> correct value is stored in `run_meta.json` and used by default.

### Two released policies (and why the paper reports `gen_best`)

We release two fully-trained seeds of the same method so reviewers can inspect the trade-off directly.
`models_g1_sit_genbest_v2` reaches a slightly higher raw success rate, but `models_g1_sit_gen_best`
(the one reported in the paper) is a **better embodiment of the paper's central claims** — *smooth,
low-effort, within-limit* motion.

Both evaluated deterministically and force-free at `beta = 0.6` (aggregated over the 8 chair lanes):

| Metric | `gen_best` (paper) | `genbest_v2` | Better | Relevance |
|---|:---:|:---:|:---:|---|
| Resets per lane | 50 | 40 | — | `gen_best` uses the full paper protocol |
| Balanced success (%) | 97.8 | **98.8** | v2 | raw task completion |
| Fall rate (%) | 2.2 | **1.3** | v2 | safety (falls) |
| Energy over first 4 s (J) | 115.6 | **107.1** | v2 | effort |
| Time-to-stand (s) | 3.55 | **3.43** | v2 | speed |
| Capture-point in support | 0.993 | **0.996** | v2 | balance |
| **Action jitter** | **1.535** | 1.774 | gen_best | **motion smoothness (core claim)** |
| **S_q — joints within limits** | **0.963** | 0.724 | gen_best | **within-limit safety (core claim)** |
| S_torque — torques within limits | 1.00 | 1.00 | tie | actuator safety |

**Why we chose `gen_best` for the paper.** The paper's contribution is not the last decimal of success
rate — it is a *smooth, human-like, within-limit* sit-to-stand that is amenable to physical
implementation (the explicit gap we identify versus prior RL stand-up work such as HoST, whose motions
are abrupt). On the two metrics that encode that claim, `genbest_v2` regresses sharply:

- **Joint-limit safety `S_q` collapses from 0.96 to 0.72** — i.e. `genbest_v2` spends roughly a quarter
  of its timesteps with at least one joint pushed outside its nominal range. That directly contradicts
  the "within-limit motion" claim, even though no torque limit is violated (`S_torque = 1.00`).
- **Action jitter rises ~16%** (1.535 → 1.774), i.e. visibly jerkier motion.

`genbest_v2` buys its extra ~1% success and lower energy by moving more aggressively. For a controller
meant to be safe and natural on hardware, that is the wrong trade, so the paper reports `gen_best`. We
ship `genbest_v2` alongside it purely for transparency; anyone can reproduce the numbers above with the
commands below (swap `--model-dir` and use `--n-resets 40` to match its bundled eval).

## Usage

### 1. Headless evaluation across all chair heights

Scores the policy with the identical protocol used in the paper — 50 deterministic, force-free resets on
each of the 8 chair lanes — and writes a per-lane + aggregate JSON.

```bash
python g1_sit_eval.py \
    --model-dir models_g1_sit_gen_best \
    --ckpt best_model \
    --vecnorm vec_normalize_best \
    --beta 0.6 \
    --n-resets 50 \
    --tag reproduce
```

This writes `models_g1_sit_gen_best/eval_reproduce.json`. Compare it against the bundled
`eval_full.json` (the exact run behind the paper's numbers). The console prints the overall
balanced-standing success (~0.98).

To reproduce the alternate seed's numbers instead, point at the other model and match its protocol:

```bash
python g1_sit_eval.py \
    --model-dir models_g1_sit_genbest_v2 \
    --ckpt best_model --vecnorm vec_normalize_best \
    --beta 0.6 --n-resets 40 --tag reproduce
```

See [Two released policies](#two-released-policies-and-why-the-paper-reports-gen_best) for how the two
compare and why the paper reports `gen_best`.

### 2. Watch a live rollout

Opens the MuJoCo viewer, runs one deterministic episode, auto-tracks the selected chair, and saves
COM/head and joint position/torque plots.

```bash
# macOS
mjpython g1_sit_viz_v3.py
# Linux
python g1_sit_viz_v3.py
```

The chair lane / start pose is chosen at the top of `g1_sit_viz_v3.py` (`KEYFRAME_CSV`,
`KEYFRAME_ROW`). It is preconfigured to load `models_g1_sit_gen_best/best_model.zip` at `beta = 0.6`.

### 3. Reproduce the COP-locus figure

Regenerates the balance figure from the paper: the center of pressure (COP) starting near the
pelvis–seat contact, transitioning into the foot-only support polygon, and settling at its centroid.

```bash
python g1_sit_cop_locus.py --model-dir models_g1_sit_gen_best --beta 0.6 --lane 0
```

### 4. Inspect training curves (TensorBoard)

```bash
tensorboard --logdir logs_g1_sit_gen_best
```

Includes the reward curves plus the curriculum signals (`curriculum/force_max`, `curriculum/beta_max`)
that decay as the policy masters progressively taller chairs.

## Expected Results

Deterministic, force-free evaluation of `best_model.zip` at `beta = 0.6`, 50 resets per lane
(reproduced from the bundled `eval_full.json`). Chair "z" is the seat-height offset in metres above the
base chair.

| Lane | Chair z (m) | Success (%) | Fall (%) | Reached height (%) | Hold (s) | CP-in-support | Action jitter | S_q | Energy (J) | t_stand (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.000 | 92  | 8 | 100 | 36.1 | 0.994 | 1.53 | 0.953 | 117.6 | 3.91 |
| 1 | 0.010 | 96  | 4 | 100 | 36.3 | 0.995 | 1.52 | 0.951 | 118.0 | 3.70 |
| 2 | 0.020 | 96  | 4 | 100 | 36.4 | 0.995 | 1.52 | 0.954 | 116.7 | 3.60 |
| 3 | 0.030 | 100 | 0 | 100 | 36.5 | 0.994 | 1.54 | 0.961 | 115.9 | 3.54 |
| 4 | 0.040 | 100 | 0 | 100 | 36.6 | 0.994 | 1.54 | 0.962 | 115.7 | 3.42 |
| 5 | 0.050 | 100 | 0 | 100 | 36.7 | 0.994 | 1.53 | 0.967 | 115.4 | 3.31 |
| 6 | 0.075 | 100 | 0 | 100 | 36.7 | 0.992 | 1.53 | 0.977 | 114.7 | 3.32 |
| 7 | 0.100 | 98  | 2 | 100 | 36.4 | 0.987 | 1.57 | 0.980 | 111.2 | 3.59 |
| **All** | — | **97.8** | 2.2 | 100 | 36.5 | 0.993 | 1.535 | 0.963 | 115.6 | 3.55 |

`S_torque = 1.00` on every lane (all joint torques stay within actuator limits throughout).

## Metric Definitions

A trial is a **balanced success** only if, over the final second of the episode, **all** of the following hold,
and the robot never triggers a fall (COM below floor, or trunk roll/pitch beyond limits) before the 40 s timeout:

- **Head height:** head site ≥ `1.2706 m` (96% of the standing head height).
- **Held:** the above is sustained for ≥ `1.0 s` at episode end.
- **Low COM velocity:** horizontal COM speed `< 0.15 m/s`.
- **Upright:** trunk tilt `< 0.4 rad`.
- **Capture point in support:** the capture point lies inside the convex hull of the foot contacts.

Other reported quantities:

- **Reached height:** peak head height crossed the gate at any point (rose up, may not have balanced).
- **Hold (s):** trailing consecutive time with head above the gate.
- **CP-in-support:** fraction of steps with the capture point inside the foot support polygon.
- **Action jitter / DoF jitter:** mean per-step change in action / joint angles.
- **S_torque / S_q:** fraction of steps with all joint torques / all joint angles within limits.
- **Energy (J):** actuation energy over the first 4 s rising window.
- **t_stand (s):** time to first cross the head-height gate.

Success and fall rates use all 50 × 8 = 400 trials; all other means exclude early-terminated trials.

## Citation

```bibtex
@inproceedings{singh2026sts,
  title     = {Natural Sit-to-Stand Motion Synthesis for Humanoids via Guided Assistance Curricula and Staged Rewards},
  author    = {Singh, Meet Pal and Ashtekar, Vyankatesh and Dutta, Ashish},
  booktitle = {International Conference on Robotics, Mechanics and Mechatronics (IPRoMM)},
  year      = {2026}
}
```

## Acknowledgements

- The MuJoCo model and STL meshes are derived from the official
  [Unitree G1](https://github.com/unitreerobotics/unitree_rl_gym) description.
- The assist-force curriculum and action-rescaler design draw on
  [HoST — Learning Humanoid Standing-up Control across Diverse Postures](https://github.com/InternRobotics/HoST) (Huang et al., RSS 2025).
- Compute provided by **PARAM Sanganak** under the National Supercomputing Mission at IIT Kanpur.

## License

Released under the [MIT License](LICENSE).
