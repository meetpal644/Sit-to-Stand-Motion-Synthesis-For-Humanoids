"""
G1 Sit-to-Stand — Evaluation Visualiser v3
==========================================
Run with: mjpython g1_sit_viz_v3.py  (RENDER=True needs a display)

Force-free (or optional assist) evaluation of a trained v3 policy. Matches the
training environment:

  - 97-D obs: proprio + previous action + beta (assist force not observed)
  - Action: ctrl = clip(qpos + a * _BETA * BETA_MAX) with constant beta
  - Start pose from keyframes/ CSV (or XML keyframe fallback); arms zeroed
  - Termination: com_z < 0.35 or trunk roll/pitch beyond limits

Saves sit_v3_eval_com.png, sit_v3_eval_states.png, and left-leg sagittal plots.
"""

import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent

# ================================================================
# CONFIG — edit these before running
# ================================================================
# Start pose: CSV from keyframes/ (full qpos). None -> XML KEYFRAME.
# The seat highlight and camera auto-follow the chosen chair lane.
KEYFRAME_CSV = BASE_DIR / "keyframes" / "chair1_pose_0.1z.csv"
KEYFRAME_ROW = "name"        # int | None (random) | "max" | "name"
KEYFRAME_NAME_TOKENS = ["tx+0.0", "p+0.0", "fx-0.03"]  # used when KEYFRAME_ROW == "name"
KEYFRAME = 7                 # XML fallback when KEYFRAME_CSV is None
MAX_STEPS = 1500             # training episode cap (40 s)
RECORD_STEPS = 450           # steps logged and plotted
SPEED = 1.0
RENDER = True
DETERMINISTIC = True         # True = mean action; False = sampled

MODEL_DIR = BASE_DIR / "models_g1_sit_gen_best"
CKPT_ZIP = MODEL_DIR / "best_model.zip"
CKPT_VN = MODEL_DIR / "vec_normalize_best.pkl"

# BETA_MAX and FORCE_MAX must match the curriculum state of the loaded checkpoint.
# beta is observed and scales the action; force is unobserved but still applied.
BETA_MAX = 0.6   # fully-decayed floor for best_model (see run_meta.json)
FORCE_MAX = 0.0  # assist force at this checkpoint

H_HEAD = 1.2706
TARGET_HEAD_HEIGHT = 1.3236
RISE_STAND_FRAC = 0.8
GATE_WINDOW = 50

# Seat lane geometry (g1_smooth.xml): seat1..seat8 along +y.
SEAT_Y_STEP = 0.5
SEAT_TOP_BASE = 0.2944
SEAT_Z_OFFSETS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1)
_SEAT_DZ = 0.012
VIZ_SEAT_MARGIN = 0.025  # 2.5 cm; env uses 5 mm

VIEWER_GEOM_ALPHA = 1.0
VIEWER_SEAT_ALPHA = 1.0
VIEWER_CAM_AZIMUTH = 100.0
VIEWER_CAM_ELEVATION = -10.0
VIEWER_CAM_DISTANCE = 3.0
# SEAT_GEOM and VIEWER_CAM_LOOKAT are set from the start pose below.
SEAT_GEOM = "seat7"
VIEWER_CAM_LOOKAT = (-0.25, 0.0, SEAT_TOP_BASE)
# ================================================================

FRAME_SKIP = 10
TORSO_ID   = 16
GYRO_ADR   = 0
ACC_ADR    = 3
_G_WORLD   = np.array([0.0, 0.0, -1.0])

TERM_FLOOR = 0.35
ROLL_LIM   = 0.5
PITCH_LIM  = 1.0

_BETA = np.array([
    0.20, 0.04, 0.04, 0.25, 0.12, 0.03,
    0.20, 0.04, 0.04, 0.25, 0.12, 0.03,
    0.03, 0.03, 0.03,
    0.04, 0.03, 0.03, 0.15, 0.02, 0.02, 0.02,
    0.04, 0.03, 0.03, 0.15, 0.02, 0.02, 0.02,
], dtype=np.float64)

JOINT_LOWER = np.array([
    -2.5307, -0.5236, -2.7576, -0.0873, -0.8727, -0.2618,
    -2.5307, -2.9671, -2.7576, -0.0873, -0.8727, -0.2618,
    -2.618,  -0.52,   -0.52,
    -2.87, -0.34, -1.30, -1.25, -1.97, -0.52, -0.43,
    -2.87, -3.11, -1.30, -1.25, -1.97, -0.52, -0.43,
], dtype=np.float64)
JOINT_UPPER = np.array([
    2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
    2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
    2.618,  0.52,   0.52,
    2.87, 3.11, 1.30, 2.61, 1.97, 0.52, 0.43,
    2.87, 0.34, 1.30, 2.61, 1.97, 0.52, 0.43,
], dtype=np.float64)

LEG_JOINT_NAMES = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
WAIST_JOINT_NAMES = ["waist_yaw", "waist_roll", "waist_pitch"]
LEFT_SAGITTAL_IDXS = (0, 3, 4)  # hip_pitch, knee, ankle_pitch
LEFT_SAGITTAL_NAMES = [LEG_JOINT_NAMES[i] for i in LEFT_SAGITTAL_IDXS]
LEFT_SAGITTAL_COLORS = ["tab:blue", "tab:orange", "tab:green"]
SAG_PLOT_WINDOW_S = 5.0
SAG_LINE_WIDTH = 2.8
SAG_LABEL_SIZE = 20
SAG_TICK_SIZE = 12
SAG_LEGEND_SIZE = 17
SAG_TITLE_SIZE = 5
SAG_FIG_WIDTH = 13.0
SAG_FIG_HEIGHT = 6.5


def load_keyframes_csv(path):
    rows = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            qpos = np.array([float(row[f"qpos_{i}"]) for i in range(36)], dtype=np.float64)
            rows.append((row["name"], qpos))
    if not rows:
        raise ValueError(f"No keyframes found in {path}")
    return rows


_CSV_KEYFRAMES = load_keyframes_csv(KEYFRAME_CSV) if KEYFRAME_CSV else None


def most_perturbed_row(keyframes):
    """Row index with the largest leg+waist deviation from the pool mean (qpos 7:22)."""
    J   = np.array([q[7:22] for _, q in keyframes])      # (N, 15) leg+waist joints
    dev = np.linalg.norm(J - J.mean(axis=0), axis=1)      # per-row L2 deviation
    return int(np.argmax(dev)), float(dev.max())


def find_row_by_tokens(keyframes, tokens):
    """First row whose name contains every token in `tokens`."""
    for i, (name, _) in enumerate(keyframes):
        if all(t in name for t in tokens):
            return i
    raise ValueError(
        f"no pool row name contains all of {tokens} in {KEYFRAME_CSV.name}"
    )


class _DummyEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(97,), dtype=np.float64)
    action_space      = gym.spaces.Box(-1.0, 1.0, shape=(29,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(97, np.float32), {}

    def step(self, action):
        return np.zeros(97, np.float32), 0.0, False, False, {}


def rise_frac(head_z, head_seated_ep):
    denom = max(TARGET_HEAD_HEIGHT - head_seated_ep, 1e-3)
    return (head_z - head_seated_ep) / denom


def head_at_rise_frac(head_seated_ep, frac):
    return head_seated_ep + frac * max(TARGET_HEAD_HEIGHT - head_seated_ep, 1e-3)


def seat_top_at_reset(data, butt_sid):
    """Seat-top height reference at episode reset."""
    butt_z = float(data.site_xpos[butt_sid][2])
    return butt_z - _SEAT_DZ


def seat_contact_viz(butt_z, seat_top):
    """Seat contact check using the viz margin (2.5 cm)."""
    return butt_z <= seat_top + VIZ_SEAT_MARGIN


def get_obs(data, last_action, beta_now):
    """97-D observation vector (must match training env)."""
    g_body  = data.xmat[TORSO_ID].reshape(3, 3).T @ _G_WORLD
    ang_vel = data.sensordata[GYRO_ADR:GYRO_ADR + 3]
    lin_acc = data.sensordata[ACC_ADR:ACC_ADR + 3]
    return np.concatenate([
        g_body, ang_vel, lin_acc,
        data.qpos[7:36], data.qvel[6:35],
        np.asarray(last_action, dtype=np.float64),
        np.array([beta_now], dtype=np.float64),
    ]).astype(np.float32)


def pelvis_roll_pitch(data):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.qpos[3:7])
    g_p = R.reshape(3, 3).T @ _G_WORLD
    roll  = math.asin(float(np.clip(-g_p[1], -1.0, 1.0)))
    pitch = math.asin(float(np.clip( g_p[0], -1.0, 1.0)))
    return abs(roll), abs(pitch)


def term_reason(data):
    if float(data.subtree_com[0][2]) < TERM_FLOOR:
        return "floor"
    roll, pitch = pelvis_roll_pitch(data)
    if roll  > ROLL_LIM:
        return "roll"
    if pitch > PITCH_LIM:
        return "pitch"
    return "none"


def reset_episode(model, data, *, qpos=None, kf=None):
    if qpos is not None:
        data.qpos[:] = qpos
    else:
        mujoco.mj_resetDataKeyframe(model, data, KEYFRAME if kf is None else kf)
    data.qpos[22:36] = 0.0
    data.qvel[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def pick_start_pose():
    if _CSV_KEYFRAMES is not None:
        if KEYFRAME_ROW is None:
            idx = random.randrange(len(_CSV_KEYFRAMES))
            name, qpos = _CSV_KEYFRAMES[idx]
        elif KEYFRAME_ROW == "max":
            idx, dev = most_perturbed_row(_CSV_KEYFRAMES)
            name, qpos = _CSV_KEYFRAMES[idx]
            name = f"{name} [MOST-PERTURBED row {idx}, dev={dev:.3f} rad]"
        elif KEYFRAME_ROW == "name":
            idx = find_row_by_tokens(_CSV_KEYFRAMES, KEYFRAME_NAME_TOKENS)
            name, qpos = _CSV_KEYFRAMES[idx]
            name = f"{name} [matched {KEYFRAME_NAME_TOKENS}, row {idx}]"
        else:
            idx = KEYFRAME_ROW
            name, qpos = _CSV_KEYFRAMES[idx]
        return name, qpos.copy(), None
    return f"XML KF{KEYFRAME}", None, KEYFRAME


def seat_and_lookat(start_qpos):
    """Derive the seat lane (highlight geom) + camera lookat from the start pose.
    The non-0z chair CSVs bake a +y offset into qpos_1 so each lands on its seat
    lane (seat1..seat8); this keeps the robot centred whichever chair is chosen."""
    y = float(start_qpos[1]) if start_qpos is not None else 0.0
    lane = max(0, min(len(SEAT_Z_OFFSETS) - 1, int(round(y / SEAT_Y_STEP))))
    seat_top = SEAT_TOP_BASE + SEAT_Z_OFFSETS[lane]
    return f"seat{lane + 1}", (-0.26, y, seat_top)


def configure_viewer(viewer, model):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = VIEWER_CAM_LOOKAT
    viewer.cam.distance  = VIEWER_CAM_DISTANCE
    viewer.cam.azimuth   = VIEWER_CAM_AZIMUTH
    viewer.cam.elevation = VIEWER_CAM_ELEVATION

    flags = viewer.opt.flags
    flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    flags[mujoco.mjtVisFlag.mjVIS_COM] = True

    ground_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    seat_gid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, SEAT_GEOM)
    for gid in range(model.ngeom):
        if gid == ground_gid:
            continue
        if gid == seat_gid:
            model.geom_rgba[gid, 3] = VIEWER_SEAT_ALPHA
        else:
            model.geom_rgba[gid, 3] = VIEWER_GEOM_ALPHA


# ----------------------------------------------------------------
# Load policy + normaliser + model
# ----------------------------------------------------------------
if not CKPT_VN.exists():
    CKPT_VN = MODEL_DIR / "vec_normalize_final.pkl"

policy = PPO.load(
    str(CKPT_ZIP),
    custom_objects={"learning_rate": 1e-4, "lr_schedule": lambda _: 1e-4},
)
vec_norm = VecNormalize.load(str(CKPT_VN), venv=DummyVecEnv([_DummyEnv]))
vec_norm.training = False
vec_norm.norm_reward = False

mj_model = mujoco.MjModel.from_xml_path(str(BASE_DIR / "g1_smooth.xml"))
mj_data  = mujoco.MjData(mj_model)
dt       = mj_model.opt.timestep * FRAME_SKIP
PELVIS_ID = mj_model.body("pelvis").id
HEAD_SID  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "head")
BUTT_SID  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "cop_pelvis")


def plot_left_sagittal_panel(
    t_axis, data, ylab, title, out_path, *, t_stand_onset=None, t_lift_off=None,
):
    """Single left-leg sagittal panel (first SAG_PLOT_WINDOW_S seconds), 1:2 aspect."""
    fig, ax = plt.subplots(figsize=(SAG_FIG_WIDTH, SAG_FIG_HEIGHT))
    for k, j in enumerate(LEFT_SAGITTAL_IDXS):
        ax.plot(
            t_axis, data[:, j],
            lw=SAG_LINE_WIDTH, color=LEFT_SAGITTAL_COLORS[k],
            label=LEFT_SAGITTAL_NAMES[k],
        )
    ax.axhline(0.0, color="0.55", lw=0.8)
    if t_lift_off is not None and t_lift_off <= SAG_PLOT_WINDOW_S:
        ax.axvline(
            t_lift_off, color="r", ls="--", lw=1.4,
            label="seat lift-off",
        )
    if t_stand_onset is not None and t_stand_onset <= SAG_PLOT_WINDOW_S:
        ax.axvline(
            t_stand_onset, color="g", ls="--", lw=1.4,
            label="standing onset",
        )
    ax.set_xlim(0.0, SAG_PLOT_WINDOW_S)
    ax.set_xlabel("time (s)", fontsize=SAG_LABEL_SIZE)
    ax.set_ylabel(ylab, fontsize=SAG_LABEL_SIZE)
    ax.set_title(title, fontsize=SAG_TITLE_SIZE, pad=10)
    ax.tick_params(axis="both", labelsize=SAG_TICK_SIZE)
    ax.grid(True, lw=0.45, alpha=0.65)
    ax.legend(fontsize=SAG_LEGEND_SIZE, loc="lower right", framealpha=0.92)
    ax.set_box_aspect(SAG_FIG_HEIGHT / SAG_FIG_WIDTH)  # 1:2 width:height
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_episode(viewer, start_qpos, start_kf):
    reset_episode(mj_model, mj_data, qpos=start_qpos, kf=start_kf)
    head_seated_ep = float(mj_data.site_xpos[HEAD_SID][2])
    head_stand_entry = head_at_rise_frac(head_seated_ep, RISE_STAND_FRAC)
    seat_top = seat_top_at_reset(mj_data, BUTT_SID)

    prev_action = np.zeros(29, dtype=np.float64)
    com_trace, head_trace, head_trace_full = [], [], []
    rise_trace = []
    pos_log, vel_log, tau_log = [], [], []
    stand_steps = 0
    stood = False
    lift_off_step = None

    for step in range(MAX_STEPS):
        action, _ = policy.predict(
            vec_norm.normalize_obs(
                get_obs(mj_data, prev_action, BETA_MAX).reshape(1, -1)
            ),
            deterministic=DETERMINISTIC,
        )
        action = np.clip(action.flatten(), -1.0, 1.0)

        mj_data.xfrc_applied[PELVIS_ID, 2] = FORCE_MAX if FORCE_MAX > 0.0 else 0.0
        mj_data.ctrl[:29] = np.clip(
            mj_data.qpos[7:36] + action * _BETA * BETA_MAX,
            JOINT_LOWER, JOINT_UPPER,
        )
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(mj_model, mj_data)
        if viewer is not None:
            viewer.sync()
            time.sleep(dt / SPEED)

        z = float(mj_data.subtree_com[0][2])
        head_z = float(mj_data.site_xpos[HEAD_SID][2])
        butt_z = float(mj_data.site_xpos[BUTT_SID][2])
        if lift_off_step is None and not seat_contact_viz(butt_z, seat_top):
            lift_off_step = step
        rf = rise_frac(head_z, head_seated_ep)
        head_trace_full.append(head_z)

        if rf >= RISE_STAND_FRAC:
            stand_steps += 1
            stood = True

        if step < RECORD_STEPS:
            com_trace.append(z)
            head_trace.append(head_z)
            rise_trace.append(rf)
            pos_log.append(mj_data.qpos[7:22].copy())
            vel_log.append(mj_data.qvel[6:21].copy())
            tau_log.append(mj_data.actuator_force[:15].copy())

        reason = term_reason(mj_data)
        if reason != "none":
            n = step + 1
            return (n, reason, stood, stand_steps / n, head_seated_ep, head_stand_entry,
                    lift_off_step, com_trace, head_trace, rise_trace, head_trace_full,
                    np.array(pos_log), np.array(vel_log), np.array(tau_log))

        prev_action = action.astype(np.float64)

    return (MAX_STEPS, "timeout", stood, stand_steps / MAX_STEPS, head_seated_ep,
            head_stand_entry, lift_off_step, com_trace, head_trace, rise_trace,
            head_trace_full, np.array(pos_log), np.array(vel_log), np.array(tau_log))


# ----------------------------------------------------------------
# Run ONE episode
# ----------------------------------------------------------------
start_label, start_qpos, start_kf = pick_start_pose()
SEAT_GEOM, VIEWER_CAM_LOOKAT = seat_and_lookat(start_qpos)
if _CSV_KEYFRAMES is not None:
    print(f"Loaded {len(_CSV_KEYFRAMES)} poses from {KEYFRAME_CSV.name}")
print(f"\nEvaluating v3  start={start_label}  seat={SEAT_GEOM}  force={FORCE_MAX}N  "
      f"beta={BETA_MAX}  cap={MAX_STEPS}  det={DETERMINISTIC}")

if RENDER:
    with mujoco.viewer.launch_passive(mj_model, mj_data, show_left_ui = True, show_right_ui = False) as viewer:
        configure_viewer(viewer, mj_model)
        result = run_episode(viewer, start_qpos, start_kf)
else:
    result = run_episode(None, start_qpos, start_kf)

(steps, reason, stood, sfrac, head_seated_ep, head_stand_entry, lift_off_step,
 com_trace, head_trace, rise_trace, head_trace_full,
 pos_log, vel_log, tau_log) = result

gate_mean = float(np.mean(head_trace_full[-GATE_WINDOW:]))
gate_pass = gate_mean >= H_HEAD

print("\n=== EPISODE SUMMARY ===")
print(f"survived       : {steps} / {MAX_STEPS} steps  ({steps * dt:.1f}s / {MAX_STEPS * dt:.0f}s)")
print(f"termination     : {reason}")
print(f"seated head h0  : {head_seated_ep:.3f} m")
print(f"standing entry  : rise_frac>={RISE_STAND_FRAC}  (~{head_stand_entry:.3f} m head)")
if lift_off_step is not None:
    print(f"seat lift-off   : step {lift_off_step}  ({lift_off_step * dt:.2f} s)  "
          f"(butt_z > seat_top + {VIZ_SEAT_MARGIN * 1000:.0f} mm)")
else:
    print(f"seat lift-off   : not detected (butt stayed within seat margin)")
print(f"reached standing: {stood}   stand% : {100 * sfrac:.1f}%")
print(f"peak COM z       : {max(com_trace) if com_trace else float('nan'):.3f} m")
print(f"peak head z      : {max(head_trace_full):.3f} m")
print(f"end-window mean head z (last {GATE_WINDOW} steps) : {gate_mean:.3f} m")
print(f"HoST gate (>= {H_HEAD}) : {'PASS' if gate_pass else 'FAIL'}")

# ----------------------------------------------------------------
# Plots
# ----------------------------------------------------------------
t = np.arange(len(com_trace)) * dt
stand_idx = next((i for i, rf in enumerate(rise_trace) if rf >= RISE_STAND_FRAC), None)
t_stand = stand_idx * dt if stand_idx is not None else None
t_lift = lift_off_step * dt if lift_off_step is not None else None

leg_colors   = plt.cm.tab10(np.arange(6))
waist_colors = plt.cm.tab10(np.arange(6, 9))

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t, com_trace, lw=1.3, color="tab:blue", label="COM z")
ax.plot(t, head_trace, lw=1.3, color="tab:orange", label="head z")
ax.axhline(head_seated_ep, color="gray", ls=":", lw=0.8, label=f"h0={head_seated_ep:.3f}")
ax.axhline(head_stand_entry, color="k", ls=":", lw=0.8,
           label=f"stand entry ({RISE_STAND_FRAC:.0%})={head_stand_entry:.3f}")
ax.axhline(TERM_FLOOR, color="r", ls=":", lw=0.8, label=f"floor={TERM_FLOOR}")
ax.axhline(H_HEAD, color="purple", ls="--", lw=0.9, label=f"H_head gate={H_HEAD}")
ax.axhline(TARGET_HEAD_HEIGHT, color="green", ls="--", lw=0.9,
           label=f"target head={TARGET_HEAD_HEIGHT}")
if t_lift is not None:
    ax.axvline(t_lift, color="r", ls="--", lw=0.9, label=f"lift-off @ {t_lift:.1f}s")
if t_stand is not None:
    ax.axvline(t_stand, color="g", ls="--", lw=0.9, label=f"standing @ {t_stand:.1f}s")
ax.set_xlabel("time (s)")
ax.set_ylabel("height z (m)")
ax.set_title(
    f"G1 sit-to-stand eval v3 — COM + head  ({start_label}, F={FORCE_MAX}N, "
    f"beta={BETA_MAX}, survived {steps * dt:.1f}s, term={reason}; first {len(t)} steps)"
)
ax.grid(True, lw=0.4)
ax.legend(fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig(BASE_DIR / "sit_v3_eval_com.png", dpi=150)

groups = [
    ("Left leg",  range(0, 6),   LEG_JOINT_NAMES,   leg_colors),
    ("Right leg", range(6, 12),  LEG_JOINT_NAMES,   leg_colors),
    ("Waist",     range(12, 15), WAIST_JOINT_NAMES, waist_colors),
]
rows = [(pos_log, "position (rad)"), (vel_log, "velocity (rad/s)"), (tau_log, "torque (N·m)")]

fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
for r, (data, ylab) in enumerate(rows):
    for c, (gname, idxs, names, colors) in enumerate(groups):
        ax = axes[r, c]
        for k, j in enumerate(idxs):
            ax.plot(t, data[:, j], lw=1.0, color=colors[k], label=names[k])
        ax.axhline(0.0, color="0.6", lw=0.6)
        if t_lift is not None:
            ax.axvline(t_lift, color="r", ls="--", lw=0.8)
        if t_stand is not None:
            ax.axvline(t_stand, color="g", ls="--", lw=0.8)
        ax.grid(True, lw=0.3)
        if r == 0:
            ax.set_title(gname, fontsize=12, fontweight="bold")
        if r == 0 or c == 2:
            ax.legend(fontsize=7, ncol=2, loc="best")
        if c == 0:
            ax.set_ylabel(ylab, fontsize=11)
        if r == 2:
            ax.set_xlabel("time (s)", fontsize=10)

for r in range(3):
    lo = min(axes[r, 0].get_ylim()[0], axes[r, 1].get_ylim()[0])
    hi = max(axes[r, 0].get_ylim()[1], axes[r, 1].get_ylim()[1])
    axes[r, 0].set_ylim(lo, hi)
    axes[r, 1].set_ylim(lo, hi)

fig.suptitle(
    f"G1 sit-to-stand eval v3 — leg + waist  ({start_label}, F={FORCE_MAX}N, "
    f"beta={BETA_MAX}, survived {steps * dt:.1f}s, term={reason}; first {len(t)} steps)\n"
    f"green dashed = rise_frac>={RISE_STAND_FRAC} onset; red dashed = seat lift-off",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = BASE_DIR / "sit_v3_eval_states.png"
fig.savefig(out, dpi=150)

# ---- left leg sagittal only (hip_pitch, knee, ankle_pitch), first 5 s ----
n_sag = min(int(round(SAG_PLOT_WINDOW_S / dt)), len(t))
t_sag = t[:n_sag]
pos_sag = pos_log[:n_sag]
tau_sag = tau_log[:n_sag]
sag_title_base = (
    f"G1 sit-to-stand — left leg sagittal  ({start_label}, β={BETA_MAX}, "
    f"first {SAG_PLOT_WINDOW_S:.0f} s)"
)
out_sag_pos = BASE_DIR / "sit_v3_eval_left_sagittal_pos.png"
out_sag_tau = BASE_DIR / "sit_v3_eval_left_sagittal_tau.png"
plot_left_sagittal_panel(
    t_sag, pos_sag, "joint position (rad)",
    f"{sag_title_base}\nhip_pitch · knee · ankle_pitch",
    out_sag_pos, t_stand_onset=t_stand, t_lift_off=t_lift,
)
plot_left_sagittal_panel(
    t_sag, tau_sag, "joint torque (N·m)",
    f"{sag_title_base}\nhip_pitch · knee · ankle_pitch",
    out_sag_tau, t_stand_onset=t_stand, t_lift_off=t_lift,
)

print(
    f"\nPlots saved → sit_v3_eval_com.png  &  {out.name}  &  "
    f"{out_sag_pos.name}  &  {out_sag_tau.name}"
)
