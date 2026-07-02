"""
G1 Sit-to-Stand — Shared Ablation Eval Harness  (headless, no training)
=======================================================================
Scores ANY trained checkpoint (full model or an ablation) with the IDENTICAL
protocol so every run is directly comparable. Replicates the v3 physics/obs
exactly (same _BETA, delta action, termination, COP), auto-detecting 97-D vs
67-D obs from the loaded policy (so it works for the obs ablation too).

For each chair lane (0=0z .. 7=0.1z) it runs N_RESETS deterministic
episodes and computes the four metric groups from ablation.md section 3:

  1. Balanced-stand success (primary): head >= GATE, held >= HOLD_S seconds,
     |v_COM_xy| < V_MAX, |tilt| < TILT_MAX, AND capture-point inside the foot hull.
  2. Generalization split: per-lane success, partitioned TRAINED vs HELD-OUT.
  3. Balance: capture-point-in-support fraction, COP-to-hull-centroid margin,
     deterministic fall rate, post-stand hold duration.
  4. Smoothness/safety: action jitter, DoF jitter, S_torque, S_DoF,
     first-4s rising energy, time-to-stand.

Except for balanced-stand success and fall rate, aggregate metric means/stds
are computed over non-terminated episodes only (`term=="timeout"`).

Writes <MODEL_DIR>/eval_<tag>.json (per-lane + aggregate, deterministic only).

Usage:
  python g1_sit_eval.py --model-dir models_g1_sit_v3_abl_full_s0 --ckpt best_model \
      --beta 0.6 --n-resets 20 --holdout 3 5
  (RENDER is intentionally off — this is a batch scorer; use g1_sit_viz_v3.py to watch.)
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

BASE_DIR = Path(__file__).parent

# ---- physics / model constants (match g1_sit_env_v3) ----
FRAME_SKIP = 10
TORSO_ID   = 16
LFOOT_ID   = 7
RFOOT_ID   = 13
GYRO_ADR   = 0
ACC_ADR    = 3
_G_WORLD   = np.array([0.0, 0.0, -1.0])
GRAV       = 9.81

TARGET_HEAD_HEIGHT = 1.3236
GATE     = 1.2706          # balanced-stand head-height gate
TERM_FLOOR = 0.35
ROLL_LIM   = 0.5
PITCH_LIM  = 1.0

# ---- balanced-stand success thresholds (ablation.md section 3) ----
HOLD_S    = 1.0            # must hold the stand this long at episode end
V_MAX     = 0.15          # |v_COM_xy| ceiling (m/s)
TILT_MAX  = 0.4           # |tilt| ceiling (rad)
ENERGY_WINDOW_S = 4.0      # rising-phase energy window; 4.0s == 200 eval steps at dt=0.02

FOOT_HALF_LEN = 0.10
FOOT_HALF_WID = 0.04

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

LANE_FILES = [
    "chair1_pose_0z.csv", "chair1_pose_0.01z.csv", "chair1_pose_0.02z.csv",
    "chair1_pose_0.03z.csv", "chair1_pose_0.04z.csv", "chair1_pose_0.05z.csv",
    "chair1_pose_0.075z.csv", "chair1_pose_0.1z.csv",
]


def load_pool(path):
    rows = []
    with Path(path).open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append(np.array([float(r[f"qpos_{i}"]) for i in range(36)], dtype=np.float64))
    return np.asarray(rows, dtype=np.float64)


def require_mj_id(model, obj_type, name):
    mid = mujoco.mj_name2id(model, obj_type, name)
    if mid < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return mid


def make_dummy_env(obs_dim):
    class _DummyEnv(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float64)
        action_space      = gym.spaces.Box(-1.0, 1.0, shape=(29,), dtype=np.float32)
        def reset(self, *, seed=None, options=None): return np.zeros(obs_dim, np.float32), {}
        def step(self, a): return np.zeros(obs_dim, np.float32), 0.0, False, False, {}
    return _DummyEnv


def get_obs(data, model, ids, last_action, beta_now, obs_dim):
    """97-D (full) or 67-D (obs ablation) — matches g1_sit_env_v3_abl._get_obs."""
    g_body  = data.xmat[TORSO_ID].reshape(3, 3).T @ _G_WORLD
    ang_vel = data.sensordata[GYRO_ADR:GYRO_ADR + 3]
    lin_acc = data.sensordata[ACC_ADR:ACC_ADR + 3]
    parts = [g_body, ang_vel, lin_acc, data.qpos[7:36], data.qvel[6:35]]
    if obs_dim == 97:
        parts.append(np.asarray(last_action, dtype=np.float64))
        parts.append(np.array([beta_now], dtype=np.float64))
    return np.concatenate(parts).astype(np.float32)


def pelvis_tilt(data):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.qpos[3:7])
    g_p = R.reshape(3, 3).T @ _G_WORLD
    roll  = abs(math.asin(float(np.clip(-g_p[1], -1.0, 1.0))))
    pitch = abs(math.asin(float(np.clip( g_p[0], -1.0, 1.0))))
    return roll, pitch


def term_reason(data):
    if float(data.subtree_com[0][2]) < TERM_FLOOR:
        return "floor"
    roll, pitch = pelvis_tilt(data)
    if roll  > ROLL_LIM:  return "roll"
    if pitch > PITCH_LIM: return "pitch"
    return "none"


def foot_corners(data):
    corners = []
    for fid in (LFOOT_ID, RFOOT_ID):
        c  = data.xpos[fid][:2]
        R2 = data.xmat[fid].reshape(3, 3)[:2, :2]
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                corners.append(c + R2 @ np.array([sx * FOOT_HALF_LEN, sy * FOOT_HALF_WID]))
    return corners


def convex_hull_xy(pts):
    uniq = sorted({(round(float(x), 6), round(float(y), 6)) for x, y in pts})
    if len(uniq) < 3:
        return uniq
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for q in uniq:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper = []
    for q in reversed(uniq):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def point_in_hull(pt, hull):
    """True if pt is inside the CCW convex hull (vertices ordered)."""
    if len(hull) < 3:
        return False
    px, py = float(pt[0]), float(pt[1])
    n = len(hull)
    for i in range(n):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % n]
        # CCW hull: inside means left of every edge (cross >= 0)
        if (bx - ax) * (py - ay) - (by - ay) * (px - ax) < -1e-9:
            return False
    return True


def signed_margin_to_hull(pt, hull):
    """SIGNED distance (m) from pt to the support-polygon boundary: POSITIVE when
    pt is inside the CCW convex hull (= clearance to the nearest edge), NEGATIVE
    when outside. Unlike a distance-to-centroid, this can go negative, so it
    detects the COP/point leaving the support polygon. Returns nan for a
    degenerate hull (<3 verts)."""
    if len(hull) < 3:
        return float("nan")
    px, py = float(pt[0]), float(pt[1])
    n = len(hull)
    m = float("inf")
    for i in range(n):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        L = math.hypot(ex, ey)
        if L < 1e-9:
            continue
        # left-perp signed distance to edge (CCW): >0 on the inner side
        d = (ex * (py - ay) - ey * (px - ax)) / L
        m = min(m, d)
    return m


def cop_xy(data, cop_adr, cop_sites):
    f = np.asarray(data.sensordata[cop_adr], dtype=np.float64)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    f = np.maximum(f, 0.0)
    tot = float(f.sum())
    if tot < 1e-6:
        return 0.5 * (data.xpos[LFOOT_ID][:2] + data.xpos[RFOOT_ID][:2])
    pos = data.site_xpos[cop_sites][:, :2]
    return (f[:, None] * pos).sum(axis=0) / tot


def _energy_window_steps(dt):
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return max(1, int(round(ENERGY_WINDOW_S / dt)))


def run_episode(mj_model, mj_data, policy, vec_norm, ids, start_qpos,
                beta_max, obs_dim, deterministic, max_steps, dt):
    """One headless episode; returns a metrics dict."""
    mj_data.qpos[:] = start_qpos
    mj_data.qpos[22:36] = 0.0          # arms neutral (matches env reset)
    mj_data.qvel[:] = 0.0
    mj_data.ctrl[:] = 0.0
    mj_data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    head_sid, cop_adr, cop_sites = ids["head"], ids["cop_adr"], ids["cop_sites"]
    tau_max = ids["tau_max"]

    prev_action = np.zeros(29)
    prev_com_xy = mj_data.subtree_com[0][:2].copy()
    prev_q = mj_data.qpos[7:36].copy()

    act_jitter = dof_jitter = energy = 0.0
    tau_ok = dof_ok = 0
    cp_in_count = 0
    cop_margin_sum = 0.0
    time_to_stand = None
    hold_window = max(1, int(round(HOLD_S / dt)))
    energy_window_steps = _energy_window_steps(dt)
    head_hist, vcom_hist, tilt_hist, cp_in_hist = [], [], [], []
    reason = "timeout"
    n = max_steps

    for step in range(max_steps):
        obs = get_obs(mj_data, mj_model, ids, prev_action, beta_max, obs_dim)
        action, _ = policy.predict(vec_norm.normalize_obs(obs.reshape(1, -1)),
                                   deterministic=deterministic)
        action = np.clip(action.flatten(), -1.0, 1.0)
        mj_data.xfrc_applied[ids["pelvis"], 2] = 0.0       # force-free eval
        mj_data.ctrl[:29] = np.clip(mj_data.qpos[7:36] + action * _BETA * beta_max,
                                    JOINT_LOWER, JOINT_UPPER)
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(mj_model, mj_data)

        # ---- measurements ----
        com   = mj_data.subtree_com[0]
        z_com = float(com[2])
        com_vel_xy = (com[:2] - prev_com_xy) / dt
        prev_com_xy = com[:2].copy()
        head_z = float(mj_data.site_xpos[head_sid][2])
        roll, pitch = pelvis_tilt(mj_data)
        tilt = max(roll, pitch)
        vcom = float(np.linalg.norm(com_vel_xy))

        hull = convex_hull_xy(foot_corners(mj_data))
        omega = math.sqrt(GRAV / max(z_com, 1e-3))
        cp = com[:2] + com_vel_xy / omega
        cp_in = point_in_hull(cp, hull)
        cp_in_count += int(cp_in)
        cop = cop_xy(mj_data, cop_adr, cop_sites)
        if len(hull) >= 3:
            # signed margin: + inside the support hull, - outside (see ablation feedback #2)
            cop_margin_sum += signed_margin_to_hull(cop, hull)

        # ---- smoothness / safety accumulation ----
        dq = mj_data.qpos[7:36] - prev_q
        prev_q = mj_data.qpos[7:36].copy()
        act_jitter += float(np.sum(np.abs(action - prev_action)))
        dof_jitter += float(np.sum(np.abs(dq)))
        tau = mj_data.actuator_force[:29]
        qd  = mj_data.qvel[6:35]
        tau_ok += int(np.all(np.abs(tau) <= tau_max + 1e-6))
        q29 = mj_data.qpos[7:36]
        dof_ok += int(np.all((q29 >= JOINT_LOWER - 1e-6) & (q29 <= JOINT_UPPER + 1e-6)))
        if step < energy_window_steps:
            energy += float(np.sum(np.abs(tau * qd))) * dt

        if head_z >= GATE and time_to_stand is None:
            time_to_stand = (step + 1) * dt

        head_hist.append(head_z); vcom_hist.append(vcom)
        tilt_hist.append(tilt);   cp_in_hist.append(cp_in)

        prev_action = action
        r = term_reason(mj_data)
        if r != "none":
            reason, n = r, step + 1
            break

    # ---- end-of-episode balanced-stand test (last hold_window steps) ----
    w = min(hold_window, len(head_hist))
    held_head = np.array(head_hist[-w:]) >= GATE
    held_v    = np.array(vcom_hist[-w:]) < V_MAX
    held_tilt = np.array(tilt_hist[-w:]) < TILT_MAX
    held_cp   = np.array(cp_in_hist[-w:], dtype=bool)
    balanced  = bool(np.all(held_head) and np.all(held_v) and
                     np.all(held_tilt) and np.all(held_cp)) and reason == "timeout"
    # post-stand hold duration: trailing consecutive steps with head>=gate
    hold = 0
    for h in reversed(head_hist):
        if h >= GATE: hold += 1
        else: break

    return {
        "balanced_success": balanced,
        "reached_height": bool(max(head_hist) >= GATE),
        "fell": reason in ("floor", "roll", "pitch"),
        "term": reason,
        "survived_steps": n,
        "peak_head": float(max(head_hist)),
        "hold_duration_s": hold * dt,
        "time_to_stand_s": (time_to_stand if time_to_stand is not None else float("nan")),
        "cp_in_support_frac": cp_in_count / n,
        "cop_boundary_margin": cop_margin_sum / n,   # signed; + inside support hull
        "action_jitter": act_jitter / n,
        "dof_jitter": dof_jitter / n,
        "S_torque": tau_ok / n,
        "S_dof": dof_ok / n,
        "energy": energy,
    }


def aggregate(trials):
    all_trial_keys = ["balanced_success", "fell"]
    nonterminated_keys = ["reached_height", "survived_steps", "peak_head", "hold_duration_s",
                          "cp_in_support_frac", "cop_boundary_margin", "action_jitter",
                          "dof_jitter", "S_torque", "S_dof", "energy"]
    out = {}
    for k in all_trial_keys:
        vals = np.array([t[k] for t in trials], dtype=np.float64)
        out[k + "_mean"] = float(np.mean(vals))
        out[k + "_std"]  = float(np.std(vals))

    nonterminated = [t for t in trials if t.get("term") == "timeout"]
    out["nonterminated_n"] = int(len(nonterminated))
    for k in nonterminated_keys:
        vals = np.array([t[k] for t in nonterminated], dtype=np.float64)
        out[k + "_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
        out[k + "_std"]  = float(np.std(vals)) if len(vals) else float("nan")
    out["energy_n"] = out["nonterminated_n"]

    tts = np.array([t["time_to_stand_s"] for t in nonterminated
                    if not math.isnan(t["time_to_stand_s"])])
    out["time_to_stand_s_mean"] = float(np.mean(tts)) if len(tts) else float("nan")
    out["n_trials"] = len(trials)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="dir with the checkpoint + vecnormalize")
    p.add_argument("--ckpt", default="best_model", help="checkpoint stem (no .zip)")
    p.add_argument("--vecnorm", default=None, help="vecnormalize pkl stem; default pairs with ckpt")
    p.add_argument("--beta", type=float, default=None,
                   help="BETA_MAX matching the ckpt's decayed curriculum floor. If omitted, "
                        "read beta_max from <model-dir>/run_meta.json (dumped by the trainer). "
                        "Wrong beta = wrong action scale + OOD obs, so this MUST match the ckpt.")
    p.add_argument("--n-resets", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--holdout", type=int, nargs="*", default=None,
                   help="held-out chair lanes for trained/heldout split. If omitted, "
                        "use run_meta.json holdout_lanes when present, else [3,5]. "
                        "Pass --holdout alone for no held-out lanes.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None, help="label for the output json")
    args = p.parse_args()

    mdir = (BASE_DIR / args.model_dir) if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    ckpt = mdir / f"{args.ckpt}.zip"
    if args.vecnorm:
        vn = mdir / f"{args.vecnorm}.pkl"
    else:
        vn = mdir / ("vec_normalize_best.pkl" if args.ckpt == "best_model"
                     else "vec_normalize_final.pkl")
        if not vn.exists():
            vn = mdir / "vec_normalize_final.pkl"
    tag = args.tag or mdir.name

    if args.n_resets <= 0:
        raise SystemExit("--n-resets must be positive")
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    if not vn.exists():
        raise SystemExit(f"VecNormalize file not found: {vn}")

    meta_path = mdir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # ---- resolve BETA_MAX (feedback #1): CLI overrides; else read the trainer's
    #      run_meta.json so eval always uses the ckpt's decayed curriculum floor. ----
    if args.beta is not None:
        beta = float(args.beta); beta_src = "cli"
    elif meta:
        beta = float(meta["beta_max"]); beta_src = "run_meta.json"
    else:
        raise SystemExit(
            f"no --beta given and no {meta_path.name} in {mdir.name}; pass --beta explicitly "
            f"(e.g. 0.6 for a fully-decayed ablation run) — wrong beta invalidates the eval.")

    train_holdout = meta.get("holdout_lanes")
    if args.holdout is None:
        if train_holdout is not None:
            holdout_lanes = [int(x) for x in train_holdout]
            holdout_src = "run_meta.json"
        else:
            holdout_lanes = [3, 5]
            holdout_src = "default"
    else:
        holdout_lanes = [int(x) for x in args.holdout]
        holdout_src = "cli"
        if train_holdout is not None and sorted(holdout_lanes) != sorted(int(x) for x in train_holdout):
            print(f"[eval][warn] CLI holdout={sorted(holdout_lanes)} differs from "
                  f"run_meta holdout_lanes={sorted(int(x) for x in train_holdout)}. "
                  "The trained/held-out split will use the CLI value, but check the training run.")
    if any(l < 0 or l >= len(LANE_FILES) for l in holdout_lanes):
        raise SystemExit(f"holdout lanes must be in 0..{len(LANE_FILES)-1}; got {holdout_lanes}")

    policy = PPO.load(str(ckpt), custom_objects={"learning_rate": 1e-4,
                                                 "lr_schedule": lambda _: 1e-4})
    obs_dim = int(policy.observation_space.shape[0])     # 97 (full) or 67 (obs ablation)
    if obs_dim not in (67, 97):
        raise SystemExit(f"unsupported policy obs_dim={obs_dim}; expected 67 or 97")
    vec_norm = VecNormalize.load(str(vn), venv=DummyVecEnv([make_dummy_env(obs_dim)]))
    vec_norm.training = False
    vec_norm.norm_reward = False

    mj_model = mujoco.MjModel.from_xml_path(str(BASE_DIR / "g1_smooth.xml"))
    mj_data  = mujoco.MjData(mj_model)
    dt = mj_model.opt.timestep * FRAME_SKIP

    cop_sensors = ["cop_f_lf0", "cop_f_lf1", "cop_f_lf2", "cop_f_lf3",
                   "cop_f_rf0", "cop_f_rf1", "cop_f_rf2", "cop_f_rf3", "cop_f_pelvis"]
    cop_sites   = ["cop_lf0", "cop_lf1", "cop_lf2", "cop_lf3",
                   "cop_rf0", "cop_rf1", "cop_rf2", "cop_rf3", "cop_pelvis"]
    ids = {
        "head":   require_mj_id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "head"),
        "pelvis": mj_model.body("pelvis").id,
        "cop_adr": np.array([mj_model.sensor_adr[require_mj_id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, n)]
                             for n in cop_sensors]),
        "cop_sites": np.array([require_mj_id(mj_model, mujoco.mjtObj.mjOBJ_SITE, n)
                               for n in cop_sites]),
        "tau_max": np.abs(mj_model.actuator_forcerange[:29, 1]),
    }
    if np.all(ids["tau_max"] == 0):   # no explicit forcerange -> use a safe ceiling
        ids["tau_max"] = np.full(29, 200.0)

    rng = np.random.default_rng(args.seed)
    pools = [load_pool(BASE_DIR / "keyframes" / f) for f in LANE_FILES]
    holdout = set(holdout_lanes)

    print(f"[eval] {tag}  obs_dim={obs_dim}  beta={beta} (from {beta_src})  "
          f"n_resets={args.n_resets}  holdout={sorted(holdout)} (from {holdout_src})")

    results = {"tag": tag, "ckpt": str(ckpt.name), "obs_dim": obs_dim,
               "beta_max": beta, "beta_src": beta_src, "holdout_lanes": sorted(holdout),
               "holdout_src": holdout_src, "train_holdout_lanes": train_holdout,
               "n_resets": args.n_resets, "energy_window_s": ENERGY_WINDOW_S,
               "energy_average": "non_terminated_only",
               "aggregate_average": "success_fall_all_trials_other_metrics_non_terminated_only",
               "deterministic": {}}

    per_lane = {}
    for lane, pool in enumerate(pools):
        trials = []
        for _ in range(args.n_resets):
            row = int(rng.integers(len(pool)))
            m = run_episode(mj_model, mj_data, policy, vec_norm, ids,
                            pool[row], beta, obs_dim, True, args.max_steps, dt)
            trials.append(m)
        per_lane[lane] = aggregate(trials)

    trained = [per_lane[l] for l in range(len(pools)) if l not in holdout]
    heldout = [per_lane[l] for l in range(len(pools)) if l in holdout]

    def split_succ(group):
        return (float(np.mean([g["balanced_success_mean"] for g in group]))
                if group else float("nan"))

    results["deterministic"] = {
        "per_lane": {str(k): v for k, v in per_lane.items()},
        "trained_success": split_succ(trained),
        "heldout_success": split_succ(heldout),
        "overall_success": float(np.mean([per_lane[l]["balanced_success_mean"]
                                           for l in range(len(pools))])),
    }
    print(f"  [deterministic] trained={results['deterministic']['trained_success']:.2f}  "
          f"held-out={results['deterministic']['heldout_success']:.2f}  "
          f"overall={results['deterministic']['overall_success']:.2f}")

    out = mdir / f"eval_{tag}.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
