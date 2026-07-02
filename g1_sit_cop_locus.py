"""
G1 Sit-to-Stand — COP Locus Figure
==================================
Single deterministic, force-free rollout on one chair lane. Plots the measured
center of pressure (COP) over the first --window-s seconds as a time-colored trail
(blue = early, red = late) against static support regions:

  - foot and seat+foot convex hulls (dashed)
  - per-foot rectangles from time-averaged contact sites
  - chair rectangle from the seat geom in g1_smooth.xml

Foot and butt coordinates are time-averaged over the window so the polygons stay
stable while the feet jitter slightly during push-off.

The --settle-steps flag holds the seated pose briefly so the butt loads onto the
seat before the policy runs (CSV poses start ~12 mm above the seat surface).

Reuses rollout helpers from g1_sit_eval.py.

Usage:
  python g1_sit_cop_locus.py --model-dir models_g1_sit_gen_best --beta 0.6 --lane 0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from g1_sit_eval import (
    BASE_DIR,
    FRAME_SKIP,
    JOINT_LOWER,
    JOINT_UPPER,
    LANE_FILES,
    TARGET_HEAD_HEIGHT,
    _BETA,
    convex_hull_xy,
    cop_xy,
    get_obs,
    load_pool,
    make_dummy_env,
    require_mj_id,
    term_reason,
)

COP_SITE_NAMES = [
    "cop_lf0", "cop_lf1", "cop_lf2", "cop_lf3",
    "cop_rf0", "cop_rf1", "cop_rf2", "cop_rf3",
    "cop_pelvis",
]
N_FOOT_SITES = 8
N_SITES_PER_FOOT = 4
# seat geom half-extents in world xy (g1_smooth.xml seat1..seat8)
SEAT_HALF_X = 0.16
SEAT_HALF_Y = 0.25
# crop empty space aft of the seat; butt/seat contact is near x ≈ -0.26
X_VIEW_MIN = -0.30
VIEW_PAD_X = 0.03
VIEW_PAD_Y = 0.03
FIG_WIDTH = 9.0
FIG_HEIGHT = 6.0          # panel aspect width:height = 1.5:1
TICK_LABELSIZE = 14
X_TICK_NBINS = 6            # sparse x-axis major ticks
Y_TICK_NBINS = 6


def _aabb_corners(xy):
    """Axis-aligned rectangle corners enclosing N 2D points (closed loop omitted)."""
    xy = np.asarray(xy, dtype=np.float64)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    return np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ], dtype=np.float64)


def _rect_from_center(center_xy, half_x, half_y):
    cx, cy = center_xy
    return np.array([
        [cx - half_x, cy - half_y],
        [cx + half_x, cy - half_y],
        [cx + half_x, cy + half_y],
        [cx - half_x, cy + half_y],
    ], dtype=np.float64)


def _resolve_beta(mdir, cli_beta):
    if cli_beta is not None:
        return float(cli_beta), "cli"
    meta_path = mdir / "run_meta.json"
    if meta_path.exists():
        return float(json.loads(meta_path.read_text())["beta_max"]), "run_meta.json"
    raise SystemExit(f"no --beta and no {meta_path.name} in {mdir.name}; pass --beta (e.g. 0.6).")


def _load_policy_bundle(mdir, ckpt_stem, vecnorm_stem):
    ckpt = mdir / f"{ckpt_stem}.zip"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    vn_path = mdir / (f"{vecnorm_stem}.pkl" if vecnorm_stem else "vec_normalize_best.pkl")
    if not vn_path.exists():
        raise SystemExit(f"VecNormalize file not found: {vn_path}")
    policy = PPO.load(str(ckpt),
                      custom_objects={"learning_rate": 1e-4, "lr_schedule": lambda _: 1e-4})
    obs_dim = int(policy.observation_space.shape[0])
    if obs_dim not in (67, 97):
        raise SystemExit(f"unsupported obs_dim={obs_dim}; expected 67 or 97")
    vec_norm = VecNormalize.load(str(vn_path), venv=DummyVecEnv([make_dummy_env(obs_dim)]))
    vec_norm.training = False
    vec_norm.norm_reward = False
    return policy, vec_norm, obs_dim, ckpt, vn_path


def _build_mj_ids(mj_model):
    cop_sensors = ["cop_f_lf0", "cop_f_lf1", "cop_f_lf2", "cop_f_lf3",
                   "cop_f_rf0", "cop_f_rf1", "cop_f_rf2", "cop_f_rf3", "cop_f_pelvis"]
    return {
        "head": require_mj_id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "head"),
        "pelvis": mj_model.body("pelvis").id,
        "cop_adr": np.array([
            mj_model.sensor_adr[require_mj_id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, n)]
            for n in cop_sensors]),
        "cop_sites": np.array([
            require_mj_id(mj_model, mujoco.mjtObj.mjOBJ_SITE, n) for n in COP_SITE_NAMES]),
    }


def _hull_closed(hull):
    pts = [tuple(p) for p in hull]
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    return np.asarray(pts, dtype=np.float64)


def rollout_cop_trace(mj_model, mj_data, policy, vec_norm, ids, start_qpos, beta_max,
                      obs_dim, *, window_s, start_after_steps, settle_steps):
    """One force-free deterministic episode; log COP + contact-site xy each step."""
    mj_data.qpos[:] = start_qpos
    mj_data.qpos[22:36] = 0.0
    mj_data.qvel[:] = 0.0
    mj_data.ctrl[:] = 0.0
    mj_data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    dt = mj_model.opt.timestep * FRAME_SKIP
    head_sid, cop_adr, cop_sites = ids["head"], ids["cop_adr"], ids["cop_sites"]

    # pre-settle: hold seated pose so the base sinks onto the seat and the butt loads
    if settle_steps > 0:
        hold = np.clip(mj_data.qpos[7:36].copy(), JOINT_LOWER, JOINT_UPPER)
        for _ in range(settle_steps):
            mj_data.ctrl[:29] = hold
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(mj_model, mj_data)

    h0 = float(mj_data.site_xpos[head_sid][2])
    prev_action = np.zeros(29, dtype=np.float64)
    n_steps = max(1, int(math.ceil(window_s / dt)))

    samples, reason, butt_peak = [], "timeout", 0.0
    butt_seat_xy = None                       # butt site xy at peak seat load
    for step_i in range(n_steps):
        obs = get_obs(mj_data, prev_action, beta_max, obs_dim)
        action, _ = policy.predict(vec_norm.normalize_obs(obs.reshape(1, -1)),
                                   deterministic=True)
        action = np.clip(action.flatten(), -1.0, 1.0)
        mj_data.ctrl[:29] = np.clip(mj_data.qpos[7:36] + action * _BETA * beta_max,
                                    JOINT_LOWER, JOINT_UPPER)
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(mj_model, mj_data)
        prev_action = action

        bf = float(max(mj_data.sensordata[cop_adr[N_FOOT_SITES]], 0.0))
        butt_peak = max(butt_peak, bf)
        if bf > 5.0:   # seat contact: keep the most-rearward loaded butt = seated pose
            bxy = mj_data.site_xpos[cop_sites[N_FOOT_SITES]][:2].astype(float)
            if butt_seat_xy is None or bxy[0] < butt_seat_xy[0]:
                butt_seat_xy = bxy.tolist()
        if step_i >= start_after_steps:
            head_z = float(mj_data.site_xpos[head_sid][2])
            samples.append({
                "t_s": (step_i + 1) * dt,
                "cop_xy": cop_xy(mj_data, cop_adr, cop_sites).tolist(),
                "site_xy": mj_data.site_xpos[cop_sites][:, :2].astype(float).tolist(),
                "rise_frac": (head_z - h0) / max(TARGET_HEAD_HEIGHT - h0, 1e-3),
            })
        reason = term_reason(mj_data)
        if reason != "none":
            break

    return {
        "dt_control_s": dt, "settle_steps": settle_steps, "h0_head_z": h0,
        "window_s": window_s, "samples": samples, "n_samples": len(samples),
        "duration_s": samples[-1]["t_s"] if samples else 0.0, "term": reason,
        "butt_force_peak_n": butt_peak, "butt_seat_xy": butt_seat_xy,
    }


def plot_cop_locus(trace, *, out_png):
    ss = trace["samples"]
    if not ss:
        raise RuntimeError("no COP samples to plot")
    site = np.array([s["site_xy"] for s in ss])          # (T, 9, 2)
    foot_avg = site[:, :N_FOOT_SITES, :].mean(axis=0)    # time-averaged foot corners
    # seat-contact point: butt site at peak seat load (encloses the early COP);
    # fall back to the time-averaged butt site if the butt never loaded.
    butt_ref = np.asarray(trace["butt_seat_xy"] if trace["butt_seat_xy"] is not None
                          else site[:, N_FOOT_SITES, :].mean(axis=0), dtype=np.float64)
    cop = np.array([s["cop_xy"] for s in ss])
    t = np.array([s["t_s"] for s in ss])

    left_foot = foot_avg[:N_SITES_PER_FOOT]
    right_foot = foot_avg[N_SITES_PER_FOOT:N_FOOT_SITES]
    left_rect = _aabb_corners(left_foot)
    right_rect = _aabb_corners(right_foot)
    chair_rect = _rect_from_center(butt_ref, SEAT_HALF_X, SEAT_HALF_Y)
    foot_hull = _hull_closed(convex_hull_xy([tuple(p) for p in foot_avg]))
    bf_hull = _hull_closed(convex_hull_xy(
        [tuple(p) for p in np.vstack([foot_avg, butt_ref])]))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # chair footprint (seat box size) centered on pelvis/seat contact
    ax.add_patch(mpatches.Polygon(chair_rect, closed=True, facecolor="#d4a574",
                                  edgecolor="#8b5a2b", linewidth=1.5, alpha=0.35,
                                  zorder=1))
    ax.plot(np.r_[chair_rect[:, 0], chair_rect[0, 0]],
            np.r_[chair_rect[:, 1], chair_rect[0, 1]],
            color="#8b5a2b", linewidth=1.5, zorder=2, label="chair (seat box)")

    # foot-only and seat+foot convex hulls (dashed outlines only)
    ax.add_patch(mpatches.Polygon(foot_hull[:-1], closed=True, fill=False,
                                  linestyle="--", edgecolor="#2b6cb0", linewidth=1.6,
                                  zorder=3))
    ax.plot(foot_hull[:, 0], foot_hull[:, 1], linestyle="--", color="#2b6cb0",
            linewidth=1.6, zorder=3, label="foot support")
    ax.add_patch(mpatches.Polygon(bf_hull[:-1], closed=True, fill=False,
                                  linestyle="--", edgecolor="0.5", linewidth=1.4,
                                  zorder=3))
    ax.plot(bf_hull[:, 0], bf_hull[:, 1], linestyle="--", color="0.5", linewidth=1.4,
            zorder=3, label="seat + foot support")

    # per-foot rectangles (drawn only; omitted from legend)
    ax.add_patch(mpatches.Polygon(left_rect, closed=True, facecolor="none",
                                  edgecolor="#2b6cb0", linewidth=1.2, linestyle="-",
                                  zorder=4))
    ax.plot(np.r_[left_rect[:, 0], left_rect[0, 0]],
            np.r_[left_rect[:, 1], left_rect[0, 1]],
            color="#2b6cb0", linewidth=1.2, zorder=4, label="_nolegend_")
    ax.add_patch(mpatches.Polygon(right_rect, closed=True, facecolor="none",
                                  edgecolor="#1a7f5a", linewidth=1.2, linestyle="-",
                                  zorder=4))
    ax.plot(np.r_[right_rect[:, 0], right_rect[0, 0]],
            np.r_[right_rect[:, 1], right_rect[0, 1]],
            color="#1a7f5a", linewidth=1.2, zorder=4, label="_nolegend_")

    ax.scatter(left_foot[:, 0], left_foot[:, 1], s=26, c="#2b6cb0", marker="o", zorder=5)
    ax.scatter(right_foot[:, 0], right_foot[:, 1], s=26, c="#1a7f5a", marker="o", zorder=5)
    ax.scatter([butt_ref[0]], [butt_ref[1]], s=60, c="#8b5a2b", marker="s", zorder=5,
               label="pelvis seat contact")

    # time-colored COP trail
    ax.plot(cop[:, 0], cop[:, 1], color="0.7", linewidth=0.7, alpha=0.6, zorder=6)
    sc = ax.scatter(cop[:, 0], cop[:, 1], c=t, cmap="coolwarm", s=52,
                    edgecolors="0.2", linewidths=0.3, zorder=7)
    ax.scatter([cop[0, 0]], [cop[0, 1]], s=240, facecolors="none", edgecolors="#2166ac",
               linewidths=2.0, zorder=8, label="COP start")
    ax.scatter([cop[-1, 0]], [cop[-1, 1]], s=320, marker="*", c="#b2182b",
               edgecolors="0.1", linewidths=0.4, zorder=8, label="COP end")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("time (s)  blue = early → red = late", fontsize=13)
    ax.set_xlabel("world x (m)", fontsize=15)
    ax.set_ylabel("world y (m)", fontsize=15)
    all_xy = np.vstack([
        cop, foot_avg, butt_ref.reshape(1, 2), chair_rect, left_rect, right_rect,
        foot_hull[:-1], bf_hull[:-1],
    ])
    ax.set_xlim(max(X_VIEW_MIN, all_xy[:, 0].min() - VIEW_PAD_X),
                all_xy[:, 0].max() + VIEW_PAD_X)
    ax.set_ylim(all_xy[:, 1].min() - VIEW_PAD_Y, all_xy[:, 1].max() + VIEW_PAD_Y)
    ax.set_aspect("auto")  # fill the 1.5:1 axes box (not equal data scaling)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=X_TICK_NBINS, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=Y_TICK_NBINS, prune=None))
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="single-window COP locus figure")
    p.add_argument("--model-dir", default="models_g1_sit_gen_best")
    p.add_argument("--ckpt", default="best_model")
    p.add_argument("--vecnorm", default="vec_normalize_best")
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--lane", type=int, default=0, help="chair lane 0..7")
    p.add_argument("--row", type=int, default=None, help="CSV row index; random if omitted")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window-s", type=float, default=1.5, help="COP trail duration (s)")
    p.add_argument("--settle-steps", type=int, default=30,
                   help="hold seated pose this many control steps before the policy "
                        "so the butt loads onto the seat (0 = off)")
    p.add_argument("--start-after-steps", type=int, default=2,
                   help="skip this many control steps before logging (reset spike)")
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if not (0 <= args.lane < len(LANE_FILES)):
        raise SystemExit(f"--lane must be in 0..{len(LANE_FILES) - 1}")
    if args.window_s <= 0:
        raise SystemExit("--window-s must be positive")

    mdir = (BASE_DIR / args.model_dir) if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    beta, beta_src = _resolve_beta(mdir, args.beta)
    policy, vec_norm, obs_dim, ckpt_path, vn_path = _load_policy_bundle(
        mdir, args.ckpt, args.vecnorm)

    pool = load_pool(BASE_DIR / "keyframes" / LANE_FILES[args.lane])
    rng = np.random.default_rng(args.seed)
    row_idx = int(args.row) if args.row is not None else int(rng.integers(len(pool)))
    if not (0 <= row_idx < len(pool)):
        raise SystemExit(f"--row must be in 0..{len(pool) - 1}")

    mj_model = mujoco.MjModel.from_xml_path(str(BASE_DIR / "g1_smooth.xml"))
    mj_data = mujoco.MjData(mj_model)
    ids = _build_mj_ids(mj_model)

    tag = args.tag or f"locus_lane{args.lane}_s{args.seed}"
    print(f"[cop_locus] {mdir.name}  lane={args.lane}  row={row_idx}  "
          f"beta={beta} ({beta_src})  obs_dim={obs_dim}  settle={args.settle_steps}")

    trace = rollout_cop_trace(mj_model, mj_data, policy, vec_norm, ids, pool[row_idx],
                              beta, obs_dim, window_s=args.window_s,
                              start_after_steps=args.start_after_steps,
                              settle_steps=args.settle_steps)
    if trace["butt_force_peak_n"] < 5.0:
        print(f"  [warn] butt never loaded (peak {trace['butt_force_peak_n']:.1f}N); "
              f"increase --settle-steps to seat the robot first.")

    out_png = Path(args.out) if args.out else mdir / f"cop_locus_{tag}.png"
    out_json = out_png.with_suffix(".json")
    if not out_png.is_absolute():
        out_png = BASE_DIR / out_png
    if not out_json.is_absolute():
        out_json = BASE_DIR / out_json
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plot_cop_locus(trace, out_png=out_png)

    meta = {"tag": tag, "model_dir": str(mdir), "ckpt": ckpt_path.name,
            "vecnorm": vn_path.name, "beta_max": beta, "beta_src": beta_src,
            "obs_dim": obs_dim, "lane": args.lane, "lane_file": LANE_FILES[args.lane],
            "row_idx": row_idx, "seed": args.seed, "png": str(out_png),
            **{k: trace[k] for k in ("dt_control_s", "settle_steps", "window_s",
                                     "h0_head_z", "n_samples", "duration_s", "term",
                                     "butt_force_peak_n", "samples")}}
    out_json.write_text(json.dumps(meta, indent=2))

    print(f"  samples={trace['n_samples']}  dur={trace['duration_s']:.2f}s  "
          f"term={trace['term']}  butt_peak={trace['butt_force_peak_n']:.0f}N")
    print(f"  wrote {out_png}\n  wrote {out_json}")


if __name__ == "__main__":
    main()
