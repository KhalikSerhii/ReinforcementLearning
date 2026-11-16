"""
python value_iteration.py
python policy_iteration.py
python results.py

Options:
    --episodes N
    --video-episodes N
    --no-video
    --show
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from mountain_car_mdp import (N_POS, N_VEL, POS_MIN, POS_MAX, VEL_MIN, VEL_MAX,
                              POS_STEP, VEL_STEP, discretize, build_P)

N_STATES = N_POS * N_VEL
OUT_DIR = "results"
ALGOS = {"vi": "Value Iteration", "pi": "Policy Iteration"}
GOAL_POS = 0.5

# межі зображення (півкрока, щоб вузли сітки були по центру клітинок)
EXTENT = [POS_MIN - POS_STEP / 2, POS_MAX + POS_STEP / 2,
          VEL_MIN - VEL_STEP / 2, VEL_MAX + VEL_STEP / 2]


def load_algo(key):
    policy = np.load(f"policy_{key}.npy").astype(int)
    value = np.load(f"value_{key}.npy")
    return policy, value


def run_episodes(policy, n_episodes, seed=0):
    env = gym.make("MountainCar-v0")
    visits = np.zeros(N_STATES)
    visits_ok = np.zeros(N_STATES)     # відвідування у успішних епізодах
    rem_sum = np.zeros(N_STATES)       # сума "скільки кроків залишалось до фінішу"
    rem_cnt = np.zeros(N_STATES)
    rewards, steps, ok = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        traj_s, total = [], 0.0
        while True:
            s = discretize(obs)
            traj_s.append(s)
            obs, r, terminated, truncated, _ = env.step(int(policy[s]))
            total += r
            if terminated or truncated:
                break
        n = len(traj_s)
        for k, s in enumerate(traj_s):
            visits[s] += 1
            if terminated:
                visits_ok[s] += 1
                rem_sum[s] += n - k
                rem_cnt[s] += 1
        rewards.append(total)
        steps.append(n)
        ok.append(bool(terminated))
    env.close()

    return dict(
        visits=visits,
        p_success=np.where(visits > 0, visits_ok / np.maximum(visits, 1), np.nan),
        rem_mean=np.where(rem_cnt > 0, rem_sum / np.maximum(rem_cnt, 1), np.nan),
        rewards=np.array(rewards), steps=np.array(steps), ok=np.array(ok),
    )


def analyze(key, n_episodes, P):
    policy, value = load_algo(key)
    return dict(key=key, name=ALGOS[key], policy=policy, value=value, P=P,
                ep=run_episodes(policy, n_episodes))


def to_img(arr):
    return np.asarray(arr, dtype=float).reshape(N_POS, N_VEL).T


def heat(ax, data, cmap, title, label=None, vmin=None, vmax=None):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("0.15")
    im = ax.imshow(np.ma.masked_invalid(to_img(data)), origin="lower", aspect="auto", extent=EXTENT,
                   cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.axvline(GOAL_POS, color="white", ls="--", lw=0.8)
    if label is not None:
        ax.figure.colorbar(im, ax=ax).set_label(label)
    return im


def plot_transition_matrix(ax, P, policy):
    rows, cols, probs = [], [], []
    for s in range(N_STATES):
        for prob, s_, _, _ in P[s][int(policy[s])]:
            rows.append(s)
            cols.append(s_)
            probs.append(prob)
    sc = ax.scatter(rows, cols, c=probs, s=1.5, cmap="viridis", vmin=0, vmax=1, linewidths=0)
    ax.set_xlim(0, N_STATES)
    ax.set_ylim(N_STATES, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("current state s")
    ax.set_ylabel("next state s'")
    ax.set_title("Transition matrix  P(s' | s, policy(s))")
    ax.figure.colorbar(sc, ax=ax).set_label("probability")


def plot_algo(r):
    ep = r["ep"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    heat(axes[0], r["value"], "magma", "State value V(s)", "V(s)")

    heat(axes[1], ep["p_success"], "RdYlGn", "P(success | state visited)", "success probability",
         vmin=0, vmax=1)

    heat(axes[2], ep["rem_mean"], "viridis", "Mean steps to finish (successful episodes)", "steps")

    fig.suptitle(f"{r['name']}: avg reward {ep['rewards'].mean():.1f}, success rate {ep['ok'].mean():.0%}",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, f"results_{r['key']}.png")
    fig.savefig(path, dpi=130)
    return fig


def build_report(a, b, n_episodes):
    def steps_ok(r):
        return r["ep"]["steps"][r["ep"]["ok"]]

    rows = [
        (f"Середня нагорода ({n_episodes} епізодів)", lambda r: f"{r['ep']['rewards'].mean():.1f}"),
        ("Частка успішних епізодів", lambda r: f"{r['ep']['ok'].mean():.1%}"),
        ("Кроків до фінішу: середнє", lambda r: f"{steps_ok(r).mean():.1f}" if steps_ok(r).size else "n/a"),
        ("Кроків до фінішу: мін / макс",
         lambda r: f"{steps_ok(r).min()} / {steps_ok(r).max()}" if steps_ok(r).size else "n/a"),
        ("Відвіданих станів", lambda r: str(int((r["ep"]["visits"] > 0).sum()))),
    ]

    width = max(len(name) for name, _ in rows) + 2
    lines = [f"{'Метрика':{width}}{'Value Iteration':>18}{'Policy Iteration':>18}", "-" * (width + 36)]
    for name, fn in rows:
        lines.append(f"{name:{width}}{fn(a):>18}{fn(b):>18}")

    visited = (a["ep"]["visits"] > 0) | (b["ep"]["visits"] > 0)
    same = a["policy"] == b["policy"]
    dv = np.abs(a["value"] - b["value"])
    lines += [
        "",
        f"Збіг політик у всіх станах:          {same.mean():.1%}",
        f"Збіг політик у відвіданих станах:    {same[visited].mean():.1%}  ({int(visited.sum())} станів)",
        f"max |V_VI - V_PI|:                   {dv.max():.3e}",
        f"середнє |V_VI - V_PI|:               {dv.mean():.3e}",
    ]
    return "\n".join(lines)


def record_videos(r, n_episodes, seed=0):
    folder = os.path.join(OUT_DIR, "videos", r["key"])
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=folder, name_prefix=r["name"].lower().replace(" ", "_"),
                        episode_trigger=lambda ep_id: True)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(int(r["policy"][discretize(obs)]))
            done = terminated or truncated
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    env = gym.make("MountainCar-v0")
    P = build_P(env)
    env.close()

    a = analyze("vi", args.episodes, P)
    b = analyze("pi", args.episodes, P)

    report = build_report(a, b, args.episodes)
    print("\n" + report + "\n")
    with open(os.path.join(OUT_DIR, "results.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    figs = [plot_algo(a), plot_algo(b)]

    if not args.no_video:
        for r in (a, b):
            record_videos(r, args.video_episodes)

    if args.show:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)


if __name__ == "__main__":
    main()
