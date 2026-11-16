"""
Changes:
- Random starting points [pos, vel, act]
- More steps during training
- Eps reduced by episode instead of visits
- gamma = 0.99, alpha = 0.01

    python monte_carlo_improved.py
    python monte_carlo_improved.py --skip-training
"""
import argparse
import os
import random

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from mountain_car_mdp import (N_POS, N_VEL, N_ACTIONS, POS_MIN, POS_MAX, VEL_MIN, VEL_MAX,
                              discretize, build_P)
from results import OUT_DIR, run_episodes, plot_algo, record_videos

import tqdm
progress = tqdm.tqdm

EPISODES = 50000
TRAIN_MAX_STEPS = 1000
EPS_START = 0.5
EPS_MIN = 0.05
EPS_DECAY_FRACTION = 0.3
COARSE_N = 50
GAMMA = 0.99
ALPHA = 0.1
FALLBACK_ACTION = 1
EVAL_EVERY = 2000
EVAL_EPISODES = 30
TEST_EPISODES = 1000
SEED = 0

N_STATES_FINE = N_POS * N_VEL
N_STATES = COARSE_N * COARSE_N
NAME = "Monte Carlo improved"
KEY = "mc_improved"

# Відображення вузла сітки 60x60 у вузол сітки COARSE_N x COARSE_N.
# Усюди (навчання, оцінка, відео) стан = coarse[discretize(obs)], тому все узгоджено.
_i, _j = np.meshgrid(np.arange(N_POS), np.arange(N_VEL), indexing="ij")
FINE_TO_COARSE = (np.rint(_i * (COARSE_N - 1) / (N_POS - 1)).astype(int) * COARSE_N
                  + np.rint(_j * (COARSE_N - 1) / (N_VEL - 1)).astype(int)).reshape(-1)


def cstate(obs):
    return int(FINE_TO_COARSE[discretize(obs)])


def epsilon_greedy_policy(state, Q, eps):
    if random.uniform(0, 1) < eps:
        return random.randrange(N_ACTIONS)
    q_row = Q[state]
    best = np.flatnonzero(q_row == q_row.max())
    return int(best[0]) if len(best) == 1 else int(random.choice(best))


def generate_episode(env, Q, eps):
    """(1) Exploring start: випадковий стан сітки + випадкова перша дія. env - це env.unwrapped."""
    pos = random.uniform(POS_MIN, 0.49)
    vel = random.uniform(VEL_MIN, VEL_MAX)
    env.state = np.array([pos, vel])
    obs = np.array([pos, vel], dtype=np.float32)

    episode = []
    terminated = False
    for t in range(TRAIN_MAX_STEPS):
        state = cstate(obs)
        if t == 0:
            action = random.randrange(N_ACTIONS)
        else:
            action = epsilon_greedy_policy(state, Q, eps)
        obs, reward, terminated, _, _ = env.step(action)
        episode.append((state, action, reward))
        if terminated:
            break

    return episode, terminated


def generate_policy(Q, N):
    """Greedy policy as an ndarray of shape (N_STATES,); -1 for unvisited states."""
    visited = N.sum(axis=-1) > 0
    # Mask untried actions with -inf so they never win argmax.
    masked_Q = np.where(N > 0, Q, -np.inf)
    max_vals = masked_Q.max(axis=-1, keepdims=True)
    ties = (masked_Q == max_vals) & (N > 0)

    policy = np.full(N_STATES, -1, dtype=np.int64)
    # Random tie-break: pick uniformly among argmax actions.
    rand = np.random.rand(N_STATES, N_ACTIONS) * ties
    policy_visited = rand.argmax(axis=-1)
    policy[visited] = policy_visited[visited]

    value = masked_Q.max(axis=-1)
    value[~visited] = np.nan
    return policy, value


def evaluate_greedy(Q, N, env):
    """Частка успішних епізодів жадібної політики у стандартному середовищі (старт з долини, ліміт 200)."""
    policy, _ = generate_policy(Q, N)
    wins = 0
    for _ in range(EVAL_EPISODES):
        obs, _ = env.reset()
        while True:
            action = int(policy[cstate(obs)])
            if action < 0:
                action = FALLBACK_ACTION
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated:
                wins += 1
            if terminated or truncated:
                break
    return wins / EVAL_EPISODES


def train():
    """First-visit Monte Carlo control. Повертає (політика з -1 для невідвіданих станів, V, N_state, історії)."""
    train_env = gym.make('MountainCar-v0').unwrapped      # без ліміту 200 кроків
    train_env.reset(seed=SEED)
    eval_env = gym.make('MountainCar-v0')
    eval_env.reset(seed=SEED)

    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    N = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    N_state = np.zeros(N_STATES, dtype=np.int64)
    train_success = []
    eval_points = []          # (номер епізоду, частка успіхів)
    decay_episodes = max(1, int(EPISODES * EPS_DECAY_FRACTION))

    for k in progress(range(EPISODES)):
        eps = max(EPS_MIN, EPS_START - (EPS_START - EPS_MIN) * k / decay_episodes)
        episode, success = generate_episode(train_env, Q, eps)
        train_success.append(float(success))

        for (state, _, _) in episode:
            N_state[state] += 1

        # First-visit indices: keep the earliest t at which each (s, a) appears.
        first_visit = {}
        for t, (state, action, _) in enumerate(episode):
            key = (state, int(action))
            if key not in first_visit:
                first_visit[key] = t

        # Compute returns G_t in a single backward pass.
        G = 0.0
        returns = [0.0] * len(episode)
        for t in range(len(episode) - 1, -1, -1):
            G = episode[t][2] + GAMMA * G
            returns[t] = G

        # Incremental update, only at the first-visit step of each (s, a).
        for (state, action), t in first_visit.items():
            N[state, action] += 1
            step = 1.0 / N[state, action] if ALPHA is None else (1.0 if N[state, action] == 1 else ALPHA)
            Q[state, action] += step * (returns[t] - Q[state, action])

        if (k + 1) % EVAL_EVERY == 0:
            eval_points.append((k + 1, evaluate_greedy(Q, N, eval_env)))

    train_env.close()
    eval_env.close()
    policy, value = generate_policy(Q, N)
    return policy, value, N_state, train_success, eval_points


def plot_training(train_success, eval_points, N_state, window=1000):
    """Крива навчання і карта станів, які агент відвідав (на сітці навчання)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ts = np.asarray(train_success, dtype=float)
    if ts.size >= window:
        smooth = np.convolve(ts, np.ones(window) / window, mode="valid")
        axes[0].plot(np.arange(window - 1, ts.size), smooth, color="lightsteelblue", lw=1.5,
                     label=f"training episodes reaching the flag (moving avg {window})")
    if eval_points:
        xs, ys = zip(*eval_points)
        axes[0].plot(xs, ys, "o-", color="tab:green", label="greedy policy, standard start (200 steps)")
    axes[0].set_title("Training progress")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="center right", fontsize=8)

    extent = [POS_MIN, POS_MAX, VEL_MIN, VEL_MAX]
    img = np.ma.masked_equal(np.log1p(N_state).reshape(COARSE_N, COARSE_N).T, 0)
    cm = plt.get_cmap("magma").copy()
    cm.set_bad("0.15")
    im = axes[1].imshow(img, origin="lower", aspect="auto", extent=extent, cmap=cm, interpolation="nearest")
    axes[1].set_title("States visited during training")
    axes[1].set_xlabel("position")
    axes[1].set_ylabel("velocity")
    fig.colorbar(im, ax=axes[1]).set_label("log(1 + visit count)")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "monte_carlo_improved.png")
    fig.savefig(path, dpi=130)
    print("Збережено", path)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true",
                        help="не навчати, а взяти збережені policy_mc_improved.npy і value_mc_improved.npy")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    unvisited_fine = None
    figs = []

    if args.skip_training:
        if not (os.path.exists("policy_mc_improved.npy") and os.path.exists("value_mc_improved.npy")):
            raise SystemExit("Не знайдено policy_mc_improved.npy / value_mc_improved.npy. "
                             "Запустіть без --skip-training.")
        policy = np.load("policy_mc_improved.npy").astype(int)
        value = np.load("value_mc_improved.npy")
        if policy.size != N_STATES_FINE:
            raise SystemExit("Розмір збереженої політики не збігається з сіткою. Запустіть без --skip-training.")
    else:
        policy_c, value_c, N_state, train_success, eval_points = train()
        print(f"Visited states (training grid): {int((policy_c >= 0).sum())} / {N_STATES}")

        policy = policy_c[FINE_TO_COARSE]
        value = value_c[FINE_TO_COARSE]
        unvisited_fine = policy < 0
        policy = np.where(unvisited_fine, FALLBACK_ACTION, policy)   # Unvisited: fall back to a default action
        np.save("policy_mc_improved.npy", policy)
        np.save("value_mc_improved.npy", value)
        figs.append(plot_training(train_success, eval_points, N_state))

    env = gym.make('MountainCar-v0')
    P = build_P(env)
    env.close()

    r = dict(key=KEY, name=NAME, policy=policy, value=value, P=P,
             ep=run_episodes(policy, TEST_EPISODES))
    print(f"Average reward: {r['ep']['rewards'].mean():.1f}")
    print(f"Success rate: {r['ep']['ok'].mean():.1%}")
    if unvisited_fine is not None:
        visits = r["ep"]["visits"]
        print(f"Share of steps with fallback action (unvisited state): "
              f"{visits[unvisited_fine].sum() / visits.sum():.1%}")

    figs.append(plot_algo(r))
    record_videos(r, 1)

    for fig in figs:
        plt.close(fig)


if __name__ == '__main__':
    main()
