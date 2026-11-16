"""
python monte_carlo.py
python monte_carlo.py --skip-training   # без навчання
"""
import argparse
import os
import random

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from mountain_car_mdp import (N_POS, N_VEL, N_ACTIONS, POS_MIN, POS_MAX, VEL_MIN, VEL_MAX,
                              POS_STEP, VEL_STEP, discretize, build_P)
from results import OUT_DIR, run_episodes, plot_algo, record_videos

import tqdm
progress = tqdm.tqdm


N_STATES = N_POS * N_VEL
EPS_MIN = 0.05
FALLBACK_ACTION = 1  # default action
EPISODES = 30000     # learning
TEST_EPISODES = 1000 # testing
SEED = 0


def epsilon_greedy_policy(state, Q, N_state):

    # GLIE decay with a floor so rare states keep exploring a bit.
    eps = max(EPS_MIN, 1.0 / (1 + N_state[state]))

    if random.uniform(0, 1) < eps:
        return random.randrange(N_ACTIONS)
    q_row = Q[state]
    best = np.flatnonzero(q_row == q_row.max())
    return int(best[0]) if len(best) == 1 else int(random.choice(best))


def generate_episode(env, Q, N_state):
    episode = []
    obs, _ = env.reset()
    state = discretize(obs)

    while True:
        action = epsilon_greedy_policy(state, Q, N_state)
        obs, reward, terminated, truncated, _ = env.step(action)

        episode.append((state, action, reward))

        if terminated or truncated:
            break

        state = discretize(obs)

    return episode


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


def train(env):
    """First-visit Monte Carlo control. Повертає (політика з -1 для невідвіданих станів, V, N_state, returns)."""
    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    N = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    N_state = np.zeros(N_STATES, dtype=np.int64)
    episode_returns = []

    for i in progress(range(EPISODES)):
        episode = generate_episode(env, Q, N_state)
        episode_returns.append(sum(r for _, _, r in episode))

        # Bump per-state visit counter for GLIE epsilon decay.
        for (state, _, _) in episode:
            N_state[state] += 1

        # First-visit indices: keep the earliest t at which each (s, a) appears.
        first_visit = {}
        for t, (state, action, _) in enumerate(episode):
            key = (state, int(action))
            if key not in first_visit:
                first_visit[key] = t

        # Compute returns G_t in a single backward pass (undiscounted).
        G = 0.0
        returns = [0.0] * len(episode)
        for t in range(len(episode) - 1, -1, -1):
            G = episode[t][2] + G
            returns[t] = G

        # Incremental-mean update, only at the first-visit step of each (s, a).
        for (state, action), t in first_visit.items():
            N[state, action] += 1
            Q[state, action] += (returns[t] - Q[state, action]) / N[state, action]

    policy, value = generate_policy(Q, N)
    return policy, value, N_state, episode_returns


def plot_training(episode_returns, N_state, window=500):
    """Крива навчання і карта станів, які агент відвідав."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    returns = np.asarray(episode_returns, dtype=float)
    axes[0].plot(returns, color="lightsteelblue", lw=0.5, label="episode return")
    if returns.size >= window:
        smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
        axes[0].plot(np.arange(window - 1, returns.size), smooth, color="tab:blue", lw=2,
                     label=f"moving average ({window})")
    axes[0].set_title("Training curve")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("return")
    axes[0].legend()

    extent = [POS_MIN - POS_STEP / 2, POS_MAX + POS_STEP / 2, VEL_MIN - VEL_STEP / 2, VEL_MAX + VEL_STEP / 2]
    img = np.ma.masked_equal(np.log1p(N_state).reshape(N_POS, N_VEL).T, 0)
    cm = plt.get_cmap("magma").copy()
    cm.set_bad("0.15")
    im = axes[1].imshow(img, origin="lower", aspect="auto", extent=extent, cmap=cm, interpolation="nearest")
    axes[1].set_title("States visited during training")
    axes[1].set_xlabel("position")
    axes[1].set_ylabel("velocity")
    fig.colorbar(im, ax=axes[1]).set_label("log(1 + visit count)")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "monte_carlo.png")
    fig.savefig(path, dpi=130)
    print("Збережено", path)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true",
                        help="не навчати, а взяти збережені policy_mc.npy і value_mc.npy")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    unvisited = None
    figs = []

    if args.skip_training:
        if not (os.path.exists("policy_mc.npy") and os.path.exists("value_mc.npy")):
            raise SystemExit("Не знайдено policy_mc.npy / value_mc.npy. Запустіть без --skip-training.")
        policy = np.load("policy_mc.npy").astype(int)
        value = np.load("value_mc.npy")
        if policy.size != N_STATES:
            raise SystemExit("Розмір збереженої політики не збігається з сіткою. Запустіть без --skip-training.")
    else:
        # Action Space Discrete(3)
        # Observation Space Box(2) -> дискретизуємо у сітку N_POS x N_VEL
        env = gym.make('MountainCar-v0')
        env.reset(seed=SEED)
        policy, value, N_state, episode_returns = train(env)
        env.close()

        unvisited = policy < 0
        print(f"Visited states: {int((~unvisited).sum())} / {N_STATES}")

        # Unvisited during training: fall back to a default action.
        policy = np.where(unvisited, FALLBACK_ACTION, policy)
        np.save("policy_mc.npy", policy)
        np.save("value_mc.npy", value)
        figs.append(plot_training(episode_returns, N_state))

    env = gym.make('MountainCar-v0')
    P = build_P(env)
    env.close()

    r = dict(key="mc", name="Monte Carlo", policy=policy, value=value, P=P,
             ep=run_episodes(policy, TEST_EPISODES))
    print(f"Average reward: {r['ep']['rewards'].mean():.1f}")
    print(f"Success rate: {r['ep']['ok'].mean():.1%}")
    if unvisited is not None:
        visits = r["ep"]["visits"]
        print(f"Share of steps with fallback action (unvisited state): "
              f"{visits[unvisited].sum() / visits.sum():.1%}")

    figs.append(plot_algo(r))
    record_videos(r, 1)

    for fig in figs:
        plt.close(fig)


if __name__ == '__main__':
    main()