"""Improved DQN for CarRacing-v3 (Box2D).

Drop-in replacement for the baseline `dqn_car_racing.py`:
  * `make_env`, `make_vec_env`, FRAME_STACK, FRAME_SIZE are IDENTICAL to the baseline
    (environment settings must not change);
  * `QNetwork(in_channels=4, a_size=5)` keeps the signature and `act()` used by the
    unchanged `test_car_racing.py`.

What is new (every item can be switched on/off through `Config`):
  * Double DQN targets
  * Dueling head
  * Soft (Polyak) target update
  * n-step returns (+ correct bootstrapping on time-limit truncation)
  * Random-shift augmentation of replay batches (DrQ-style)
  * Checkpoint selection by greedy evaluation instead of the noisy epsilon-greedy
    training average, plus a top-k list of candidates for the final selection
"""
import copy
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass, asdict

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
    RecordVideo,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # autotune convs for fixed input size

FRAME_STACK = 4
FRAME_SIZE = 84

# The test script builds QNetwork(a_size=...) with defaults and loads the checkpoint
# strictly, so the default architecture must match the architecture of the final
# checkpoint. Set to False if the final model is trained with dueling=False.
#   DQN_DUELING=1 (default) -> dueling default; DQN_DUELING=0 -> baseline-style head, used to
#   evaluate the ORIGINAL baseline checkpoint with the unchanged test script.
DEFAULT_DUELING = os.environ.get("DQN_DUELING", "1") == "1"


# ----------------------------------------------------------------------------
# Environment (UNCHANGED w.r.t. the baseline)
# ----------------------------------------------------------------------------
def make_env(render_mode=None):
    """CarRacing with grayscale, resize and frame-stacking wrappers."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (FRAME_SIZE, FRAME_SIZE))
    env = FrameStackObservation(env, FRAME_STACK)
    return env


def make_vec_env(num_envs):
    """Several CarRacing instances stepped in parallel subprocesses."""
    return gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class QNetwork(nn.Module):
    """CNN mapping a stack of frames to action values.

    dueling=False -> Nature DQN head (identical parameter names to the baseline,
                     so baseline checkpoints load into it);
    dueling=True  -> separate value / advantage streams.
    """

    def __init__(self, in_channels=FRAME_STACK, a_size=5, dueling=None):
        super().__init__()
        if dueling is None:
            dueling = DEFAULT_DUELING
        self.a_size = a_size
        self.dueling = dueling
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        if dueling:
            self.val1 = nn.Linear(64 * 7 * 7, 512)
            self.val2 = nn.Linear(512, 1)
            self.adv1 = nn.Linear(64 * 7 * 7, 512)
            self.adv2 = nn.Linear(512, a_size)
        else:
            self.fc1 = nn.Linear(64 * 7 * 7, 512)
            self.fc2 = nn.Linear(512, a_size)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        if self.dueling:
            v = self.val2(F.relu(self.val1(x)))
            a = self.adv2(F.relu(self.adv1(x)))
            return v + a - a.mean(dim=1, keepdim=True)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def act(self, state, epsilon=0.0):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.a_size)
        state = torch.tensor(np.asarray(state), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(state)
        return torch.argmax(q_values, dim=1).item()

    def act_batch(self, states, epsilon=0.0):
        """Epsilon-greedy actions for a batch of states (one per env)."""
        states = torch.tensor(np.asarray(states), dtype=torch.float32, device=device)
        with torch.no_grad():
            actions = self.forward(states).argmax(dim=1).cpu().numpy()
        explore = np.random.rand(len(actions)) < epsilon
        actions[explore] = np.random.randint(self.a_size, size=int(explore.sum()))
        return actions


# ----------------------------------------------------------------------------
# Replay + n-step
# ----------------------------------------------------------------------------
class ReplayBuffer:
    """Preallocated ring buffer; frames stay uint8.

    Besides the reward it stores `term` (true termination -> no bootstrap) and
    `discount` (= gamma ** n_actual_steps) so that n-step targets are exact.
    """

    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terms = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, term, discount):
        i = self.pos
        self.states[i] = state
        self.next_states[i] = next_state
        self.actions[i] = action
        self.rewards[i] = reward
        self.terms[i] = term
        self.discounts[i] = discount
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        states = torch.from_numpy(self.states[idx]).to(device, non_blocking=True)
        next_states = torch.from_numpy(self.next_states[idx]).to(device, non_blocking=True)
        actions = torch.from_numpy(self.actions[idx]).to(device).unsqueeze(1)
        rewards = torch.from_numpy(self.rewards[idx]).to(device).unsqueeze(1)
        terms = torch.from_numpy(self.terms[idx]).to(device).unsqueeze(1)
        discounts = torch.from_numpy(self.discounts[idx]).to(device).unsqueeze(1)
        return states, actions, rewards, next_states, terms, discounts

    def __len__(self):
        return self.size


class NStepCollector:
    """Per-environment n-step accumulator feeding a ReplayBuffer.

    For n=1 it degenerates to plain 1-step transitions. On episode end the window
    is flushed with shorter horizons. `terminated` stops bootstrapping; a time-limit
    `truncated` still bootstraps from the final observation.
    """

    def __init__(self, num_envs, n, gamma, buffer):
        self.n = n
        self.gamma = gamma
        self.buffer = buffer
        self.queues = [deque() for _ in range(num_envs)]

    def _emit(self, i, next_state, term):
        q = self.queues[i]
        ret = 0.0
        for k, (_, _, r) in enumerate(q):
            ret += (self.gamma ** k) * r
        s0, a0, _ = q[0]
        self.buffer.push(s0, a0, ret, next_state, float(term), self.gamma ** len(q))
        q.popleft()

    def add(self, i, state, action, reward, next_state, terminated, truncated):
        q = self.queues[i]
        q.append((state, action, float(reward)))
        if len(q) == self.n:
            self._emit(i, next_state, terminated)
        if terminated or truncated:
            while q:
                self._emit(i, next_state, terminated)


def random_shift(x, pad):
    """DrQ-style random shift. x: float tensor (N, C, H, W)."""
    n, c, h, w = x.size()
    x = F.pad(x, (pad,) * 4, mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device, dtype=x.dtype)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
    shift *= 2.0 / (h + 2 * pad)
    return F.grid_sample(x, base_grid + shift, padding_mode="zeros", align_corners=False)


# ----------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------
@dataclass
class Config:
    name: str = "exp"
    # --- loop (defaults = baseline) ---
    num_envs: int = 8
    n_episodes: int = 1000
    updates_per_step: int = 2
    lr: float = 1e-4
    gamma: float = 0.99
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995        # multiplied per finished episode
    buffer_size: int = 50000
    batch_size: int = 128
    learn_start: int = 128
    target_update: int = 1000           # hard update period (in learn steps)
    # --- improvements (all off = baseline) ---
    tau: float = 0.0                    # >0 -> soft target update, replaces hard update
    double: bool = False
    dueling: bool = False
    n_step: int = 1
    aug_pad: int = 0                    # >0 -> random shift of this many pixels
    # --- bookkeeping ---
    validate_every: int = 25
    eval_episodes: int = 5
    top_k: int = 3
    seed: int = 0
    out_dir: str = "runs"


def eps_decay_for(n_episodes, eps_min=0.05, frac=0.6):
    """Per-episode decay that reaches eps_min after `frac` of the budget.
    For n_episodes=1000 this gives ~0.995, i.e. the baseline schedule."""
    return eps_min ** (1.0 / (frac * n_episodes))


class DQNAgent:
    def __init__(self, cfg, action_dim, obs_shape):
        self.cfg = cfg
        self.action_dim = action_dim
        self.epsilon = 1.0
        self.model = QNetwork(a_size=action_dim, dueling=cfg.dueling).to(device)
        self.target_model = QNetwork(a_size=action_dim, dueling=cfg.dueling).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.buffer_size, obs_shape)
        self.scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        self.learn_steps = 0

    def decay_epsilon(self):
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def replay(self):
        cfg = self.cfg
        if len(self.memory) < max(cfg.batch_size, cfg.learn_start):
            return
        states, actions, rewards, next_states, terms, discounts = self.memory.sample(cfg.batch_size)
        if cfg.aug_pad > 0:
            states = random_shift(states.float(), cfg.aug_pad)
            next_states = random_shift(next_states.float(), cfg.aug_pad)

        with torch.autocast(device.type, enabled=device.type == "cuda"):
            q_values = self.model(states).gather(1, actions)
            with torch.no_grad():
                if cfg.double:
                    best = self.model(next_states).argmax(1, keepdim=True)
                    next_q = self.target_model(next_states).gather(1, best)
                else:
                    next_q = self.target_model(next_states).max(1, keepdim=True)[0]
                target = rewards + discounts * next_q.float() * (1.0 - terms)
            loss = F.smooth_l1_loss(q_values.float(), target)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.learn_steps += 1
        if cfg.tau > 0:
            with torch.no_grad():
                for p, tp in zip(self.model.parameters(), self.target_model.parameters()):
                    tp.mul_(1.0 - cfg.tau).add_(p, alpha=cfg.tau)
        elif self.learn_steps % cfg.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())


# ----------------------------------------------------------------------------
# Evaluation (same loop as test_car_racing.py, greedy, no video)
# ----------------------------------------------------------------------------
def evaluate_policy(model, n_episodes=5, env=None):
    own_env = env is None
    if own_env:
        env = make_env()
    was_training = model.training
    model.eval()
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = model.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        rewards.append(total)
    if was_training:
        model.train()
    if own_env:
        env.close()
    return np.array(rewards)


def select_best(paths, n_episodes=20, dueling=None):
    """Re-evaluate candidate checkpoints on fresh episodes and return a ranking."""
    env = make_env()
    out = []
    for p in paths:
        m = QNetwork(a_size=env.action_space.n, dueling=dueling).to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        r = evaluate_policy(m, n_episodes, env=env)
        print(f"{os.path.basename(p)}: mean {r.mean():.1f}  std {r.std():.1f}  min {r.min():.1f}")
        out.append((float(r.mean()), p))
    env.close()
    return sorted(out, reverse=True)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg):
    """Train with several CarRacing envs stepped in parallel (as in the baseline).

    Returns a dict with paths of the logs / checkpoints. Checkpoints:
      <name>_best_train.pth : best 100-episode average of the (noisy) training score
                              -> the baseline's selection rule
      <name>_best_eval.pth  : best greedy evaluation score
      <name>_ep*_eval*.pth  : top-k greedy-evaluation candidates
      <name>_last.pth       : final weights
    """
    set_seed(cfg.seed)
    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(cfg.out_dir, f"{cfg.name}.json")

    env = make_vec_env(cfg.num_envs)
    eval_env = make_env()
    agent = DQNAgent(cfg, env.single_action_space.n, env.single_observation_space.shape)
    collector = NStepCollector(cfg.num_envs, cfg.n_step, cfg.gamma, agent.memory)

    log = {"config": asdict(cfg), "train": [], "eval": []}
    scores_deque = deque(maxlen=100)
    best_train, best_eval = -np.inf, -np.inf
    top = []  # (score, path)
    t0 = time.time()

    def save_log():
        with open(log_path, "w") as f:
            json.dump(log, f)

    states, _ = env.reset(seed=cfg.seed)
    prev_done = np.zeros(cfg.num_envs, dtype=bool)
    episode_rewards = np.zeros(cfg.num_envs)
    completed = 0

    while completed < cfg.n_episodes:
        actions = agent.model.act_batch(states, epsilon=agent.epsilon)
        next_states, rewards, terminations, truncations, _ = env.step(actions)
        dones = np.logical_or(terminations, truncations)

        for i in range(cfg.num_envs):
            if prev_done[i]:
                continue  # autoreset step: next_states[i] is a fresh episode start
            collector.add(i, states[i], actions[i], rewards[i], next_states[i],
                          bool(terminations[i]), bool(truncations[i]))
            episode_rewards[i] += rewards[i]
            if not dones[i]:
                continue

            score = float(episode_rewards[i])
            episode_rewards[i] = 0
            completed += 1
            scores_deque.append(score)
            agent.decay_epsilon()
            avg = float(np.mean(scores_deque))
            log["train"].append({"episode": completed, "score": score, "avg100": avg,
                                 "eps": agent.epsilon, "minutes": (time.time() - t0) / 60})
            print(f"[{cfg.name}] ep {completed}\tscore {score:7.1f}\tavg100 {avg:7.1f}\teps {agent.epsilon:.3f}")

            if avg > best_train and len(scores_deque) >= 10:
                best_train = avg
                torch.save(agent.model.state_dict(), os.path.join(ckpt_dir, f"{cfg.name}_best_train.pth"))

            if completed % cfg.validate_every == 0 or completed == cfg.n_episodes:
                r = evaluate_policy(agent.model, cfg.eval_episodes, env=eval_env)
                s = float(r.mean())
                log["eval"].append({"episode": completed, "mean": s, "std": float(r.std()),
                                    "min": float(r.min()), "max": float(r.max())})
                print(f"[{cfg.name}]   greedy eval @ {completed}: mean {s:.1f} (min {r.min():.1f}, max {r.max():.1f})")
                if s > best_eval:
                    best_eval = s
                    torch.save(agent.model.state_dict(), os.path.join(ckpt_dir, f"{cfg.name}_best_eval.pth"))
                if len(top) < cfg.top_k or s > min(top)[0]:
                    path = os.path.join(ckpt_dir, f"{cfg.name}_ep{completed}_eval{s:.0f}.pth")
                    torch.save(agent.model.state_dict(), path)
                    top.append((s, path))
                    if len(top) > cfg.top_k:
                        worst = min(top)
                        top.remove(worst)
                        os.remove(worst[1])
                save_log()
            if completed >= cfg.n_episodes:
                break

        states = next_states
        prev_done = dones
        for _ in range(cfg.updates_per_step):
            agent.replay()

    torch.save(agent.model.state_dict(), os.path.join(ckpt_dir, f"{cfg.name}_last.pth"))
    env.close()
    eval_env.close()
    log["top"] = [p for _, p in sorted(top, reverse=True)]
    log["minutes"] = (time.time() - t0) / 60
    save_log()
    return {
        "log": log_path,
        "best_train": os.path.join(ckpt_dir, f"{cfg.name}_best_train.pth"),
        "best_eval": os.path.join(ckpt_dir, f"{cfg.name}_best_eval.pth"),
        "top": log["top"],
        "last": os.path.join(ckpt_dir, f"{cfg.name}_last.pth"),
    }


def render(checkpoint, num_episodes=3, video_folder="videos", dueling=None):
    """Record the trained agent playing to MP4 files (works headless)."""
    env = make_env(render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, name_prefix="car_racing",
                      episode_trigger=lambda e: True)
    policy = QNetwork(a_size=env.action_space.n, dueling=dueling).to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device))
    policy.eval()
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            state, _, terminated, truncated, _ = env.step(policy.act(state))
            done = terminated or truncated
    env.close()
    print(f"Saved videos to {video_folder}/")
