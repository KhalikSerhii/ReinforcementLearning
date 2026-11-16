import numpy as np
import gymnasium as gym

N_POS = 100
N_VEL = 100
POS_MIN, POS_MAX = -1.2, 0.6
VEL_MIN, VEL_MAX = -0.07, 0.07

# Actions = [move left, do nothing, move right]
N_ACTIONS = 3

POS_STEP = (POS_MAX - POS_MIN) / (N_POS - 1)
VEL_STEP = (VEL_MAX - VEL_MIN) / (N_VEL - 1)


def discretize(obs):
    i = int(np.clip(round((obs[0] - POS_MIN) / POS_STEP), 0, N_POS - 1))
    j = int(np.clip(round((obs[1] - VEL_MIN) / VEL_STEP), 0, N_VEL - 1))
    return i * N_VEL + j


def build_P(env):
    """
    Будує модель середовища у тому ж форматі, що й env.unwrapped.P у FrozenLake:
        P[s][a] = [(prob, next_state, reward, done), ...]

    Динаміка Mountain Car детермінована, але наступний стан зазвичай потрапляє між вузлами сітки.
    Тому кожен перехід розподіляється між 4 сусідніми вузлами (білінійна інтерполяція),
    а ваги вузлів використовуються як ймовірності prob.
    Сама динаміка береться зі справжнього env.step(), нічого не вигадується вручну.
    """
    unwrapped = env.unwrapped
    unwrapped.reset(seed=0)

    P = [[[] for _ in range(N_ACTIONS)] for _ in range(N_POS * N_VEL)]

    for i in range(N_POS):
        for j in range(N_VEL):
            s = i * N_VEL + j
            pos = POS_MIN + i * POS_STEP
            vel = VEL_MIN + j * VEL_STEP

            for a in range(N_ACTIONS):
                # step
                unwrapped.state = np.array([pos, vel], dtype=np.float64)
                obs, reward, terminated, _, _ = unwrapped.step(a)

                if terminated:
                    P[s][a].append((1.0, s, reward, True))
                    continue

                fp = (float(obs[0]) - POS_MIN) / POS_STEP
                fv = (float(obs[1]) - VEL_MIN) / VEL_STEP
                i0 = int(np.clip(np.floor(fp), 0, N_POS - 2))
                j0 = int(np.clip(np.floor(fv), 0, N_VEL - 2))
                tp = float(np.clip(fp - i0, 0.0, 1.0))
                tv = float(np.clip(fv - j0, 0.0, 1.0))

                neighbours = [
                    ((1 - tp) * (1 - tv), i0 * N_VEL + j0),
                    ((1 - tp) * tv,       i0 * N_VEL + j0 + 1),
                    (tp * (1 - tv),       (i0 + 1) * N_VEL + j0),
                    (tp * tv,             (i0 + 1) * N_VEL + j0 + 1),
                ]
                for prob, s_ in neighbours:
                    if prob > 1e-12:
                        P[s][a].append((prob, s_, reward, False))

    return P
