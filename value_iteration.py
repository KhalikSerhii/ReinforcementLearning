import gymnasium as gym
import numpy as np
import time
import json
from mountain_car_mdp import build_P, discretize, N_POS, N_VEL, N_ACTIONS



def value_iteration(P, num_iterations=3000, threshold=1e-6, gamma=0.99):
    n_states = len(P)
    n_actions = len(P[0])
    value_table = np.zeros(n_states)

    for i in range(num_iterations):
        updated_value_table = np.copy(value_table)

        for s in range(n_states):
            Q_values = [sum([prob * (r + gamma * updated_value_table[s_] * (not done))
                             for prob, s_, r, done in P[s][a]])
                        for a in range(n_actions)]

            value_table[s] = max(Q_values)

        delta = np.max(np.fabs(updated_value_table - value_table))
        if i % 50 == 0:
            print(f"iteration {i}: delta = {delta:.6f}")

        if delta <= threshold:
            break

    print(f"Value iteration converged after {i + 1} iterations")
    return value_table, i + 1


def extract_policy(P, value_table, gamma=0.99):
    n_states = len(P)
    n_actions = len(P[0])

    policy = np.zeros(n_states)

    for s in range(n_states):
        Q_values = [sum([prob * (r + gamma * value_table[s_] * (not done))
                         for prob, s_, r, done in P[s][a]])
                    for a in range(n_actions)]

        policy[s] = np.argmax(np.array(Q_values))

    return policy


def test(env, optimal_policy, render=True):

    state, _ = env.reset()
    state = discretize(state)

    total_reward = 0
    counter = 0
    success = False
    for _ in range(200):
        action = int(optimal_policy[state])
        obs, reward, terminated, truncated, _ = env.step(action)
        state = discretize(obs)

        if render:
            env.render()

        total_reward += reward
        counter += 1
        if terminated:
            success = True
        if terminated or truncated:
            break

    return total_reward, counter, success


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    
    P = build_P(env)

    start = time.time()
    optimal_value_function, n_iter = value_iteration(P)
    optimal_policy = extract_policy(P, optimal_value_function)
    elapsed = time.time() - start
    print(f"Value iteration time: {elapsed:.2f} s")

    np.save("policy_vi.npy", optimal_policy)
    np.save("value_vi.npy", optimal_value_function)

    # tests in enviroment
    n_episodes = 1000
    sum_reward = 0
    successes = 0
    steps_on_success = 0
    for _ in range(n_episodes):
        total_reward, counter, success = test(env, optimal_policy, render=False)
        sum_reward += total_reward
        if success:
            successes += 1
            steps_on_success += counter

    print("Average reward:", sum_reward / n_episodes)
    print("Success rate:", successes / n_episodes)
    if successes > 0:
        print("Average steps (successful episodes):", steps_on_success / successes)

    env.close()
