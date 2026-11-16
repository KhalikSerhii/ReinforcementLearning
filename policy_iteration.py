import gymnasium as gym
import numpy as np

import time

from mountain_car_mdp import build_P, discretize, N_POS, N_VEL, N_ACTIONS


def show_random_agent(env):
    state, _ = env.reset()

    for t in range(200):
        state, reward, terminated, truncated, _ = env.step(np.random.randint(0, N_ACTIONS))
        env.render()
        if terminated or truncated:
            break
        time.sleep(0.02)

    env.close()


def compute_value_function(policy, P, value_table=None, num_iterations=3000, threshold=1e-6, gamma=0.99):
    n_states = len(P)

    if value_table is None:
        value_table = np.zeros(n_states)

    for i in range(num_iterations):
        updated_value_table = np.copy(value_table)

        for s in range(n_states):
            a = int(policy[s])

            value_table[s] = sum([prob * (r + gamma * updated_value_table[s_] * (not done))
                                  for prob, s_, r, done in P[s][a]])

        if np.max(np.fabs(updated_value_table - value_table)) <= threshold:
            break

    return value_table


def extract_policy(value_table, P, old_policy=None, gamma=0.99):
    n_states = len(P)
    n_actions = len(P[0])

    policy = np.zeros(n_states)

    for s in range(n_states):
        Q_values = [sum([prob * (r + gamma * value_table[s_] * (not done))
                         for prob, s_, r, done in P[s][a]])
                    for a in range(n_actions)]

        best = int(np.argmax(np.array(Q_values)))

        if old_policy is not None and Q_values[int(old_policy[s])] >= Q_values[best] - 1e-9: # rounding error
            policy[s] = old_policy[s]
        else:
            policy[s] = best

    return policy


def policy_iteration(P):
    num_iterations = 200
    n_states = len(P)
    n_actions = len(P[0])

    policy = np.random.randint(0, n_actions, size=(n_states))
    value_function = None

    for i in range(num_iterations):
        value_function = compute_value_function(policy, P, value_table=value_function)
        new_policy = extract_policy(value_function, P, old_policy=policy)

        print(f"iteration {i}: changed actions = {int(np.sum(policy != new_policy))}")

        if np.all(policy == new_policy):
            break
        policy = new_policy

    print(f"Policy iteration converged after {i + 1} iterations")
    return policy, value_function, i + 1


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
    optimal_policy, value_function, n_iter = policy_iteration(P)
    elapsed = time.time() - start
    print(f"Policy iteration time: {elapsed:.2f} s")

    np.save("policy_pi.npy", optimal_policy)
    np.save("value_pi.npy", value_function)

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
