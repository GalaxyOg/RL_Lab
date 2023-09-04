# 开发时间：2023/8/21 21:06 by Og
import time

import numpy as np

from maze_env_5_5 import Maze

gamma = 0.9


def update(env, q_state_action):
    [_, mapW] = np.shape(env.map)
    v_p_k = np.ones_like(env.state_value)
    v_p_kp1 = np.zeros_like(env.state_value)

    # vk_j = np.empty_like(env.state_value)
    # vk_j_new = np.ones_like(env.state_value)
    k = 0
    env.policy_uniformization()
    print(k, "iters PI", 'loss:', np.abs(np.linalg.norm(v_p_kp1, 2) - np.linalg.norm(v_p_k, 2)))
    while np.abs(np.linalg.norm(v_p_kp1, 2) - np.linalg.norm(v_p_k, 2)) >= 0.1:
        v_p_k = np.copy(v_p_kp1)
        j = 0
        v_p_k_j0 = np.ones_like(env.state_value)
        print("Policy evaluation:")
        # policy evaluation
        v_p_kp1 = np.zeros_like(env.state_value)
        while np.abs(np.linalg.norm(v_p_kp1 - v_p_k_j0)) >= 0.01:  # j <= 40:  #- np.linalg.norm(, 2))
            v_p_k_j0 = np.copy(v_p_kp1)
            for i in range(0, q_state_action.shape[0]):
                s = [1 + i // mapW, 1 + i % mapW]
                env.reset(s)
                action = env.policy_state_action[i, :].argmax() + 1  #
                s_, r, done = env.step(action)
                v_p_kp1[i] = r + gamma * v_p_k_j0[(s_[0] - 1) * mapW + s_[1] - 1]
                # for action in range(1, env.n_actions + 1):
                #     env.reset(s)
                #     s_, r, done = env.step(action)
                #     v_p_kp1[i] += env.policy_state_action[i, action - 1] * (
                #             r + gamma * v_p_k_j0[(s_[0] - 1) * mapW + s_[1] - 1])
            env.state_value = np.copy(v_p_kp1)
            # print("state_value", env.state_value)
            # print(j, ":", v_p_kp1, v_p_k_j0)
            j += 1
            if j % 20 == 0:
                print("     j=", j, "iters PE", "loss:", np.abs(np.linalg.norm(v_p_kp1) - np.linalg.norm(v_p_k_j0, 2)))

        print("Policy improvement:")
        # policy improvement
        for i in range(0, q_state_action.shape[0]):
            s = [1 + i // mapW, 1 + i % mapW]
            for action in range(1, env.n_actions + 1):
                env.reset(s)
                s_, r, done = env.step(action)
                q_state_action[i, action - 1] = r + gamma * v_p_kp1[(s_[0] - 1) * mapW + s_[1] - 1]
            a_star = q_state_action[i, :].argmax()
            env.policy_state_action[i, :] = 0
            env.policy_state_action[i, a_star] = 1
        # print("env.policy_state_action=", env.policy_state_action)
        k += 1
        env.policy_uniformization()
        print(k, "iters PI", 'loss:', np.abs(np.linalg.norm(v_p_kp1, 2) - np.linalg.norm(v_p_k, 2)))
    env.show_policy()
    env.show_state_value()


if __name__ == "__main__":
    env = Maze()
    [mapH, mapW] = np.shape(env.map)
    q_state_action = np.zeros([mapH * mapW, env.n_actions])
    env.after(100, update(env, q_state_action))
    env.mainloop()
