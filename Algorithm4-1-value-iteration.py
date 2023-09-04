# P71
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from maze_env_5_5 import Maze

gamma = 0.9


def update_by_order(env, q_state_action):
    [_, mapW] = np.shape(env.map)
    vk = np.ones_like(env.state_value)
    vk_new = np.zeros_like(env.state_value)
    k = 0
    value = [np.abs(np.linalg.norm(vk_new, 2) - np.linalg.norm(vk, 2))]
    print(np.abs(np.linalg.norm(vk_new - vk, 2)))
    while np.abs(np.linalg.norm(vk_new - vk, 2)) >= 0.001:  # k <= 3:  #- np.linalg.norm(vk, 2)) >= 0.001
        vk = np.copy(vk_new)
        for i in range(0, q_state_action.shape[0]):
            env.reset([1 + i // mapW, 1 + i % mapW])
            s = [1 + i // mapW, 1 + i % mapW]

            for action in range(1, q_state_action.shape[1] + 1):
                env.reset([1 + i // mapW, 1 + i % mapW])
                s_, r, done = env.step(action)  # env.step(action=1,2,3,4,5)
                if not done:
                    s_ = s
                env.render()
                q_state_action[i, action - 1] = r + gamma * env.state_value[(s_[0] - 1) * mapW + s_[1] - 1]
                # print(i, env.s, j, s_, r, done)
            # Maximum action value
            a_star = q_state_action[i, :].argmax()
            # Policy update
            env.policy_state_action[i, :] = 0
            env.policy_state_action[i, a_star] = 1
            # Value update
            max_index = np.argwhere(q_state_action[i, :] == np.amax(q_state_action[i, :]))
            vk_new[i] = max(q_state_action[i, random.choice(max_index)])
            # print("k=", k, "s=", s)
            # print('vk_new-vk')
            # print(np.linalg.norm(vk_new - vk))
            # print("policy_state_action")
            # print(env.policy_state_action)
            # print("q_state_action")
            # print(q_state_action)

        env.state_value = np.copy(vk_new)
        k += 1
        value.append(np.abs(np.linalg.norm(vk_new - vk, 2)))
        print(k, "iter", 'loss:', abs(np.linalg.norm(vk_new, 2) - np.linalg.norm(vk, 2)))

        # if k % 10 == 0:
        #     print("10*", k // 10, "times already!")

        # time.sleep(20)
        # env.render()

        # end of game

        # print('vk_new-vk')
        # print(np.linalg.norm(vk_new - vk))

    env.show_policy()
    env.show_state_value()
    # plt.figure(figsize=(10, 10))  # 设置绘图大小为20*15
    # plt.xlabel('time(s)')  # 设置x、y轴标签
    # plt.ylabel('|loss|')
    # plt.grid()
    # plt.plot(value)
    # plt.show()

    # time.sleep(0)


# env.destroy()


# 错误的思想，错误的算法
# def update_by_trajectory(env, q_state_action):
#     [_, mapW] = np.shape(env.map)
#     vk = env.state_value
#     vk_new = np.ones_like(env.state_value)
#     k = 0
#     s = env.start
#     while k <= 500:  # np.linalg.norm(vk_new - vk) >= 0.00005:  #
#         s = env.s
#         i = 5 * (s[0] - 1) + s[1] - 1
#         for j in range(1, q_state_action.shape[1] + 1):
#             # env.reset([1 + i // mapW, 1 + i % mapW])
#             s_, r, done = env.step(j)  # env.step(action=1,2,3,4,5)
#             print(s, j, s_, r)
#             if not done:
#                 s_ = s
#             env.render()
#             q_state_action[i, j - 1] = r + gamma * env.state_value[i]
#         # Maximum action value
#         a_star = q_state_action[i, :].argmax()
#         # Policy update
#         env.policy_state_action[i, :] = 0
#         env.policy_state_action[i, a_star] = 1
#         # Value update
#         vk = vk_new
#         vk_new[i] = max(q_state_action[i, :])
#         # print("k=", k, "s=", s)
#         # print('vk_new-vk')
#         # print(np.linalg.norm(vk_new - vk))
#         # print("policy_state_action")
#         # print(env.policy_state_action)
#         # print("q_state_action")
#         # print(q_state_action)
#     env.state_value = vk_new
#     k += 1
#     if k % 10 == 0:
#         print("10*", k // 10, "times already!")
#     env.show_policy()
#     # env.show_state_value()
#     time.sleep(0)
#     env.render()


if __name__ == "__main__":
    env = Maze()
    [mapH, mapW] = np.shape(env.map)
    q_state_action = np.zeros([mapH * mapW, env.n_actions])
    env.after(100, update_by_order(env, q_state_action))
    env.mainloop()
