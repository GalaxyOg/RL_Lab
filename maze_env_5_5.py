# 2023-8-20 By Og
#
# maze size 5*5
# action={up,right,down,left,keep}

import numpy as np
import time
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 80  # pixels
MAZE_H = 5  # grid height
MAZE_W = 5  # grid width
# define reward
normal_step_reward = -1
keep_step_reward = -1
forbidden_step_reward = -10
boundary_step_reward = -10
target_step_reward = 0


class Maze(tk.Tk, object):
    # 2023.9.3将start和s更改为np.array,早期为list
    start = np.array([1, 1])  # start position
    s = np.array([1, 1])
    target = np.array([4, 3])
    map = np.array([[0, 0, 0, 0, 0],
                    [0, -1, -1, 0, 0],
                    [0, 0, -1, 0, 0],
                    [0, -1, 1, -1, 0],
                    [0, -1, 0, 0, 0]])
    state_value = np.zeros([MAZE_H * MAZE_W, 1], dtype=float)
    policy_state_action = np.zeros([MAZE_H * MAZE_W, 5], dtype=float)
    policy_state_action.fill(0.2)

    # 随机创建非负policy，调用前需先归一化policy_uniformization
    # policy_state_action = np.abs(np.empty([MAZE_H * MAZE_W, 5]))
    # policy_state_action = np.concatenate((np.ones([MAZE_H * MAZE_W, 1]), np.zeros([MAZE_H * MAZE_W, 4])), 1)

    # policy_state_action = policy_uniformization(np.ones([MAZE_H * MAZE_W, 5]))

    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'r', 'd', 'l', 'k']  # 上、右、下、左、保持
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def policy_uniformization(self):  # policy uniformization
        if np.min(self.policy_state_action) < 0:
            print("Illegal policy!!!")
            exit()
        for i in range(np.shape(self.policy_state_action)[0]):
            r = self.policy_state_action[i, :]
            self.policy_state_action[i, :] = r / sum(r)

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT + 1,
                                width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        origin = np.array([UNIT / 2, UNIT / 2])

        # create forbidden areas and target areas
        for i in range(0, self.map.shape[0]):
            for j in range(0, self.map.shape[1]):
                if self.map[i][j] == -1:
                    temp_center = origin + np.array([UNIT * j, UNIT * i])
                    self.canvas.create_rectangle(
                        temp_center[0] - UNIT / 2, temp_center[1] - UNIT / 2,
                        temp_center[0] + UNIT / 2, temp_center[1] + UNIT / 2,
                        fill='orange')
                if self.map[i][j] == 1:
                    temp_center = origin + np.array([UNIT * j, UNIT * i])
                    self.canvas.create_rectangle(
                        temp_center[0] - UNIT / 2, temp_center[1] - UNIT / 2,
                        temp_center[0] + UNIT / 2, temp_center[1] + UNIT / 2,
                        fill='cyan')

        # create start point and  initial agent
        temp_center = origin + np.array([UNIT * (self.start[1] - 1), UNIT * (self.start[0] - 1)])
        self.canvas.create_rectangle(
            temp_center[0] - UNIT / 2, temp_center[1] - UNIT / 2,
            temp_center[0] + UNIT / 2, temp_center[1] + UNIT / 2,
            fill='pink')
        agent_center = temp_center
        self.agent = self.canvas.create_oval(agent_center[0] - 20, agent_center[1] - 20,
                                             agent_center[0] + 20, agent_center[1] + 20,
                                             fill='lightblue')
        self.canvas.pack()

    def reset(self, state):  # 重置当前位置
        if state[0] > MAZE_H or state[0] <= 0 or state[1] > MAZE_W or state[1] <= 0:
            return False
        self.update()
        # time.sleep(0.5)
        self.canvas.delete(self.agent)
        origin = np.array([UNIT / 2, UNIT / 2])
        agent_center = origin + np.array([UNIT * (state[1] - 1), UNIT * (state[0] - 1)])
        self.agent = self.canvas.create_oval(agent_center[0] - 20, agent_center[1] - 20,
                                             agent_center[0] + 20, agent_center[1] + 20,
                                             fill='blue')
        self.s = np.copy(state)
        # return observation
        return True

    def step(self, action):
        s_ = np.copy(self.s)
        reward = None
        done = None
        if action < 1 or action > 5:
            print("Illegal action input!")
            print("Input action is", action)
            # exit()
        origin = np.array([UNIT / 2, UNIT / 2])
        agent_center = origin + np.array([UNIT * (s_[1] - 1), UNIT * (s_[0] - 1)])
        if action == 1:  # up
            if self.s[0] - 1 >= 1:
                s_[0] = self.s[0] - 1
                agent_center = origin + np.array([UNIT * (s_[1] - 1), UNIT * (s_[0] - 1)])
                if self.map[s_[0] - 1, s_[1] - 1] == 0:
                    reward = normal_step_reward
                elif self.map[s_[0] - 1, s_[1] - 1] == 1:
                    reward = target_step_reward
                else:
                    reward = forbidden_step_reward
                done = True
                pass
            else:
                reward = boundary_step_reward
                done = False
                pass

        if action == 2:  # right
            if self.s[1] + 1 <= MAZE_W:
                s_[1] = self.s[1] + 1
                agent_center = origin + np.array([UNIT * (s_[1] - 1), UNIT * (s_[0] - 1)])
                if self.map[s_[0] - 1, s_[1] - 1] == 0:
                    reward = normal_step_reward
                elif self.map[s_[0] - 1, s_[1] - 1] == 1:
                    reward = target_step_reward
                else:
                    reward = forbidden_step_reward
                done = True
                pass
            else:
                reward = boundary_step_reward
                done = False
                pass
        if action == 3:  # down
            if self.s[0] + 1 <= MAZE_H:
                s_[0] = self.s[0] + 1
                agent_center = origin + np.array([UNIT * (s_[1] - 1), UNIT * (s_[0] - 1)])
                if self.map[s_[0] - 1, s_[1] - 1] == 0:
                    reward = normal_step_reward
                elif self.map[s_[0] - 1, s_[1] - 1] == 1:
                    reward = target_step_reward
                else:
                    reward = forbidden_step_reward
                done = True
                pass
            else:
                reward = boundary_step_reward
                done = False
                pass
        if action == 4:  # left
            if self.s[1] - 1 >= 1:
                s_[1] = self.s[1] - 1
                agent_center = origin + np.array([UNIT * (s_[1] - 1), UNIT * (s_[0] - 1)])
                if self.map[s_[0] - 1, s_[1] - 1] == 0:
                    reward = normal_step_reward
                elif self.map[s_[0] - 1, s_[1] - 1] == 1:
                    reward = target_step_reward
                else:
                    reward = forbidden_step_reward
                done = True
                pass
            else:
                reward = boundary_step_reward
                done = False
                pass
        if action == 5:  # keep still
            s_ = self.s
            agent_center = origin + np.array([UNIT * (self.s[1] - 1), UNIT * (self.s[0] - 1)])
            if self.map[s_[0] - 1, s_[1] - 1] == 1:
                reward = target_step_reward
            elif self.map[s_[0] - 1, s_[1] - 1] == -1:
                reward = forbidden_step_reward
            else:
                reward = keep_step_reward
            done = True

        if done:
            self.s = np.copy(s_)
            self.canvas.delete(self.agent)
            self.agent = self.canvas.create_oval(agent_center[0] - 20, agent_center[1] - 20,
                                                 agent_center[0] + 20, agent_center[1] + 20,
                                                 fill='blue')

            # self.canvas.move(self.agent, 100, 0,)  # move agent

        if not done:
            s_ = np.copy(self.s)
        self.render()
        return s_, reward, done

    def render(self):
        # time.sleep(0.1)
        self.update()

    def show_state_value(self):

        for r in range(0, MAZE_H):
            for c in range(0, MAZE_W):
                text_center = np.array([c * UNIT + UNIT / 2, r * UNIT + UNIT / 2])
                self.canvas.create_text(text_center[0], text_center[1] + 25,
                                        text='{:.3f}'.format(self.state_value[r * MAZE_W + c, 0]))

    # show policy
    def show_policy(self):

        for r in range(0, MAZE_H):
            for c in range(0, MAZE_W):
                text_center = np.array([c * UNIT + UNIT / 2, r * UNIT + UNIT / 2])
                if self.policy_state_action[MAZE_W * r + c, :].argmax() == 0:
                    self.canvas.create_bitmap(text_center[0], text_center[1], bitmap="@pic/up.xbm")
                if self.policy_state_action[MAZE_W * r + c, :].argmax() == 1:
                    self.canvas.create_bitmap(text_center[0], text_center[1], bitmap="@pic/right.xbm")
                if self.policy_state_action[MAZE_W * r + c, :].argmax() == 2:
                    self.canvas.create_bitmap(text_center[0], text_center[1], bitmap="@pic/down.xbm")
                if self.policy_state_action[MAZE_W * r + c, :].argmax() == 3:
                    self.canvas.create_bitmap(text_center[0], text_center[1], bitmap="@pic/left.xbm")
                if self.policy_state_action[MAZE_W * r + c, :].argmax() == 4:
                    self.canvas.create_bitmap(text_center[0], text_center[1], bitmap="@pic/keep.xbm")

        pass

    # key_control() and move() allow keyboard control
    def key_control(self):

        self.canvas.bind_all("<Key>", self.move)
        print("Keyboard Control!")

    def move(self, event):
        if event.char == "w":
            s_, r, _ = self.step(1)
            print("up :", r)
        if event.char == "d":
            s_, r, _ = self.step(2)
            print("right :", r)
        if event.char == "s":
            s_, r, _ = self.step(3)
            print("down :", r)
        if event.char == "a":
            s_, r, _ = self.step(4)
            print("left :", r)
        if event.char == "0":
            s_, r, _ = self.step(5)
            print("keep still :", r)
        print("->", s_)


if __name__ == '__main__':
    env = Maze()
    # print(env.policy_state_action)
    # env.policy_uniformization()
    # print(env.policy_state_action)

    env.key_control()
    # env.show_state_value()
    # env.render()
    # env.show_policy()
    # env.show_state_value()
    env.mainloop()

# q table for test
# r1 = np.concatenate((np.ones([5, 1]), np.zeros([5, 4])), 1)
# r2 = np.concatenate((np.zeros([5, 1]), np.ones([5, 1]), np.zeros([5, 3])), 1)
# r3 = np.concatenate((np.zeros([5, 2]), np.ones([5, 1]), np.zeros([5, 2])), 1)
# r4 = np.concatenate((np.zeros([5, 3]), np.ones([5, 1]), np.zeros([5, 1])), 1)
# r5 = np.concatenate((np.zeros([5, 4]), np.ones([5, 1])), 1)
# q=np.concatenate((r1, r2, r3, r4, r5), 0)
# env.policy_state_action = np.concatenate((r1, r2, r3, r4, r5), 0)
# 开发时间：2023/8/21 20:01
