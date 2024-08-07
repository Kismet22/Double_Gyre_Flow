import random

import numpy as np
from math import *
from FlowEnvironment import Double_gyre_Flow
from RRTController import RRT
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 常数设置
U_swim = 0.9
A = 2 * U_swim / 3
epsilon = 0.3
L = 1
target_center = (0.5 * L, 0.5 * L)
start_center = (.5 * L, 1.5 * L)
D_range = 0.5 * L
omega = 20 * pi * U_swim / (3 * L)
D = 1
dt = 0.1

env_in = Double_gyre_Flow(U_swim=U_swim, L=L, epsilon=epsilon, dt=dt)


# 结合RRT控制器使用的最优化控制器

class Optimal_Controller:
    def __init__(self, start, target, env, map_dimensions, smooth_matrix, target_matrix, u_min,
                 u_max, time_steps=50, use_rrt=False):
        self.start = start
        self.goal = target
        self.env = env
        self.map_dimensions = map_dimensions
        self.smooth_matrix = smooth_matrix
        self.target_matrix = target_matrix
        self.u_min = u_min
        self.u_max = u_max
        self.N = time_steps
        self.use_rrt = use_rrt
        self.initial_guess = np.zeros(self.N)  # 初始化猜想

    def _init_initial_guess(self):
        if self.use_rrt:
            len_old = self.N
            action_list_init = np.zeros(self.N)
            for _ in range(50):
                rrt = RRT(self.start, self.goal, self.env, self.map_dimensions,
                          expand_distance=self.env.L / 50, obstacle_list=None)
                path, time_list, action_list = rrt.plan()
                len_new = len(action_list)
                if len_new < len_old:
                    action_list_init[:len_new] = action_list
                    len_old = len_new

            self.initial_guess[:len(action_list_init)] = action_list_init

        else:
            self.initial_guess = np.zeros(self.N)

    def object(self, U):
        Q_to_target = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        U = U.reshape(self.N, -1)
        x_old = self.start
        cost = 0
        for number in range(self.N):
            x_new = self.env.agent_dynamics_withtime(x_old, U[number])
            cost += (x_new - np.array(x_old)).T @ self.smooth_matrix @ (x_new - np.array(x_old)) +\
                    (x_new - self.goal).T @ self.target_matrix @ (x_new - self.goal)
            x_old = x_new
            distance_to_target = (x_old - self.goal).T @ Q_to_target @ (x_old - self.goal)
            if distance_to_target <= (self.env.L / 50) ** 2:
                break
        cost += (x_old - self.goal).T @ self.target_matrix @ (x_old - self.goal)
        return cost

    def constraint_u_min(self, U):
        U = U.reshape(self.N, -1)
        return (U - self.u_min).flatten()

    def constraint_u_max(self, U):
        U = U.reshape(self.N, -1)
        return (self.u_max - U).flatten()

    def plan(self):
        constraints = [{'type': 'ineq', 'fun': self.constraint_u_min},
                       {'type': 'ineq', 'fun': self.constraint_u_max}]
        objective = self.object
        self._init_initial_guess()
        initial_guess = self.initial_guess.copy()  # 确保初始猜想是一个新的数组
        # 使用SLSQP方法进行优化
        for i in range(9):
            result = minimize(objective, initial_guess, method='SLSQP', constraints=constraints)
            initial_guess = result.x
        result = minimize(objective, initial_guess, method='SLSQP', constraints=constraints)
        return result


r = random.uniform(0, 0.25 * L)
angle = random.uniform(0, 2 * pi)
x0 = np.array([0.5 * L + r * cos(angle), 1.25 * L + r * sin(angle), 0])
xf = np.array([0.5 * L + r * cos(angle), 0.25 * L + r * sin(angle), 0])
Q_smooth = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 10]])
# Q_smooth = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 10]])
Q_target = np.array([[50, 0, 0], [0, 50, 0], [0, 0, 0]])
u_min = np.array([-pi])
u_max = np.array([pi])
N = 50
controller = Optimal_Controller(start=x0, target=xf, env=env_in, map_dimensions=[0, 3], smooth_matrix=Q_smooth,
                                target_matrix=Q_target, u_min=u_min, u_max=u_max, time_steps=N,
                                use_rrt=True)

result = controller.plan()

# 绘制轨迹
x = x0
trajectory = [x]
U_opt = result.x.reshape(N, -1)
print(U_opt)
for k in range(N):
    x = env_in.agent_dynamics_withtime(x, U_opt[k])
    trajectory.append(x)
    if env_in.distance_to_goal(trajectory[-1], xf) <= L / 50:
        break

state_trajectory = np.array(trajectory)
print("末位置", trajectory[-1])
print("时间步长", k)
print(f'距离差:{env_in.distance_to_goal(trajectory[-1], xf)};限制距离:{L / 50}')

plt.plot(state_trajectory[:, 0], state_trajectory[:, 1], marker='o')
plt.plot(x0[0], x0[1], 'ro', label='Start Position')
plt.plot(xf[0], xf[1], 'go', label='Target Position')
circle = plt.Circle((xf[0], xf[1]), radius=L / 50, color='g', fill=False, linestyle='--', linewidth=1.5)
plt.gca().add_patch(circle)
plt.title('系统状态轨迹')
plt.xlabel('位置 x')
plt.ylabel('位置 y')
plt.grid(True)
plt.show()
