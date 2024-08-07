import numpy as np
from scipy.optimize import minimize
from math import *
import matplotlib.pyplot as plt
import random
import csv
import pandas as pd
from FlowEnvironment import Double_gyre_Flow

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 常数设置
U_swim = 0.9
A = 2 * U_swim / 3
epsilon = 0.3
L = 1
target_center = (0.5 * L, 0.5 * L)
start_center = (1.5 * L, 0.5 * L)
D_range = 0.5 * L

omega = 20 * pi * U_swim / (3 * L)
D = 1
dt = 0.1

env = Double_gyre_Flow(U_swim=U_swim, L=L, epsilon=epsilon, dt=dt)

# 梯度下降优化的目标：
# 1、确保智能体到达终点范围
# 2、最小化时间步数


# 初始状态和目标状态
x0 = np.array([0.5 * L, 1.5 * L, 0])
xf = np.array([0.25 * L, 0.5 * L, 1.2])

# 最终状态权重矩阵
# @ 矩阵乘法
Q_f = np.array([[100, 0, 0], [0, 100, 0], [0, 0, 300]])

N = 16


# 定义目标函数
def objective(U):
    U = U.reshape(N, -1)
    x = x0
    for k in range(N):
        x = env.agent_dynamics_withtime(x, U[k])
    cost = (x - xf).T @ Q_f @ (x - xf)
    return cost


u_min = np.array([-pi])
u_max = np.array([pi])


# 定义约束条件函数
def constraint_u_min(U):
    U = U.reshape(N, -1)
    return (U - u_min).flatten()


def constraint_u_max(U):
    U = U.reshape(N, -1)
    return (u_max - U).flatten()


action_dir = './RRT_output/29步/actions.csv'
df = pd.read_csv(action_dir, header=None, dtype=str)  # 读取为字符串类型
data = df.values.flatten().tolist()  # 将数据转换为一维列表

# 初始化猜想
initial_guess = np.zeros(N)
print("初始化随机猜想", initial_guess)

"""""""""
# RRT轨迹运行验证
# print(type(data))
# print(len(data))
pos = []
for i in range(len(data)):
    print(x0)
    print("action", float(data[i]))
    x_new = agent_dynamics_withtime(x0, float(data[i]))
    x0 = x_new
    pos.append([x0[0], x0[1]])
print(x0)
"""

for i in range(N):
    initial_guess[i] = data[i]
print("RRT控制输入", initial_guess)

# 定义约束条件
constraints = [{'type': 'ineq', 'fun': constraint_u_min},
               {'type': 'ineq', 'fun': constraint_u_max}]

# 使用SLSQP方法进行优化
for i in range(9):
    result = minimize(objective, initial_guess, method='SLSQP', constraints=constraints)
    initial_guess = result.x
result = minimize(objective, initial_guess, method='SLSQP', constraints=constraints)
# 输出优化结果
print("最优控制输入：", result.x)
print("最优目标函数值：", result.fun)

# 绘制轨迹
x = x0
trajectory = [x]
U_opt = result.x.reshape(N, -1)
for k in range(N):
    x = env.agent_dynamics_withtime(x, U_opt[k])
    trajectory.append(x)

state_trajectory = np.array(trajectory)
print("末位置", trajectory[-1])
print(f'距离差:{env.distance_to_goal(trajectory[-1], xf)};限制距离:{L / 50}')

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
