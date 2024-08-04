import numpy as np
from scipy.optimize import minimize
from math import *
import matplotlib.pyplot as plt
import random
import csv
import pandas as pd

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


def random_start():
    # 目标区域在一个圆形的范围内
    x_center = 1.5 * L
    y_center = 0.5 * L

    # 生成随机角度
    angle = random.uniform(0, 2 * pi)

    # 生成随机半径
    # radius = random.uniform(0, 0.25 * L)
    radius = 0

    # 将极坐标转换为直角坐标
    x = x_center + radius * cos(angle)
    y = y_center + radius * sin(angle)
    startpoint = [x, y]
    return startpoint


def random_target():
    # 目标区域在一个圆形的范围内
    x_center = 0.5 * L
    y_center = 0.5 * L

    # 生成随机角度
    angle = random.uniform(0, 2 * pi)

    # 生成随机半径
    # radius = random.uniform(0, 0.25 * L)
    radius = 0

    # 将极坐标转换为直角坐标
    x = x_center + radius * cos(angle)
    y = y_center + radius * sin(angle)
    startpoint = [x, y]
    return startpoint


# 动力学函数
def f_1(pos_x, t):
    """

    param pos_x:          position x
    param t:        simulation time

    return: f(x,t)
    """
    out_put = epsilon * sin(omega * t) * pos_x ** 2 + pos_x - 2 * epsilon * sin(omega * t) * pos_x
    return out_put


def psi(pos_x, pos_y, t):
    """

    param pos_x:          position x
    param pos_y:          position y
    param t:        simulation time

    return: ψ(x,y,t)
    """

    out_put = A * sin(pi * f_1(pos_x, t)) * sin(pi * pos_y)
    return out_put


def U_flow_x(pos_x, pos_y, t):
    """

    param pos_x:          position x
    param pos_y:          position y
    param t:        simulation time

    return: vflow_x
    """

    out_put = -pi * A * sin(pi * f_1(pos_x, t)) * cos(pi * pos_y)
    return out_put


def U_flow_y(pos_x, pos_y, t):
    """

    param pos_x:          position x
    param pos_y:          position y
    param t:        simulation time

    return: vflow_y
    """

    out_put = pi * A * cos(pi * f_1(pos_x, t)) * sin(pi * pos_y) * (
            2 * epsilon * sin(omega * t) * pos_x
            + 1 - 2 * epsilon * sin(omega * t))
    return out_put


def agent_dynamics_withtime(state, action):
    pos_x, pos_y, t = state[0], state[1], state[2]
    flow_x = U_flow_x(pos_x, pos_y, t)
    flow_y = U_flow_y(pos_x, pos_y, t)
    dx = (U_swim * cos(action) + flow_x) * dt
    dy = (U_swim * sin(action) + flow_y) * dt
    new_x = pos_x + dx
    new_y = pos_y + dy
    new_t = t + dt
    state_new = [new_x, new_y, new_t]
    return state_new


# 计算距离
def distance_to_goal(current, goal):
    dx = current[0] - goal[0]
    dy = current[1] - goal[1]
    # hypot(dx, dy),返回欧式距离
    return hypot(dx, dy)


# 初始状态和目标状态
x0 = np.array([1.5 * L, 0.5 * L, 0])
xf = np.array([0.5 * L, 0.5 * L, 0])


# 计算轨迹的总时间步长
def compute_total_time(path):
    total_time = 0
    for i in range(1, len(path)):
        total_time += np.linalg.norm(path[i] - path[i - 1])
    return total_time


# 计算轨迹的平滑性
def compute_smoothness(path):
    smoothness = 0
    path = np.array(path)  # 确保路径是 NumPy 数组
    for i in range(1, len(path) - 1):
        smoothness += np.linalg.norm(path[i - 1] - 2 * path[i] + path[i + 1])
    return smoothness


# 定义目标函数
def objective(control_in):
    # 目标：到达终点k1(1)
    # 目标：缩短时间步长k2(10)
    # 目标：平滑轨迹k3(1)
    k1 = 1000
    k2 = 5
    k3 = 1
    path = []
    time_cost = 0
    state = x0
    path.append([state[0], state[1]])
    for i in range(len(control_in)):
        state = agent_dynamics_withtime(state, control_in[i])
        time_cost += 1
        path.append([state[0], state[1]])
    smooth_cost = compute_smoothness(path)
    goal_cost = distance_to_goal(path[-1], [xf[0], xf[1]])
    # 对损失函数的定义
    print("time_cost", time_cost)
    print("smooth_cost", smooth_cost)
    print("goal_cost", goal_cost)
    cost = k2 * time_cost + k3 * smooth_cost + k1 * goal_cost
    return cost


# 计算目标函数的梯度
def compute_gradient(control_in):
    eps = 1e-8
    object_gradient = np.zeros_like(control_in)
    for i in range(len(control_in)):
        control_in[i] += eps
        f1 = objective(control_in)
        control_in[i] -= 2 * eps
        f2 = objective(control_in)
        object_gradient[i] = (f1 - f2) / (2 * eps)
        control_in[i] += eps
    return object_gradient


# 投影函数：将控制输入投影到约束集内
def project(control_in, input_min, input_max):
    return np.clip(control_in, input_min, input_max)


# 设置控制输入的上下限
u_min = -pi
u_max = pi

# 学习率
learning_rate = 0.01

# 最大迭代次数
max_iter = 100

action_dir = './RRT_output/19步/actions.csv'
df = pd.read_csv(action_dir, header=None, dtype=str)  # 读取为字符串类型
data = df.values.flatten().astype(float).tolist()

initial_guess = data
print("RRT控制输入", initial_guess)

# 投影梯度下降算法
control_inputs = initial_guess
for _ in range(max_iter):
    gradient = compute_gradient(control_inputs)
    control_inputs = control_inputs - learning_rate * gradient
    control_inputs = project(control_inputs, u_min, u_max)

    # 动态调整路径的点数
    new_control_points = [control_inputs[0]]
    for i in range(1, len(control_inputs)):
        if np.linalg.norm(control_inputs[i] - control_inputs[i - 1]) > 0.1:  # 合并相邻的近似点
            new_control_points.append(control_inputs[i])
    control_inputs = np.array(new_control_points)

# 输出优化结果
print("最优控制输入：", control_inputs)
print("最优目标函数值：", objective(control_inputs))

# 绘制轨迹
x = x0
trajectory = [x]
U_opt = control_inputs
for k in range(len(control_inputs)):
    x = agent_dynamics_withtime(x, U_opt[k])
    trajectory.append(x)

state_trajectory = np.array(trajectory)
print("末位置", trajectory[-1])
print(f'距离差:{distance_to_goal(trajectory[-1], xf)};限制距离:{L / 50}')

plt.plot(state_trajectory[:, 0], state_trajectory[:, 1], marker='o')
plt.title('系统状态轨迹')
plt.xlabel('位置 x')
plt.ylabel('位置 y')
plt.grid(True)
plt.show()
