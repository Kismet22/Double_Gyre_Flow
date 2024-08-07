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
N = 12
x0 = np.array([1.5 * L, 0.5 * L, 0])
xf = np.array([0.5 * L, 0.5 * L, dt * N])


u_min = np.array([-pi])
u_max = np.array([pi])


# 定义约束条件函数
def constraint_u_min(U):
    return (U - u_min).flatten()


def constraint_u_max(U):
    return (u_max - U).flatten()


# 定义确保到达目标的约束
def constraint_goal_reached(U):
    num_controls = len(U)
    path = []
    state = x0
    path.append([state[0], state[1]])

    for number in range(num_controls):
        state = agent_dynamics_withtime(state, U[number])
        path.append([state[0], state[1]])

    final_position = path[-1]
    return L / 20 - distance_to_goal(final_position, xf)  # 确保距离小于L/20


# 定义目标函数
# 最小化时间步长，将智能体到达目标终点加入到限制条件中
def objective(actions):
    state = x0  # 初始状态
    total_time = 0
    for action in actions:
        state = agent_dynamics_withtime(state, action)
        total_time = state[2]  # 更新时间
    return total_time


# 读取控制输入数据
action_dir = './RRT_output/19步/actions.csv'
df = pd.read_csv(action_dir, header=None, dtype=str)  # 读取为字符串类型
data = df.values.flatten().astype(float).tolist()  # 将数据转换为一维浮点数列表

# 初始化猜想
initial_guess = np.zeros(len(data))
print("初始化控制输入", initial_guess)
print("初始化控制输入的时间步长", len(initial_guess))
print('\n')

for i in range(len(data)):
    initial_guess[i] = data[i]
print("RRT控制输入:", initial_guess)
print("RRT控制输入的时间步长:", len(initial_guess))
print('\n')

# 定义约束条件
con1 = {'type': 'ineq', 'fun': constraint_u_min}
con2 = {'type': 'ineq', 'fun': constraint_u_max}
con3 = {'type': 'ineq', 'fun': constraint_goal_reached}
constraints = [con1, con2, con3]

# 记录目标函数值的列表
objective_values = []


# 定义回调函数
def callback(xk):
    obj_value = objective(xk)
    objective_values.append(obj_value)
    print(f"当前目标函数值: {obj_value}")


# 使用SLSQP方法进行优化
result = minimize(objective, initial_guess, method='SLSQP', constraints=constraints, callback=callback)
# 输出优化结果
optimal_time = len(result.x)
optimal_output = result.x
print("最优时间步长:", optimal_time)
print("最优控制输入:", optimal_output)
print("最优目标函数值:", result.fun)

# 绘制目标函数值的变化图
plt.plot(objective_values)
plt.xlabel('迭代次数')
plt.ylabel('目标函数值')
plt.title('优化过程中目标函数值的变化')
plt.show()

# 绘制轨迹
x = x0
trajectory = [x]
U_opt = optimal_output
for k in range(optimal_time):
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
