import casadi as ca
import numpy as np
import time
from math import *
import matplotlib.pyplot as plt
import random
import casadi.tools as ca_tools
import csv
import pandas as pd
from FlowEnvironment import Double_gyre_Flow

# 常数设置
U_swim = 0.9
A = 2 * U_swim / 3
epsilon = 0.3
L = 1
target_center = (0.5 * L, 0.5 * L)
start_center = (1.5 * L, 0.5 * L)
dt = 0.1
omega = 20 * pi * U_swim / (3 * L)
R = 0.25 * L


def random_start():
    # 目标区域在一个圆形的范围内
    x_center = 1.5 * L
    y_center = 0.5 * L

    # 生成随机角度
    angle = random.uniform(0, 2 * pi)

    # 生成随机半径
    radius = random.uniform(0, 0.25 * L)
    # radius = 0

    # 将极坐标转换为直角坐标
    x_start = x_center + radius * cos(angle)
    y_start = y_center + radius * sin(angle)
    startpoint = [x_start, y_start]
    return startpoint


def random_target():
    # 目标区域在一个圆形的范围内
    x_center = 0.5 * L
    y_center = 0.5 * L

    # 生成随机角度
    angle = random.uniform(0, 2 * pi)

    # 生成随机半径
    radius = random.uniform(0, 0.25 * L)
    # radius = 0

    # 将极坐标转换为直角坐标
    x_target = x_center + radius * cos(angle)
    y_target = y_center + radius * sin(angle)
    startpoint = [x_target, y_target]
    return startpoint


# 动力学函数
env = Double_gyre_Flow(U_swim=U_swim, epsilon=epsilon, L=L, dt=dt, mode='casadi')


def shift_movement(delta_t, t_in, x_in, u, function):
    # 运动到下一个状态
    f_value = function(x_in, u[:, 0])
    st = x_in + delta_t * f_value
    # 时间增加
    t_move = t_in + delta_t
    # 准备下一个估计的最优控制
    # 更新控制输入矩阵，去掉第一列控制输入，在末尾添加上（复制）最后一列控制输入
    u_end = ca.horzcat(u[:, 1:], u[:, -1])
    return t_move, st, u_end.T


#####################
# 环境变量声明
dt = 0.1  # （模拟的）系统采样时间【秒】
N = 5  # 需要预测的步长【超参数】
t0 = 0  # 初始时间
# tf = dt * 150  # 结束时间
#####################


"""""""""
# 代码测试
action_dir = './RRT_output/19步/actions.csv'
df = pd.read_csv(action_dir, header=None, dtype=str)  # 读取为字符串类型
data = df.values.flatten().astype(float).tolist()  # 将数据转换为一维浮点数列表
pos = []
x0 = np.array([1.5, 0.5, 0.0])
pos.append([x0[0], x0[1]])
for i in range(len(data)):
    x_new = x0 + dt * agent_dynamics_withtime(x0, data[i])
    print("x_new", x0)
    x0 = x_new
"""

# 根据数学模型建模
# 1 系统状态
x = ca.SX.sym('x')  # x坐标
y = ca.SX.sym('y')  # y坐标
t = ca.SX.sym('time')  # 时间步

states = ca.vertcat(x, y, t)

n_states = states.size()[0]  # 获得系统状态的尺寸，向量以（n_states, 1）的格式呈现

# 2 控制输入
theta = ca.SX.sym('theta')
controls = ca.vertcat(theta)
n_controls = controls.size()[0]  # 控制向量尺寸

# 3 运动学模型
# 定义右手函数
# rhs = agent_dynamics_withtime(states, controls)
rhs = env.agent_dynamics_withtime(states, controls)

# 利用CasADi构建一个函数
f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

# 4 开始构建MPC
# 4.1 相关变量，格式(状态长度， 步长)
U = ca.SX.sym('U', n_controls, N)  # N步内的控制输出
X = ca.SX.sym('X', n_states, N + 1)  # N+1步的系统状态，通常长度比控制多1
P = ca.SX.sym('P', n_states + n_states)  # 构建问题的相关参数,在这里每次只需要给定当前/初始位置和目标终点位置

# 4.2 Single Shooting 约束条件
X[:, 0] = P[:n_states]

# 4.2剩余N状态约束条件
for i in range(N):
    # 通过前述函数获得下个时刻系统状态变化。
    # 这里需要注意引用的index为[:, i]，因为X为(n_states, N+1)矩阵
    delta_X = f(X[:, i], U[:, i])  # delta_X
    X[:, i + 1] = X[:, i] + delta_X * dt

# 4.3获得输入（控制输入，参数）和输出（系统状态）之间关系的函数ff
ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])

# NLP问题
# 位置惩罚矩阵
Q = np.zeros((2, 2))  # 全零 2x2 矩阵，用于惩罚位置
Q[:2, :2] = np.array([[10.0, 0.0],  # 对状态矩阵的前两个进行惩罚
                      [0.0, 10.0]])
# 时间惩罚矩阵
R = np.array([10.0])

# 优化目标
obj = 0  # 初始化优化目标值
for i in range(N):
    # 在N步内对获得优化目标表达式
    # .T表示矩阵转置,计算惩罚函数 对应误差的平方与系数相乘再相加
    # ca.mtimes,矩阵乘法操作
    obj = obj + ca.mtimes([(X[:2, i] - P[n_states:-1]).T, Q, X[:2, i] - P[n_states:-1]]) + \
          ca.mtimes([X[-1, i].T, R, X[-1, i]])

# 约束条件定义
g = []  # 用list来存储优化目标的向量
for i in range(N + 1):
    # 这里的约束条件只有小车的坐标（x,y）必须在-2至2之间
    # 由于xy没有特异性，所以在这个例子中顺序不重要（但是在更多实例中，这个很重要）
    # g中表示需要约束的内容
    # g.append(X[0, i])  # 第一行第n列
    # g.append(X[1, i])  # 第二行第n列
    pass

# 定义NLP问题，'f'为需要优化的目标函数，'x'为需要优化的目标变量，'p'为包含已知不变的参数，'g'为额外约束条件
# 需要注意的是，用SX表达必须将所有表示成标量或者是一维矢量的形式
# .reshape(U, -1, 1):-1表示该维度由另一位维度推演而来, 1 一列
nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p': P, 'g': ca.vertcat(*g)}

# ipopt设置:
# ipopt.max_iter: 最大迭代次数
# ipopt.print_level: 输出信息的详细级别，0 表示关闭输出
# print_time: 控制是否输出求解时间
# ipopt.acceptable_tol: KKT条件变化的容忍度
# ipopt.acceptable_obj_change_tol: 接受的目标函数变化的容忍度
opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                'ipopt.acceptable_obj_change_tol': 1e-8}

# 最终目标，获得求解器:
# solver' 是求解器的名称
# ipopt' 指定了所使用的求解器为 IPOPT
# nlp_prob 是定义好的非线性优化问题
# opts_setting 是求解器的设置参数，告诉求解器如何进行求解
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

# 5 开始仿真
#   控制约束
control_max = ca.pi
lbx = []  # 最低约束条件
ubx = []  # 最高约束条件
for _ in range(N):
    #   在一个循环里进行连续定义
    lbx.append(-control_max)
    ubx.append(control_max)

# 仿真条件和相关变量
t0 = 0.0  # 仿真时间
x0 = np.array([1.5, 0.5, 0.0]).reshape(-1, 1)  # 初始始状态
xs = np.array([0.5, 0.75, np.nan]).reshape(-1, 1)  # 末状态
u0 = np.array([0.0] * N).reshape(-1, n_controls)  # 系统初始控制状态，为了统一本例中所有numpy有关,N行,n_controls列,每个值都是0
# 变量都会定义成（N,状态数）的形式方便索引和print
x_c = []  # 存储系统的状态
position_record = []  # 记录坐标位置
u_c = []  # 存储控制全部计算后的控制指令
t_c = []  # 保存时间
xx = []  # 存储每一步位置
sim_time = 20  # 仿真时长
index_t = []  # 存储时间戳，以便计算每一步求解的时间

# 6 开始仿真
mpciter = 0  # 迭代计数器
start_time = time.time()  # 获取开始仿真时间
# 终止条件为目标的欧式距离小于D/50或者仿真超时
while np.linalg.norm(x0[:2] - xs[:2]) > L / 50 and mpciter - sim_time / dt < 0.0:
    print("'''''''''''''''''''''''''")
    print("mpc_iter", mpciter)
    print("控制器输入", u0)
    # 初始化优化参数
    # c_p中存储的是当前的位置信息和目标点的位置信息
    c_p = np.concatenate((x0, xs))
    # 初始化优化目标变量
    init_control = ca.reshape(u0, -1, 1)
    # 计算结果并且
    t_ = time.time()
    res = solver(x0=init_control, p=c_p, lbx=lbx, ubx=ubx)
    index_t.append(time.time() - t_)
    # 获得最优控制结果u
    u_sol = ca.reshape(res['x'], n_controls, N)  # 记住将其恢复U的形状定义
    # 每一列代表了系统在每个时间步上的最优控制输入
    ###
    ff_value = ff(u_sol, c_p)  # 利用之前定义ff函数获得根据优化后的结果
    # 之后N+1步后的状态（n_states, N+1）
    # 存储结果
    x_c.append(ff_value)
    u_c.append(u_sol[:, 0])
    t_c.append(t0)
    # 根据数学模型和MPC计算的结果移动并且准备好下一个循环的初始化目标
    t0, x0, u0 = shift_movement(dt, t0, x0, u_sol, f)
    # 存储位置
    x0 = ca.reshape(x0, -1, 1)
    xx.append(x0.full())
    position_record.append([x0.full()[0], x0.full()[1]])
    # 打印状态值
    print("Current State:", x0.full())
    # 计数器+1
    mpciter = mpciter + 1

# 绘制轨迹图
xx = np.array(xx)  # 转换为numpy数组

plt.figure(figsize=(10, 6))

# 绘制位置记录
position_record = np.array(position_record)
plt.plot(position_record[:, 0], position_record[:, 1], 'bo-', label='Trajectory')

# 绘制目标位置
plt.plot(xs[0], xs[1], 'go', label='Target Position')

# 添加标题和标签
plt.title('System State Trajectories')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid()
plt.show()
