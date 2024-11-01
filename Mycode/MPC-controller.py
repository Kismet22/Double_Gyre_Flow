import casadi as ca
import numpy as np
import time
from math import *
from FlowEnvironment import Double_gyre_Flow
from Real_environment import DoubleGyreEnvironment

# 常数设置
dt = 0.01
U_swim = 0.9
epsilon = 0.3
A = 2 * U_swim / 3
L = 1
r = 0.25 * L
omega = 20 * pi * U_swim / (3 * L)
target_center = (0.5 * L, 0.5 * L)
start_center = (1.5 * L, 0.5 * L)


# 使用IPOPT方法的封装实现
class MPCController:
    def __init__(self, delta_t, states_object, actions_object, p_n_steps,
                 state_matrix, action_matrix, agent_dynamics_function,
                 low_limit, high_limit,
                 max_iter=100, print_level=0, print_time=0, acceptable_tol=1e-8, acceptable_obj_change_tol=1e-8):
        self.dt = delta_t  # 系统采样时间
        self.N = p_n_steps  # 预测步长
        self.Q = states_object  # 状态优化目标矩阵
        self.R = actions_object  # 控制优化目标矩阵
        self.max_iter = max_iter
        self.print_level = print_level
        self.print_time = print_time
        self.acceptable_tol = acceptable_tol
        self.acceptable_obj_change_tol = acceptable_obj_change_tol
        self.lbx = low_limit
        self.ubx = high_limit

        # 1 系统状态,状态矩阵根据外部输入
        self.states = state_matrix
        self.n_states = self.states.size()[0]

        # 2 控制输入,控制矩阵根据外部输入
        self.controls = action_matrix
        self.n_controls = self.controls.size()[0]

        self.agent_dynamics_function = agent_dynamics_function  # 状态更新方式

        # 3 运动学模型
        self.rhs = self.agent_dynamics_function(self.states, self.controls)

        # 利用CasADi构建一个函数
        self.f = ca.Function('f', [self.states, self.controls], [self.rhs], ['input_state', 'control_input'], ['rhs'])
        # 4 构建MPC
        self._construct_mpc()

    def _construct_mpc(self):
        # 相关变量，格式(状态长度，步长)
        U = ca.SX.sym('U', self.n_controls, self.N)  # 控制矩阵
        X = ca.SX.sym('X', self.n_states, self.N + 1)  # 状态矩阵
        P = ca.SX.sym('P', self.n_states + self.n_states)  # 目标矩阵

        # Single Shooting 约束条件
        X[:, 0] = P[:self.n_states]

        # 剩余N状态约束条件
        for i_step in range(self.N):
            delta_X = self.f(X[:, i_step], U[:, i_step])
            X[:, i_step + 1] = X[:, i_step] + delta_X * self.dt

        # 获得输入（控制输入，参数）和输出（系统状态）之间关系的函数
        self.ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])

        ######################### 定义优化目标参数 #########################
        obj = 0
        """""""""
        # 这是目前效果最好的目标函数设置
        # 状态约束
        for i in range(self.N):
            obj = obj + ca.mtimes([(X[:, i] - P[self.n_states:]).T, self.Q, X[:, i] - P[self.n_states:]])
        """

        # 状态约束,使智能体靠近目标终点
        for i_step in range(self.N):
            obj = obj + ca.mtimes([(X[:, i_step] - P[self.n_states:]).T, self.Q, X[:, i_step] - P[self.n_states:]])

        # 动作约束，减少动作与动作之间输出的变化幅度,在靠近终点的末端能起到一定的限制效果
        for i_step in range(1, self.N):
            obj += ca.mtimes([(U[:, i_step] - U[:, i_step - 1]).T, self.R, (U[:, i_step] - U[:, i_step - 1])])

        #################################################################

        # 约束条件定义
        g = []

        # 定义NLP问题
        nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p': P, 'g': ca.vertcat(*g)}

        # ipopt设置
        opts_setting = {'ipopt.max_iter': self.max_iter, 'ipopt.print_level': self.print_level,
                        'print_time': self.print_time, 'ipopt.acceptable_tol': self.acceptable_tol,
                        'ipopt.acceptable_obj_change_tol': self.acceptable_obj_change_tol}

        # 获得求解器
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    ######################### MPC内置的环境更新模拟器 #########################
    def shift_movement(self, delta_t, time_in, x_in, u):
        # 内置环境更新模拟器，用于模拟真实环境的更新
        # 当存在真实环境的更新时，可以不调用
        f_value = self.f(x_in, u[:, 0])  # 实际执行预测序列的第一步
        st = x_in + delta_t * f_value  # 状态更新
        time_move = time_in + delta_t  # 时间更新(实际并不需要)，时间已经包含在状态中被更新
        u_end = ca.horzcat(u[:, 1:], u[:, -1])  # 将预测的序列用于下一次的更新控制
        # 具体来说，去掉已经执行的第一步动作，复制第N步(最后一步)的执行动作
        # 在末尾添加上（复制）最后一列控制输入
        return time_move, st, u_end.T

    #################################################################

    def solve_mpc(self, start_state, target_state):
        # 初始化优化目标变量(控制变量)
        init_control = np.zeros((self.n_controls, self.N))
        # 初始化优化参数
        # c_p中存储的是当前的位置信息和目标点的位置信息
        c_p = np.concatenate((start_state, target_state))
        # Solve the optimization problem
        sol = self.solver(x0=ca.reshape(init_control, -1, 1), p=c_p, lbx=self.lbx, ubx=self.ubx)  # 当前阶段N步优化器
        # sol['f']输出当前预测的优化目标值
        u_opt = ca.reshape(sol['x'], self.n_controls, self.N)  # 当前阶段N步的动作策略
        s_opt = self.ff(u_opt, c_p)  # 当前阶段的N步预测结果
        return u_opt, sol['f'], s_opt


x = ca.SX.sym('x')  # x坐标
y = ca.SX.sym('y')  # y坐标
t = ca.SX.sym('time')  # 时间步
states = ca.vertcat(x, y, t)

theta = ca.SX.sym('theta')
actions = ca.vertcat(theta)

# 状态目标矩阵
Q = np.zeros((3, 3))  # 全零 2x2 矩阵，用于惩罚位置
# 终点目标
Q[:2, :2] = np.array([[10.0, 0.0],
                      [0.0, 10.0]])
# 时间目标
Q[2][2] = 0.0
# 动作目标
R = np.array([1.0])

N = 50  # 每个状态进行预测的步数
control_max = ca.pi
lbx = []  # 最低约束条件
ubx = []  # 最高约束条件
for _ in range(N):
    #   在一个循环里进行连续定义
    lbx.append(-control_max)
    ubx.append(control_max)

env = DoubleGyreEnvironment(render_mode='human', _init_t=0.01, is_fixed_start_and_target=True, save_render=True)
env.reset()
env_1 = Double_gyre_Flow(U_swim=U_swim, epsilon=epsilon, L=L, dt=dt, mode='casadi')
mpc = MPCController(delta_t=dt, p_n_steps=N, states_object=Q, actions_object=R,
                    state_matrix=states, action_matrix=actions,
                    agent_dynamics_function=env_1.agent_dynamics_withtime,
                    low_limit=lbx, high_limit=ubx)

# 仿真条件和相关变量
t0 = env.t0  # 仿真开始时间
print("MPC开始时间", t0)
x_env_start = env.agent_pos
x0 = np.append(x_env_start, t0).reshape(-1, 1)  # 起点状态
print("智能体起点状态", x0)
x_env_target = env.target
xs = np.append(x_env_target, 0.0).reshape(-1, 1)  # 末状态
print("智能体目标状态", xs)
n_controls = 1
u0 = np.array([0.0] * N).reshape(-1, n_controls)  # 系统初始控制状态，为了统一本例中所有numpy有关,N行,n_controls列,每个值都是0
x_c = []  # 存储每一次的N步序列预测结果
position_record = [[x0[0], x0[1]]]  # 存储真实的运动轨迹序列
u_c = []  # 存储真实的控制序列
t_c = []  # 保存时间
xx = []  # 存储真实的状态序列
sim_time = t0 + 4  # 仿真时长
index_t = []  # 存储时间戳，以便计算每一步求解的时间

# 6 开始仿真
mpciter = 0  # 迭代计数器
start_time = time.time()  # 获取开始仿真时间
terminated = False
truncated = False
# 终止条件为目标的欧式距离小于D/50或者仿真超时
while not (terminated or truncated):
    print("'''''''''''''''''''''''''")
    print("时间步", mpciter)
    print("控制器输入", u0)
    # 计算结果并且
    t_ = time.time()
    u_sol, res, ff_value = mpc.solve_mpc(x0, xs)
    print(f'当前优化目标值:{res}')
    print(f'后{N}步预测结果:{ff_value}')
    index_t.append(time.time() - t_)
    # 存储结果
    x_c.append(ff_value)
    u_c.append(u_sol[:, 0])
    t_c.append(t0)
    # 根据数学模型和MPC计算的结果移动并且准备好下一个循环的初始化目标
    # u_sol的后N-1步并没有参与到MPC的更新预测环节中，作为N步序列可以起到参考的作用
    # 真实的状态转移
    _, _, terminated, truncated, _ = env.step(u_c[-1])
    # 存储位置
    x0 = np.append(env.agent_pos, t0 + env.t_step * dt).reshape(-1, 1)
    x0 = ca.reshape(x0, -1, 1)
    xx.append(x0.full())
    position_record.append([x0.full()[0], x0.full()[1]])
    # 打印状态值
    print("当前状态:", x0.full())
    # 计数器+1
    mpciter = mpciter + 1
