import numpy as np
import random
from math import *
from FlowEnvironment import Double_gyre_Flow
from RRTController import RRT
from scipy.optimize import minimize
from Real_environment import DoubleGyreEnvironment
import matplotlib.pyplot as plt


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 常数设置
U_swim = 0.9
epsilon = 0.3
L = 1
dt = 0.01
A = 2 * U_swim / 3
omega = 20 * pi * U_swim / (3 * L)

env_in = Double_gyre_Flow(U_swim=U_swim, L=L, epsilon=epsilon, dt=dt)


# 结合RRT控制器使用的最优化控制器
class Optimal_Controller:
    def __init__(self, start, target, env, map_dimensions, u_min,
                 u_max, time_steps, rrt_times=50, use_rrt=False):

        self.start = start
        self.goal = target
        self.env = env
        self.map_dimensions = map_dimensions
        self.u_min = u_min
        self.u_max = u_max
        self.N = time_steps
        self.N_optimal = time_steps
        self.rrt_times = rrt_times
        self.use_rrt = use_rrt
        self.initial_guess = np.zeros(self.N)  # 初始化猜想
        self.rrt_result = []
        self.rrt_path = []

    def _init_initial_guess(self):
        if self.use_rrt:
            len_old = self.N
            action_list_init = np.zeros(self.N)
            for i in range(self.rrt_times):
                print(f'RRT LOOP {i}')
                print("当前最短时间步:", len_old)
                rrt = RRT(self.start, self.goal, self.env, self.map_dimensions,
                          expand_distance=self.env.L / 50, obstacle_list=None)
                path, time_list, action_list = rrt.plan()
                # 如果成功寻找到了轨迹
                if path:
                    print("action_list", action_list)
                    len_new = len(action_list)
                    if len_new <= len_old:
                        action_list_init[:len_new] = action_list
                        action_list_init = action_list
                        len_old = len_new
                        self.rrt_path = path
                        self.rrt_result = action_list
                        self.initial_guess[:len(action_list_init)] = action_list_init
            self.initial_guess = self.initial_guess[:len_old + 1]
            self.N_optimal = len(self.initial_guess)

        else:
            self.initial_guess = np.zeros(self.N)

    def object(self, U):
        U = U.reshape(U.shape[-1], -1)  # U 的长度可能变化
        x_old = self.start
        for number in range(U.shape[0]):
            x_new = self.env.agent_dynamics_withtime(x_old, U[number])
            x_old = x_new
        cost = self.env.dt * U.shape[0]
        return cost

    def constraint_u_min(self, U):
        U = U.reshape(U.shape[-1], -1)  # U 的长度可能变化
        return (U - self.u_min).flatten()

    def constraint_u_max(self, U):
        U = U.reshape(U.shape[-1], -1)  # U 的长度可能变化
        return (self.u_max - U).flatten()

    def constraint_final_position(self, U):
        U = U.reshape(U.shape[-1], -1)  # U 的长度可能变化
        x_old = self.start
        for number in range(U.shape[0]):
            x_new = self.env.agent_dynamics_withtime(x_old, U[number])
            x_old = x_new
        dx = x_old[0] - self.goal[0]
        dy = x_old[1] - self.goal[1]
        return hypot(dx, dy) - self.env.L / 50  # 末位置到达目标

    def plan(self):
        self._init_initial_guess()
        initial_guess = self.initial_guess.copy()  # 确保初始猜想是一个新的数组
        out_put = None
        for step in range(self.N_optimal, 0, -1):  # 从 N_optimal 开始，逐步减少
            self.N_optimal = step
            constraints = [{'type': 'ineq', 'fun': self.constraint_u_min},
                           {'type': 'ineq', 'fun': self.constraint_u_max},
                           {'type': 'eq', 'fun': self.constraint_final_position}]
            objective = self.object
            guess = initial_guess[:step]
            print("\n")
            print(f"################## START TO OPTIMIZE WITH MAX_STEPS={self.N_optimal} ##################")
            result = minimize(objective, guess, method='SLSQP', constraints=constraints)
            if result.success:
                print("\n")
                print(f"################## SUCCESS WITH MAX_STEPS={self.N_optimal} ##################")
                out_put = result
            else:
                break
        return out_put, self.rrt_result


env_real = DoubleGyreEnvironment(render_mode='human', _init_t=0.01, is_fixed_start_and_target=True)
env_real.reset()
# 仿真条件和相关变量
t0 = env_real.t0  # 仿真开始时间
print("开始时间", t0)
x_env_start = env_real.agent_pos
x0 = np.append(x_env_start, t0)  # 起点状态
print("智能体起点状态", x0)
x_env_target = env_real.target
xf = np.append(x_env_target, 0)  # 末状态
print("智能体目标状态", xf)

"""""""""""
r = random.uniform(0, 0.25 * L)
angle = random.uniform(0, 2 * pi)
t_in = np.random.uniform(0, 0.33)
x0 = np.array([1.5 * L + r * cos(angle), 0.5 * L + r * sin(angle), t_in])
xf = np.array([0.5 * L + r * cos(angle), 0.25 * L + r * sin(angle), 0])
# x0 = np.array([1.5 * L, 0.5 * L, 0])
# xf = np.array([0.75 * L, 0.5 * L, 0])
"""

u_min_in = np.array([-pi])
u_max_in = np.array([pi])
N = 200
controller = Optimal_Controller(start=x0, target=xf, env=env_in, map_dimensions=[0, 3],
                                u_min=u_min_in, u_max=u_max_in, time_steps=N, rrt_times=10,
                                use_rrt=True)

optimal_result, rrt_result = controller.plan()
#rrt_path = controller.rrt_path

# 绘制轨迹
U_opt = optimal_result.x
optimal_iter = 0  # 迭代计数器
terminated = False
truncated = False
while not (terminated or truncated) or optimal_iter < len(U_opt):
    print("'''''''''''''''''''''''''")
    print("时间步", optimal_iter)
    print("控制器输入", U_opt[optimal_iter])
    # 真实的状态转移
    _, _, terminated, truncated, _ = env_real.step(U_opt[optimal_iter])
    # 计数器+1
    optimal_iter += 1

"""""""""
x = x0
trajectory = [x]
N_step = len(U_opt)
for k in range(N_step):
    x = env_in.agent_dynamics_withtime(x, U_opt[k], mode='math')
    trajectory.append(x)
    if env_in.distance_to_goal(x, xf) <= L / 50:
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
"""
