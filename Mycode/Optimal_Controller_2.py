import numpy as np
import random
from math import *
from RRTController_2 import RRT
from scipy.optimize import minimize
from Real_environment import DoubleGyreEnvironment
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

save_dir_1 = './output/position.csv'
save_dir_2 = './output/strategy.csv'

# 常数设置
epsilon = 0.3
L = 1
dt = 0.01


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
                          expand_distance=0.9 * self.env.L / 50, obstacle_list=None)
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
        return 0.99 * self.env.L / 50 - hypot(dx, dy)  # 末位置到达目标

    def plan(self):
        self._init_initial_guess()
        initial_guess = self.initial_guess.copy()  # 确保初始猜想是一个新的数组
        out_put = None
        for step in range(self.N_optimal, 0, -1):  # 从 N_optimal 开始，逐步减少
            self.N_optimal = step
            constraints = [{'type': 'ineq', 'fun': self.constraint_u_min},
                           {'type': 'ineq', 'fun': self.constraint_u_max},
                           {'type': 'ineq', 'fun': self.constraint_final_position}]
            objective = self.object
            guess = initial_guess[:step]
            print("\n")
            print(f"################## START TO OPTIMIZE WITH MAX_STEPS={self.N_optimal} ##################")
            result = minimize(objective, guess, method='SLSQP', constraints=constraints)
            if result.success:
                print("\n")
                print(f"################## SUCCESS WITH MAX_STEPS={self.N_optimal} ##################")
                out_put = result
                initial_guess = out_put.x
            else:
                break
        return out_put, self.rrt_result


#################### 开始测试 ######################
#is_swap = 1
is_swap = 0
is_fixed_time = 0
#test_swim_vel = 0.9
#test_swim_vel = 0.8
test_swim_vel = 0.7


# 文件保存路径
save_file_path = f'./output/swap_{is_swap}_swim_speed_{test_swim_vel}_is_fixed_time_{is_fixed_time}.csv'
# 定义保存结果的 DataFrame
columns = ['start_x', 'start_y', 'start_t', 'target_x', 'target_y', 'navigation_time']
results_df = pd.DataFrame(columns=columns)
results_df.to_csv(save_file_path, index=False)  # 写入文件头部

# 设置重复次数
num_repeats = 450

# 设置环境
env_real = DoubleGyreEnvironment(render_mode=None, _init_t=None, _init_zero=False, is_fixed_start_and_target=False,
                                 swim_vel=test_swim_vel,
                                 swap=False)

# 设置控制器参数
u_min_in = np.array([-pi])
u_max_in = np.array([pi])
N = 400

# 执行操作
for i in range(num_repeats):
    print("################################################")
    print(f"episode:{i + 1}")
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
    controller = Optimal_Controller(start=x0, target=xf, env=env_real, map_dimensions=[0, 3],
                                    u_min=u_min_in, u_max=u_max_in, time_steps=N, rrt_times=5,
                                    use_rrt=True)
    optimal_result, rrt_result = controller.plan()
    if not optimal_result:
        continue
    U_opt = optimal_result.x
    navigation_time = len(U_opt) * env_real.dt
    # 保存到 DataFrame
    result = {
        'start_x': x0[0],
        'start_y': x0[1],
        'start_t': x0[2],
        'target_x': xf[0],
        'target_y': xf[1],
        'navigation_time': navigation_time
    }
    result_df = pd.DataFrame([result])
    result_df.to_csv(save_file_path, mode='a', header=False, index=False)  # 追加写入
    print(f"Round {i + 1} results saved.")
    print("\n")

print(f"All results saved to {save_file_path}")
##########################################


"""""""""
data = {
    'start_x': [x0[0]],
    'start_y': [x0[1]],
    'start_t': [x0[2]],
    'target_x': [xf[0]],
    'target_y': [xf[1]],
}
df = pd.DataFrame(data)
df.to_csv(save_dir_1, index=False)
"""

"""""""""
# 保存优化控制输入 U_opt
U_opt_data = {
    'U_opt': U_opt
}
U_opt_df = pd.DataFrame(U_opt_data)
U_opt_df.to_csv(save_dir_2, index=False)
"""

"""""""""
# 文件保存路径
point_pair = [1.5, 0.5, 0.5, 0.5]
#point_pair = [1.6, 0.6, 0.4, 0.4]
#point_pair = [0.4, 0.4, 1.4, 0.6]
save_file_path = f'./output/swim_speed_{test_swim_vel}_start_x_{point_pair[0]}.csv'
# 定义保存结果的 DataFrame
columns = ['pos_x', 'pos_y', 'navigation_t']
results_df = pd.DataFrame(columns=columns)
results_df.to_csv(save_file_path, index=False)  # 写入文件头部

env_real = DoubleGyreEnvironment(render_mode='human', _init_t=None, _init_zero=True, is_fixed_start_and_target=True,
                                 swim_vel=test_swim_vel,
                                 swap=False, _init_pair=None)
env_real.reset()
u_min_in = np.array([-pi])
u_max_in = np.array([pi])
N = 400
# 仿真条件和相关变量
t0 = env_real.t0  # 仿真开始时间
print("开始时间", t0)
x_env_start = env_real.agent_pos
x0 = np.append(x_env_start, t0)  # 起点状态
print("智能体起点状态", x0)
x_env_target = env_real.target
xf = np.append(x_env_target, 0)  # 末状态
print("智能体目标状态", xf)
controller = Optimal_Controller(start=x0, target=xf, env=env_real, map_dimensions=[0, 3],
                                u_min=u_min_in, u_max=u_max_in, time_steps=N, rrt_times=5,
                                use_rrt=True)


result = {
    'pos_x': env_real.agent_pos[0],
    'pos_y': env_real.agent_pos[1],
    'navigation_t': env_real.t_step * env_real.dt
}
result_df = pd.DataFrame([result])
result_df.to_csv(save_file_path, mode='a', header=False, index=False)  # 追加写入


optimal_result, rrt_result = controller.plan()
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

    result = {
        'pos_x': env_real.agent_pos[0],
        'pos_y': env_real.agent_pos[1],
        'navigation_t': env_real.t_step * env_real.dt
    }
    result_df = pd.DataFrame([result])
    result_df.to_csv(save_file_path, mode='a', header=False, index=False)  # 追加写入

    # 计数器+1
    optimal_iter += 1
"""
