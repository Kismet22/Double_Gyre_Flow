import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import csv
from math import *
import pandas as pd
from FlowEnvironment import Double_gyre_Flow

# plot中文字体设置
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


# RRT过程
# Node类表示树中的一个节点
class Node:
    def __init__(self, x, y, t, a):
        # 节点x坐标
        self.x = x
        # 节点y坐标
        self.y = y
        # 节点时刻
        self.t = t
        # 到达该节点进行的动作(action to this node)
        self.at = a
        # 指向父节点的指针parent
        self.parent = None


"""""""""
class TimeNode(Node):
    def __init__(self, x, y, t):
        super().__init__(x, y)
        self.t = t
"""


class RRT:
    def __init__(self, start, goal, map_dimensions, delta_t, obstacle_list, expand_distance=None,
                 path_resolution=0.5,
                 goal_sample_rate=5, max_iter=500):
        # 地图起点
        self.start = Node(start[0], start[1], 0, float('inf'))
        # 地图终点
        self.goal = Node(goal[0], goal[1], float('inf'), float('inf'))
        # 采样时间
        self.time_step = delta_t
        # 地图尺寸
        self.min_rand = map_dimensions[0]
        self.max_rand = map_dimensions[1]
        # 扩展距离
        self.expand_distance = expand_distance
        # 路径分辨率
        self.path_resolution = path_resolution
        # 目标采样率
        self.goal_sample_rate = goal_sample_rate
        # 最大迭代次数
        self.max_iter = max_iter
        # 障碍物列表
        self.obstacle_list = obstacle_list
        self.node_list = [self.start]

    def plan(self):
        # 在最大迭代次数内
        t = 0  # 初始化时间
        for i in range(self.max_iter):
            # print("#######################################")
            # print(f'第{i}步')
            # 生成随机节点
            rnd_node = self.get_random_node(t)
            # print("当前时间", rnd_node.t)
            # 找到离随机节点最近的现有节点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            # print("最近点的时间", nearest_node.t)
            # 生成从现有节点到随机节点的新节点
            if nearest_node.t == t:
                # print("不需要多步模拟")
                new_node = self.steer(nearest_node, rnd_node, self.time_step, nearest_node.t)
                # print("更新后的时间", new_node.t)
                # 不与障碍物碰撞，加入节点list中
                if self.check_collision(new_node, self.obstacle_list):
                    self.node_list.append(new_node)
                    # 如果new_node离目标足够近，尝试直接连接到目标，并检查是否碰撞
                    if self.distance_to_goal(new_node.x, new_node.y) <= self.expand_distance:
                        """""""""
                         final_node, final_action = self.steer(new_node, self.goal, self.expand_distance,
                                                               round(t + self.time_step, 1))
                         self.action_list.append(final_action)
                         if self.check_collision(final_node, self.obstacle_list):
                             # 如果找到路径，生成最终路径
                             return self.generate_final_course(len(self.node_list) - 1)
                         """
                        return self.generate_final_course(len(self.node_list) - 1)
            else:
                # print("需要多步模拟")
                new_node = self.steer(nearest_node, rnd_node, self.time_step, nearest_node.t)
                # print("第一步模拟后的时间", new_node.t)
                # 不与障碍物碰撞，加入节点list中
                if self.check_collision(new_node, self.obstacle_list):
                    self.node_list.append(new_node)
                    # 如果new_node离目标足够近，尝试直接连接到目标，并检查是否碰撞
                    if self.distance_to_goal(new_node.x, new_node.y) <= self.expand_distance:
                        """""""""
                        final_node, final_action = self.steer(new_node, self.goal, self.expand_distance,
                                                              round(t + self.time_step, 1))
                        self.action_list.append(final_action)
                        if self.check_collision(final_node, self.obstacle_list):
                            # 如果找到路径，生成最终路径
                            return self.generate_final_course(len(self.node_list) - 1)
                        """
                        return self.generate_final_course(len(self.node_list) - 1)
                time_step = round((t - nearest_node.t) / self.time_step)  # 四舍五入向上取整
                # print("开始循环，循环内需要进行的次数", time_step)
                # print("不取整的循环次数", (t - nearest_node.t) / self.time_step)
                for _ in range(time_step):
                    # print("进入循环")
                    current_time = new_node.t
                    # print("此时的时间", current_time)
                    new_node = self.steer(new_node, rnd_node, self.time_step, current_time)
                    # print("更新时间", new_node.t)
                    if self.check_collision(new_node, self.obstacle_list):
                        self.node_list.append(new_node)
                        # 如果new_node离目标足够近，尝试直接连接到目标，并检查是否碰撞
                        if self.distance_to_goal(new_node.x, new_node.y) <= self.expand_distance:
                            """""""""
                            final_node, final_action = self.steer(new_node, self.goal, self.expand_distance,
                                                                  round(t + self.time_step, 1))
                            self.action_list.append(final_action)
                            if self.check_collision(final_node, self.obstacle_list):
                                # 如果找到路径，生成最终路径
                                return self.generate_final_course(len(self.node_list) - 1)
                            """
                            return self.generate_final_course(len(self.node_list) - 1)
            t = round(t + self.time_step, 1)  # 更新时间并四舍五入保留一位小数
        return None

    # steer方法从from_node向to_node生成一个新的节点new_node
    def steer(self, from_node, to_node, time_step, t):
        new_node = Node(from_node.x, from_node.y, round(t + self.time_step, 1), 0)
        new_node.x, new_node.y, new_node.at = self.nonlinear_motion(from_node, to_node, time_step, t)
        new_node.parent = from_node
        return new_node

    def nonlinear_motion(self, from_node, to_node, delta_t, t):
        d, theta = self.calc_distance_and_angle(from_node, to_node)
        new_action = theta
        flow_x, flow_y = self.velocity_function(from_node.x, from_node.y, t)
        dx = (U_swim * cos(theta) + flow_x) * delta_t
        dy = (U_swim * sin(theta) + flow_y) * delta_t
        new_x = from_node.x + dx
        new_y = from_node.y + dy
        return new_x, new_y, new_action

    @staticmethod
    def velocity_function(x, y, t):
        v_x = env.U_flow_x(x, y, t)
        v_y = env.U_flow_y(x, y, t)
        # v_x, v_y = env.get_speed([x, y, t])
        return v_x, v_y

    # generate_final_course方法生成从起点到终点的路径
    # 从终点开始，通过父节点指针逐步回溯到起点，生成路径
    def generate_final_course(self, goal_ind):
        print("start to generate")
        path = [[self.goal.x, self.goal.y]]
        path_time = []
        actions = []
        node = self.node_list[goal_ind]
        actions.append(node.at)
        path.append([node.x, node.y])
        path_time.append(node.t)
        node = node.parent
        while node.parent is not None:
            path.append([node.x, node.y])
            path_time.append(node.t)
            actions.append(node.at)
            node = node.parent
        path.append([self.start.x, self.start.y])
        path_time.append(0)
        return path[::-1], path_time[::-1], actions[::-1]

    # get_random_node方法生成一个随机节点
    # 有一定概率直接返回目标节点，以提高算法效率
    def get_random_node(self, t):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = Node(random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand), t,
                       float('inf'))
        else:
            rnd = Node(self.goal.x, self.goal.y, t, float('inf'))
        return rnd

    @staticmethod
    # 返回距离最小的节点索引
    def get_nearest_node_index(node_list, rnd_node):
        """""""""
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 + (rnd_node.t - node.t) * 100
                 for node in node_list]
        """
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    # 计算两个节点之间的角度和距离
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = hypot(dx, dy)
        theta = atan2(dy, dx)
        return d, theta

    # 判断是否与障碍物发生碰撞
    # size为距离半径
    def check_collision(self, node, obstacleList):
        """""""""
        for (ox, oy, size) in obstacleList:
            dx = ox - node.x
            dy = oy - node.y
            d = dx * dx + dy * dy
            if d <= size ** 2:
                return False  # collision
        """
        return True  # safe

    # 计算节点到目标的距离
    def distance_to_goal(self, x, y):
        dx = x - self.goal.x
        dy = y - self.goal.y
        # hypot(dx, dy),返回欧式距离
        return hypot(dx, dy)

    def draw_graph(self, all_path=False, final_path=None, time_list=None, obstacle_list=None,
                   goal_range=None, save_dir=None):
        fig, ax = plt.subplots()
        if obstacle_list:
            for (ox, oy, size) in obstacle_list:
                circle = plt.Circle((ox, oy), size, color='b')
                ax.add_artist(circle)

        if goal_range:
            for ((ox, oy), size) in goal_range:
                circle = plt.Circle((ox, oy), size, color='gold')
                ax.add_artist(circle)

        if all_path:
            for node in self.node_list:
                if node.parent is not None:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "k--")

        if final_path is None:
            print("无法找到轨迹")

        else:
            print("轨迹寻找成功")
            print("time_list", time_list)
            path = final_path
            for i in range(len(path) - 1):
                plt.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], "r-")

        plt.plot(self.goal.x, self.goal.y, "gx")
        for (ox, oy, size) in self.obstacle_list:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.axis("equal")
        plt.grid(True)
        if save_dir:
            plt.savefig(save_dir)

        if not save_dir:
            plt.show()


def main():
    start = env.get_start(center=[0.5 * L, 1.5 * L], radius=0, angle=0)
    goal = env.get_target(center=[0.5 * L, 0.5 * L], radius=0, angle=0)
    goal_range = [(goal, 1.0 / 50)]
    map_dimensions = [0, 3]

    """""""""
    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (7, 5, 2),
    ]
    """
    obstacle_list = []

    action_dir = './RRT_output/actions.csv'
    path_dir = './RRT_output/trajectory.csv'
    img_dir = './RRT_output/trajectory.png'

    len_old = 40
    for _ in range(20):
        rrt = RRT(start, goal, map_dimensions, dt, expand_distance=L / 50, obstacle_list=obstacle_list)
        path, time_list, action_list = rrt.plan()
        len_new = len(action_list)
        if len_new < len_old:
            # 保存为CSV文件
            with open(action_dir, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(action_list)

            with open(path_dir, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(path)

            rrt.draw_graph(all_path=True, final_path=path, time_list=time_list, goal_range=goal_range,
                           save_dir=img_dir)

            len_old = len_new

    """""""""
    rrt = RRT(start, goal, map_dimensions, dt, expand_distance=L / 50, obstacle_list=obstacle_list)
    path, time_list, action_list = rrt.plan()
    """

    """""""""
    df = pd.read_csv(action_dir, header=None, dtype=str)  # 读取为字符串类型
    action_list = df.values.flatten().tolist()  # 将数据转换为一维列表
    t = 0
    path = [1.5, 0.5]
    # 验证轨迹运动状态
    for i in range(len(action_list)):
        print("########################################")
        # print(path[i + 1])
        # print(step(path[i], action_list[i], time_list[i]))
        print("action", float(action_list[i]))
        new_path = step(path, float(action_list[i]), t)
        print(new_path)
        path = new_path
        t = t + 0.1
    # rrt.draw_graph(all_path=True, final_path=path, time_list=time_list, goal_range=goal_range)
    """


if __name__ == '__main__':
    main()
