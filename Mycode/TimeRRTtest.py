import matplotlib.pyplot as plt
import numpy as np
import random
from math import *

# 常数设置
U_swim = 0.9
A = 2 * U_swim / 3
epsilon = 0.3
L = 1
omega = 20 * pi * U_swim / (3 * L)
D = 1
dt = 0.1


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


# RRT过程
# Node类表示树中的一个节点
class Node:
    def __init__(self, x, y, t):
        # 节点x坐标
        self.x = x
        # 节点y坐标
        self.y = y
        # 节点时刻
        self.t = t
        # 指向父节点的指针parent
        self.parent = None


"""""""""
class TimeNode(Node):
    def __init__(self, x, y, t):
        super().__init__(x, y)
        self.t = t
"""


class RRT:
    def __init__(self, start, goal, map_dimensions, delta_t, obstacle_list, expand_distance=1.0 / 6,
                 path_resolution=0.5,
                 goal_sample_rate=5, max_iter=500):
        # 地图起点
        self.action_list = []
        self.start = Node(start[0], start[1], 0)
        # 地图终点
        self.goal = Node(goal[0], goal[1], float('inf'))
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

    """""""""
    def plan(self):
        # 在最大迭代次数内
        t = 0  # 初始化时间
        for i in range(self.max_iter):
            print("#######################################")
            # 生成随机节点
            rnd_node = self.get_random_node(t)
            print("rnd_node_t", rnd_node.t)
            # 找到离随机节点最近的现有节点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            print("nearest_node_t", nearest_node.t)
            # 生成从现有节点到随机节点的新节点
            new_node, new_action = self.steer(nearest_node, rnd_node, self.time_step, t)
            print("new_node_t", new_node.t)
            # 不与障碍物碰撞，加入节点list中
            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)
                self.action_list.append(new_action)
            # 如果new_node离目标足够近，尝试直接连接到目标，并检查是否碰撞
            if self.distance_to_goal(new_node.x, new_node.y) <= self.expand_distance:
                final_node, final_action = self.steer(new_node, self.goal, self.expand_distance, (t + self.time_step))
                self.action_list.append(final_action)
                if self.check_collision(final_node, self.obstacle_list):
                    # 如果找到路径，生成最终路径
                    return self.generate_final_course(len(self.node_list) - 1)

            t += self.time_step  # 更新时间
        return None
    """

    def plan(self):
        # 在最大迭代次数内
        t = 0  # 初始化时间
        for i in range(self.max_iter):
            # print("#######################################")
            # 生成随机节点
            rnd_node = self.get_random_node(t)
            # print("rnd_node_t", rnd_node.t)
            # 找到离随机节点最近的现有节点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            # print("nearest_node_t", nearest_node.t)
            # 生成从现有节点到随机节点的新节点
            if nearest_node.t == t:
                new_node, new_action = self.steer(nearest_node, rnd_node, self.time_step, nearest_node.t)
                # print("new_node_t", new_node.t)
                # 不与障碍物碰撞，加入节点list中
                if self.check_collision(new_node, self.obstacle_list):
                    self.node_list.append(new_node)
                    self.action_list.append(new_action)
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
                new_node, new_action = self.steer(nearest_node, rnd_node, self.time_step, nearest_node.t)
                # print("new_node_t", new_node.t)
                # 不与障碍物碰撞，加入节点list中
                if self.check_collision(new_node, self.obstacle_list):
                    self.node_list.append(new_node)
                    self.action_list.append(new_action)
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
                time_step = ceil((t - nearest_node.t) / self.time_step)  # 四舍五入向上取整
                for _ in range(time_step):
                    new_node, new_action = self.steer(new_node, rnd_node, self.time_step, new_node.t)
                    # print("new_node_t", new_node.t)
                    if self.check_collision(new_node, self.obstacle_list):
                        self.node_list.append(new_node)
                        self.action_list.append(new_action)
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
        new_node = Node(from_node.x, from_node.y, round(t + self.time_step, 1))
        new_node.x, new_node.y, action = self.nonlinear_motion(from_node, to_node, time_step, t)
        new_node.parent = from_node
        return new_node, action

    def nonlinear_motion(self, from_node, to_node, delta_t, t):
        d, theta = self.calc_distance_and_angle(from_node, to_node)
        new_action = theta
        flow_x, flow_y = self.velocity_function(from_node.x, from_node.y, t)
        # dx = (U_swim * cos(theta) + flow_x) * (delta_t + to_node.t - from_node.t)
        # dy = (U_swim * sin(theta) + flow_y) * (delta_t + to_node.t - from_node.t)
        dx = (U_swim * cos(theta) + flow_x) * delta_t
        dy = (U_swim * sin(theta) + flow_y) * delta_t
        new_x = from_node.x + dx
        new_y = from_node.y + dy
        return new_x, new_y, new_action

    def velocity_function(self, x, y, t):
        v_x = U_flow_x(x, y, t)
        v_y = U_flow_y(x, y, t)
        return v_x, v_y

    # generate_final_course方法生成从起点到终点的路径
    # 从终点开始，通过父节点指针逐步回溯到起点，生成路径
    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        path_time = []
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            path_time.append([node.t])
            node = node.parent
        path.append([self.start.x, self.start.y])
        path_time.append([0])
        return path[::-1], path_time[::-1]

    # get_random_node方法生成一个随机节点
    # 有一定概率直接返回目标节点，以提高算法效率
    def get_random_node(self, t):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = Node(random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand), t)
        else:
            rnd = Node(self.goal.x, self.goal.y, t)
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
        return hypot(dx, dy)

    def draw_graph(self, final_path=None, time_list=None, obstacle_list=None, goal_range=None):
        fig, ax = plt.subplots()
        if obstacle_list:
            for (ox, oy, size) in obstacle_list:
                circle = plt.Circle((ox, oy), size, color='b')
                ax.add_artist(circle)

        if goal_range:
            for (ox, oy, size) in goal_range:
                circle = plt.Circle((ox, oy), size, color='gold')
                ax.add_artist(circle)

        for node in self.node_list:
            if node.parent is not None:
                plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "k-")

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
        plt.show()


def main():
    start = [1, 1]
    goal = [3, 3]
    goal_range = [(3, 3, 1.0 / 6)]
    map_dimensions = [0, 4]

    """""""""
    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (7, 5, 2),
    ]
    """
    obstacle_list = []

    rrt = RRT(start, goal, map_dimensions, dt, obstacle_list)
    path, time_list = rrt.plan()

    rrt.draw_graph(path, time_list, goal_range=goal_range)


if __name__ == '__main__':
    main()
