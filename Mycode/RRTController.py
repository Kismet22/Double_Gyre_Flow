import random
from math import *


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


class RRT:
    def __init__(self, start, goal, env, map_dimensions, obstacle_list, expand_distance=None,
                 goal_sample_rate=5, max_iter=5000):
        # 地图起点
        self.start = Node(start[0], start[1], 0, float('inf'))
        # 地图终点
        self.goal = Node(goal[0], goal[1], float('inf'), float('inf'))
        self.env = env
        # 采样时间
        self.time_step = env.dt
        # 地图尺寸
        self.min_rand = map_dimensions[0]
        self.max_rand = map_dimensions[1]
        # 期望距离
        self.expand_distance = expand_distance
        # 目标采样率
        self.goal_sample_rate = goal_sample_rate
        # 最大迭代次数
        self.max_iter = max_iter
        # 障碍物列表
        self.obstacle_list = obstacle_list
        self.node_list = [self.start]

    def plan(self):
        # 在最大迭代次数内
        t = self.start.t  # 初始化时间
        for i in range(self.max_iter):
            # 生成随机节点
            rnd_node = self.get_random_node(t)
            # 找到离随机节点最近的现有节点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            # 生成从现有节点到随机节点的新节点
            if nearest_node.t == t:
                # print("不需要多步模拟")
                new_node = self.steer(nearest_node, rnd_node, nearest_node.t)
                # 不与障碍物碰撞，加入节点list中
                if self.check_collision(new_node, self.obstacle_list):
                    self.node_list.append(new_node)
                    # 如果new_node离目标足够近，尝试直接连接到目标，并检查是否碰撞
                    if self.distance_to_goal(new_node.x, new_node.y) <= self.expand_distance:
                        print("FIND PATH")
                        return self.generate_final_course(len(self.node_list) - 1)

            else:
                new_node = self.steer(nearest_node, rnd_node, nearest_node.t)
                # 不与障碍物碰撞，加入节点list中
                if self.check_collision(new_node, self.obstacle_list):
                    self.node_list.append(new_node)
                    # 如果new_node离目标足够近，尝试直接连接到目标，并检查是否碰撞
                    if self.distance_to_goal(new_node.x, new_node.y) <= self.expand_distance:
                        print("FIND PATH")
                        return self.generate_final_course(len(self.node_list) - 1)
                while abs(rnd_node.t - new_node.t) > 0.1 * self.env.dt:
                    current_time = new_node.t
                    new_node = self.steer(new_node, rnd_node, current_time)
                    if self.check_collision(new_node, self.obstacle_list):
                        self.node_list.append(new_node)
                        # 如果new_node离目标足够近，尝试直接连接到目标，并检查是否碰撞
                        if self.distance_to_goal(new_node.x, new_node.y) <= self.expand_distance:
                            print("FIND PATH")
                            return self.generate_final_course(len(self.node_list) - 1)
            t = new_node.t
        print("FAIL TO FIND PATH")
        return None, None, None

    # steer方法从from_node向to_node生成一个新的节点new_node
    def steer(self, from_node, to_node, t):
        new_node = Node(from_node.x, from_node.y, from_node.t, 0)
        new_node.x, new_node.y, new_node.t, new_node.at = self.nonlinear_motion(from_node, to_node, t)
        new_node.parent = from_node
        return new_node

    def nonlinear_motion(self, from_node, to_node, t):
        d, theta = self.calc_distance_and_angle(from_node, to_node)
        new_action = theta
        new_state = self.env.agent_dynamics_withtime([from_node.x, from_node.y, t], theta)
        new_x = new_state[0]
        new_y = new_state[1]
        new_t = new_state[2]
        return new_x, new_y, new_t, new_action

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
        path_time.append(self.start.t)
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
    def check_collision(self, node, obstaclelist):
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