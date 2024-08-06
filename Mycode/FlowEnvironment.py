import numpy as np
import random
from math import *
import casadi as ca


# 双环流智能体更新环境
class Double_gyre_Flow:
    def __init__(self, U_swim, epsilon, dt, L, omega=None, A=None, R=None, mode='standard'):
        # 流场常数设置
        # 包含 U_swim A epsilon L omega dt的内容,其中U_swim epsilon dt L为必须输入的值,A和omega的值可以输入也可以通过U_swim进行推导
        self.U_swim = U_swim
        self.epsilon = epsilon
        self.L = L
        self.dt = dt
        self.mode = mode
        if omega:
            self.omega = omega
        else:
            self.omega = 20 * pi * U_swim / (3 * L)
        if A:
            self.A = A
        else:
            self.A = 2 * U_swim / 3
        if R:
            self.R = R
        else:
            self.R = 0.25 * L

    # 动力学函数
    def f_1(self, pos_x, t):
        if self.mode == 'casadi' and ca is not None:
            return self.epsilon * ca.sin(self.omega * t) * pos_x ** 2 + pos_x - \
                   2 * self.epsilon * ca.sin(self.omega * t) * pos_x
        return self.epsilon * sin(self.omega * t) * pos_x ** 2 + pos_x - 2 * self.epsilon * sin(self.omega * t) * pos_x

    def psi(self, pos_x, pos_y, t):
        if self.mode == 'casadi' and ca is not None:
            pos_t = t
            return self.A * ca.sin(ca.pi * self.f_1(pos_x, pos_t)) * ca.sin(ca.pi * pos_y)
        return self.A * sin(pi * self.f_1(pos_x, t)) * sin(pi * pos_y)

    def U_flow_x(self, pos_x, pos_y, t):
        if self.mode == 'casadi' and ca is not None:
            pos_t = t
            return -ca.pi * self.A * ca.sin(ca.pi * self.f_1(pos_x, pos_t)) * ca.cos(ca.pi * pos_y)
        return -pi * self.A * sin(pi * self.f_1(pos_x, t)) * cos(pi * pos_y)

    def U_flow_y(self, pos_x, pos_y, t):
        if self.mode == 'casadi' and ca is not None:
            pos_t = t
            return ca.pi * self.A * ca.cos(ca.pi * self.f_1(pos_x, pos_t)) * ca.sin(ca.pi * pos_y) * (
                    2 * self.epsilon * ca.sin(self.omega * pos_t) * pos_x
                    + 1 - 2 * self.epsilon * ca.sin(self.omega * pos_t))
        return pi * self.A * cos(pi * self.f_1(pos_x, t)) * sin(pi * pos_y) * (
                2 * self.epsilon * sin(self.omega * t) * pos_x
                + 1 - 2 * self.epsilon * sin(self.omega * t))

    def agent_dynamics_withtime(self, state, action):
        if self.mode == 'casadi' and ca is not None:
            pos_x, pos_y, pos_t = state[0], state[1], state[2]
            v_flow_x = self.U_flow_x(pos_x, pos_y, pos_t)
            v_flow_y = self.U_flow_y(pos_x, pos_y, pos_t)

            delta_pos_x = self.U_swim * ca.cos(action) + v_flow_x
            delta_pos_y = self.U_swim * ca.sin(action) + v_flow_y
            delta_t = 1
            return ca.vertcat(delta_pos_x, delta_pos_y, delta_t)
        pos_x, pos_y, t = state
        flow_x = self.U_flow_x(pos_x, pos_y, t)
        flow_y = self.U_flow_y(pos_x, pos_y, t)
        dx = (self.U_swim * cos(action) + flow_x) * self.dt
        dy = (self.U_swim * sin(action) + flow_y) * self.dt
        new_x = pos_x + dx
        new_y = pos_y + dy
        new_t = t + self.dt
        return [new_x, new_y, new_t]

    @staticmethod
    def distance_to_goal(current, goal):
        # 计算两点之间的欧式距离
        dx = current[0] - goal[0]
        dy = current[1] - goal[1]
        return hypot(dx, dy)

    def get_start(self, center=None, angle=None, radius=None):
        if center:
            x_center = center[0]
            y_center = center[1]
        else:
            # 目标区域在一个圆形的范围内
            x_center = 1.5 * self.L
            y_center = 0.5 * self.L

        if angle:
            angle = angle
        else:
            # 生成随机角度
            angle = random.uniform(0, 2 * pi)

        if radius:
            radius = radius
        else:
            # 生成随机半径
            radius = random.uniform(0, self.R)

        # 将极坐标转换为直角坐标
        x_start = x_center + radius * cos(angle)
        y_start = y_center + radius * sin(angle)
        start_point = [x_start, y_start]
        return start_point

    def get_target(self, center=None, angle=None, radius=None):
        if center:
            x_center = center[0]
            y_center = center[1]
        else:
            # 目标区域在一个圆形的范围内
            x_center = 0.5 * self.L
            y_center = 0.5 * self.L

        if angle:
            angle = angle
        else:
            # 生成随机角度
            angle = random.uniform(0, 2 * pi)

        if radius:
            radius = radius
        else:
            # 生成随机半径
            radius = random.uniform(0, self.R)

        # 将极坐标转换为直角坐标
        x_start = x_center + radius * cos(angle)
        y_start = y_center + radius * sin(angle)
        target_point = [x_start, y_start]
        return target_point
