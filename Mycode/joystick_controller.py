import pygame
import math


class JoystickSimulator:
    def __init__(self):
        # 初始化pygame
        pygame.init()
        # 创建窗口
        self.screen = pygame.display.set_mode((600, 600))
        # 中心点坐标
        self.center_x, self.center_y = self.screen.get_width() / 2, self.screen.get_height() / 2
        self.radius = 250  # 控制圈的半径
        self.angle = 0

        self.screen.fill((0, 0, 0))  # 清屏，填充为黑色
        # 绘制中心点
        pygame.draw.circle(self.screen, (255, 0, 0), (self.center_x, self.center_y), 5)
        # 绘制控制圈
        pygame.draw.circle(self.screen, (255, 255, 255), (self.center_x, self.center_y), self.radius, 1)
        pygame.display.flip()

    def get_angle(self):
        """
        使用鼠标位置模拟一个摇杆用于输出角度
        :return: 输出角度-pi~pi
        """
        for event in pygame.event.get():  # This line is necessary to process internal queue events
            if event.type == pygame.QUIT:
                self.kill()
                return None
        # 获取鼠标位置
        mouse_x, mouse_y = pygame.mouse.get_pos()
        # 计算鼠标与中心点的距离
        distance = math.sqrt((mouse_x - self.center_x) ** 2 + (mouse_y - self.center_y) ** 2)
        # 如果鼠标在圆内，计算角度并显示
        if distance <= self.radius:
            dx = mouse_x - self.center_x
            dy = mouse_y - self.center_y
            self.angle = math.atan2(-dy, dx)
            # print("Direction: {:.2f} degrees".format(self.angle))
        # pygame.display.flip()  # 更新屏幕显示
        return self.angle


if __name__ == '__main__':
    joy_stick = JoystickSimulator()
    ang = 0
    while ang is not None:
        ang = joy_stick.get_angle()
        print(ang)
