import pygame
import os
from settings import CONFIG

class Background:
    """
    独立背景管理类
    负责背景图片的加载、异常回退（浅灰色背景）、动态向下循环滚动及渲染绘制。
    """
    def __init__(self, width, height, image_path=None, speed=None):
        """
        初始化背景对象
        :param width: 屏幕或窗口的宽度
        :param height: 屏幕或窗口的高度
        :param image_path: 背景图片的相对或绝对路径
        :param speed: 背景垂直滚动的速度（像素/帧），推荐 1-3
        """
        self.width = width
        self.height = height
        self.speed = CONFIG.bg_speed if speed is None else speed
        self.image = None
        self.y_offset = 0                  # 记录滚动的垂直偏移量
        self.fallback_color = CONFIG.bg_fallback_color # 默认回退颜色：浅灰色
        # 未传入路径时使用项目内默认背景图
        if image_path is None:
            image_path = CONFIG.bg_image_path
        
        # 尝试加载图片
        if image_path:
            if os.path.exists(image_path):
                try:
                    # load().convert() 能加速 Pygame 渲染
                    loaded_image = pygame.image.load(image_path).convert()
                    # 强制缩放图片使其完全适应屏幕尺寸
                    self.image = pygame.transform.scale(loaded_image, (width, height))
                    print(f"成功加载背景图片: {image_path}")
                except pygame.error as e:
                    print(f"背景图片加载失败 ({e})，将使用默认浅灰色背景。")
            else:
                print(f"未找到路径为 '{image_path}' 的图片，将使用默认浅灰色背景。")

    def update(self):
        """
        更新背景滚动状态 (在主循环的逻辑更新阶段调用)
        原理：偏移量不断增加，当偏移量超过屏幕高度时归零，形成无缝闭环。
        """
        if self.image: # 只有存在图片时才需要计算滚动
            self.y_offset -= self.speed
            # 边界处理：向上循环滚动，偏移量小于等于 -self.height 时归零
            if self.y_offset <= -self.height:
                self.y_offset += self.height
                # 用加法保证连续性，防止跳变

    def draw(self, surface):
        """
        将背景绘制到指定的显示图层上 (在主循环的渲染阶段调用)
        注意：此方法必须在每帧渲染的第一步调用，以确保其处于最底层，不遮挡其他元素。
        """
        if self.image:
            # 向上循环滚动：同时绘制两张图片拼接
            # 第一张：正逐渐向上离开屏幕
            surface.blit(self.image, (0, self.y_offset))
            # 第二张：紧跟在第一张下方（y坐标加屏幕高度），随第一张一起向上滑入屏幕
            surface.blit(self.image, (0, self.y_offset + self.height))
        else:
            # 如果没有图片，直接用浅灰色填充整个图层
            surface.fill(self.fallback_color)
