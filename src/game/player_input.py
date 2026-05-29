import pygame

from src.game.game_core import GameCore


def move_horizontal(game_core: GameCore, direction: int) -> bool:
    """
    检查方块是否越界或发生碰撞。如果没有，就往指定的左右方向移动 1 格。

    参数:
    game_core (GameCore): 游戏核心对象，提供网格和碰撞判定方法
    direction (int): 横向移动增量：1 表示右移，-1 表示左移

    返回:
    bool: 成功移动返回 True，无法执行或碰撞受阻返回 False
    """
    if game_core.state != "RUNNING" or game_core.current_piece is None:
        return False

    if game_core.current_piece.check_collision(dx=direction, dy=0, grid=game_core.grid):
        return False

    game_core.current_piece.x += direction
    return True


class PlayerInputController:
    """
    管理人类玩家实时输入的控制器：
    - 按下 Left/Right 触发单次移动(无连续自动移动长按)
    - Up 引发顺时针旋转，Down 请求方下降，Space 引发立即下坠(硬降)并锁定
    """

    def __init__(self, config=None):
        # 保留 config 参数兼容旧调用方。
        self.config = config

    def handle_keydown(self, key: int, game_core: GameCore):
        if key == pygame.K_a:
            move_horizontal(game_core, -1)
        elif key == pygame.K_d:
            move_horizontal(game_core, 1)
        elif key == pygame.K_l:
            if game_core.state == "RUNNING" and game_core.current_piece is not None:
                game_core.current_piece.rotate(game_core.grid)
        elif key in (pygame.K_s, pygame.K_SPACE):
            if game_core.state == "RUNNING" and game_core.current_piece is not None:
                game_core.current_piece.y = game_core.current_piece.get_drop_position(game_core.grid)
                game_core.lock_shape()

    def handle_keyup(self, key: int):
        return None

    def update(self, dt: int, pressed_keys, game_core: GameCore):
        return None
