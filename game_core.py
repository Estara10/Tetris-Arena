from collections.abc import Callable
import random

import pygame

from settings import CONFIG, GameConfig
from tetromino import Tetromino


class GameCore:
    """
    俄罗斯方块核心逻辑引擎类。
    独立管理网格、方块、计分和状态机，保持与渲染解耦。
    """

    def __init__(
        self,
        config: GameConfig | None = None,
        piece_factory: Callable[[], Tetromino] | None = None,
    ):
        self.config = config if config is not None else CONFIG
        self.piece_factory = piece_factory or self._default_piece_factory
        self.base_fall_speed = self.config.fall_speed_ms
        self.score_mapping = dict(self.config.score_mapping)

        self.state = "RUNNING"
        self.grid = []
        self.score = 0
        self.lines_cleared_total = 0
        self.last_cleared_lines = 0
        self.last_cleared_rows: list[int] = []
        self.last_locked_cells: list[tuple[int, int]] = []
        self.lock_count = 0
        self.current_piece = None
        self.next_piece = None
        self.fall_time = 0
        self.fall_speed = self.base_fall_speed
        self.incoming_garbage = []
        self.garbage_received_total = 0

        self.reset_game()

    def _default_piece_factory(self) -> Tetromino:
        return Tetromino(config=self.config)

    def _create_piece(self) -> Tetromino:
        piece = self.piece_factory()
        piece.config = self.config
        return piece

    def reset_game(self):
        """重新开始游戏：清空网格，重置分数与状态，生成初始方块。"""
        self.grid = [
            [0 for _ in range(self.config.grid_cols)]
            for _ in range(self.config.grid_rows)
        ]
        self.score = 0
        self.lines_cleared_total = 0
        self.last_cleared_lines = 0
        self.last_cleared_rows = []
        self.last_locked_cells = []
        self.lock_count = 0
        self.state = "RUNNING"
        self.fall_time = 0
        self.fall_speed = self.base_fall_speed
        self.incoming_garbage = []
        self.garbage_received_total = 0
        self.current_piece = self._create_piece()
        self.next_piece = self._create_piece()

    def spawn_piece(self):
        """
        生成新方块：将 next_piece 提升为 current_piece,并生成新的 next_piece。
        如果新方块出生即碰撞当前网格，判定游戏结束，状态变为 GAME_OVER。
        """
        self.current_piece = self.next_piece
        self.next_piece = self._create_piece()

        if self.current_piece.check_collision(grid=self.grid):
            self.state = "GAME_OVER"

    def handle_input(self, event):
        """
        单局输入处理，保留给兼容场景使用。
        对抗模式由 main.py 统一接管全局输入。
        """
        if event.type != pygame.KEYDOWN:
            return

        if event.key == pygame.K_r:
            self.reset_game()
            return

        if event.key == pygame.K_p:
            if self.state == "RUNNING":
                self.state = "PAUSED"
            elif self.state == "PAUSED":
                self.state = "RUNNING"
            return

        if self.state != "RUNNING" or self.current_piece is None:
            return

        if event.key == pygame.K_LEFT:
            if not self.current_piece.check_collision(dx=-1, dy=0, grid=self.grid):
                self.current_piece.x -= 1
        elif event.key == pygame.K_RIGHT:
            if not self.current_piece.check_collision(dx=1, dy=0, grid=self.grid):
                self.current_piece.x += 1
        elif event.key == pygame.K_UP:
            self.current_piece.rotate(self.grid)
        elif event.key == pygame.K_DOWN:
            if not self.current_piece.check_collision(dx=0, dy=1, grid=self.grid):
                self.current_piece.y += 1
        elif event.key == pygame.K_SPACE:
            self.current_piece.y = self.current_piece.get_drop_position(self.grid)
            self.lock_shape()

    def update(self, dt):
        """
        游戏时间轴的逻辑更新。
        :param dt: 距离上一帧过去的时间（毫秒）
        """
        if self.state != "RUNNING" or self.current_piece is None:
            return

        self.fall_time += dt
        if self.fall_time < self.fall_speed:
            return

        self.fall_time = 0
        if not self.current_piece.check_collision(dx=0, dy=1, grid=self.grid):
            self.current_piece.y += 1
        else:
            self.lock_shape()

    def lock_shape(self):
        """
        固化方块：将当前触底/发生碰撞的方块写入固定网格缓存 (self.grid)，
        之后调用 clear_lines() 检查是否有可以消除的行，
        最后调用 spawn_piece() 生成新的方块继续循环。
        """
        self.last_locked_cells = []
        for row_idx, row in enumerate(self.current_piece.matrix):
            for col_idx, cell in enumerate(row):
                if cell != "X":
                    continue

                board_x = self.current_piece.x + col_idx
                board_y = self.current_piece.y + row_idx
                if board_y >= 0:
                    self.grid[board_y][board_x] = self.current_piece.shape_name
                    self.last_locked_cells.append((board_x, board_y))

        self.clear_lines()
        if getattr(self.config, "shared_arena_score_on_lock", False):
            self.score += 1
        self.lock_count += 1
        self.spawn_piece()

    def clear_lines(self) -> int:
        """
        消行判定与积分计算：
        遍历 grid 所有的行，如果有不含 0（代表空隙）的行将被剔除。
        由于消除行造成的悬空将由上面自动掉落补全（通过重新计算并在最上层插入空行），
        最后根据消除的行数发放对应的分数。
        返回当次消除的总行数。
        """
        full_row_indices = [i for i, row in enumerate(self.grid) if 0 not in row]
        new_grid = [row for i, row in enumerate(self.grid) if i not in full_row_indices]
        lines_cleared = len(full_row_indices)
        self.last_cleared_lines = lines_cleared
        self.last_cleared_rows = full_row_indices

        if lines_cleared <= 0:
            return 0

        for _ in range(lines_cleared):
            new_grid.insert(0, [0 for _ in range(self.config.grid_cols)])

        self.grid = new_grid
        self.lines_cleared_total += lines_cleared

        earned_score = self.score_mapping.get(lines_cleared, 0)
        self.score += earned_score

        if self.config.debug_logging:
            print(
                f"消除了 {lines_cleared} 行！获得 {earned_score} 分！总分: {self.score}"
            )

        return lines_cleared

    def set_fall_speed(self, fall_speed_ms: int):
        """允许外部为特殊模式动态调整下落速度。"""
        self.fall_speed = max(80, int(fall_speed_ms))

    def queue_garbage(self, hole_columns: list[int]):
        """将收到的垃圾行洞位加入待结算队列。"""
        for hole in hole_columns:
            self.incoming_garbage.append(int(hole))

    def cancel_incoming_garbage(self, attack_lines: int) -> int:
        """用当前进攻抵消自己待接收的垃圾行。"""
        if attack_lines <= 0 or not self.incoming_garbage:
            return 0

        canceled = min(attack_lines, len(self.incoming_garbage))
        del self.incoming_garbage[:canceled]
        return canceled

    def pop_incoming_garbage(self, max_lines: int) -> list[int]:
        """弹出最多 max_lines 行待结算垃圾。"""
        if max_lines <= 0 or not self.incoming_garbage:
            return []

        count = min(max_lines, len(self.incoming_garbage))
        popped = self.incoming_garbage[:count]
        del self.incoming_garbage[:count]
        return popped

    def apply_incoming_garbage(self, max_lines: int | None = None) -> int:
        """将队列中的垃圾行结算到棋盘，返回实际结算行数。"""
        if not self.incoming_garbage:
            return 0

        if max_lines is None:
            holes = self.pop_incoming_garbage(len(self.incoming_garbage))
        else:
            holes = self.pop_incoming_garbage(max_lines)
        return self.inject_garbage(len(holes), holes)

    def inject_garbage(self, lines: int, hole_columns: list[int] | None = None) -> int:
        """
        从底部注入垃圾行。
        - 垃圾行为 G，仅一个洞位为 0。
        - 被顶出顶部的非空格会触发 GAME_OVER。
        """
        if lines <= 0:
            return 0

        applied = 0
        overflow = False
        holes = hole_columns or []

        for idx in range(lines):
            hole_x = holes[idx] if idx < len(holes) else random.randint(0, self.config.grid_cols - 1)
            hole_x = max(0, min(self.config.grid_cols - 1, int(hole_x)))

            top_row = self.grid.pop(0)
            if any(cell != 0 for cell in top_row):
                overflow = True

            garbage_row = ["G" for _ in range(self.config.grid_cols)]
            garbage_row[hole_x] = 0
            self.grid.append(garbage_row)
            applied += 1

        if self.current_piece is not None:
            self.current_piece.y -= applied
            if self.current_piece.check_collision(grid=self.grid):
                overflow = True

        self.garbage_received_total += applied
        if overflow:
            self.state = "GAME_OVER"

        return applied
