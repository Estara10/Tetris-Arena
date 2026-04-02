import pygame

from settings import CONFIG, GameConfig
from ui_fonts import build_ui_font


class Render:
    """
    俄罗斯方块渲染模块：
    只负责“画什么、怎么画”，不负责玩法逻辑。
    """

    def __init__(
        self,
        surface,
        config: GameConfig | None = None,
        cell_size=None,
        grid_cols=None,
        grid_rows=None,
        side_panel_width=None,
    ):
        self.config = config if config is not None else CONFIG
        self.surface = surface
        self.cell_size = self.config.cell_size if cell_size is None else cell_size
        self.grid_cols = self.config.grid_cols if grid_cols is None else grid_cols
        self.grid_rows = self.config.grid_rows if grid_rows is None else grid_rows
        self.game_width = self.grid_cols * self.cell_size
        self.game_height = self.grid_rows * self.cell_size
        self.side_panel_width = (
            self.config.side_panel_width if side_panel_width is None else side_panel_width
        )

        self.bg_color = self.config.render_bg_color
        self.grid_color = self.config.grid_color
        self.text_color = self.config.text_color
        self.panel_color = self.config.panel_color
        self.overlay_color = self.config.overlay_color
        self.ghost_alpha = self.config.ghost_alpha
        self.block_colors = dict(self.config.block_colors)

        pygame.font.init()
        self.font_title = build_ui_font(self.config, self.config.font_size_title, bold=True)
        self.font_main = build_ui_font(self.config, self.config.font_size_main, bold=True)
        self.font_hint = build_ui_font(self.config, self.config.font_size_hint, bold=True)

        self._alpha_block_cache = {}
        self._board_overlay = pygame.Surface(
            (self.game_width, self.game_height),
            pygame.SRCALPHA,
        )
        self._board_overlay.fill((10, 14, 24, 104))

    def draw(
        self,
        game_core,
        background=None,
        title="PLAYER",
        title_color=None,
        info_lines=None,
        overlay_hint=None,
    ):
        """
        对外统一入口：按固定层级完成整帧绘制。
        """
        if background is not None:
            background.draw(self.surface)
        else:
            self.surface.fill(self.bg_color)

        self.surface.blit(self._board_overlay, (0, 0))
        self.draw_grid()

        if game_core.state != "GAME_OVER" and game_core.current_piece is not None:
            self.draw_ghost_piece(game_core.current_piece, game_core.grid)

        self.draw_locked_blocks(game_core.grid)

        if game_core.state != "GAME_OVER" and game_core.current_piece is not None:
            self.draw_piece(game_core.current_piece)

        self.draw_ui(
            title=title,
            title_color=title_color,
            score=game_core.score,
            lines_cleared=game_core.lines_cleared_total,
            next_piece=game_core.next_piece,
            state=game_core.state,
            info_lines=info_lines or [],
            overlay_hint=overlay_hint,
        )

    def draw_grid(self):
        """
        绘制游戏主区域的网格线
        """
        # 绘制垂直网格线
        for x in range(0, self.game_width + 1, self.cell_size):
            pygame.draw.line(
                self.surface,
                self.grid_color,
                (x, 0),
                (x, self.game_height),
                1,
            )

        # 绘制水平网格线
        for y in range(0, self.game_height + 1, self.cell_size):
            pygame.draw.line(
                self.surface,
                self.grid_color,
                (0, y),
                (self.game_width, y),
                1,
            )

    def draw_block(self, grid_x, grid_y, color, alpha=255, border_color=(255, 255, 255)):
        """
        在指定网格坐标绘制单个方块单元
        :param grid_x: 网格 X 坐标
        :param grid_y: 网格 Y 坐标
        :param color: 方块填充颜色
        :param alpha: 透明度 (0-255)
        :param border_color: 边框颜色
        """
        # 如果方块在顶部屏幕外（例如刚生成的方块），不绘制
        if grid_y < 0:
            return

        # 计算实际像素坐标
        px = grid_x * self.cell_size
        py = grid_y * self.cell_size

        # 越界检查，防止绘制到游戏区外部
        if px < 0 or px >= self.game_width or py < 0 or py >= self.game_height:
            return

        rect = pygame.Rect(px, py, self.cell_size, self.cell_size)

        # 全不透明时直接绘制，提高性能
        if alpha >= 255:
            pygame.draw.rect(self.surface, color, rect)
            pygame.draw.rect(self.surface, border_color, rect, 1)
            return

        # 处理带透明度的方块（如阴影方块），使用缓存避免重复创建 Surface
        key = (color, alpha, self.cell_size)
        block_surface = self._alpha_block_cache.get(key)
        if block_surface is None:
            # 创建支持透明通道的 Surface
            block_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            block_surface.fill((color[0], color[1], color[2], alpha))
            pygame.draw.rect(
                block_surface,
                (border_color[0], border_color[1], border_color[2], alpha),
                block_surface.get_rect(),
                1,
            )
            # 存入缓存
            self._alpha_block_cache[key] = block_surface

        # 将半透明方块绘制到主 Surface 上
        self.surface.blit(block_surface, (px, py))

    def draw_piece(self, piece, alpha=255):
        """
        绘制当前正在下落的方块
        """
        color = self.block_colors.get(piece.shape_name, (220, 220, 220))
        for row_idx, row in enumerate(piece.matrix):
            for col_idx, cell in enumerate(row):
                if cell == "X":  # 仅绘制形状矩阵中被标记为 'X' 的部分
                    self.draw_block(
                        piece.x + col_idx,
                        piece.y + row_idx,
                        color,
                        alpha=alpha,
                    )

    def draw_ghost_piece(self, piece, grid):
        """
        绘制幽灵方块（即方块下落的最终位置预览）
        """
        # 计算幽灵方块的 Y 轴落地位置
        ghost_y = piece.get_drop_position(grid)
        ghost_color = self.block_colors.get(piece.shape_name, (220, 220, 220))

        for row_idx, row in enumerate(piece.matrix):
            for col_idx, cell in enumerate(row):
                if cell == "X":
                    self.draw_block(
                        piece.x + col_idx,
                        ghost_y + row_idx,
                        ghost_color,
                        alpha=self.ghost_alpha,
                        border_color=(240, 240, 240),
                    )

    def draw_locked_blocks(self, grid):
        """
        绘制已经锁定在网格中的方块堆
        """
        for y in range(self.grid_rows):
            for x in range(self.grid_cols):
                cell = grid[y][x]
                if cell != 0:  # 0 表示该位置为空
                    color = self.block_colors.get(cell, (200, 200, 200))
                    self.draw_block(x, y, color, alpha=255)

    def draw_ui(
        self,
        title,
        title_color,
        score,
        lines_cleared,
        next_piece,
        state,
        info_lines,
        overlay_hint,
    ):
        """
        绘制游戏右侧的 UI 面板（含得分、行数、下一个方块预览等）及状态层叠加
        """
        # 设置面板区域及背景
        panel_x = self.game_width
        panel_rect = pygame.Rect(panel_x, 0, self.side_panel_width, self.game_height)
        pygame.draw.rect(self.surface, self.panel_color, panel_rect)
        pygame.draw.line(
            self.surface,
            (90, 90, 90),
            (panel_x, 0),
            (panel_x, self.game_height),
            2,
        )

        # 绘制玩家标题
        title_surface = self.font_title.render(
            title,
            True,
            title_color or self.text_color,
        )
        self.surface.blit(title_surface, (panel_x + 20, 20))

        # 绘制得分
        score_title = self.font_hint.render("得分", True, self.text_color)
        score_value = self.font_main.render(str(score), True, self.text_color)
        self.surface.blit(score_title, (panel_x + 20, 78))
        self.surface.blit(score_value, (panel_x + 20, 104))

        # 绘制消除行数
        lines_title = self.font_hint.render("消行", True, self.text_color)
        lines_value = self.font_main.render(str(lines_cleared), True, self.text_color)
        self.surface.blit(lines_title, (panel_x + 20, 154))
        self.surface.blit(lines_value, (panel_x + 20, 180))

        # 绘制额外信息行（例如连击、Back-to-Back 提示等）
        info_y = 238
        for info in info_lines[:4]:
            info_surface = self.font_hint.render(info, True, (225, 225, 225))
            self.surface.blit(info_surface, (panel_x + 20, info_y))
            info_y += 30

        # 准备下一个方块的预览区域
        next_y = max(360, info_y + 18)
        next_title = self.font_hint.render("下一个", True, self.text_color)
        self.surface.blit(next_title, (panel_x + 20, next_y))

        preview_size = min(130, self.side_panel_width - 40)
        preview_x = panel_x + 20
        preview_y = next_y + 28
        preview_rect = pygame.Rect(preview_x, preview_y, preview_size, preview_size)
        pygame.draw.rect(self.surface, (16, 16, 20), preview_rect)
        pygame.draw.rect(self.surface, (120, 120, 120), preview_rect, 1)

        # 在预览区绘制下一个方块
        if next_piece is not None:
            self._draw_next_piece_preview(next_piece, preview_rect)

        # 如果游戏处于暂停或结束状态，绘制全屏半透明遮罩与提示文字
        if state in ("PAUSED", "GAME_OVER"):
            overlay = pygame.Surface((self.game_width, self.game_height), pygame.SRCALPHA)
            overlay.fill(self.overlay_color)
            self.surface.blit(overlay, (0, 0))

            if state == "PAUSED":
                text = self.font_title.render("已暂停", True, (255, 235, 80))
                default_hint = "按 P 继续"
            else:
                text = self.font_title.render("游戏结束", True, (255, 80, 80))
                default_hint = "按 R 重开"

            hint = self.font_hint.render(
                overlay_hint or default_hint,
                True,
                (245, 245, 245),
            )
            # 居中对齐文字
            text_rect = text.get_rect(center=(self.game_width // 2, self.game_height // 2 - 14))
            hint_rect = hint.get_rect(center=(self.game_width // 2, self.game_height // 2 + 20))
            self.surface.blit(text, text_rect)
            self.surface.blit(hint, hint_rect)

    def _draw_next_piece_preview(self, piece, preview_rect):
        """
        内部方法：在 UI 的预览框中心绘制下一个方块
        """
        matrix_h = len(piece.matrix)
        matrix_w = len(piece.matrix[0]) if matrix_h > 0 else 0
        if matrix_w == 0:
            return

        # 根据方块尺寸和可用空间，动态计算渲染网格大小，限制在最大 24
        preview_cell = min(
            24,
            (preview_rect.width - 20) // max(1, matrix_w),
            (preview_rect.height - 20) // max(1, matrix_h),
        )

        draw_w = matrix_w * preview_cell
        draw_h = matrix_h * preview_cell
        
        # 计算偏移量以居中显示
        offset_x = preview_rect.x + (preview_rect.width - draw_w) // 2
        offset_y = preview_rect.y + (preview_rect.height - draw_h) // 2
        
        color = self.block_colors.get(piece.shape_name, (220, 220, 220))

        # 遍历形状矩阵绘制对应格子
        for row_idx, row in enumerate(piece.matrix):
            for col_idx, cell in enumerate(row):
                if cell != "X":
                    continue

                px = offset_x + col_idx * preview_cell
                py = offset_y + row_idx * preview_cell
                rect = pygame.Rect(px, py, preview_cell, preview_cell)
                pygame.draw.rect(self.surface, color, rect)
                pygame.draw.rect(self.surface, (255, 255, 255), rect, 1)
