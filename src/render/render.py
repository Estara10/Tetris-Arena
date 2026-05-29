import pygame

from settings import CONFIG, GameConfig
from src.render.ui_fonts import build_ui_font
from src.render.ui_primitives import draw_card, draw_soft_glow, shade, tint


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
        self.font_title = build_ui_font(self.config, min(24, self.config.font_size_title), bold=True)
        self.font_main = build_ui_font(self.config, min(22, self.config.font_size_main), bold=True)
        self.font_hint = build_ui_font(self.config, min(17, self.config.font_size_hint), bold=True)

        self._block_cache = {}
        self._board_rect = pygame.Rect(0, 0, self.game_width, self.game_height)
        self._panel_rect = pygame.Rect(self.game_width, 0, self.side_panel_width, self.game_height)
        self._board_backdrop = self._build_vertical_gradient_surface(
            self.game_width,
            self.game_height,
            (6, 10, 18, 242),
            (2, 5, 11, 250),
        )
        self._board_overlay = pygame.Surface((self.game_width, self.game_height), pygame.SRCALPHA)
        self._board_overlay.fill((4, 7, 13, 76))
        pygame.draw.rect(self._board_overlay, (65, 120, 190, 22), pygame.Rect(0, 0, self.game_width, 4))
        pygame.draw.rect(self._board_overlay, (20, 32, 52, 80), pygame.Rect(0, 0, self.game_width, self.game_height), 1)
        self._panel_backdrop = self._build_vertical_gradient_surface(
            self.side_panel_width,
            self.game_height,
            (13, 20, 33, 252),
            (7, 10, 18, 252),
        )
        self._panel_overlay = pygame.Surface((self.side_panel_width, self.game_height), pygame.SRCALPHA)
        pygame.draw.rect(self._panel_overlay, (92, 132, 190, 24), pygame.Rect(0, 0, self.side_panel_width, 4))
        for y in range(0, self.game_height, 56):
            pygame.draw.line(self._panel_overlay, (255, 255, 255, 7), (0, y), (self.side_panel_width, y), 1)

    def _build_vertical_gradient_surface(self, width, height, top_rgba, bottom_rgba):
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        for y in range(height):
            ratio = y / max(1, height - 1)
            color = tuple(
                int(top + (bottom - top) * ratio)
                for top, bottom in zip(top_rgba, bottom_rgba)
            )
            pygame.draw.line(surface, color, (0, y), (width, y))
        return surface

    def _tint(self, color, amount: float):
        return tint(color, amount)

    def _shade(self, color, amount: float):
        return shade(color, amount)

    def _fit_text(self, font, text: str, max_width: int) -> str:
        if font.size(text)[0] <= max_width:
            return text

        suffix = "..."
        clipped = text
        while clipped and font.size(clipped + suffix)[0] > max_width:
            clipped = clipped[:-1]
        return clipped + suffix if clipped else suffix

    def _draw_fitted_text(self, font, text, color, x, y, max_width):
        fitted = self._fit_text(font, str(text), max_width)
        surface = font.render(fitted, True, color)
        self.surface.blit(surface, (x, y))
        return surface

    def _draw_soft_glow(self, rect, color, spread=16, alpha=32, border_radius=8):
        draw_soft_glow(self.surface, rect, color, spread, alpha, border_radius)

    def _draw_card(self, rect, fill_color, border_color, glow_color=None, border_radius=8):
        draw_card(self.surface, rect, fill_color, border_color,
                  glow_color=glow_color, border_radius=border_radius,
                  shadow_offset=(14, 10), shadow_alpha=84,
                  inner_border_color=(42, 56, 78))

    def _draw_stat_card(self, rect, label, value, accent_color):
        self._draw_card(
            rect,
            (17, 23, 36),
            self._tint(accent_color, 0.18),
            glow_color=accent_color,
            border_radius=8,
        )
        accent_rect = pygame.Rect(rect.x + 14, rect.y + 14, rect.width - 28, 6)
        pygame.draw.rect(self.surface, accent_color, accent_rect, border_radius=2)

        label_surface = self.font_hint.render(label, True, (177, 191, 212))
        value_surface = self.font_main.render(str(value), True, (245, 248, 255))
        self.surface.blit(label_surface, (rect.x + 16, rect.y + 26))
        self.surface.blit(value_surface, (rect.x + 16, rect.y + 56))

    def _draw_info_chip(self, x, y, text, accent_color):
        max_width = self.side_panel_width - 52
        fitted = self._fit_text(self.font_hint, text, max_width - 32)
        label = self.font_hint.render(fitted, True, (232, 238, 248))
        chip_rect = pygame.Rect(x, y, min(max_width, label.get_width() + 32), label.get_height() + 12)
        pygame.draw.rect(self.surface, (22, 30, 46), chip_rect, border_radius=7)
        pygame.draw.rect(
            self.surface,
            self._tint(accent_color, 0.25),
            chip_rect,
            1,
            border_radius=7,
        )
        dot_center = (chip_rect.x + 12, chip_rect.centery)
        pygame.draw.circle(self.surface, accent_color, dot_center, 4)
        self.surface.blit(label, (chip_rect.x + 22, chip_rect.y + 6))
        return chip_rect.height

    def _build_block_surface(self, color, alpha, border_color):
        key = (color, alpha, border_color, self.cell_size)
        cached = self._block_cache.get(key)
        if cached is not None:
            return cached

        surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        outer_rect = pygame.Rect(1, 1, self.cell_size - 2, self.cell_size - 2)
        shadow_rect = outer_rect.move(0, 2)
        radius = max(4, self.cell_size // 5)
        base_alpha = max(0, min(255, alpha))
        dark_color = self._shade(color, 0.42)
        light_color = self._tint(color, 0.35)
        core_rect = outer_rect.inflate(-2, -2)
        highlight_rect = pygame.Rect(
            core_rect.x + 2,
            core_rect.y + 2,
            max(4, core_rect.width - 4),
            max(4, self.cell_size // 3),
        )

        pygame.draw.rect(
            surface,
            (dark_color[0], dark_color[1], dark_color[2], max(40, int(base_alpha * 0.78))),
            shadow_rect,
            border_radius=radius,
        )
        pygame.draw.rect(
            surface,
            (color[0], color[1], color[2], base_alpha),
            core_rect,
            border_radius=radius,
        )
        pygame.draw.rect(
            surface,
            (light_color[0], light_color[1], light_color[2], max(50, int(base_alpha * 0.72))),
            highlight_rect,
            border_radius=max(4, radius - 1),
        )
        pygame.draw.rect(
            surface,
            (border_color[0], border_color[1], border_color[2], base_alpha),
            core_rect,
            1,
            border_radius=radius,
        )
        self._block_cache[key] = surface
        return surface

    def _draw_board_backdrop(self):
        self.surface.blit(self._board_backdrop, (0, 0))
        self.surface.blit(self._board_overlay, (0, 0))
        pygame.draw.rect(self.surface, (56, 78, 112), self._board_rect, 2)

    def _draw_panel_backdrop(self):
        self.surface.blit(self._panel_backdrop, (self._panel_rect.x, self._panel_rect.y))
        self.surface.blit(self._panel_overlay, (self._panel_rect.x, self._panel_rect.y))
        pygame.draw.line(
            self.surface,
            (78, 108, 156),
            (self._panel_rect.x, 14),
            (self._panel_rect.x, self.game_height - 14),
            2,
        )

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

        self._draw_board_backdrop()
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
        for x in range(0, self.game_width + 1, self.cell_size):
            is_major = x % (self.cell_size * 5) == 0
            pygame.draw.line(
                self.surface,
                (76, 92, 118) if is_major else (44, 56, 78),
                (x, 0),
                (x, self.game_height),
                2 if is_major else 1,
            )

        for y in range(0, self.game_height + 1, self.cell_size):
            is_major = y % (self.cell_size * 5) == 0
            pygame.draw.line(
                self.surface,
                (76, 92, 118) if is_major else (42, 54, 76),
                (0, y),
                (self.game_width, y),
                2 if is_major else 1,
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

        block_surface = self._build_block_surface(color, alpha, border_color)
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
        panel_x = self.game_width
        self._draw_panel_backdrop()

        title_bar_rect = pygame.Rect(panel_x + 18, 18, self.side_panel_width - 36, 58)
        accent_color = title_color or self.text_color
        self._draw_card(
            title_bar_rect,
            (17, 23, 36),
            self._tint(accent_color, 0.12),
            glow_color=accent_color,
            border_radius=8,
        )
        title_text = self._fit_text(
            self.font_title,
            title,
            title_bar_rect.width - 32,
        )
        title_surface = self.font_title.render(title_text, True, accent_color)
        self.surface.blit(title_surface, (title_bar_rect.x + 16, title_bar_rect.y + 8))
        badge_surface = self.font_hint.render("HUD", True, (198, 212, 238))
        self.surface.blit(badge_surface, (title_bar_rect.x + 16, title_bar_rect.y + 32))

        stat_gap = 10
        stat_width = (self.side_panel_width - 18 * 2 - stat_gap) // 2
        score_rect = pygame.Rect(panel_x + 18, 92, stat_width, 92)
        lines_rect = pygame.Rect(score_rect.right + stat_gap, 92, stat_width, 92)
        self._draw_stat_card(score_rect, "得分", score, accent_color)
        self._draw_stat_card(lines_rect, "消行", lines_cleared, self._tint(accent_color, 0.1))

        info_card_rect = pygame.Rect(panel_x + 18, 198, self.side_panel_width - 36, 184)
        self._draw_card(
            info_card_rect,
            (17, 23, 36),
            (66, 92, 126),
            glow_color=(90, 132, 190),
            border_radius=8,
        )
        info_title = self.font_hint.render("状态信息", True, (214, 224, 242))
        self.surface.blit(info_title, (info_card_rect.x + 16, info_card_rect.y + 14))

        info_y = info_card_rect.y + 46
        for info in info_lines[:4]:
            chip_h = self._draw_info_chip(
                info_card_rect.x + 14,
                info_y,
                info,
                accent_color,
            )
            info_y += chip_h + 8
            if info_y > info_card_rect.bottom - 28:
                break

        next_card_y = info_card_rect.bottom + 14
        next_card_rect = pygame.Rect(panel_x + 18, next_card_y, self.side_panel_width - 36, 162)
        self._draw_card(
            next_card_rect,
            (17, 23, 36),
            (74, 106, 150),
            glow_color=(104, 152, 214),
            border_radius=8,
        )
        next_title = self.font_hint.render("下一个方块", True, (214, 224, 242))
        self.surface.blit(next_title, (next_card_rect.x + 16, next_card_rect.y + 14))

        preview_size = min(108, next_card_rect.width - 36)
        preview_x = next_card_rect.x + (next_card_rect.width - preview_size) // 2
        preview_y = next_card_rect.y + 42
        preview_rect = pygame.Rect(preview_x, preview_y, preview_size, preview_size)
        pygame.draw.rect(self.surface, (10, 14, 22), preview_rect, border_radius=8)
        pygame.draw.rect(self.surface, (86, 112, 152), preview_rect, 2, border_radius=8)
        preview_inner = preview_rect.inflate(-14, -14)
        pygame.draw.rect(self.surface, (14, 20, 30), preview_inner, border_radius=6)
        pygame.draw.rect(self.surface, (45, 60, 84), preview_inner, 1, border_radius=6)

        if next_piece is not None:
            self._draw_next_piece_preview(next_piece, preview_inner)

        if state in ("PAUSED", "GAME_OVER"):
            overlay = pygame.Surface((self.game_width, self.game_height), pygame.SRCALPHA)
            overlay.fill((6, 10, 18, 166))
            self.surface.blit(overlay, (0, 0))

            if state == "PAUSED":
                text = self.font_title.render("已暂停", True, (255, 235, 80))
                default_hint = "按 P 继续"
                accent = (255, 214, 82)
            else:
                text = self.font_title.render("游戏结束", True, (255, 80, 80))
                default_hint = "按 R 重开"
                accent = (255, 96, 102)

            hint = self.font_hint.render(
                overlay_hint or default_hint,
                True,
                (245, 245, 245),
            )
            card_rect = pygame.Rect(0, 0, min(320, self.game_width - 60), 132)
            card_rect.center = (self.game_width // 2, self.game_height // 2)
            self._draw_card(
                card_rect,
                (16, 22, 34),
                self._tint(accent, 0.2),
                glow_color=accent,
                border_radius=10,
            )
            accent_bar = pygame.Rect(card_rect.x + 22, card_rect.y + 18, card_rect.width - 44, 6)
            pygame.draw.rect(self.surface, accent, accent_bar, border_radius=2)
            text_rect = text.get_rect(center=(card_rect.centerx, card_rect.y + 58))
            hint_rect = hint.get_rect(center=(card_rect.centerx, card_rect.y + 92))
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

        preview_cell = min(
            24,
            (preview_rect.width - 20) // max(1, matrix_w),
            (preview_rect.height - 20) // max(1, matrix_h),
        )

        draw_w = matrix_w * preview_cell
        draw_h = matrix_h * preview_cell
        
        offset_x = preview_rect.x + (preview_rect.width - draw_w) // 2
        offset_y = preview_rect.y + (preview_rect.height - draw_h) // 2
        color = self.block_colors.get(piece.shape_name, (220, 220, 220))

        for row_idx, row in enumerate(piece.matrix):
            for col_idx, cell in enumerate(row):
                if cell != "X":
                    continue

                px = offset_x + col_idx * preview_cell
                py = offset_y + row_idx * preview_cell
                preview_surface = self._build_block_surface(
                    color,
                    255,
                    (255, 255, 255),
                )
                if preview_cell != self.cell_size:
                    preview_surface = pygame.transform.smoothscale(
                        preview_surface,
                        (preview_cell, preview_cell),
                    )
                self.surface.blit(preview_surface, (px, py))
