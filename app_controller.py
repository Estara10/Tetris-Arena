import sys
from dataclasses import replace

import pygame

from game_modes import AI_LEVELS, GAME_MODES, MENU_CLASSIC_LEVEL, MENU_MODE_SELECT
from menu_screens import draw_classic_level_menu, draw_mode_select_menu
from shared_arena_match import SharedArenaMatch
from settings import CONFIG, GameConfig
from ui_fonts import build_menu_fonts
from versus_match import VersusMatch


def build_match_config(base_config: GameConfig) -> GameConfig:
    """为对抗模式生成更适合双屏布局的配置。"""
    match_config = replace(base_config)

    display_info = pygame.display.Info()
    current_panel_width = match_config.screen_width
    current_height = match_config.screen_height

    available_panel_width = max(
        current_panel_width,
        min(
            match_config.versus_target_panel_width,
            (max(display_info.current_w, current_panel_width * 2) - match_config.versus_gap - 64) // 2,
        ),
    )
    available_height = max(
        current_height,
        min(
            match_config.versus_target_height,
            max(display_info.current_h, current_height) - 48,
        ),
    )

    match_config.auto_tune_cell_size(
        target_width=available_panel_width,
        target_height=available_height,
    )
    return match_config


class TetrisApp:
    """管理菜单、模式切换和整局对战流程的主应用程序类。"""

    def __init__(self):
        """初始化 Pygame，准备窗口环境和共享状态数据。"""
        pygame.init()

        self.match_config = build_match_config(CONFIG)
        self.screen = pygame.display.set_mode(
            (self.match_config.versus_screen_width, self.match_config.screen_height),
            pygame.DOUBLEBUF,
        )
        pygame.display.set_caption("俄罗斯方块 AI 对战")
        self.clock = pygame.time.Clock()
        self.menu_fonts = build_menu_fonts(self.match_config)
        self.menu_cache = {}

        self.running = True
        self.menu_state = MENU_MODE_SELECT
        self.current_match = None
        self.current_mode = None
        self.current_ai_level = None
        self._menu_entry_ms = pygame.time.get_ticks()

    def run(self):
        """游戏的主循环。处理事件、更新逻辑并负责绘制画面。"""
        while self.running:
            dt = self.clock.tick(self.match_config.fps)

            for event in pygame.event.get():
                self._handle_event(event)
                if not self.running:
                    break

            if not self.running:
                break

            pressed_keys = pygame.key.get_pressed()
            self._render_frame(dt, pressed_keys)
            pygame.display.flip()

        pygame.quit()
        sys.exit(0)

    def _handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return

        if event.type == pygame.KEYUP and self.current_match is not None:
            self.current_match.handle_keyup(event.key)
            return

        if event.type != pygame.KEYDOWN:
            return

        if event.key in (pygame.K_ESCAPE, pygame.K_q):
            self.running = False
            return

        if self.current_match is None:
            self._handle_menu_keydown(event.key)
            return

        if event.key == pygame.K_m:
            self._return_to_menu()
            return

        if event.key == pygame.K_r:
            self._restart_match()
            return

        self.current_match.handle_keydown(event.key)

    def _handle_menu_keydown(self, key: int):
        """处理菜单状态下的所有键盘输入操作（模式选择、AI等级选择）。"""
        if self.menu_state == MENU_MODE_SELECT:
            if key in (pygame.K_1, pygame.K_KP1):
                self.current_mode = GAME_MODES["CLASSIC"]
                self.current_ai_level = None
                self.menu_state = MENU_CLASSIC_LEVEL
                self._menu_entry_ms = pygame.time.get_ticks()
            elif key in (pygame.K_2, pygame.K_KP2):
                self._start_match(GAME_MODES["TRADITIONAL"])
            return

        if self.menu_state == MENU_CLASSIC_LEVEL:
            if key == pygame.K_m:
                self.current_mode = None
                self.current_ai_level = None
                self.menu_state = MENU_MODE_SELECT
                self._menu_entry_ms = pygame.time.get_ticks()
            elif key in (pygame.K_1, pygame.K_KP1):
                self._start_match(self.current_mode, AI_LEVELS["EASY"])
            elif key in (pygame.K_2, pygame.K_KP2):
                self._start_match(self.current_mode, AI_LEVELS["NORMAL"])
            elif key in (pygame.K_3, pygame.K_KP3):
                self._start_match(self.current_mode, AI_LEVELS["HARD"])

    def _start_match(self, mode, ai_level=None):
        """传入选定的游戏模式和可能的 AI 等级，以此初始化对局环境 (VersusMatch)。"""
        self.current_mode = mode
        self.current_ai_level = ai_level if mode.key == "CLASSIC" else None
        if mode.key == "TRADITIONAL":
            self.current_match = SharedArenaMatch(
                self.screen,
                self.current_mode,
                self.match_config,
            )
            return

        self.current_match = VersusMatch(
            self.screen,
            self.current_mode,
            self.match_config,
            ai_level=self.current_ai_level,
        )

    def _restart_match(self):
        if self.current_mode is None:
            return

        if self.current_mode.key == "TRADITIONAL":
            self.current_match = SharedArenaMatch(
                self.screen,
                self.current_mode,
                self.match_config,
            )
            return

        self.current_match = VersusMatch(
            self.screen,
            self.current_mode,
            self.match_config,
            ai_level=self.current_ai_level,
        )

    def _return_to_menu(self):
        self.current_match = None
        self.current_mode = None
        self.current_ai_level = None
        self.menu_state = MENU_MODE_SELECT
        self._menu_entry_ms = pygame.time.get_ticks()

    def _get_menu_surface(self, menu_state: str):
        cached = self.menu_cache.get(menu_state)
        if cached is not None:
            return cached

        surface = pygame.Surface(self.screen.get_size()).convert()
        if menu_state == MENU_MODE_SELECT:
            draw_mode_select_menu(surface, self.match_config, self.menu_fonts)
        else:
            draw_classic_level_menu(surface, self.match_config, self.menu_fonts)

        self.menu_cache[menu_state] = surface
        return surface

    def _render_frame(self, dt: int, pressed_keys):
        """核心渲染调用，根据状态决定绘制主菜单或是绘制当前的对抗比赛画面。"""
        if self.current_match is None:
            menu_surface = self._get_menu_surface(self.menu_state)
            now = pygame.time.get_ticks()
            elapsed = now - self._menu_entry_ms
            duration = 500
            if elapsed < duration:
                progress = elapsed / duration
                eased = 1.0 - (1.0 - progress) ** 3
                alpha = int(255 * eased)
                temp = menu_surface.copy()
                temp.set_alpha(alpha)
                self.screen.fill((20, 20, 24))
                self.screen.blit(temp, (0, 0))
            else:
                self.screen.blit(menu_surface, (0, 0))
            return

        self.current_match.update(dt, pressed_keys)
        self.current_match.draw()


def main():
    TetrisApp().run()
