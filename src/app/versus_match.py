import random
from pathlib import Path

import pygame

from src.ai.ai_controller import AIController
from src.render.background import Background
from src.render.effects import VisualEffects
from src.game.game_core import GameCore
from src.game.game_modes import AILevel, MatchMode
from src.game.piece_sequence import SharedShapeSequence
from src.game.player_input import PlayerInputController
from src.render.render import Render
from settings import GameConfig
from src.render.ui_fonts import build_ui_font
from src.render.ui_primitives import draw_card, draw_soft_glow, draw_vertical_gradient, lerp_color


class VersusMatch:
    """封装一局玩家 vs AI 的对战会话。"""

    def __init__(
        self,
        screen,
        mode: MatchMode,
        config: GameConfig,
        ai_level: AILevel | None = None,
    ):
        self.screen = screen
        self.mode = mode
        self.config = config
        self.ai_level = ai_level if mode.key == "CLASSIC" else None

        panel_width = self.config.screen_width
        panel_height = self.config.screen_height

        self.left_rect = pygame.Rect(0, 0, panel_width, panel_height)
        self.right_rect = pygame.Rect(
            panel_width + self.config.versus_gap,
            0,
            panel_width,
            panel_height,
        )

        self.player_surface = self.screen.subsurface(self.left_rect)
        self.ai_surface = self.screen.subsurface(self.right_rect)

        self.player_background = Background(panel_width, panel_height)
        self.ai_background = Background(panel_width, panel_height)

        self.shared_sequence = SharedShapeSequence(config=self.config)
        self.player_core = GameCore(
            config=self.config,
            piece_factory=self.shared_sequence.make_piece_factory("player"),
        )
        self.ai_core = GameCore(
            config=self.config,
            piece_factory=self.shared_sequence.make_piece_factory("ai"),
        )
        self.player_renderer = Render(surface=self.player_surface, config=self.config)
        self.ai_renderer = Render(surface=self.ai_surface, config=self.config)
        self.player_input = PlayerInputController(config=self.config)

        initial_action_interval = self.mode.ai_action_interval_ms
        initial_mistake_chance = self.mode.ai_mistake_chance
        if self.ai_level is not None:
            initial_action_interval = self.ai_level.action_interval_ms
            initial_mistake_chance = self.ai_level.mistake_chance

        configured_model_path = Path(self.config.ai_model_path)
        if not configured_model_path.is_absolute():
            configured_model_path = Path(__file__).resolve().parent.parent.parent / configured_model_path

        # 经典模式采用轻量启发式控制，避免模型分支回退导致主线程卡顿。
        controller_mode = "heuristic" if self.mode.key == "CLASSIC" else "model"
        controller_lookahead_weight = 0.0 if self.mode.key == "CLASSIC" else 0.28

        self.ai_controller = AIController(
            config=self.config,
            action_interval_ms=initial_action_interval,
            mistake_chance=initial_mistake_chance,
            lookahead_weight=controller_lookahead_weight,
            mode=controller_mode,
            model_path=str(configured_model_path),
        )

        pygame.font.init()
        self.mode_font = build_ui_font(self.config, 22, bold=True)
        self.result_title_font = build_ui_font(self.config, 40, bold=True)
        self.result_body_font = build_ui_font(self.config, 24, bold=True)
        self.result_hint_font = build_ui_font(self.config, 18, bold=True)

        self.finished = False
        self.result_title = ""
        self.result_lines = []

        self.player_combo = -1
        self.ai_combo = -1
        self.player_b2b = False
        self.ai_b2b = False

        self.player_trap_energy = 0
        self.ai_trap_energy = 0
        self.player_trap_cooldown_ms = 0
        self.ai_trap_cooldown_ms = 0
        self.player_trap_state = None
        self.ai_trap_state = None

        self.player_warning_text = ""
        self.player_warning_ttl = 0
        self.ai_warning_text = ""
        self.ai_warning_ttl = 0

        self._processed_player_locks = self.player_core.lock_count
        self._processed_ai_locks = self.ai_core.lock_count

        self.fx = VisualEffects()

        self._fx_font = build_ui_font(self.config, 26, bold=True)

    def _lerp_color(self, color_a, color_b, ratio: float):
        return lerp_color(color_a, color_b, ratio)

    def _draw_vertical_gradient(self, rect, top_color, bottom_color):
        draw_vertical_gradient(self.screen, rect, top_color, bottom_color)

    def _draw_soft_glow(self, rect, color, spread=18, alpha=30, border_radius=8):
        draw_soft_glow(self.screen, rect, color, spread, alpha, border_radius)

    def _draw_card(self, rect, fill_color, border_color, glow_color=None, border_radius=8):
        draw_card(self.screen, rect, fill_color, border_color,
                  glow_color=glow_color, border_radius=border_radius,
                  shadow_offset=(16, 10), shadow_alpha=94,
                  inner_border_color=(48, 62, 86))

    def _draw_scene_backdrop(self):
        full_rect = self.screen.get_rect()
        self._draw_vertical_gradient(full_rect, (6, 9, 16), (13, 18, 29))

        glow_overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        for y in range(0, full_rect.height, 42):
            pygame.draw.line(glow_overlay, (120, 142, 178, 10), (0, y), (full_rect.width, y), 1)
        for x in range(0, full_rect.width, 42):
            pygame.draw.line(glow_overlay, (120, 142, 178, 8), (x, 0), (x, full_rect.height), 1)
        pygame.draw.rect(glow_overlay, (68, 126, 192, 30), pygame.Rect(0, 0, full_rect.width, 4))

        self.screen.blit(glow_overlay, (0, 0))

    def toggle_pause(self):
        if self.finished:
            return

        should_resume = any(
            core.state == "PAUSED" for core in (self.player_core, self.ai_core)
        )

        for core in (self.player_core, self.ai_core):
            if should_resume and core.state == "PAUSED":
                core.state = "RUNNING"
            elif not should_resume and core.state == "RUNNING":
                core.state = "PAUSED"

    def handle_keydown(self, key: int):
        if key in (pygame.K_p, pygame.K_w):
            self.toggle_pause()
            return

        if self.finished:
            return

        if key == pygame.K_c:
            self._try_activate_trap(is_player=True)
            return

        self.player_input.handle_keydown(key, self.player_core)

    def handle_keyup(self, key: int):
        if self.finished:
            return

        self.player_input.handle_keyup(key)

    def update(self, dt: int, pressed_keys):
        if self.finished:
            return

        self._tick_battle_states(dt)
        self._update_ai_profile()

        if any(core.state == "RUNNING" for core in (self.player_core, self.ai_core)):
            self.player_background.update()
            self.ai_background.update()

        self.player_input.update(dt, pressed_keys, self.player_core)
        self.ai_controller.update(self.ai_core, dt)
        self.player_core.update(dt)
        self.ai_core.update(dt)

        self._process_lock_events()
        self._try_ai_activate_trap()
        self._update_match_result()

        self.fx.update(dt)

    def draw(self):
        self._draw_scene_backdrop()
        self._draw_gap_background()

        self.player_renderer.draw(
            self.player_core,
            background=self.player_background,
            title="玩家",
            title_color=(115, 220, 255),
            info_lines=self._player_info_lines(),
            overlay_hint=self._overlay_hint(self.player_core, waiting_for="AI 对手"),
        )
        self.ai_renderer.draw(
            self.ai_core,
            background=self.ai_background,
            title="AI 对手",
            title_color=(255, 132, 118),
            info_lines=self._ai_info_lines(),
            overlay_hint=self._overlay_hint(self.ai_core, waiting_for="玩家"),
        )

        self._draw_board_frames()
        self.fx.draw(self.screen, font=self._fx_font)

        if self.finished:
            self._draw_result_overlay()

        if self.fx.is_shaking:
            offset = self.fx.get_shake_offset()
            temp = self.screen.copy()
            self.screen.fill((6, 9, 16))
            self.screen.blit(temp, offset)

    def _update_ai_profile(self):
        action_interval_ms = self.mode.ai_action_interval_ms
        mistake_chance = self.mode.ai_mistake_chance
        player_fall_speed = self.config.fall_speed_ms
        ai_fall_speed = self.config.fall_speed_ms

        if self.mode.key == "CLASSIC" and self.ai_level is not None:
            action_interval_ms = self.ai_level.action_interval_ms
            mistake_chance = self.ai_level.mistake_chance
            classic_fall_speed = max(80, int(self.ai_level.fall_speed_ms))
            player_fall_speed = classic_fall_speed
            ai_fall_speed = classic_fall_speed
            if self.ai_controller.mode == "model":
                action_interval_ms = int(self.config.ai_model_action_interval_ms)
        elif self.mode.key == "CHALLENGE":
            target_lines = self.mode.objective_lines or 1
            progress = min(1.0, self.player_core.lines_cleared_total / target_lines)
            pressure = 0.0
            if self.player_core.score > self.ai_core.score:
                pressure = min(1.0, (self.player_core.score - self.ai_core.score) / 1600)

            action_interval_ms = int(
                self.mode.ai_action_interval_ms - progress * 30 - pressure * 28
            )
            mistake_chance = max(
                0.03,
                self.mode.ai_mistake_chance - progress * 0.10 - pressure * 0.08,
            )
            ai_fall_speed = int(
                self.config.fall_speed_ms - progress * 140 - pressure * 100
            )

        self.player_core.set_fall_speed(player_fall_speed)
        self.ai_core.set_fall_speed(ai_fall_speed)
        self.ai_controller.set_profile(action_interval_ms, mistake_chance)

    def _update_match_result(self):
        if self.mode.key == "CLASSIC":
            if (
                self.player_core.state == "GAME_OVER"
                and self.ai_core.state == "GAME_OVER"
            ):
                self.finished = True
                self._set_classic_result()
            return

        target_lines = self.mode.objective_lines or 0
        if self.player_core.lines_cleared_total >= target_lines:
            self.finished = True
            self.result_title = "挑战成功"
            self.result_lines = [
                f"目标完成：{self.player_core.lines_cleared_total}/{target_lines} 行",
                f"当前得分  玩家 {self.player_core.score}  |  AI {self.ai_core.score}",
                "按 R 重试当前模式，或按 M 返回菜单。",
            ]
        elif self.player_core.state == "GAME_OVER":
            self.finished = True
            self.result_title = "挑战失败"
            self.result_lines = [
                f"目标未完成：{self.player_core.lines_cleared_total}/{target_lines} 行",
                f"当前得分  玩家 {self.player_core.score}  |  AI {self.ai_core.score}",
                "按 R 重试当前模式，或按 M 返回菜单。",
            ]

    def _set_classic_result(self):
        if self.player_core.score > self.ai_core.score:
            self.result_title = "玩家获胜"
        elif self.player_core.score < self.ai_core.score:
            self.result_title = "AI 获胜"
        else:
            self.result_title = "平局"

        self.result_lines = [
            f"最终得分  玩家 {self.player_core.score}  |  AI {self.ai_core.score}",
            f"消除总行数  玩家 {self.player_core.lines_cleared_total}  |  AI {self.ai_core.lines_cleared_total}",
            "按 R 重开本局，或按 M 返回菜单。",
        ]

    def _player_info_lines(self):
        incoming_text = f"来袭垃圾：{len(self.player_core.incoming_garbage)}"
        trap_text = self._trap_bar_text(self.player_trap_energy, self.player_trap_cooldown_ms)

        lines = []
        if self.mode.key == "CHALLENGE":
            target_lines = self.mode.objective_lines or 0
            remain_lines = max(0, target_lines - self.player_core.lines_cleared_total)
            lines.append(f"挑战剩余：{remain_lines} 行")
        else:
            level_label = self.ai_level.label if self.ai_level is not None else "默认"
            lines.append(f"经典对战 · {level_label}")

        lines.append(incoming_text)
        lines.append(trap_text)

        combo = self.player_combo
        if combo >= 0:
            lines.append(f"Combo x{combo + 1}" + ("  B2B!" if self.player_b2b else ""))
        elif self.player_b2b:
            lines.append("B2B 就绪")

        lines.append("控制：A/D/L/S，W暂停")
        return lines

    def _ai_info_lines(self):
        tempo_text = f"操作间隔：{self.ai_controller.action_interval_ms} ms"
        strategy_text = "策略：模型推理" if self.ai_controller.mode == "model" else "策略：启发式基线"
        incoming_text = f"来袭垃圾：{len(self.ai_core.incoming_garbage)}"
        trap_text = self._trap_bar_text(self.ai_trap_energy, self.ai_trap_cooldown_ms)
        warning = self.ai_warning_text if self.ai_warning_ttl > 0 else "防守：消行可抵消来袭"

        lines = [strategy_text, tempo_text, incoming_text, trap_text]
        combo = self.ai_combo
        if combo >= 0:
            lines.append(f"Combo x{combo + 1}" + ("  B2B!" if self.ai_b2b else ""))
        elif self.ai_b2b:
            lines.append("B2B 就绪")
        lines.append(warning)
        return lines

    def _trap_bar_text(self, energy: int, cooldown_ms: int) -> str:
        cost = self.config.versus_trap_energy_cost
        filled = min(energy, cost)
        empty = cost - filled
        bar = "█" * filled + "░" * empty
        if energy >= cost and cooldown_ms <= 0:
            return f"陷阱 [{bar}]  C 释放"
        return f"陷阱 [{bar}]  {energy}/{cost}"

    def _overlay_hint(self, core: GameCore, waiting_for: str):
        if core.state == "PAUSED":
            return "按 P 继续"
        if not self.finished and core.state == "GAME_OVER":
            return f"等待 {waiting_for}"
        return None

    def _tick_battle_states(self, dt: int):
        self.player_trap_cooldown_ms = max(0, self.player_trap_cooldown_ms - dt)
        self.ai_trap_cooldown_ms = max(0, self.ai_trap_cooldown_ms - dt)
        self.player_warning_ttl = max(0, self.player_warning_ttl - dt)
        self.ai_warning_ttl = max(0, self.ai_warning_ttl - dt)

    def _process_lock_events(self):
        if self.player_core.lock_count > self._processed_player_locks:
            self._processed_player_locks = self.player_core.lock_count
            self._on_lock_resolved(is_player=True, lines_cleared=self.player_core.last_cleared_lines)

        if self.ai_core.lock_count > self._processed_ai_locks:
            self._processed_ai_locks = self.ai_core.lock_count
            self._on_lock_resolved(is_player=False, lines_cleared=self.ai_core.last_cleared_lines)

    def _on_lock_resolved(self, is_player: bool, lines_cleared: int):
        core = self.player_core if is_player else self.ai_core
        owner = "玩家" if is_player else "AI"
        board_rect = self.left_rect if is_player else self.right_rect

        # Lock flash
        if core.last_locked_cells:
            self.fx.spawn_lock_flash(
                core.last_locked_cells,
                self.config.cell_size,
                board_rect.x,
                board_rect.y,
            )

        if lines_cleared > 0:
            self._update_combo_b2b(is_player, lines_cleared)
            self._gain_trap_energy(is_player, lines_cleared)

            # Particle burst on each cleared row
            for row_idx in core.last_cleared_rows:
                row_py = board_rect.y + row_idx * self.config.cell_size + self.config.cell_size // 2
                self.fx.spawn_line_clear_particles(
                    board_rect.x,
                    row_py,
                    board_rect.width,
                    (255, 255, 200),
                    count=12,
                )

            # Screen shake proportional to lines cleared
            shake_power = {1: 2.0, 2: 4.0, 3: 6.0, 4: 10.0}.get(lines_cleared, 4.0)
            self.fx.trigger_shake(shake_power, 280 if lines_cleared < 4 else 420)

            # Floating text
            combo = self.player_combo if is_player else self.ai_combo
            texts = []
            if lines_cleared == 4:
                texts.append(("TETRIS!", (255, 255, 100)))
            if combo >= 1:
                texts.append((f"COMBO x{combo + 1}", (255, 200, 100)))
            b2b = self.player_b2b if is_player else self.ai_b2b
            if b2b and lines_cleared == 4:
                texts.append(("B2B!", (255, 150, 255)))

            earned = core.score_mapping.get(lines_cleared, 0)
            if earned > 0:
                texts.append((f"+{earned}", (255, 255, 255)))

            text_x = board_rect.centerx
            text_y = board_rect.y + (core.last_cleared_rows[0] if core.last_cleared_rows else 10) * self.config.cell_size
            for i, (msg, clr) in enumerate(texts):
                self.fx.spawn_floating_text(msg, text_x, text_y - i * 32, clr, 1400)
        else:
            if is_player:
                self.player_combo = -1
                self.player_b2b = False
            else:
                self.ai_combo = -1
                self.ai_b2b = False

        attack_lines = self._compute_attack(is_player, lines_cleared)
        if attack_lines > 0:
            canceled = core.cancel_incoming_garbage(attack_lines)
            remaining = attack_lines - canceled
            if canceled > 0:
                self._set_warning(is_player, f"{owner} 抵消来袭 {canceled} 行")
            if remaining > 0:
                self._send_attack(is_player, remaining)

        if lines_cleared == 0 and self.config.versus_garbage_apply_on_nonclear:
            settled = core.apply_incoming_garbage(self.config.versus_garbage_cap_per_lock)
            if settled > 0:
                self._set_warning(is_player, f"{owner} 承受垃圾 {settled} 行")

    def _update_combo_b2b(self, is_player: bool, lines_cleared: int):
        combo = self.player_combo if is_player else self.ai_combo
        combo = combo + 1 if lines_cleared > 0 else -1

        if is_player:
            self.player_combo = combo
        else:
            self.ai_combo = combo

        if lines_cleared == 4:
            if is_player:
                self.player_b2b = True
            else:
                self.ai_b2b = True
        elif lines_cleared > 0:
            if is_player:
                self.player_b2b = False
            else:
                self.ai_b2b = False

    def _compute_attack(self, is_player: bool, lines_cleared: int) -> int:
        if lines_cleared <= 0:
            return 0

        base = self.config.versus_attack_mapping.get(lines_cleared, 0)
        combo = self.player_combo if is_player else self.ai_combo
        combo_idx = max(0, min(combo, len(self.config.versus_combo_bonus) - 1))
        combo_bonus = self.config.versus_combo_bonus[combo_idx]

        b2b = self.player_b2b if is_player else self.ai_b2b
        b2b_bonus = self.config.versus_b2b_bonus if b2b and lines_cleared == 4 else 0
        return max(0, int(base + combo_bonus + b2b_bonus))

    def _send_attack(self, from_player: bool, lines: int):
        target_core = self.ai_core if from_player else self.player_core
        holes = self._build_attack_holes(from_player, lines, target_core.config.grid_cols)
        target_core.queue_garbage(holes)

        if from_player:
            self._set_warning(False, f"玩家发来 {lines} 行垃圾")
        else:
            self._set_warning(True, f"AI 发来 {lines} 行垃圾")

    def _build_attack_holes(self, from_player: bool, lines: int, cols: int) -> list[int]:
        holes = []
        trap_state = self.player_trap_state if from_player else self.ai_trap_state

        for _ in range(lines):
            if trap_state is not None and trap_state["remaining"] > 0:
                holes.append(self._forced_hole(trap_state["pattern"], cols))
                trap_state["remaining"] -= 1
                if trap_state["remaining"] <= 0:
                    if from_player:
                        self.player_trap_state = None
                    else:
                        self.ai_trap_state = None
            else:
                holes.append(random.randint(0, cols - 1))
        return holes

    def _forced_hole(self, pattern: str, cols: int) -> int:
        one_third = max(1, cols // 3)
        if pattern == "left":
            return random.randint(0, one_third - 1)
        if pattern == "right":
            return random.randint(cols - one_third, cols - 1)

        left = max(0, one_third)
        right = min(cols - 1, cols - one_third - 1)
        if left > right:
            return cols // 2
        return random.randint(left, right)

    def _gain_trap_energy(self, is_player: bool, lines_cleared: int):
        gain = lines_cleared * self.config.versus_trap_energy_gain_per_line
        if is_player:
            self.player_trap_energy += gain
        else:
            self.ai_trap_energy += gain

    def _try_activate_trap(self, is_player: bool):
        core = self.player_core if is_player else self.ai_core
        if core.state != "RUNNING":
            return False

        energy = self.player_trap_energy if is_player else self.ai_trap_energy
        cooldown = self.player_trap_cooldown_ms if is_player else self.ai_trap_cooldown_ms
        if energy < self.config.versus_trap_energy_cost or cooldown > 0:
            return False

        cols = core.config.grid_cols
        center = cols // 2
        pattern = "center"
        if core.current_piece is not None:
            if core.current_piece.x < center - 2:
                pattern = "left"
            elif core.current_piece.x > center + 1:
                pattern = "right"

        state = {
            "pattern": pattern,
            "remaining": self.config.versus_trap_forced_lines,
        }
        if is_player:
            self.player_trap_state = state
            self.player_trap_energy -= self.config.versus_trap_energy_cost
            self.player_trap_cooldown_ms = self.config.versus_trap_cooldown_ms
            self._set_warning(False, "玩家触发陷阱：洞位受限")
        else:
            self.ai_trap_state = state
            self.ai_trap_energy -= self.config.versus_trap_energy_cost
            self.ai_trap_cooldown_ms = self.config.versus_trap_cooldown_ms
            self._set_warning(True, "AI 触发陷阱：注意防守")

        return True

    def _try_ai_activate_trap(self):
        if self.ai_core.state != "RUNNING":
            return
        if self.ai_trap_state is not None:
            return
        if self.ai_trap_energy < self.config.versus_trap_ai_use_threshold:
            return
        if self.ai_trap_cooldown_ms > 0:
            return
        if random.random() <= self.config.versus_trap_ai_use_chance:
            self._try_activate_trap(is_player=False)

    def _set_warning(self, for_player: bool, text: str):
        if for_player:
            self.player_warning_text = text
            self.player_warning_ttl = self.config.versus_warning_duration_ms
        else:
            self.ai_warning_text = text
            self.ai_warning_ttl = self.config.versus_warning_duration_ms

    def _draw_gap_background(self):
        gap_rect = pygame.Rect(
            self.left_rect.right,
            0,
            self.config.versus_gap,
            self.config.screen_height,
        )
        self._draw_vertical_gradient(gap_rect, (13, 18, 32), (8, 12, 22))
        pygame.draw.line(
            self.screen,
            (66, 90, 128),
            (gap_rect.centerx, 28),
            (gap_rect.centerx, self.config.screen_height - 28),
            2,
        )
        badge_rect = pygame.Rect(0, 0, min(112, gap_rect.width - 6), 52)
        badge_rect.center = (gap_rect.centerx, self.config.screen_height // 2)
        self._draw_card(
            badge_rect,
            (18, 24, 38),
            (92, 126, 178),
            glow_color=(92, 160, 255),
            border_radius=8,
        )
        vs_surface = self.result_body_font.render("VS", True, (242, 246, 255))
        vs_rect = vs_surface.get_rect(center=badge_rect.center)
        self.screen.blit(vs_surface, vs_rect)

    def _draw_mode_pill(self):
        text_value = self.mode.label
        if self.mode.key == "CLASSIC" and self.ai_level is not None:
            text_value = f"经典模式 · {self.ai_level.label}"

        if self.ai_controller.mode == "model":
            text_value = f"{text_value} · 模型AI"

        text = self.mode_font.render(text_value, True, (235, 240, 255))
        pill_rect = pygame.Rect(0, 0, text.get_width() + 42, text.get_height() + 18)
        pill_rect.centerx = self.screen.get_width() // 2
        pill_rect.y = 18

        self._draw_card(
            pill_rect,
            (17, 23, 36),
            (105, 138, 190),
            glow_color=(102, 168, 255),
            border_radius=8,
        )
        accent_bar = pygame.Rect(pill_rect.x + 18, pill_rect.y + 10, pill_rect.width - 36, 4)
        pygame.draw.rect(self.screen, (128, 196, 255), accent_bar, border_radius=2)
        self.screen.blit(text, (pill_rect.x + 21, pill_rect.y + 10))

    def _draw_board_frames(self):
        left_color = (74, 112, 148)
        right_color = (148, 84, 78)

        if self.finished and self.result_title in ("玩家获胜", "挑战成功"):
            left_color = (110, 225, 255)
        elif self.finished and self.result_title == "AI 获胜":
            right_color = (255, 130, 120)

        for rect, color in ((self.left_rect, left_color), (self.right_rect, right_color)):
            self._draw_soft_glow(rect, color, spread=12, alpha=24, border_radius=0)
            pygame.draw.rect(self.screen, self._lerp_color(color, (255, 255, 255), 0.18), rect, 1)
            pygame.draw.rect(self.screen, color, rect, 3)
            corner = 20
            pygame.draw.line(self.screen, color, (rect.x, rect.y), (rect.x + corner, rect.y), 4)
            pygame.draw.line(self.screen, color, (rect.x, rect.y), (rect.x, rect.y + corner), 4)
            pygame.draw.line(self.screen, color, (rect.right, rect.y), (rect.right - corner, rect.y), 4)
            pygame.draw.line(self.screen, color, (rect.right, rect.y), (rect.right, rect.y + corner), 4)
            pygame.draw.line(self.screen, color, (rect.x, rect.bottom), (rect.x + corner, rect.bottom), 4)
            pygame.draw.line(self.screen, color, (rect.x, rect.bottom), (rect.x, rect.bottom - corner), 4)
            pygame.draw.line(self.screen, color, (rect.right, rect.bottom), (rect.right - corner, rect.bottom), 4)
            pygame.draw.line(self.screen, color, (rect.right, rect.bottom), (rect.right, rect.bottom - corner), 4)

    def _draw_result_overlay(self):
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((6, 10, 18, 172))
        pygame.draw.circle(
            overlay,
            (100, 178, 255, 32),
            (self.screen.get_width() // 2, self.screen.get_height() // 2 - 40),
            240,
        )
        self.screen.blit(overlay, (0, 0))

        accent = (104, 170, 255)
        if self.result_title in ("玩家获胜", "挑战成功"):
            accent = (102, 228, 255)
        elif self.result_title == "AI 获胜":
            accent = (255, 132, 118)

        card_rect = pygame.Rect(0, 0, 760, 254)
        card_rect.center = (
            self.screen.get_width() // 2,
            self.screen.get_height() // 2,
        )

        self._draw_card(
            card_rect,
            (17, 23, 36),
            (96, 140, 214),
            glow_color=accent,
            border_radius=10,
        )
        accent_bar = pygame.Rect(card_rect.x + 28, card_rect.y + 18, card_rect.width - 56, 6)
        pygame.draw.rect(self.screen, accent, accent_bar, border_radius=2)

        title_surface = self.result_title_font.render(
            self.result_title,
            True,
            (248, 250, 255),
        )
        title_rect = title_surface.get_rect(center=(card_rect.centerx, card_rect.y + 64))
        self.screen.blit(title_surface, title_rect)

        text_y = card_rect.y + 118
        for line in self.result_lines:
            text_surface = self.result_body_font.render(line, True, (220, 228, 240))
            text_rect = text_surface.get_rect(center=(card_rect.centerx, text_y))
            self.screen.blit(text_surface, text_rect)
            text_y += 38

        hint_surface = self.result_hint_font.render(
            "操作：R 重开  |  M 返回菜单  |  Esc 退出",
            True,
            (165, 182, 214),
        )
        hint_rect = hint_surface.get_rect(center=(card_rect.centerx, card_rect.bottom - 28))
        self.screen.blit(hint_surface, hint_rect)
