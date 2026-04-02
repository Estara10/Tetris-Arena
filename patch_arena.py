import dataclasses

code = '''import time
from dataclasses import replace
import random

import pygame

from ai_controller import generate_candidates
from game_modes import MatchMode
from piece_sequence import SharedShapeSequence
from settings import GameConfig
from shared_game_core import SharedGameCore, ArenaEntity
from ui_fonts import build_ui_font


class ArenaAIState:
    def __init__(self):
        self.planned_piece_id = None
        self.target_x = 0
        self.rotation_steps_remaining = 0


class SharedArenaMatch:
    """同屏大乱斗模式：支持多个实体在同一个棋盘上互相推挤"""

    def __init__(self, screen, mode: MatchMode, config: GameConfig):
        self.screen = screen
        self.mode = mode
        self.config = config

        self.arena_config = self._build_arena_config(config)

        sides = ["player", "ai1", "ai2"]
        self.shared_sequence = SharedShapeSequence(self.arena_config, sides=sides)
        
        spawns = getattr(self.arena_config, "shared_arena_spawns", [9, 17, 25])
        while len(spawns) < 3: spawns.append(spawns[-1] + 8)

        self.entities = [
            ArenaEntity(id="player", is_player=True, spawn_x=spawns[0], piece_factory=self.shared_sequence.make_piece_factory("player")),
            ArenaEntity(id="ai1", is_player=False, spawn_x=spawns[1], piece_factory=self.shared_sequence.make_piece_factory("ai1")),
            ArenaEntity(id="ai2", is_player=False, spawn_x=spawns[2], piece_factory=self.shared_sequence.make_piece_factory("ai2")),
        ]

        self.core = SharedGameCore(
            config=self.arena_config,
            grid_cols=self.arena_config.grid_cols,
            entities=self.entities,
        )

        self.cell_size = 24
        
        self.board_width = self.arena_config.grid_cols * self.cell_size
        self.board_height = self.arena_config.grid_rows * self.cell_size
        
        screen_w, screen_h = screen.get_size()
        center_x, center_y = screen_w // 2, screen_h // 2

        self.board_rect = pygame.Rect(
            center_x - self.board_width // 2,
            center_y - self.board_height // 2 + 10,
            self.board_width,
            self.board_height,
        )
        
        self.paused = False
        self.finished = False

        self.duration_ms = getattr(self.arena_config, "shared_arena_duration_ms", 180000)
        self.remaining_ms = self.duration_ms

        self.base_font = build_ui_font(28)
        self.large_font = build_ui_font(48)
        self.small_font = build_ui_font(20)

        # Limits and Timers
        self.cooldown_ms = {
            "player": getattr(self.arena_config, "shared_arena_player_cooldown_ms", 200),
            "ai1": getattr(self.arena_config, "shared_arena_ai_cooldown_ms", 300),
            "ai2": getattr(self.arena_config, "shared_arena_ai_cooldown_ms", 300),
        }
        self.move_timers = {ent.id: 0 for ent in self.entities}
        self.gravity_interval_ms = getattr(self.arena_config, "shared_arena_fall_speed_ms", 500)
        
        self.gravity_accumulators = {ent.id: 0 for ent in self.entities}
        if len(self.entities) > 1:
            self.gravity_accumulators["ai1"] += 90
        if len(self.entities) > 2:
            self.gravity_accumulators["ai2"] += 180

        # AI tracking
        self.ai_states = {ent.id: ArenaAIState() for ent in self.entities if not ent.is_player}

        # Player inputs (continuous movement decoupled from repeat)
        self._player_left_held = False
        self._player_right_held = False

        self.winner_id = None

    def _build_arena_config(self, config: GameConfig) -> GameConfig:
        return replace(
            config,
            grid_cols=getattr(config, "shared_arena_grid_cols", 33),
            grid_rows=getattr(config, "shared_arena_grid_rows", 25),
        )

    def handle_event(self, event) -> str | None:
        if event.type == pygame.KEYDOWN:
            key = event.key
            if key in (pygame.K_p, pygame.K_w):
                self.paused = not self.paused
                return None
            if self.paused or self.finished:
                if self.finished and key == pygame.K_ESCAPE:
                    return "QUIT"
                return None

            if key == pygame.K_ESCAPE:
                return "QUIT"
            elif key == pygame.K_a:
                self._player_left_held = True
            elif key == pygame.K_d:
                self._player_right_held = True
            elif key == pygame.K_l:
                # Rotate
                p_ent = self._get_entity("player")
                if p_ent and p_ent.piece:
                    self.core.rotate_piece(p_ent.piece)
            elif key == pygame.K_s:
                # Hard drop
                p_ent = self._get_entity("player")
                if p_ent and p_ent.piece:
                    self.core.hard_drop_piece(p_ent)
                    self.move_timers["player"] = self.cooldown_ms["player"] # Delay after hard drop

        elif event.type == pygame.KEYUP:
            key = event.key
            if key == pygame.K_a:
                self._player_left_held = False
            elif key == pygame.K_d:
                self._player_right_held = False

        return None
        
    def _get_entity(self, eid: str) -> ArenaEntity | None:
        for e in self.entities:
            if e.id == eid:
                return e
        return None

    def update(self, dt: int, _pressed_keys):
        if self.finished or self.paused:
            return

        self.remaining_ms = max(0, self.remaining_ms - dt)
        
        # Update cooldown timers
        for eid in self.move_timers:
            self.move_timers[eid] = max(0, self.move_timers[eid] - dt)
            
        # Update gravity
        for ent in self.entities:
            self.gravity_accumulators[ent.id] += dt
            if self.gravity_accumulators[ent.id] >= self.gravity_interval_ms:
                self.gravity_accumulators[ent.id] -= self.gravity_interval_ms
                self.core.soft_drop_piece(ent)
                
        if self.core.check_game_over() or self.remaining_ms <= 0:
            self.finished = True
            self._settle_result()
            return
            
        self._run_cycle()

    def _run_cycle(self):
        cmds = {}
        for ent in self.entities:
            cmds[ent.id] = {"dx": 0, "rotate": False, "hard_drop": False}
            
            if ent.is_player:
                if self.move_timers[ent.id] == 0:
                    if self._player_left_held and not self._player_right_held:
                        cmds[ent.id]["dx"] = -1
                        self.move_timers[ent.id] = self.cooldown_ms[ent.id]
                    elif self._player_right_held and not self._player_left_held:
                        cmds[ent.id]["dx"] = 1
                        self.move_timers[ent.id] = self.cooldown_ms[ent.id]
            else:
                if self.move_timers[ent.id] == 0:
                    ai_cmd = self._build_ai_command(ent)
                    if ai_cmd["rotate"]:
                        self.core.rotate_piece(ent.piece)
                        self.move_timers[ent.id] = self.cooldown_ms[ent.id] // 2
                    elif ai_cmd["hard_drop"]:
                        self.core.hard_drop_piece(ent)
                        self.move_timers[ent.id] = self.cooldown_ms[ent.id]
                    elif ai_cmd["dx"] != 0:
                        cmds[ent.id]["dx"] = ai_cmd["dx"]
                        self.move_timers[ent.id] = self.cooldown_ms[ent.id]
        
        # Apply movements with pushing logic
        # For simplicity, we process dx commands one by one, giving priority to player, then randomly
        # A more realistic physics simulation pushes the chain
        
        agents_moving = [e for e in self.entities if cmds[e.id]["dx"] != 0]
        # Sort by x coordinate roughly depending on direction?
        # Actually, let's process everyone's desired direction. 
        # If A wants to move right (+1), it pushes anything in its way by +1 if valid.
        
        for ent in agents_moving:
            dx = cmds[ent.id]["dx"]
            if dx == 0 or ent.piece is None: continue
            
            # Find the chain of pushed entities
            chain = self._find_push_chain(ent, dx)
            if chain is not None:
                # Chain is valid, move everyone in the chain
                for pushed_ent in chain:
                    pushed_ent.piece.x += dx

    def _find_push_chain(self, start_ent: ArenaEntity, dx: int) -> list[ArenaEntity] | None:
        """Finds all entities pushed by start_ent moving dx.
        Returns a list of entities. If the push hits a wall, returns None (invalid move)."""
        visited = set()
        chain = []
        
        queue = [start_ent]
        while queue:
            curr = queue.pop(0)
            if curr.id in visited: continue
            visited.add(curr.id)
            chain.append(curr)
            
            # Check if this single piece moving dx collides with walls
            if not self.core.is_valid_position(curr.piece, state_override={"x": curr.piece.x + dx}):
                # Wait, it might be colliding with another ACTIVE piece, OR a static wall.
                # Let's check against active pieces first.
                pass
                
            # For every block in curr.piece, if it moves dx, does it hit static wall?
            if not self._can_shift_single_ignoring_actives(curr.piece, dx):
                return None # Hit a static wall/block! The entire chain cannot move.
                
            # Now who does it push?
            for other in self.entities:
                if other.id in visited or other.piece is None: continue
                if self._pieces_overlap(curr.piece, dx, other.piece, 0):
                    queue.append(other)
                    
        return chain

    def _can_shift_single_ignoring_actives(self, piece, dx: int) -> bool:
        """Check if moving dx hits static blocks or screen boundaries"""
        if piece is None: return False
        for row_idx, row in enumerate(piece.matrix):
            for col_idx, cell in enumerate(row):
                if cell == "X":
                    bx = piece.x + col_idx + dx
                    by = piece.y + row_idx
                    if bx < 0 or bx >= self.arena_config.grid_cols: return False
                    if by >= self.arena_config.grid_rows: return False
                    if by >= 0 and self.core.grid[by][bx] != 0: return False
        return True

    def _pieces_overlap(self, p1, dx1, p2, dx2) -> bool:
        """Check if p1 (moved dx1) overlaps p2 (moved dx2)"""
        s1 = set()
        for r, row in enumerate(p1.matrix):
            for c, cell in enumerate(row):
                if cell == "X": s1.add((p1.x + c + dx1, p1.y + r))
                
        for r, row in enumerate(p2.matrix):
            for c, cell in enumerate(row):
                if cell == "X": 
                    if (p2.x + c + dx2, p2.y + r) in s1:
                        return True
        return False

    def _build_ai_command(self, ent: ArenaEntity) -> dict[str, int | bool]:
        piece = ent.piece
        state = self.ai_states[ent.id]
        
        if piece is None or self.core.state != "RUNNING":
            return {"dx": 0, "rotate": False, "hard_drop": False}

        if state.planned_piece_id != id(piece):
            self._plan_ai_piece(ent)

        cmd = {"dx": 0, "rotate": False, "hard_drop": False}
        if state.rotation_steps_remaining > 0:
            cmd["rotate"] = True
            state.rotation_steps_remaining -= 1
            return cmd

        if piece.x < state.target_x:
            cmd["dx"] = 1
        elif piece.x > state.target_x:
            cmd["dx"] = -1
        else:
            cmd["hard_drop"] = True
            state.planned_piece_id = None
            
        return cmd

    def _plan_ai_piece(self, ent: ArenaEntity):
        piece = ent.piece
        if piece is None: return
        
        state = self.ai_states[ent.id]
        candidates = generate_candidates(self.core.grid, piece, self.arena_config)
        if not candidates:
            state.rotation_steps_remaining = 0
            state.target_x = piece.x
            state.planned_piece_id = id(piece)
            return

        for candidate in candidates:
            # Simple heuristic
            candidate["total_score"] = candidate["surface_score"]

        candidates.sort(key=lambda item: item["total_score"], reverse=True)
        selected = candidates[0]

        if len(candidates) > 1 and random.random() < self.mode.ai_mistake_chance:
            selected = random.choice(candidates[: min(3, len(candidates))])

        state.rotation_steps_remaining = selected["rotation_steps"]
        state.target_x = selected["target_x"]
        state.planned_piece_id = id(piece)

    def _settle_result(self, reason="结算"):
        high_score = -1
        winners = []
        for ent in self.entities:
            if ent.score > high_score:
                high_score = ent.score
                winners = [ent.id]
            elif ent.score == high_score:
                winners.append(ent.id)
                
        if len(winners) == 1:
            self.winner_id = winners[0]
        else:
            self.winner_id = "Draw"

    def draw(self):
        self.screen.fill((10, 14, 22))
        self._draw_board()
        self._draw_entities()
        self._draw_panel()

        if self.paused and not self.finished:
            self._draw_overlay("已暂停", "按 W 或 P 继续")

        if self.finished:
            winner_text = "平局!" if self.winner_id == "Draw" else f"{self.winner_id} 获胜!"
            self._draw_overlay(winner_text, "按 ESC 退出")

    def _draw_board(self):
        pygame.draw.rect(self.screen, (20, 28, 42), self.board_rect)
        pygame.draw.rect(self.screen, (94, 124, 166), self.board_rect, 2)

        for x in range(self.arena_config.grid_cols + 1):
            px = self.board_rect.x + x * self.cell_size
            pygame.draw.line(self.screen, (56, 70, 94), (px, self.board_rect.y), (px, self.board_rect.bottom), 1)
        for y in range(self.arena_config.grid_rows + 1):
            py = self.board_rect.y + y * self.cell_size
            pygame.draw.line(self.screen, (56, 70, 94), (self.board_rect.x, py), (self.board_rect.right, py), 1)

        # Draw static grid
        for row_idx, row in enumerate(self.core.grid):
            for col_idx, cell in enumerate(row):
                if cell != 0:
                    self._draw_block(col_idx, row_idx, (100, 100, 100))

    def _draw_entities(self):
        colors = {
            "player": (0, 255, 200),
            "ai1": (255, 100, 100),
            "ai2": (255, 200, 100)
        }
        for ent in self.entities:
            c = colors.get(ent.id, (255, 255, 255))
            piece = getattr(ent, "piece", None)
            if piece:
                for r, row in enumerate(piece.matrix):
                    for col, cell in enumerate(row):
                        if cell == "X":
                            self._draw_block(piece.x + col, piece.y + r, c)
                            # ghost block?
                
                ghost_y = self.core.get_ghost_y(piece)
                for r, row in enumerate(piece.matrix):
                    for col, cell in enumerate(row):
                        if cell == "X" and ghost_y >= 0:
                            bx = self.board_rect.x + (piece.x + col) * self.cell_size
                            by = self.board_rect.y + (ghost_y + r) * self.cell_size
                            rect = pygame.Rect(bx, by, self.cell_size, self.cell_size)
                            pygame.draw.rect(self.screen, c, rect, 1)

    def _draw_block(self, col: int, row: int, color):
        if row < 0: return
        x = self.board_rect.x + col * self.cell_size
        y = self.board_rect.y + row * self.cell_size
        rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

    def _draw_panel(self):
        sw, sh = self.screen.get_size()
        panel_rect = pygame.Rect(self.board_rect.right + 20, self.board_rect.y, 250, self.board_height)
        pygame.draw.rect(self.screen, (20, 28, 42), panel_rect)
        pygame.draw.rect(self.screen, (94, 124, 166), panel_rect, 2)
        
        mins, secs = divmod(self.remaining_ms // 1000, 60)
        t_surf = self.base_font.render(f"时间: {mins:02d}:{secs:02d}", True, (255, 255, 255))
        self.screen.blit(t_surf, (panel_rect.x + 20, panel_rect.y + 20))
        
        y_offset = 80
        for ent in self.entities:
            s_surf = self.small_font.render(f"{ent.id}: {ent.score} 分", True, (255, 255, 255))
            self.screen.blit(s_surf, (panel_rect.x + 20, panel_rect.y + y_offset))
            y_offset += 40

    def _draw_overlay(self, title: str, subtitle: str):
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        sw, sh = self.screen.get_size()
        t_surf = self.large_font.render(title, True, (255, 255, 255))
        s_surf = self.base_font.render(subtitle, True, (200, 200, 200))
        self.screen.blit(t_surf, (sw // 2 - t_surf.get_width() // 2, sh // 2 - 50))
        self.screen.blit(s_surf, (sw // 2 - s_surf.get_width() // 2, sh // 2 + 20))
'''

with open('shared_arena_match.py', 'w') as f:
    f.write(code)

