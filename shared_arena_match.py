import time
from dataclasses import replace
from pathlib import Path
import random

import pygame

from ai_controller import generate_candidates
from dqn_model import load_inference_model
from game_modes import MatchMode
from next_state_features import NEXT_STATE_AUX_FEATURE_COUNT, extract_next_state_features
from piece_sequence import SharedShapeSequence
from settings import GameConfig
from shared_game_core import SharedGameCore, ArenaEntity
from ui_fonts import build_ui_font

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class ArenaAIState:
    def __init__(self):
        self.planned_piece_id = None
        self.target_x = 0
        self.rotation_steps_remaining = 0
        self.plan_signature = None

    def clear_plan(self):
        self.planned_piece_id = None
        self.target_x = 0
        self.rotation_steps_remaining = 0
        self.plan_signature = None


class SharedArenaMatch:
    """同屏大乱斗模式：支持多个实体在同一个棋盘上互相推挤"""

    def __init__(self, screen, mode: MatchMode, config: GameConfig):
        self.screen = screen
        self.mode = mode
        self.config = config

        self.arena_config = self._build_arena_config(config)

        if self.mode.key == "TRADITIONAL":
            sides = ["player", "ai1"]
            spawns = [9, 17]
            self.shared_sequence = SharedShapeSequence(self.arena_config, sides=sides)
            self.entities = [
                ArenaEntity(id="player", is_player=True, spawn_x=spawns[0], piece_factory=self.shared_sequence.make_piece_factory("player")),
                ArenaEntity(id="ai1", is_player=False, spawn_x=spawns[1], piece_factory=self.shared_sequence.make_piece_factory("ai1")),
            ]
        else:
            sides = ["player", "ai1", "ai2"]
            spawns = [9, 17, 25]
            self.shared_sequence = SharedShapeSequence(self.arena_config, sides=sides)
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

        self._player_left_held = False
        self._player_right_held = False
        self._player_down_held = False

        self.duration_ms = getattr(self.arena_config, "shared_arena_duration_ms", 180000)
        self.remaining_ms = self.duration_ms

        self.base_font = build_ui_font(self.arena_config, 28)
        self.large_font = build_ui_font(self.arena_config, 48)
        self.small_font = build_ui_font(self.arena_config, 20)
        self.model_enabled = self.mode.key == "TRADITIONAL"
        self._model = None
        self._model_input_dim = 0
        self._model_error = ""
        # In shared-arena mode, model is out-of-distribution relative to next-state training.
        # Keep heuristic as a safety prior and let model only re-rank a shortlist.
        self._model_rerank_topk = 8
        self._model_rerank_scale = 35.0
        configured_model_path = Path(self.config.ai_model_path)
        if not configured_model_path.is_absolute():
            configured_model_path = Path(__file__).resolve().parent / configured_model_path
        self._model_path = configured_model_path

        # Limits and Timers
        self.cooldown_ms = {
            "player": 100 if self.mode.key == "TRADITIONAL" else 200,
            "ai1": 200 if self.mode.key == "TRADITIONAL" else 300,
            "ai2": 200 if self.mode.key == "TRADITIONAL" else 300,
        }
        self.move_timers = {ent.id: 0 for ent in self.entities}
        self.gravity_interval_ms = 500
        
        # 初始撞人权归咱
        self.push_rights = {ent.id: ent.is_player for ent in self.entities}
        if self.mode.key != "TRADITIONAL":
            # 三体版不管撞人权限制
            self.push_rights = {ent.id: True for ent in self.entities}

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

        if self.model_enabled:
            for ent in self.entities:
                if not ent.is_player:
                    self.cooldown_ms[ent.id] = max(40, int(self.config.ai_model_action_interval_ms))
            self._try_load_shared_model()

    def _build_arena_config(self, config: GameConfig) -> GameConfig:
        from dataclasses import replace
        if self.mode.key == "TRADITIONAL":
            cols, rows = 25, 25
            score_on_lock = False
            score_mapping = {1: 1, 2: 2, 3: 3, 4: 4}
            duration_ms = 180000 * 999  # 相当于无限制时间，直到溢出
        else:
            cols, rows = 33, 25
            score_on_lock = True
            score_mapping = {1: 0, 2: 0, 3: 0, 4: 0} # 只有落子得分
            duration_ms = 180000

        return replace(
            config,
            grid_cols=cols,
            grid_rows=rows,
            shared_arena_score_on_lock=score_on_lock,
            score_mapping=score_mapping,
            shared_arena_duration_ms=duration_ms,
        )

    def handle_keydown(self, key: int) -> str | None:
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
        elif key == pygame.K_s:
            self._player_down_held = True
        elif key == pygame.K_l:
            p_ent = self._get_entity("player")
            if p_ent and p_ent.piece:
                self.core.rotate_piece(p_ent.piece)
        elif key == pygame.K_w:
            p_ent = self._get_entity("player")
            if p_ent and p_ent.piece:
                self.core.hard_drop_piece(p_ent)
                self.move_timers["player"] = self.cooldown_ms["player"] // 2
        return None

    def handle_keyup(self, key: int):
        if key == pygame.K_a:
            self._player_left_held = False
        elif key == pygame.K_d:
            self._player_right_held = False
        elif key == pygame.K_s:
            self._player_down_held = False
        
    def _get_entity(self, eid: str) -> ArenaEntity | None:
        for e in self.entities:
            if e.id == eid:
                return e
        return None

    def _soft_drop_or_lock(self, ent: ArenaEntity) -> bool:
        """Try one soft drop step; lock immediately if the piece is grounded."""
        moved = self.core.soft_drop_piece(ent)
        if moved:
            return True

        piece = ent.piece
        if piece is None:
            return False
        if not self.core.is_valid_position(piece):
            return False

        self.core.lock_piece(ent)
        return False

    def _try_load_shared_model(self):
        if torch is None:
            self._model_error = "torch 不可用，自动回退启发式策略"
            self.model_enabled = False
            return

        self._model, self._model_error = load_inference_model(
            self._model_path,
            map_location=self.config.ai_model_device,
            expected_input_dim=23,
            expected_action_dim=1,
        )
        if self._model is None:
            self.model_enabled = False
            if not self._model_error:
                self._model_error = "模型加载失败，自动回退启发式策略"
            return

        try:
            self._model_input_dim = int(self._model.net[0].in_features)
        except Exception:
            self._model_error = "模型结构不支持，自动回退启发式策略"
            self._model = None
            self.model_enabled = False

    def _build_planned_command(self, ent: ArenaEntity) -> dict[str, int | bool]:
        piece = ent.piece
        state = self.ai_states[ent.id]
        cmd = {"dx": 0, "rotate": False, "soft_drop": False, "hard_drop": False}

        if piece is None:
            return cmd

        if state.rotation_steps_remaining > 0:
            cmd["rotate"] = True
            return cmd

        if piece.x < state.target_x:
            cmd["dx"] = 1
        elif piece.x > state.target_x:
            cmd["dx"] = -1
        else:
            cmd["hard_drop"] = True

        return cmd

    def _capture_grid_signature(self):
        return tuple(tuple(row) for row in self.core.grid)

    def _capture_piece_signature(self, piece) -> tuple | None:
        if piece is None:
            return None
        return (
            piece.shape_name,
            int(piece.x),
            int(piece.y),
            tuple(tuple(row) for row in piece.matrix),
        )

    def _capture_plan_signature(self, ent: ArenaEntity):
        other_piece_signatures = tuple(
            (other.id, self._capture_piece_signature(other.piece))
            for other in self.entities
            if other.id != ent.id
        )
        return (
            self._capture_grid_signature(),
            self._capture_piece_signature(ent.piece),
            other_piece_signatures,
            tuple(sorted(self.push_rights.items())),
        )

    def _refresh_ai_plan_signature(self, ent: ArenaEntity):
        if ent.is_player:
            return
        self.ai_states[ent.id].plan_signature = self._capture_plan_signature(ent)

    def _invalidate_ai_plan(self, ent: ArenaEntity):
        if ent.is_player:
            return
        self.ai_states[ent.id].clear_plan()

    def _should_replan_ai(self, ent: ArenaEntity) -> bool:
        piece = ent.piece
        if piece is None:
            return False

        state = self.ai_states[ent.id]
        if state.planned_piece_id != id(piece):
            return True

        if self.mode.key != "TRADITIONAL":
            return False

        return state.plan_signature != self._capture_plan_signature(ent)

    def _model_board_cols(self) -> int:
        return max(1, int(self._model_input_dim) - NEXT_STATE_AUX_FEATURE_COUNT)

    def _crop_candidate_grid(self, grid, center_col: int, target_cols: int):
        full_cols = len(grid[0]) if grid else 0
        if full_cols <= 0:
            return [], 0

        if full_cols <= target_cols:
            left_pad = max(0, (target_cols - full_cols) // 2)
            right_pad = max(0, target_cols - full_cols - left_pad)
            padded = []
            for row in grid:
                padded.append(([0] * left_pad) + row[:] + ([0] * right_pad))
            return padded, -left_pad

        start = max(0, min(full_cols - target_cols, int(center_col) - target_cols // 2))
        cropped = [row[start : start + target_cols] for row in grid]
        return cropped, start

    def _piece_coords_in_window(self, piece, window_start: int, window_cols: int) -> list[float]:
        if piece is None:
            return [0.0, 0.0]

        local_x = piece.x - window_start
        local_x = max(0, min(window_cols - 1, local_x))
        return [float(local_x), float(piece.y)]

    def _other_piece_for(self, ent: ArenaEntity):
        others = [other for other in self.entities if other.id != ent.id and other.piece is not None]
        if not others:
            return None
        if ent.piece is None:
            return others[0].piece

        others.sort(key=lambda other: abs(other.piece.x - ent.piece.x))
        return others[0].piece

    def _build_model_features(self, ent: ArenaEntity, candidate) -> list[float]:
        target_cols = self._model_board_cols()
        piece = ent.piece
        piece_width = len(piece.matrix[0]) if piece is not None and piece.matrix else 1
        center_col = int(candidate["target_x"]) + max(0, piece_width // 2)
        cropped_grid, window_start = self._crop_candidate_grid(candidate["result_grid"], center_col, target_cols)
        return extract_next_state_features(
            cropped_grid,
            player_pos=self._piece_coords_in_window(piece, window_start, target_cols),
            ai_pos=self._piece_coords_in_window(self._other_piece_for(ent), window_start, target_cols),
            can_trap=False,
            lines_cleared=float(candidate.get("lines_cleared", 0)),
        )

    def _score_candidate_with_model(self, ent: ArenaEntity, candidate) -> float | None:
        if self._model is None or torch is None:
            return None

        features = self._build_model_features(ent, candidate)
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        try:
            model_device = next(self._model.parameters()).device
            tensor = tensor.to(model_device)
        except Exception:
            pass

        try:
            with torch.no_grad():
                output = self._model(tensor)
        except Exception as exc:
            self._model_error = f"模型评分失败: {exc}"
            self._model = None
            self.model_enabled = False
            return None

        if hasattr(output, "detach"):
            values = output.detach().flatten().tolist()
        elif isinstance(output, (list, tuple)):
            values = list(output)
        else:
            self._model_error = "模型输出格式不支持"
            self._model = None
            self.model_enabled = False
            return None

        if not values:
            self._model_error = "模型输出为空"
            self._model = None
            self.model_enabled = False
            return None

        return float(values[0])

    def update(self, dt: int, _pressed_keys):
        if self.finished or self.paused:
            return

        self.remaining_ms = max(0, self.remaining_ms - dt)
        
        # Update cooldown timers
        for eid in self.move_timers:
            self.move_timers[eid] = max(0, self.move_timers[eid] - dt)
            
        # Update gravity
        for ent in self.entities:
            eff_dt = dt
            if ent.is_player and getattr(self, "_player_down_held", False):
                eff_dt *= 10  # Soft drop is much faster
                
            self.gravity_accumulators[ent.id] += eff_dt
            if self.gravity_accumulators[ent.id] >= self.gravity_interval_ms:
                self.gravity_accumulators[ent.id] -= self.gravity_interval_ms
                self._soft_drop_or_lock(ent)
                
        if self.core.check_game_over() or self.remaining_ms <= 0:
            self.finished = True
            self._settle_result()
            return
            
        self._run_cycle()

    def _run_cycle(self):
        cmds = {}
        for ent in self.entities:
            cmds[ent.id] = {"dx": 0, "rotate": False, "soft_drop": False, "hard_drop": False}
            
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
                        before_signature = self._capture_piece_signature(ent.piece)
                        self.core.rotate_piece(ent.piece)
                        after_signature = self._capture_piece_signature(ent.piece)
                        if before_signature != after_signature:
                            state = self.ai_states[ent.id]
                            state.rotation_steps_remaining = max(0, state.rotation_steps_remaining - 1)
                            self._refresh_ai_plan_signature(ent)
                        else:
                            self._invalidate_ai_plan(ent)
                        self.move_timers[ent.id] = self.cooldown_ms[ent.id] // 2
                    elif ai_cmd["soft_drop"]:
                        moved = self._soft_drop_or_lock(ent)
                        if moved:
                            self._refresh_ai_plan_signature(ent)
                        else:
                            self._invalidate_ai_plan(ent)
                        self.move_timers[ent.id] = self.cooldown_ms[ent.id]
                    elif ai_cmd["hard_drop"]:
                        self.core.hard_drop_piece(ent)
                        self._invalidate_ai_plan(ent)
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
                # Check push rights for TRADITIONAL mode
                if self.mode.key == "TRADITIONAL" and len(chain) > 1:
                    # The initiator must have push rights
                    if not self.push_rights[ent.id]:
                        if not ent.is_player:
                            self._invalidate_ai_plan(ent)
                        continue  # Cannot push!
                    
                    # If valid, initiator loses right, those pushed gain right
                    self.push_rights[ent.id] = False
                    for pushed_ent in chain:
                        if pushed_ent.id != ent.id:
                            self.push_rights[pushed_ent.id] = True

                # Chain is valid, move everyone in the chain
                for pushed_ent in chain:
                    pushed_ent.piece.x += dx
                for pushed_ent in chain:
                    if pushed_ent.is_player:
                        continue
                    if pushed_ent.id == ent.id:
                        self._refresh_ai_plan_signature(pushed_ent)
                    else:
                        self._invalidate_ai_plan(pushed_ent)
            elif not ent.is_player:
                self._invalidate_ai_plan(ent)

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
        
        if piece is None or self.core.state != "RUNNING":
            return {"dx": 0, "rotate": False, "soft_drop": False, "hard_drop": False}

        if self._should_replan_ai(ent):
            self._plan_ai_piece(ent)
        return self._build_planned_command(ent)

    def _plan_ai_piece(self, ent: ArenaEntity):
        piece = ent.piece
        if piece is None:
            return
        
        state = self.ai_states[ent.id]
        candidates = generate_candidates(self.core.grid, piece, self.arena_config)
        if not candidates:
            state.clear_plan()
            state.target_x = piece.x
            state.planned_piece_id = id(piece)
            self._refresh_ai_plan_signature(ent)
            return

        for candidate in candidates:
            candidate["total_score"] = float(candidate["surface_score"])

        use_model_scores = self.model_enabled and self._model is not None
        if use_model_scores:
            candidates.sort(key=lambda item: item["surface_score"], reverse=True)
            top_k = max(1, min(len(candidates), int(self._model_rerank_topk)))
            shortlist = candidates[:top_k]

            scored_pairs = []
            for candidate in shortlist:
                model_score = self._score_candidate_with_model(ent, candidate)
                if model_score is None:
                    use_model_scores = False
                    break
                scored_pairs.append((candidate, float(model_score)))

            if use_model_scores and scored_pairs:
                values = [item[1] for item in scored_pairs]
                mean_value = sum(values) / max(1, len(values))
                var_value = sum((value - mean_value) ** 2 for value in values) / max(1, len(values))
                std_value = (var_value + 1e-6) ** 0.5

                for candidate, model_score in scored_pairs:
                    normalized_model_score = (model_score - mean_value) / std_value
                    candidate["total_score"] = float(candidate["surface_score"]) + normalized_model_score * float(self._model_rerank_scale)

        candidates.sort(key=lambda item: item["total_score"], reverse=True)
        selected = candidates[0]

        if (not use_model_scores) and len(candidates) > 1 and random.random() < self.mode.ai_mistake_chance:
            selected = random.choice(candidates[: min(3, len(candidates))])

        state.rotation_steps_remaining = selected["rotation_steps"]
        state.target_x = selected["target_x"]
        state.planned_piece_id = id(piece)
        self._refresh_ai_plan_signature(ent)

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
        strategy = "模型落点评分" if self.model_enabled and self._model is not None else "启发式回退"
        strategy_surf = self.small_font.render(f"AI策略: {strategy}", True, (220, 220, 220))
        self.screen.blit(strategy_surf, (panel_rect.x + 20, panel_rect.y + y_offset))
        y_offset += 34

        model_name = self._model_path.name
        path_surf = self.small_font.render(f"模型: {model_name}", True, (210, 210, 210))
        self.screen.blit(path_surf, (panel_rect.x + 20, panel_rect.y + y_offset))
        y_offset += 34

        if self._model_error:
            clipped = self._model_error[:24]
            error_surf = self.small_font.render(f"状态: {clipped}", True, (255, 180, 160))
            self.screen.blit(error_surf, (panel_rect.x + 20, panel_rect.y + y_offset))
            y_offset += 34

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
