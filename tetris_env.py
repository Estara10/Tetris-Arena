from __future__ import annotations

import random
from dataclasses import dataclass

from next_state_features import analyze_board, extract_next_state_features
from ai_controller import AIController, board_profile
from game_core import GameCore
from piece_sequence import SharedShapeSequence
from settings import CONFIG, GameConfig


ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_ROTATE = 3
ACTION_SOFT_DROP = 4
ACTION_HARD_DROP = 5
ACTION_TRAP = 6


@dataclass
class StepOutcome:
    state: list[float]
    reward: float
    done: bool
    info: dict[str, float | int | bool | str]


class TetrisEnv:
    """
    无渲染的训练环境封装。

    设计目标:
    1) 复用现有双盘对战规则（攻击、抵消、陷阱、垃圾结算）。
    2) 提供 reset()/step() 接口，便于后续接入 DQN/PPO 训练器。
    3) 状态使用数值向量，避免图像像素输入带来的训练负担。
    """

    action_meaning = {
        ACTION_NOOP: "noop",
        ACTION_LEFT: "left",
        ACTION_RIGHT: "right",
        ACTION_ROTATE: "rotate",
        ACTION_SOFT_DROP: "soft_drop",
        ACTION_HARD_DROP: "hard_drop",
        ACTION_TRAP: "trap",
    }

    piece_order = ["I", "J", "L", "O", "S", "T", "Z"]

    def __init__(
        self,
        config: GameConfig | None = None,
        seed: int | None = None,
        step_dt_ms: int = 90,
        max_episode_steps: int = 6000,
        opponent_mode: str = "heuristic",
        opponent_action_interval_ms: int | None = None,
        opponent_mistake_chance: float | None = None,
    ):
        self.config = config if config is not None else CONFIG
        self.seed = seed
        self.rng = random.Random(seed)

        self.step_dt_ms = max(20, int(step_dt_ms))
        self.max_episode_steps = max(300, int(max_episode_steps))

        self.opponent_mode = opponent_mode
        self.opponent_action_interval_ms = opponent_action_interval_ms
        self.opponent_mistake_chance = opponent_mistake_chance
        self.shared_sequence = None
        self.player_core = None
        self.ai_core = None
        self.ai_controller = None

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

        self._processed_player_locks = 0
        self._processed_ai_locks = 0
        self._recent_events: list[dict[str, int | bool | str]] = []

        self.episode_step = 0
        self._prev_player_lines = 0
        self._prev_ai_lines = 0
        self._prev_player_score = 0
        self._prev_ai_score = 0
        self._prev_player_garbage = 0
        self._prev_ai_garbage = 0
        self._prev_player_holes = 0
        self._prev_ai_holes = 0
        self._prev_player_max_height = 0
        self._prev_ai_max_height = 0
        self._prev_player_aggregate_height = 0
        self._prev_ai_aggregate_height = 0
        self._prev_player_covered_holes = 0
        self._prev_ai_covered_holes = 0
        self._prev_player_transitions = 0
        self._prev_ai_transitions = 0
        self._prev_player_side_imbalance = 0
        self._prev_ai_side_imbalance = 0

        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)

        bag_seed = self.seed if self.seed is not None else self.rng.randint(0, 2**31 - 1)
        self.shared_sequence = SharedShapeSequence(config=self.config, seed=bag_seed)

        self.player_core = GameCore(
            config=self.config,
            piece_factory=self.shared_sequence.make_piece_factory("player"),
        )
        self.ai_core = GameCore(
            config=self.config,
            piece_factory=self.shared_sequence.make_piece_factory("ai"),
        )

        self.ai_controller = AIController(
            config=self.config,
            action_interval_ms=(
                self.opponent_action_interval_ms
                if self.opponent_action_interval_ms is not None
                else self.config.ai_model_action_interval_ms
            ),
            mistake_chance=(
                self.opponent_mistake_chance
                if self.opponent_mistake_chance is not None
                else 0.15
            ),
            mode=self.opponent_mode,
            model_path=self.config.ai_model_path,
        )

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
        self._recent_events = []

        self.episode_step = 0
        self._sync_prev_metrics()
        self._prev_player_locks = self.player_core.lock_count

        return self._get_state_vector()

    def step(self, action: int) -> StepOutcome:
        if self._is_done():
            return StepOutcome(
                state=self._get_state_vector(),
                reward=0.0,
                done=True,
                info={"reason": "episode_already_done"},
            )

        self.episode_step += 1
        self._recent_events = []

        self._tick_battle_states(self.step_dt_ms)
        self._apply_player_action(action)

        self.ai_controller.update(self.ai_core, self.step_dt_ms)
        self.player_core.update(self.step_dt_ms)
        self.ai_core.update(self.step_dt_ms)

        self._process_lock_events()
        self._try_ai_activate_trap()

        reward = self._compute_reward()
        done = self._is_done()
        info = self._build_info(action=action, reward=reward, done=done)

        self._sync_prev_metrics()
        return StepOutcome(
            state=self._get_state_vector(),
            reward=reward,
            done=done,
            info=info,
        )

    def sample_action(self) -> int:
        return self.rng.randint(ACTION_NOOP, ACTION_TRAP)

    def set_opponent_profile(self, action_interval_ms: int, mistake_chance: float):
        self.opponent_action_interval_ms = int(action_interval_ms)
        self.opponent_mistake_chance = float(mistake_chance)
        if self.ai_controller is not None:
            self.ai_controller.set_profile(
                action_interval_ms=self.opponent_action_interval_ms,
                mistake_chance=self.opponent_mistake_chance,
            )

    def state_size(self) -> int:
        return len(self._get_state_vector())

    def action_size(self) -> int:
        return len(self.action_meaning)

    def _apply_player_action(self, action: int):
        if self.player_core.state != "RUNNING" or self.player_core.current_piece is None:
            return

        piece = self.player_core.current_piece
        if action == ACTION_LEFT:
            if not piece.check_collision(dx=-1, dy=0, grid=self.player_core.grid):
                piece.x -= 1
            return

        if action == ACTION_RIGHT:
            if not piece.check_collision(dx=1, dy=0, grid=self.player_core.grid):
                piece.x += 1
            return

        if action == ACTION_ROTATE:
            piece.rotate(self.player_core.grid)
            return

        if action == ACTION_SOFT_DROP:
            if not piece.check_collision(dx=0, dy=1, grid=self.player_core.grid):
                piece.y += 1
            else:
                self.player_core.lock_shape()
            return

        if action == ACTION_HARD_DROP:
            piece.y = piece.get_drop_position(self.player_core.grid)
            self.player_core.lock_shape()
            return

        if action == ACTION_TRAP:
            self._try_activate_trap(is_player=True)

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

        if lines_cleared > 0:
            self._update_combo_b2b(is_player, lines_cleared)
            self._gain_trap_energy(is_player, lines_cleared)
        else:
            if is_player:
                self.player_combo = -1
                self.player_b2b = False
            else:
                self.ai_combo = -1
                self.ai_b2b = False

        attack_lines = self._compute_attack(is_player, lines_cleared)
        canceled = 0
        sent = 0
        settled = 0

        if attack_lines > 0:
            canceled = core.cancel_incoming_garbage(attack_lines)
            remaining = attack_lines - canceled
            if canceled > 0:
                self._set_warning(is_player, f"抵消来袭 {canceled} 行")
            if remaining > 0:
                self._send_attack(is_player, remaining)
                sent = remaining

        if lines_cleared == 0 and self.config.versus_garbage_apply_on_nonclear:
            settled = core.apply_incoming_garbage(self.config.versus_garbage_cap_per_lock)
            if settled > 0:
                self._set_warning(is_player, f"承受垃圾 {settled} 行")

        self._recent_events.append(
            {
                "side": "player" if is_player else "ai",
                "lines_cleared": lines_cleared,
                "attack_sent": sent,
                "attack_canceled": canceled,
                "garbage_settled": settled,
            }
        )

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
                holes.append(self.rng.randint(0, cols - 1))
        return holes

    def _forced_hole(self, pattern: str, cols: int) -> int:
        one_third = max(1, cols // 3)
        if pattern == "left":
            return self.rng.randint(0, one_third - 1)
        if pattern == "right":
            return self.rng.randint(cols - one_third, cols - 1)

        left = max(0, one_third)
        right = min(cols - 1, cols - one_third - 1)
        if left > right:
            return cols // 2
        return self.rng.randint(left, right)

    def _gain_trap_energy(self, is_player: bool, lines_cleared: int):
        gain = lines_cleared * self.config.versus_trap_energy_gain_per_line
        if is_player:
            self.player_trap_energy += gain
        else:
            self.ai_trap_energy += gain

    def _try_activate_trap(self, is_player: bool) -> bool:
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
            self._set_warning(False, "玩家触发陷阱")
        else:
            self.ai_trap_state = state
            self.ai_trap_energy -= self.config.versus_trap_energy_cost
            self.ai_trap_cooldown_ms = self.config.versus_trap_cooldown_ms
            self._set_warning(True, "AI 触发陷阱")

        self._recent_events.append(
            {
                "side": "player" if is_player else "ai",
                "lines_cleared": 0,
                "attack_sent": 0,
                "attack_canceled": 0,
                "garbage_settled": 0,
                "trap_activated": True,
            }
        )
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
        if self.rng.random() <= self.config.versus_trap_ai_use_chance:
            self._try_activate_trap(is_player=False)

    def _set_warning(self, for_player: bool, text: str):
        if for_player:
            self.player_warning_text = text
            self.player_warning_ttl = self.config.versus_warning_duration_ms
        else:
            self.ai_warning_text = text
            self.ai_warning_ttl = self.config.versus_warning_duration_ms

    def _compute_reward(self) -> float:
        player_lines_gain = self.player_core.lines_cleared_total - getattr(self, '_prev_player_lines', 0)
        player_stats = analyze_board(self.player_core.grid)
        rows = max(1.0, float(player_stats["rows"]))
        cols = max(1.0, float(player_stats["cols"]))
        player_holes = float(player_stats["holes"])
        player_bumpiness = float(player_stats["bumpiness"])
        player_aggregate_height = float(player_stats["aggregate_height"])
        player_max_height = float(player_stats["max_height"])
        player_covered_holes = float(player_stats["covered_holes"])
        player_transitions = float(player_stats["row_transitions"]) + float(player_stats["col_transitions"])
        player_danger_cells = float(player_stats["danger_cells"])
        # 新增：深层空洞统计
        player_deep_holes = float(player_stats.get("deep_holes", 0.0))
        player_deep_hole_depth = float(player_stats.get("deep_hole_depth_sum", 0.0))
        heights = player_stats["heights"]
        split = max(1, len(heights) // 2)
        player_side_imbalance = float(abs(sum(heights[:split]) - sum(heights[split:])))
        near_top_excess = sum(max(0.0, float(height) - (rows - 6.0)) for height in heights)
        high_columns = float(sum(1 for height in heights if float(height) >= rows - 4.0))
        avg_height = player_aggregate_height / cols
        danger_density = player_danger_cells / max(1.0, float(player_stats["danger_rows"]) * cols)

        reward = 0.25

        line_rewards = {1: 30.0, 2: 130.0, 3: 420.0, 4: 1300.0}
        reward += line_rewards.get(player_lines_gain, 0.0)
        reward -= avg_height * 0.22
        reward -= max(0.0, player_max_height - rows * 0.55) * 1.1
        reward -= player_danger_cells * 3.5
        reward -= near_top_excess * 1.8
        reward -= high_columns * 2.5
        reward -= danger_density * 6.0
        reward -= player_side_imbalance * 0.08
        # 新增：深层空洞惩罚（底部空洞更严重，因为难以消除）
        reward -= player_deep_holes * 8.0
        reward -= player_deep_hole_depth * 12.0

        if player_max_height <= rows * 0.42 and player_danger_cells <= 0.0:
            reward += 2.0
        
        # 新增：底部干净奖励（没有深层空洞时给予额外奖励）
        if player_deep_holes == 0.0 and player_holes <= 2.0:
            reward += 3.0

        if player_lines_gain == 1 and (player_danger_cells > 0.0 or player_max_height >= rows - 5.0):
            reward -= 18.0 + player_danger_cells * 1.5 + near_top_excess * 1.2

        current_locks = getattr(self.player_core, 'lock_count', 0)
        prev_locks = getattr(self, '_prev_player_locks', 0)
        is_locked = current_locks > prev_locks

        if is_locked:
            delta_holes = player_holes - getattr(self, '_prev_player_holes', 0)
            delta_bumpiness = player_bumpiness - getattr(self, '_prev_player_bumpiness', 0)
            delta_aggregate_height = player_aggregate_height - getattr(self, '_prev_player_aggregate_height', 0)
            delta_max_height = player_max_height - getattr(self, '_prev_player_max_height', 0)
            delta_covered_holes = player_covered_holes - getattr(self, '_prev_player_covered_holes', 0)
            delta_transitions = player_transitions - getattr(self, '_prev_player_transitions', 0)
            delta_side_imbalance = player_side_imbalance - getattr(self, '_prev_player_side_imbalance', 0)
            delta_danger_cells = player_danger_cells - getattr(self, '_prev_player_danger_cells', 0.0)
            delta_high_columns = high_columns - getattr(self, '_prev_player_high_columns', 0.0)
            # 新增：深层空洞变化
            delta_deep_holes = player_deep_holes - getattr(self, '_prev_player_deep_holes', 0.0)
            delta_deep_hole_depth = player_deep_hole_depth - getattr(self, '_prev_player_deep_hole_depth', 0.0)

            reward -= max(0.0, delta_holes) * 5.0
            reward -= max(0.0, delta_covered_holes) * 0.30
            reward -= max(0.0, delta_bumpiness) * 0.40
            reward -= max(0.0, delta_aggregate_height) * 0.30
            reward -= max(0.0, delta_max_height) * 1.5
            reward -= max(0.0, delta_transitions) * 0.12
            reward -= max(0.0, delta_side_imbalance) * 0.30
            reward -= max(0.0, delta_danger_cells) * 3.5
            reward -= max(0.0, delta_high_columns) * 3.0
            # 新增：深层空洞增量惩罚（新增深层空洞惩罚更重）
            reward -= max(0.0, delta_deep_holes) * 10.0
            reward -= max(0.0, delta_deep_hole_depth) * 15.0

            reward += max(0.0, -delta_aggregate_height) * 0.55
            reward += max(0.0, -delta_bumpiness) * 0.45
            reward += max(0.0, -delta_holes) * 2.0
            reward += max(0.0, -delta_side_imbalance) * 0.42
            reward += max(0.0, -delta_danger_cells) * 1.8
            reward += max(0.0, -delta_high_columns) * 2.0
            # 新增：消除深层空洞奖励
            reward += max(0.0, -delta_deep_holes) * 5.0
            reward += max(0.0, -delta_deep_hole_depth) * 8.0

            piece_y = getattr(self, '_recent_player_piece_y', 0)
            reward += (piece_y / max(1.0, float(self.config.grid_rows - 1))) * 1.0

        for event in self._recent_events:
            if event.get("side") != "player":
                continue
            reward += float(event.get("attack_sent", 0)) * 12.0
            reward += float(event.get("attack_canceled", 0)) * 8.0
            reward -= float(event.get("garbage_settled", 0)) * 10.0
            if event.get("trap_activated"):
                reward += 2.0

        if self.player_core.state == "GAME_OVER":
            reward = min(reward - 500.0, -500.0)
        elif self.ai_core.state == "GAME_OVER":
            reward += 350.0

        return reward

    def _build_info(self, action: int, reward: float, done: bool) -> dict[str, float | int | bool | str]:
        return {
            "action": int(action),
            "action_name": self.action_meaning.get(int(action), "unknown"),
            "reward": float(reward),
            "done": bool(done),
            "step": int(self.episode_step),
            "player_score": int(self.player_core.score),
            "ai_score": int(self.ai_core.score),
            "player_lines": int(self.player_core.lines_cleared_total),
            "ai_lines": int(self.ai_core.lines_cleared_total),
            "player_incoming": int(len(self.player_core.incoming_garbage)),
            "ai_incoming": int(len(self.ai_core.incoming_garbage)),
            "player_state": self.player_core.state,
            "ai_state": self.ai_core.state,
        }

    def get_next_states(self) -> dict:
        """
        获取当前方块穷举的所有合法落座状态。
        返回: dict{(rotation_steps, target_x): state_vector}
        """
        states = {}
        for cand in self.get_next_state_candidates():
            states[cand["key"]] = cand["state"]
        return states

    def get_next_state_candidates(self) -> list[dict]:
        from ai_controller import generate_candidates

        piece = self.player_core.current_piece
        if not piece:
            return []

        candidates = generate_candidates(self.player_core.grid, piece, self.config)
        items = []
        for cand in candidates:
            rot = int(cand["rotation_steps"])
            tx = int(cand["target_x"])
            sim_grid = cand["result_grid"]
            lines_cleared = int(cand.get("lines_cleared", 0))
            items.append(
                {
                    "key": (rot, tx),
                    "state": self._extract_board_features(sim_grid, lines_cleared),
                    "surface_score": float(cand.get("surface_score", 0.0)),
                    "lines_cleared": lines_cleared,
                    "result_grid": sim_grid,
                }
            )
        return items

    def _extract_board_features(self, grid, lines_cleared=0) -> list[float]:
        can_trap = (
            self.player_trap_energy >= self.config.versus_trap_energy_cost
            and self.player_trap_cooldown_ms <= 0
        )
        return extract_next_state_features(
            grid,
            player_piece=self.player_core.current_piece,
            ai_piece=self.ai_core.current_piece,
            can_trap=can_trap,
            lines_cleared=float(lines_cleared),
        )

    def step_next_state(self, rotation_steps: int, target_x: int):
        """一键抵达目标状态的跃迁动作"""
        if self._is_done():
            return self.step(ACTION_NOOP)
            
        self.episode_step += 1
        self._recent_events = []
        
        self._tick_battle_states(self.step_dt_ms)
        
        piece = self.player_core.current_piece
        if piece:
            # 必须绕过碰撞检测强制旋转矩阵，否则如果在出生点旋转由于空间不足会被拒绝旋转
            for _ in range(rotation_steps):
                piece.matrix = piece.get_rotated_matrix()
            piece.x = target_x
            piece.y = piece.get_drop_position(self.player_core.grid)
            self._recent_player_piece_y = piece.y
            self.player_core.lock_shape()
            
        self.ai_controller.update(self.ai_core, self.step_dt_ms)
        self.player_core.update(self.step_dt_ms)
        self.ai_core.update(self.step_dt_ms)
        
        self._process_lock_events()
        self._try_ai_activate_trap()
        
        reward = self._compute_reward()
        done = self._is_done()
        info = self._build_info(action=ACTION_HARD_DROP, reward=reward, done=done)
        
        self._sync_prev_metrics()
        
        # state vector 其实应该传真实的当前盘面，我们可以复用刚才的 _extract
        return StepOutcome(
            state=self._get_state_vector(),
            reward=reward,
            done=done,
            info=info
        )

    def _is_done(self) -> bool:
        if self.player_core.state == "GAME_OVER":
            return True
        if self.ai_core.state == "GAME_OVER":
            return True
        if self.episode_step >= self.max_episode_steps:
            return True
        return False

    def _get_state_vector(self) -> list[float]:
        state: list[float] = []
        grid = self.player_core.grid
        cols = self.config.grid_cols
        rows = self.config.grid_rows

        heights = []
        holes = 0
        for x in range(cols):
            col_height = 0
            col_holes = 0
            found_block = False
            for y in range(rows):
                if grid[y][x]:
                    if not found_block:
                        col_height = rows - y
                        found_block = True
                else:
                    if found_block:
                        col_holes += 1
            heights.append(col_height)
            holes += col_holes
        
        # 每一列的当前最高高度
        state.extend([h for h in heights])
        
        # 全盘的空洞总数
        state.append(holes)

        # 相邻列的高度差绝对值之和（平整度）
        bumpiness = sum(abs(heights[i] - heights[i-1]) for i in range(1, cols))
        state.append(bumpiness)

        # AI 当前方块的坐标 (X, Y) (指 player_core)
        p_piece = self.player_core.current_piece
        if p_piece:
            state.append(p_piece.x)
            state.append(p_piece.y)
        else:
            state.extend([0, 0])

        # 对手当前方块的坐标 (X, Y)
        ai_piece = self.ai_core.current_piece
        if ai_piece:
            state.append(ai_piece.x)
            state.append(ai_piece.y)
        else:
            state.extend([0, 0])

        # 当前是否拥有“撞人权”（1有，0无）
        can_trap = 1.0 if (self.player_trap_energy >= self.config.versus_trap_energy_cost and self.player_trap_cooldown_ms <= 0) else 0.0
        state.append(can_trap)
        state.append(0.0) # padding lines_cleared for regular state queries

        return state
    def _board_binary(self, grid: list[list[int | str]]) -> list[float]:
        values: list[float] = []
        for row in grid:
            for cell in row:
                values.append(0.0 if cell == 0 else 1.0)
        return values

    def _piece_onehot(self, piece) -> list[float]:
        vec = [0.0] * len(self.piece_order)
        if piece is None:
            return vec
        if piece.shape_name in self.piece_order:
            vec[self.piece_order.index(piece.shape_name)] = 1.0
        return vec

    def _sync_prev_metrics(self):
        self._prev_player_lines = self.player_core.lines_cleared_total
        self._prev_ai_lines = self.ai_core.lines_cleared_total
        self._prev_player_score = self.player_core.score
        self._prev_ai_score = self.ai_core.score
        self._prev_player_garbage = self.player_core.garbage_received_total
        self._prev_ai_garbage = self.ai_core.garbage_received_total

        player_stats = analyze_board(self.player_core.grid)
        ai_stats = analyze_board(self.ai_core.grid)
        self._prev_player_holes = float(player_stats["holes"])
        self._prev_ai_holes = float(ai_stats["holes"])
        self._prev_player_max_height = float(player_stats["max_height"])
        self._prev_ai_max_height = float(ai_stats["max_height"])
        self._prev_player_bumpiness = float(player_stats["bumpiness"])
        self._prev_ai_bumpiness = float(ai_stats["bumpiness"])
        self._prev_player_aggregate_height = float(player_stats["aggregate_height"])
        self._prev_ai_aggregate_height = float(ai_stats["aggregate_height"])
        self._prev_player_covered_holes = float(player_stats["covered_holes"])
        self._prev_ai_covered_holes = float(ai_stats["covered_holes"])
        self._prev_player_transitions = float(player_stats["row_transitions"]) + float(player_stats["col_transitions"])
        self._prev_ai_transitions = float(ai_stats["row_transitions"]) + float(ai_stats["col_transitions"])
        self._prev_player_danger_cells = float(player_stats["danger_cells"])
        self._prev_ai_danger_cells = float(ai_stats["danger_cells"])
        # 新增：深层空洞追踪
        self._prev_player_deep_holes = float(player_stats.get("deep_holes", 0.0))
        self._prev_ai_deep_holes = float(ai_stats.get("deep_holes", 0.0))
        self._prev_player_deep_hole_depth = float(player_stats.get("deep_hole_depth_sum", 0.0))
        self._prev_ai_deep_hole_depth = float(ai_stats.get("deep_hole_depth_sum", 0.0))
        player_heights = player_stats["heights"]
        ai_heights = ai_stats["heights"]
        player_split = max(1, len(player_heights) // 2)
        ai_split = max(1, len(ai_heights) // 2)
        self._prev_player_side_imbalance = float(abs(sum(player_heights[:player_split]) - sum(player_heights[player_split:])))
        self._prev_ai_side_imbalance = float(abs(sum(ai_heights[:ai_split]) - sum(ai_heights[ai_split:])))
        self._prev_player_high_columns = float(sum(1 for height in player_heights if float(height) >= float(player_stats["rows"]) - 4.0))
        self._prev_ai_high_columns = float(sum(1 for height in ai_heights if float(height) >= float(ai_stats["rows"]) - 4.0))
        self._prev_player_locks = getattr(self.player_core, 'lock_count', self.player_core.lines_cleared_total * 0)  # fallback if not using lock_count
        if hasattr(self, '_recent_player_piece_y'):
            self._prev_player_piece_y = self._recent_player_piece_y
