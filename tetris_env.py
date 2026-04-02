from __future__ import annotations

import random
from dataclasses import dataclass

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
        player_lines_gain = self.player_core.lines_cleared_total - self._prev_player_lines
        ai_lines_gain = self.ai_core.lines_cleared_total - self._prev_ai_lines

        player_score_gain = self.player_core.score - self._prev_player_score
        ai_score_gain = self.ai_core.score - self._prev_ai_score

        player_garbage_gain = self.player_core.garbage_received_total - self._prev_player_garbage
        ai_garbage_gain = self.ai_core.garbage_received_total - self._prev_ai_garbage

        player_heights, player_holes = board_profile(self.player_core.grid)
        ai_heights, ai_holes = board_profile(self.ai_core.grid)
        player_max_height = max(player_heights, default=0)
        ai_max_height = max(ai_heights, default=0)

        reward = 0.0

        score_gap_gain = (player_score_gain - ai_score_gain)
        reward += score_gap_gain * 1.8

        reward += player_lines_gain * 1.2
        reward -= ai_lines_gain * 0.9

        reward -= player_garbage_gain * 0.8
        reward += ai_garbage_gain * 0.3

        reward -= max(0, player_holes - self._prev_player_holes) * 0.10
        reward += max(0, self._prev_ai_holes - ai_holes) * 0.04

        reward -= max(0, player_max_height - self._prev_player_max_height) * 0.06
        reward += max(0, self._prev_ai_max_height - ai_max_height) * 0.02

        trap_used = any(event.get("trap_activated") for event in self._recent_events)
        if trap_used:
            reward += 0.06

        reward -= 0.002

        if self.player_core.state == "GAME_OVER":
            reward -= 120.0
        if self.ai_core.state == "GAME_OVER" and self.player_core.state != "GAME_OVER":
            reward += 120.0

        if self.episode_step >= self.max_episode_steps:
            final_gap = self.player_core.score - self.ai_core.score
            if final_gap > 0:
                reward += 30.0
            elif final_gap < 0:
                reward -= 30.0

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

        state.extend(self._board_binary(self.player_core.grid))
        state.extend(self._board_binary(self.ai_core.grid))

        state.extend(self._piece_onehot(self.player_core.current_piece))
        state.extend(self._piece_onehot(self.player_core.next_piece))
        state.extend(self._piece_onehot(self.ai_core.current_piece))
        state.extend(self._piece_onehot(self.ai_core.next_piece))

        player_heights, player_holes = board_profile(self.player_core.grid)
        ai_heights, ai_holes = board_profile(self.ai_core.grid)

        player_aggregate = sum(player_heights)
        ai_aggregate = sum(ai_heights)
        player_max_height = max(player_heights, default=0)
        ai_max_height = max(ai_heights, default=0)

        max_board = max(1, self.config.grid_rows * self.config.grid_cols)
        state.append(player_aggregate / max_board)
        state.append(ai_aggregate / max_board)
        state.append(player_max_height / max(1, self.config.grid_rows))
        state.append(ai_max_height / max(1, self.config.grid_rows))
        state.append(player_holes / max_board)
        state.append(ai_holes / max_board)

        state.append(min(1.0, len(self.player_core.incoming_garbage) / 20.0))
        state.append(min(1.0, len(self.ai_core.incoming_garbage) / 20.0))

        state.append(min(1.0, self.player_trap_energy / 12.0))
        state.append(min(1.0, self.ai_trap_energy / 12.0))
        state.append(min(1.0, self.player_trap_cooldown_ms / max(1, self.config.versus_trap_cooldown_ms)))
        state.append(min(1.0, self.ai_trap_cooldown_ms / max(1, self.config.versus_trap_cooldown_ms)))

        state.append(self.player_combo / 10.0)
        state.append(self.ai_combo / 10.0)
        state.append(1.0 if self.player_b2b else 0.0)
        state.append(1.0 if self.ai_b2b else 0.0)

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

        player_heights, player_holes = board_profile(self.player_core.grid)
        ai_heights, ai_holes = board_profile(self.ai_core.grid)
        self._prev_player_holes = player_holes
        self._prev_ai_holes = ai_holes
        self._prev_player_max_height = max(player_heights, default=0)
        self._prev_ai_max_height = max(ai_heights, default=0)
