from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
import time
from typing import Sequence

try:
    import torch  # type: ignore[import-not-found]
    from torch import nn  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    torch = None
    nn = None

from src.ai.model_paths import mode_model_dir
from src.ai.replay_memory import ReplayMemory
from settings import CONFIG, GameConfig
from src.ai.tetris_env import TetrisEnv


LATEST_CHECKPOINT_NAME = "latest_checkpoint.pth"


def _no_grad_if_available():
    if torch is None:
        def _decorator(func):
            return func

        return _decorator
    return torch.no_grad()


@dataclass
class TrainerConfig:
    episodes: int
    warmup_steps: int
    batch_size: int
    gamma: float
    learning_rate: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    train_frequency: int
    target_sync_interval: int
    replay_capacity: int
    max_episode_steps: int
    eval_interval: int
    eval_episodes: int
    checkpoint_interval: int
    model_dir: str
    curriculum_episodes: int
    opponent_start_interval_ms: int
    opponent_end_interval_ms: int
    opponent_start_mistake: float
    opponent_end_mistake: float
    seed: int | None = None
    opponent_mode: str = "heuristic"
    game_mode: str = "CLASSIC"
    device: str = "cpu"
    checkpoint_name: str = LATEST_CHECKPOINT_NAME


class ActionMapper:
    """将 Agent 的标准动作索引映射到环境动作索引。"""

    canonical_order = (
        "noop",        # 0: 不动
        "left",        # 1: 左移
        "right",       # 2: 右移
        "soft_drop",   # 3: 加速下落
        "hard_drop",   # 4: 一键到底
        "rotate",      # 5: 旋转
    )

    def __init__(self, env):
        self.agent_to_env = self._build_mapping(env)
        self.action_dim = len(self.agent_to_env)

    @staticmethod
    def _normalize_name(name: str) -> str:
        return str(name).strip().lower()

    def _build_mapping(self, env) -> list[int]:
        action_meaning = getattr(env, "action_meaning", None)
        if isinstance(action_meaning, dict) and action_meaning:
            reverse: dict[str, int] = {}
            for idx, name in action_meaning.items():
                reverse[self._normalize_name(name)] = int(idx)

            mapped: list[int] = []
            for canonical in self.canonical_order:
                idx = reverse.get(canonical)
                if idx is None and canonical == "soft_drop":
                    idx = reverse.get("down")
                if idx is None and canonical == "hard_drop":
                    idx = reverse.get("drop")
                if idx is None:
                    break
                mapped.append(int(idx))

            if len(mapped) == len(self.canonical_order):
                return mapped

        # 回退策略：如果环境没有语义动作名，至少保证 0..5 可用。
        env_action_size = int(getattr(env, "action_size")())
        fallback_size = max(1, min(6, env_action_size))
        return list(range(fallback_size))

    def to_env_action(self, agent_action: int) -> int:
        return int(self.agent_to_env[int(agent_action)])


class CNNStateAdapter:
    """
    将环境返回的 state 向量切分为：
    1) 棋盘张量 board: [C, H, W]
    2) 附加特征 aux: [F]

    说明：
    - 兼容现有工程中的不同状态编码格式（向量前缀是棋盘，后缀是附加特征）。
    - 优先尝试双通道棋盘（玩家 + 对手），不满足时回退到单通道。
    """

    def __init__(
        self,
        board_rows: int,
        board_cols: int,
        board_channels: int,
        aux_dim: int,
    ):
        self.board_rows = max(1, int(board_rows))
        self.board_cols = max(1, int(board_cols))
        self.board_channels = max(1, int(board_channels))
        self.aux_dim = max(1, int(aux_dim))
        self.board_dim = self.board_rows * self.board_cols * self.board_channels

    @classmethod
    def from_state(
        cls,
        state: Sequence[float],
        board_rows: int,
        board_cols: int,
        prefer_two_channels: bool = True,
    ) -> "CNNStateAdapter":
        state_len = len(list(state))
        single_board_dim = max(1, int(board_rows) * int(board_cols))

        board_channels = 1
        if prefer_two_channels and state_len >= single_board_dim * 2:
            board_channels = 2

        board_dim = single_board_dim * board_channels
        aux_dim = max(1, state_len - board_dim)
        return cls(
            board_rows=board_rows,
            board_cols=board_cols,
            board_channels=board_channels,
            aux_dim=aux_dim,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "board_rows": int(self.board_rows),
            "board_cols": int(self.board_cols),
            "board_channels": int(self.board_channels),
            "aux_dim": int(self.aux_dim),
        }

    def _coerce_state(self, state: Sequence[float]) -> list[float]:
        return [float(v) for v in state]

    def _split_state(self, state: Sequence[float]) -> tuple[list[float], list[float]]:
        flat = self._coerce_state(state)

        board_flat = flat[: self.board_dim]
        if len(board_flat) < self.board_dim:
            board_flat.extend([0.0] * (self.board_dim - len(board_flat)))

        aux_flat = flat[self.board_dim :]
        if len(aux_flat) < self.aux_dim:
            aux_flat.extend([0.0] * (self.aux_dim - len(aux_flat)))
        elif len(aux_flat) > self.aux_dim:
            aux_flat = aux_flat[: self.aux_dim]

        return board_flat, aux_flat

    def encode_single(self, state: Sequence[float], device) -> tuple:
        board_flat, aux_flat = self._split_state(state)
        board = torch.tensor(board_flat, dtype=torch.float32, device=device).view(
            1,
            self.board_channels,
            self.board_rows,
            self.board_cols,
        )
        aux = torch.tensor(aux_flat, dtype=torch.float32, device=device).view(1, self.aux_dim)
        return board, aux

    def encode_batch(
        self,
        states: Sequence[Sequence[float]],
        device,
    ) -> tuple:
        boards: list[list[float]] = []
        auxes: list[list[float]] = []
        for st in states:
            board_flat, aux_flat = self._split_state(st)
            boards.append(board_flat)
            auxes.append(aux_flat)

        board_tensor = torch.tensor(boards, dtype=torch.float32, device=device).view(
            len(boards),
            self.board_channels,
            self.board_rows,
            self.board_cols,
        )
        aux_tensor = torch.tensor(auxes, dtype=torch.float32, device=device).view(len(auxes), self.aux_dim)
        return board_tensor, aux_tensor

    def player_holes(self, state: Sequence[float]) -> int:
        """估算玩家棋盘空洞数：某列中，已出现方块后其下方的空格计为空洞。"""
        board_flat, _ = self._split_state(state)
        one_channel_dim = self.board_rows * self.board_cols
        player_flat = board_flat[:one_channel_dim]

        holes = 0
        for col in range(self.board_cols):
            seen_block = False
            for row in range(self.board_rows):
                val = player_flat[row * self.board_cols + col]
                if val > 0.5:
                    seen_block = True
                elif seen_block:
                    holes += 1
        return int(holes)


class BoardCNNQNetwork(nn.Module):
    """
    用于 25x25（或同类网格）棋盘输入的 CNN Q 网络。

    输入：
    - board: [B, C, H, W]
    - aux:   [B, F]，例如当前方块姿态、坐标、has_push_right 等附加特征

    输出：
    - q_values: [B, action_dim]
    """

    def __init__(self, board_channels: int, aux_dim: int, action_dim: int):
        super().__init__()
        self.board_channels = int(board_channels)
        self.aux_dim = int(aux_dim)
        self.action_dim = int(action_dim)

        self.board_encoder = nn.Sequential(
            nn.Conv2d(self.board_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        self.aux_encoder = nn.Sequential(
            nn.Linear(self.aux_dim, 128),
            nn.ReLU(),
        )

        fused_dim = (64 * 4 * 4) + 128
        self.q_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(module.weight, a=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, board, aux):
        if board.dim() == 3:
            board = board.unsqueeze(0)
        if aux.dim() == 1:
            aux = aux.unsqueeze(0)

        board_feat = self.board_encoder(board)
        aux_feat = self.aux_encoder(aux)
        fused = torch.cat([board_feat, aux_feat], dim=1)
        return self.q_head(fused)

    @_no_grad_if_available()
    def greedy_action(self, board, aux) -> int:
        q_values = self.forward(board, aux)
        return int(q_values.argmax(dim=1).item())


class DQNAgent:
    """封装 DQN 的动作选择、优化与目标网络同步逻辑。"""

    def __init__(
        self,
        board_channels: int,
        aux_dim: int,
        action_dim: int,
        gamma: float,
        learning_rate: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        device,
    ):
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = max(1, int(epsilon_decay_steps))
        self.device = device
        self.action_dim = int(action_dim)

        self.policy_net = BoardCNNQNetwork(board_channels, aux_dim, action_dim).to(self.device)
        self.target_net = BoardCNNQNetwork(board_channels, aux_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=float(learning_rate))
        self.loss_fn = nn.SmoothL1Loss()

        self.current_epsilon = float(self.epsilon_start)

    def epsilon_by_step(self, step: int) -> float:
        if step >= self.epsilon_decay_steps:
            return float(self.epsilon_end)

        ratio = float(step) / float(max(1, self.epsilon_decay_steps))
        return float(self.epsilon_start + (self.epsilon_end - self.epsilon_start) * ratio)

    def select_action(
        self,
        board,
        aux,
        global_step: int,
        explore: bool,
    ) -> int:
        epsilon = self.epsilon_by_step(global_step) if explore else 0.0
        self.current_epsilon = float(epsilon)

        if explore and random.random() < epsilon:
            return int(random.randrange(self.action_dim))

        self.policy_net.eval()
        with torch.no_grad():
            action = self.policy_net.greedy_action(board, aux)
        self.policy_net.train()
        return int(action)

    def optimize_once(self, batch, state_adapter: CNNStateAdapter) -> float:
        states = [item.state for item in batch]
        next_states = [item.next_state for item in batch]

        boards, aux = state_adapter.encode_batch(states, self.device)
        next_boards, next_aux = state_adapter.encode_batch(next_states, self.device)

        actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=self.device)

        q_values = self.policy_net(boards, aux).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_boards, next_aux).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_boards, next_aux).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


class DQNTrainer:
    """DQN 训练器：负责采样、学习、评估、检查点保存与自动恢复。"""

    def __init__(
        self,
        game_config: GameConfig | None = None,
        trainer_config: TrainerConfig | None = None,
    ):
        if torch is None:
            raise RuntimeError("PyTorch 不可用，请先安装 torch 再进行训练")

        self.game_config = game_config if game_config is not None else CONFIG
        self.config = trainer_config or self._build_trainer_config_from_game(self.game_config)

        if self.config.seed is not None:
            random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

        self.device = self._resolve_device(self.config.device)

        if self.config.game_mode == "TRADITIONAL":
            from src.ai.shared_tetris_env import SharedTetrisEnv
            self.env = SharedTetrisEnv(
                config=self.game_config,
                mode_key=self.config.game_mode,
                seed=self.config.seed,
            )
        else:
            self.env = TetrisEnv(
                config=self.game_config,
                seed=self.config.seed,
                max_episode_steps=self.config.max_episode_steps,
                opponent_mode=self.config.opponent_mode,
                opponent_action_interval_ms=self.config.opponent_start_interval_ms,
                opponent_mistake_chance=self.config.opponent_start_mistake,
            )

        # 用一次 reset 采样状态维度，构建 CNN 输入适配器。
        sample_state = self.env.reset(seed=self.config.seed)
        board_rows, board_cols = self._resolve_board_shape()
        self.state_adapter = CNNStateAdapter.from_state(
            state=sample_state,
            board_rows=board_rows,
            board_cols=board_cols,
            prefer_two_channels=(self.config.game_mode == "CLASSIC"),
        )

        self.action_mapper = ActionMapper(self.env)

        self.agent = DQNAgent(
            board_channels=self.state_adapter.board_channels,
            aux_dim=self.state_adapter.aux_dim,
            action_dim=self.action_mapper.action_dim,
            gamma=self.config.gamma,
            learning_rate=self.config.learning_rate,
            epsilon_start=self.config.epsilon_start,
            epsilon_end=self.config.epsilon_end,
            epsilon_decay_steps=self.config.epsilon_decay_steps,
            device=self.device,
        )

        self.replay = ReplayMemory(capacity=self.config.replay_capacity, seed=self.config.seed)

        self.total_steps = 0
        self.start_episode = 1
        self.current_episode = 0

        self.best_eval_reward = float("-inf")
        self.best_eval_score: tuple[float, float, float] = (float("-inf"), float("-inf"), float("-inf"))
        self.best_eval_stats: dict[str, float] = {}
        self.history: list[dict[str, float | int]] = []

        self.model_dir = mode_model_dir(self.config.model_dir, self.config.game_mode)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_device(requested: str | None):
        # 按要求保留自动硬件适配写法。
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        normalized = str(requested or "").strip().lower()
        if normalized == "cpu":
            return torch.device("cpu")
        if normalized == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return device

    def _resolve_board_shape(self) -> tuple[int, int]:
        if self.config.game_mode == "TRADITIONAL":
            return (
                int(self.game_config.shared_arena_grid_rows),
                int(self.game_config.shared_arena_grid_cols),
            )
        return int(self.game_config.grid_rows), int(self.game_config.grid_cols)

    @staticmethod
    def _build_trainer_config_from_game(game_config: GameConfig) -> TrainerConfig:
        return TrainerConfig(
            episodes=game_config.rl_train_episodes,
            warmup_steps=game_config.rl_warmup_steps,
            batch_size=game_config.rl_batch_size,
            gamma=game_config.rl_gamma,
            learning_rate=game_config.rl_learning_rate,
            epsilon_start=game_config.rl_epsilon_start,
            epsilon_end=game_config.rl_epsilon_end,
            epsilon_decay_steps=game_config.rl_epsilon_decay_steps,
            train_frequency=game_config.rl_train_frequency,
            target_sync_interval=game_config.rl_target_sync_interval,
            replay_capacity=game_config.rl_replay_capacity,
            max_episode_steps=game_config.rl_max_episode_steps,
            eval_interval=game_config.rl_eval_interval,
            eval_episodes=game_config.rl_eval_episodes,
            checkpoint_interval=game_config.rl_checkpoint_interval,
            model_dir=game_config.rl_model_dir,
            curriculum_episodes=game_config.rl_curriculum_episodes,
            opponent_start_interval_ms=game_config.rl_opponent_start_interval_ms,
            opponent_end_interval_ms=game_config.rl_opponent_end_interval_ms,
            opponent_start_mistake=game_config.rl_opponent_start_mistake,
            opponent_end_mistake=game_config.rl_opponent_end_mistake,
            seed=game_config.rl_eval_seed,
            opponent_mode="heuristic",
            device=game_config.ai_model_device,
            checkpoint_name=LATEST_CHECKPOINT_NAME,
        )

    def _opponent_profile_for_episode(self, episode: int, for_eval: bool = False) -> tuple[int, float]:
        if for_eval:
            return int(self.config.opponent_end_interval_ms), float(self.config.opponent_end_mistake)

        if self.config.curriculum_episodes <= 0:
            return int(self.config.opponent_end_interval_ms), float(self.config.opponent_end_mistake)

        progress = min(1.0, episode / self.config.curriculum_episodes)
        interval = int(
            self.config.opponent_start_interval_ms
            + (self.config.opponent_end_interval_ms - self.config.opponent_start_interval_ms) * progress
        )
        mistake = float(
            self.config.opponent_start_mistake
            + (self.config.opponent_end_mistake - self.config.opponent_start_mistake) * progress
        )
        return interval, mistake

    def auto_resume_if_available(self, enabled: bool = True) -> bool:
        if not enabled:
            return False

        checkpoint = self._find_latest_checkpoint()
        if checkpoint is None:
            return False

        return self.load_checkpoint(str(checkpoint))

    def _find_latest_checkpoint(self) -> Path | None:
        candidates = [
            self.model_dir / self.config.checkpoint_name,
            self.model_dir / "last.pt",
        ]
        existing = [p for p in candidates if p.exists()]
        if not existing:
            return None
        return max(existing, key=lambda p: p.stat().st_mtime)

    def train(self) -> list[dict[str, float | int]]:
        started_at = time.time()
        try:
            for episode in range(self.start_episode, self.config.episodes + 1):
                self.current_episode = episode
                ep_stats = self._run_training_episode(episode)
                self.history.append(ep_stats)

                if episode % self.config.eval_interval == 0:
                    curr_eval_stats = self.evaluate(
                        episodes=self.config.eval_episodes,
                        episode=episode,
                        profile="curriculum",
                    )
                    hard_eval_stats = self.evaluate(
                        episodes=max(5, self.config.eval_episodes // 2),
                        episode=episode,
                        profile="hard",
                    )

                    hard_score = self._evaluation_score(hard_eval_stats)
                    if hard_score > self.best_eval_score:
                        self.best_eval_score = hard_score
                        self.best_eval_reward = hard_eval_stats["avg_reward"]
                        self.best_eval_stats = {
                            "curr_avg_reward": curr_eval_stats["avg_reward"],
                            "curr_win_rate": curr_eval_stats["win_rate"],
                            "curr_avg_score_gap": curr_eval_stats["avg_score_gap"],
                            "hard_avg_reward": hard_eval_stats["avg_reward"],
                            "hard_win_rate": hard_eval_stats["win_rate"],
                            "hard_avg_score_gap": hard_eval_stats["avg_score_gap"],
                        }
                        self._save_checkpoint("best.pt", episode=episode, extra=self.best_eval_stats)

                    print(
                        "[EVAL_CURR] "
                        f"episode={episode} "
                        f"avg_reward={curr_eval_stats['avg_reward']:.3f} "
                        f"win_rate={curr_eval_stats['win_rate']:.3f} "
                        f"avg_gap={curr_eval_stats['avg_score_gap']:.2f} "
                        f"opp_interval={int(curr_eval_stats['opp_interval'])} "
                        f"opp_mistake={curr_eval_stats['opp_mistake']:.3f}"
                    )
                    print(
                        "[EVAL_HARD] "
                        f"episode={episode} "
                        f"avg_reward={hard_eval_stats['avg_reward']:.3f} "
                        f"win_rate={hard_eval_stats['win_rate']:.3f} "
                        f"avg_gap={hard_eval_stats['avg_score_gap']:.2f} "
                        f"opp_interval={int(hard_eval_stats['opp_interval'])} "
                        f"opp_mistake={hard_eval_stats['opp_mistake']:.3f}"
                    )

                print(
                    "[TRAIN] "
                    f"episode={episode} "
                    f"steps={ep_stats['steps']} "
                    f"reward={ep_stats['reward']:.3f} "
                    f"epsilon={ep_stats['epsilon']:.4f} "
                    f"loss={ep_stats['loss']:.4f} "
                    f"opp_interval={ep_stats['opp_interval']} "
                    f"opp_mistake={ep_stats['opp_mistake']:.3f}"
                )

                # 每回合写入 latest/last，保证中断后可无缝恢复。
                self._save_checkpoint(self.config.checkpoint_name, episode=episode)
                self._save_checkpoint("last.pt", episode=episode)
                self._save_history(started_at)
                self._cleanup_extra_checkpoints()

        except KeyboardInterrupt:
            print("\n[INFO] 检测到 Ctrl+C，正在安全保存训练进度...")
            fallback_episode = max(1, self.current_episode)
            self._save_checkpoint(self.config.checkpoint_name, episode=fallback_episode)
            self._save_checkpoint("last.pt", episode=fallback_episode)
            self._save_history(started_at)
            self._cleanup_extra_checkpoints()
            self._print_recommended_checkpoint()
            return self.history

        self._print_recommended_checkpoint()
        return self.history

    def _run_training_episode(self, episode: int) -> dict[str, float | int]:
        opp_interval, opp_mistake = self._opponent_profile_for_episode(episode)
        self.env.set_opponent_profile(opp_interval, opp_mistake)

        state = self.env.reset(seed=(self.config.seed or 0) + episode)
        done = False
        total_reward = 0.0
        total_loss = 0.0
        optimize_count = 0
        step_count = 0

        prev_lock_count = self._player_lock_count()
        prev_player_lines = self._player_lines()
        prev_holes = self.state_adapter.player_holes(state)

        while not done and step_count < self.config.max_episode_steps:
            board, aux = self.state_adapter.encode_single(state, self.device)
            agent_action = self.agent.select_action(board, aux, self.total_steps, explore=True)
            env_action = self.action_mapper.to_env_action(agent_action)

            outcome = self.env.step(env_action)
            next_state = outcome.state
            done = outcome.done

            curr_lock_count = self._player_lock_count()
            curr_player_lines = self._player_lines()
            curr_holes = self.state_adapter.player_holes(next_state)

            reward = self._compose_reward(
                base_reward=outcome.reward,
                prev_lock_count=prev_lock_count,
                curr_lock_count=curr_lock_count,
                prev_player_lines=prev_player_lines,
                curr_player_lines=curr_player_lines,
                prev_holes=prev_holes,
                curr_holes=curr_holes,
            )

            self.replay.append(state, agent_action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step_count += 1
            self.total_steps += 1

            if curr_lock_count > prev_lock_count:
                prev_lock_count = curr_lock_count
                prev_player_lines = curr_player_lines
                prev_holes = curr_holes

            if (
                self.total_steps >= self.config.warmup_steps
                and self.total_steps % self.config.train_frequency == 0
                and self.replay.can_sample(self.config.batch_size)
            ):
                batch = self.replay.sample(self.config.batch_size)
                loss = self.agent.optimize_once(batch, self.state_adapter)
                total_loss += loss
                optimize_count += 1

            if self.total_steps % self.config.target_sync_interval == 0:
                self.agent.sync_target()

        avg_loss = total_loss / optimize_count if optimize_count > 0 else 0.0
        return {
            "episode": int(episode),
            "steps": int(step_count),
            "reward": float(total_reward),
            "loss": float(avg_loss),
            "epsilon": float(self.agent.current_epsilon),
            "player_score": int(getattr(self.env.player_core, "score", 0)),
            "ai_score": self._best_opponent_score(),
            "player_lines": int(getattr(self.env.player_core, "lines_cleared_total", 0)),
            "ai_lines": self._best_opponent_lines(),
            "opp_interval": int(opp_interval),
            "opp_mistake": float(round(opp_mistake, 4)),
        }

    def _compose_reward(
        self,
        base_reward: float,
        prev_lock_count: int,
        curr_lock_count: int,
        prev_player_lines: int,
        curr_player_lines: int,
        prev_holes: int,
        curr_holes: int,
    ) -> float:
        """
        核心奖励整形（按你的需求）：
        - 己方消行：+1.0 * 行数
        - 己方溢出 Game Over：-5.0
        - 放置后新增空洞：-0.1 * 新增空洞数
        - 放置位置越靠底（Y 越大）：给微小正奖励

        说明：
        - 仅在 lock 发生时应用“消行/空洞/Y奖励”，避免每帧重复累计。
        - 同时保留环境基础奖励 base_reward，方便与现有机制兼容。
        """
        reward = float(base_reward)

        locked_now = curr_lock_count > prev_lock_count
        if locked_now:
            line_gain = max(0, curr_player_lines - prev_player_lines)
            reward += float(line_gain) * 1.0

            new_holes = max(0, curr_holes - prev_holes)
            reward -= float(new_holes) * 0.1

            reward += self._landing_y_bonus()

        if self._player_game_over():
            reward -= 5.0

        return float(reward)

    def _landing_y_bonus(self) -> float:
        """
        估算“落点越低奖励越高”。
        优先读取核心对象中的落点字段；如果环境未提供该字段，退化为 0。
        """
        core = getattr(self.env, "player_core", None)
        if core is None:
            return 0.0

        y_value = None
        for attr in ("last_lock_y", "last_piece_y", "last_placement_y", "last_drop_y"):
            candidate = getattr(core, attr, None)
            if isinstance(candidate, (int, float)):
                y_value = float(candidate)
                break

        if y_value is None:
            return 0.0

        denom = max(1.0, float(self.state_adapter.board_rows - 1))
        normalized = max(0.0, min(1.0, y_value / denom))
        return float(0.03 * normalized)

    def _player_game_over(self) -> bool:
        core = getattr(self.env, "player_core", None)
        return bool(getattr(core, "state", "") == "GAME_OVER")

    def _player_lines(self) -> int:
        core = getattr(self.env, "player_core", None)
        return int(getattr(core, "lines_cleared_total", 0))

    def _player_lock_count(self) -> int:
        core = getattr(self.env, "player_core", None)
        return int(getattr(core, "lock_count", 0))

    def _best_opponent_score(self) -> int:
        if hasattr(self.env, "opponent_entities"):
            return int(max((ent.score for ent in self.env.opponent_entities), default=0))
        return int(getattr(self.env.ai_core, "score", 0))

    def _best_opponent_lines(self) -> int:
        if hasattr(self.env, "opponent_entities"):
            return int(max((ent.lines_cleared_total for ent in self.env.opponent_entities), default=0))
        return int(getattr(self.env.ai_core, "lines_cleared_total", 0))

    @_no_grad_if_available()
    def evaluate(
        self,
        episodes: int = 5,
        episode: int | None = None,
        profile: str = "curriculum",
    ) -> dict[str, float]:
        rewards: list[float] = []
        wins = 0
        score_gaps: list[float] = []

        eval_episode = self.config.episodes if episode is None else int(episode)
        if profile == "hard":
            opp_interval, opp_mistake = self._opponent_profile_for_episode(
                episode=self.config.episodes,
                for_eval=True,
            )
        else:
            opp_interval, opp_mistake = self._opponent_profile_for_episode(
                episode=eval_episode,
                for_eval=False,
            )
        self.env.set_opponent_profile(opp_interval, opp_mistake)

        self.agent.policy_net.eval()
        for idx in range(episodes):
            state = self.env.reset(seed=(self.config.seed or 0) + 10000 + idx)
            done = False
            total_reward = 0.0
            steps = 0

            prev_lock_count = self._player_lock_count()
            prev_player_lines = self._player_lines()
            prev_holes = self.state_adapter.player_holes(state)

            while not done and steps < self.config.max_episode_steps:
                board, aux = self.state_adapter.encode_single(state, self.device)
                agent_action = self.agent.select_action(board, aux, self.total_steps, explore=False)
                env_action = self.action_mapper.to_env_action(agent_action)

                outcome = self.env.step(env_action)
                state = outcome.state
                done = outcome.done

                curr_lock_count = self._player_lock_count()
                curr_player_lines = self._player_lines()
                curr_holes = self.state_adapter.player_holes(state)

                reward = self._compose_reward(
                    base_reward=outcome.reward,
                    prev_lock_count=prev_lock_count,
                    curr_lock_count=curr_lock_count,
                    prev_player_lines=prev_player_lines,
                    curr_player_lines=curr_player_lines,
                    prev_holes=prev_holes,
                    curr_holes=curr_holes,
                )
                total_reward += reward
                steps += 1

                if curr_lock_count > prev_lock_count:
                    prev_lock_count = curr_lock_count
                    prev_player_lines = curr_player_lines
                    prev_holes = curr_holes

            rewards.append(total_reward)
            if hasattr(self.env, "competitive_score_gap"):
                score_gap = float(self.env.competitive_score_gap())
                did_win = bool(self.env.did_player_win())
            else:
                score_gap = float(getattr(self.env.player_core, "score", 0) - getattr(self.env.ai_core, "score", 0))
                did_win = score_gap > 0

            score_gaps.append(score_gap)
            if did_win:
                wins += 1

        self.agent.policy_net.train()

        avg_reward = sum(rewards) / max(1, len(rewards))
        win_rate = wins / max(1, len(rewards))
        avg_gap = sum(score_gaps) / max(1, len(score_gaps))
        return {
            "avg_reward": float(avg_reward),
            "win_rate": float(win_rate),
            "avg_score_gap": float(avg_gap),
            "opp_interval": float(opp_interval),
            "opp_mistake": float(opp_mistake),
        }

    @staticmethod
    def _evaluation_score(stats: dict[str, float]) -> tuple[float, float, float]:
        return (
            float(stats.get("win_rate", 0.0)),
            float(stats.get("avg_score_gap", float("-inf"))),
            float(stats.get("avg_reward", float("-inf"))),
        )

    def _save_checkpoint(self, filename: str, episode: int, extra: dict | None = None):
        payload = {
            "policy_state_dict": self.agent.policy_net.state_dict(),
            "target_state_dict": self.agent.target_net.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "episode": int(episode),
            "total_steps": int(self.total_steps),
            "epsilon": float(self.agent.current_epsilon),
            "config": self.config.__dict__,
            "state_adapter": self.state_adapter.to_dict(),
            "agent_to_env_action": list(self.action_mapper.agent_to_env),
            "best_eval_reward": float(self.best_eval_reward),
            "best_eval_score": list(self.best_eval_score),
            "best_eval_stats": self.best_eval_stats,
        }
        if extra:
            payload["extra"] = extra

        path = self.model_dir / filename
        torch.save(payload, path)

    def load_checkpoint(self, filename: str) -> bool:
        path = Path(filename)
        if not path.is_absolute():
            path = self.model_dir / filename

        if not path.exists():
            print(f"[LOAD] Checkpoint {path} not found. Starting from scratch.")
            return False

        print(f"[LOAD] Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        try:
            self.agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
            self.agent.target_net.load_state_dict(checkpoint["target_state_dict"])
            self.agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as exc:
            print(f"[LOAD] Checkpoint 与当前模型结构不兼容，已忽略恢复: {exc}")
            return False

        restored_episode = int(checkpoint.get("episode", 0))
        self.start_episode = max(1, restored_episode + 1)
        self.current_episode = restored_episode
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.agent.current_epsilon = float(
            checkpoint.get("epsilon", self.agent.epsilon_by_step(self.total_steps))
        )

        self.best_eval_reward = float(checkpoint.get("best_eval_reward", self.best_eval_reward))

        raw_best_score = checkpoint.get("best_eval_score", None)
        if isinstance(raw_best_score, (list, tuple)) and len(raw_best_score) == 3:
            self.best_eval_score = (
                float(raw_best_score[0]),
                float(raw_best_score[1]),
                float(raw_best_score[2]),
            )

        best_stats = checkpoint.get("best_eval_stats", None)
        if isinstance(best_stats, dict):
            self.best_eval_stats = {str(k): float(v) for k, v in best_stats.items()}

        return True

    def _save_history(self, started_at: float):
        summary = {
            "started_at": started_at,
            "ended_at": time.time(),
            "total_steps": self.total_steps,
            "best_eval_reward": self.best_eval_reward,
            "best_eval_score": self.best_eval_score,
            "best_eval_stats": self.best_eval_stats,
            "history": self.history,
            "adapter": self.state_adapter.to_dict(),
            "agent_to_env_action": self.action_mapper.agent_to_env,
            "checkpoint_name": self.config.checkpoint_name,
        }

        history_path = self.model_dir / "training_history.json"
        history_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _cleanup_extra_checkpoints(self):
        keep_names = {
            "best.pt",
            "last.pt",
            self.config.checkpoint_name,
            "training_history.json",
        }
        for path in self.model_dir.iterdir():
            if path.name in keep_names:
                continue
            if path.is_file() and path.suffix in (".pt", ".pth"):
                path.unlink(missing_ok=True)

    def _recommended_checkpoint_path(self) -> Path:
        best_path = self.model_dir / "best.pt"
        if best_path.exists():
            return best_path

        latest_path = self.model_dir / self.config.checkpoint_name
        if latest_path.exists():
            return latest_path

        return self.model_dir / "last.pt"

    def _print_recommended_checkpoint(self):
        recommended = self._recommended_checkpoint_path()
        if recommended.exists():
            print(f"[DONE] 推荐使用模型: {recommended}")
        else:
            print(f"[DONE] 当前模式 {self.config.game_mode} 还没有可用 checkpoint")


def run_training(
    episodes: int | None = None,
    seed: int | None = None,
    mode: str = "CLASSIC",
    opponent_mode: str = "heuristic",
    device: str | None = None,
    resume: bool = True,
):
    if torch is None:
        raise RuntimeError("PyTorch 不可用，无法启动训练")

    game_config = CONFIG
    trainer_cfg = DQNTrainer._build_trainer_config_from_game(game_config)
    if episodes is not None:
        trainer_cfg.episodes = int(episodes)
    if seed is not None:
        trainer_cfg.seed = int(seed)
    trainer_cfg.opponent_mode = opponent_mode
    trainer_cfg.game_mode = mode
    if device is not None:
        trainer_cfg.device = device

    trainer = DQNTrainer(game_config=game_config, trainer_config=trainer_cfg)
    resumed = trainer.auto_resume_if_available(enabled=resume)

    print(
        "[CONFIG] "
        f"device={trainer.device} "
        f"episodes={trainer.config.episodes} "
        f"batch={trainer.config.batch_size} "
        f"warmup={trainer.config.warmup_steps} "
        f"curriculum={trainer.config.curriculum_episodes} "
        f"board={trainer.state_adapter.board_rows}x{trainer.state_adapter.board_cols} "
        f"channels={trainer.state_adapter.board_channels} "
        f"aux_dim={trainer.state_adapter.aux_dim} "
        f"action_dim={trainer.action_mapper.action_dim} "
        f"resume={'yes' if resumed else 'no'}"
    )
    return trainer.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN agent for TetrisEnv")
    parser.add_argument("--episodes", type=int, default=None, help="训练回合数，默认取配置")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument(
        "--mode",
        type=str,
        default="CLASSIC",
        choices=("CLASSIC", "TRADITIONAL"),
        help="目标游戏模式，输入 TRADITIONAL 为模式2训练",
    )
    parser.add_argument(
        "--opponent-mode",
        type=str,
        default="heuristic",
        choices=("heuristic", "model"),
        help="训练时 AI 对手模式",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("cpu", "cuda"),
        help="训练设备，默认自动选择 cuda/cpu",
    )

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="自动检测并恢复 checkpoint（默认开启）",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="禁用自动恢复，从头训练",
    )
    parser.set_defaults(resume=True)

    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run_training(
            episodes=args.episodes,
            seed=args.seed,
            mode=args.mode,
            opponent_mode=args.opponent_mode,
            device=args.device,
            resume=args.resume,
        )
    except RuntimeError as exc:
        print(f"训练启动失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
