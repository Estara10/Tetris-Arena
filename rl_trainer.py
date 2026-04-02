from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
import time

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


def _no_grad_if_available():
    if torch is None:
        def _decorator(func):
            return func

        return _decorator
    return torch.no_grad()

from dqn_model import DQNModelConfig, TetrisDQN
from replay_memory import ReplayMemory
from settings import CONFIG, GameConfig
from tetris_env import TetrisEnv


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
    device: str = "cpu"


class DQNTrainer:
    """DQN 训练器：负责采样、学习、评估和检查点保存。"""

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

        requested_device = self.config.device.strip().lower()
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        self.env = TetrisEnv(
            config=self.game_config,
            seed=self.config.seed,
            max_episode_steps=self.config.max_episode_steps,
            opponent_mode=self.config.opponent_mode,
            opponent_action_interval_ms=self.config.opponent_start_interval_ms,
            opponent_mistake_chance=self.config.opponent_start_mistake,
        )

        state_dim = self.env.state_size()
        action_dim = self.env.action_size()

        model_cfg = DQNModelConfig(
            input_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=(512, 256),
            dueling=True,
        )
        self.policy_net = TetrisDQN(model_cfg).to(self.device)
        self.target_net = TetrisDQN(model_cfg).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
        )
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = ReplayMemory(
            capacity=self.config.replay_capacity,
            seed=self.config.seed,
        )

        self.total_steps = 0
        self.best_eval_reward = float("-inf")
        self.best_eval_score: tuple[float, float, float] = (float("-inf"), float("-inf"), float("-inf"))
        self.best_eval_stats: dict[str, float] = {}
        self.history: list[dict[str, float | int]] = []

        self.model_dir = Path(self.config.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

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
        )

    def _opponent_profile_for_episode(self, episode: int, for_eval: bool = False) -> tuple[int, float]:
        if for_eval:
            return (
                int(self.config.opponent_end_interval_ms),
                float(self.config.opponent_end_mistake),
            )

        if self.config.curriculum_episodes <= 0:
            return (
                int(self.config.opponent_end_interval_ms),
                float(self.config.opponent_end_mistake),
            )

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

    def train(self) -> list[dict[str, float | int]]:
        started_at = time.time()
        for episode in range(1, self.config.episodes + 1):
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
                    self._save_checkpoint("best.pt", extra=self.best_eval_stats)

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

            if episode % self.config.checkpoint_interval == 0:
                self._save_checkpoint(f"episode_{episode}.pt")

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

        self._save_checkpoint("last.pt")
        self._save_history(started_at)
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

        while not done and step_count < self.config.max_episode_steps:
            epsilon = self._epsilon_by_step(self.total_steps)
            action = self._select_action(state, epsilon)

            outcome = self.env.step(action)
            self.replay.append(state, action, outcome.reward, outcome.state, outcome.done)

            state = outcome.state
            done = outcome.done
            total_reward += outcome.reward
            step_count += 1
            self.total_steps += 1

            if (
                self.total_steps >= self.config.warmup_steps
                and self.total_steps % self.config.train_frequency == 0
                and self.replay.can_sample(self.config.batch_size)
            ):
                loss = self._optimize_once()
                total_loss += loss
                optimize_count += 1

            if self.total_steps % self.config.target_sync_interval == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        avg_loss = total_loss / optimize_count if optimize_count > 0 else 0.0
        return {
            "episode": episode,
            "steps": step_count,
            "reward": total_reward,
            "loss": avg_loss,
            "epsilon": self._epsilon_by_step(self.total_steps),
            "player_score": self.env.player_core.score,
            "ai_score": self.env.ai_core.score,
            "player_lines": self.env.player_core.lines_cleared_total,
            "ai_lines": self.env.ai_core.lines_cleared_total,
            "opp_interval": opp_interval,
            "opp_mistake": round(opp_mistake, 4),
        }

    def _epsilon_by_step(self, step: int) -> float:
        if step >= self.config.epsilon_decay_steps:
            return self.config.epsilon_end

        ratio = step / max(1, self.config.epsilon_decay_steps)
        return self.config.epsilon_start + (self.config.epsilon_end - self.config.epsilon_start) * ratio

    def _select_action(self, state: list[float], epsilon: float) -> int:
        if random.random() < epsilon:
            return self.env.sample_action()

        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            action = self.policy_net.greedy_action(state_tensor)
        self.policy_net.train()
        return action

    def _optimize_once(self) -> float:
        batch = self.replay.sample(self.config.batch_size)

        states = torch.tensor([item.state for item in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([item.next_state for item in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.config.gamma * next_q

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        return float(loss.item())

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

        eval_episode = self.config.episodes if episode is None else episode
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

        self.policy_net.eval()
        for idx in range(episodes):
            state = self.env.reset(seed=(self.config.seed or 0) + 10000 + idx)
            done = False
            total_reward = 0.0
            steps = 0

            while not done and steps < self.config.max_episode_steps:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                action = self.policy_net.greedy_action(state_tensor)
                outcome = self.env.step(action)
                state = outcome.state
                total_reward += outcome.reward
                done = outcome.done
                steps += 1

            rewards.append(total_reward)
            score_gap = float(self.env.player_core.score - self.env.ai_core.score)
            score_gaps.append(score_gap)
            if score_gap > 0:
                wins += 1

        self.policy_net.train()

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
        # 优先比较胜率，再比较分差，最后比较奖励，避免奖励塑形项掩盖真实胜负表现。
        return (
            float(stats.get("win_rate", 0.0)),
            float(stats.get("avg_score_gap", float("-inf"))),
            float(stats.get("avg_reward", float("-inf"))),
        )

    def _save_checkpoint(self, filename: str, extra: dict | None = None):
        payload = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "config": self.config.__dict__,
        }
        if extra:
            payload["extra"] = extra

        path = self.model_dir / filename
        torch.save(payload, path)

    def _save_history(self, started_at: float):
        summary = {
            "started_at": started_at,
            "ended_at": time.time(),
            "total_steps": self.total_steps,
            "best_eval_reward": self.best_eval_reward,
            "best_eval_score": self.best_eval_score,
            "best_eval_stats": self.best_eval_stats,
            "history": self.history,
        }

        history_path = self.model_dir / "training_history.json"
        history_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def run_training(
    episodes: int | None = None,
    seed: int | None = None,
    opponent_mode: str = "heuristic",
    device: str | None = None,
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
    if device is not None:
        trainer_cfg.device = device

    trainer = DQNTrainer(game_config=game_config, trainer_config=trainer_cfg)
    print(
        "[CONFIG] "
        f"device={trainer.device} "
        f"episodes={trainer.config.episodes} "
        f"batch={trainer.config.batch_size} "
        f"warmup={trainer.config.warmup_steps} "
        f"curriculum={trainer.config.curriculum_episodes}"
    )
    return trainer.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN agent for TetrisEnv")
    parser.add_argument("--episodes", type=int, default=None, help="训练回合数，默认取配置")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
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
        help="训练设备，默认使用配置项",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run_training(
            episodes=args.episodes,
            seed=args.seed,
            opponent_mode=args.opponent_mode,
            device=args.device,
        )
    except RuntimeError as exc:
        print(f"训练启动失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
