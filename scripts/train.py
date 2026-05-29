"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import dataclass
import json
import os
import random
from collections import deque
from pathlib import Path

# 训练模式下避免 Pygame 打开窗口/音频设备。
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
import torch.nn.functional as F  # type: ignore[import-not-found]

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    try:
        from tensorboardX import SummaryWriter  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        SummaryWriter = None

from src.ai.ai_controller import board_profile, evaluate_grid, generate_candidates
from src.ai.deep_q_network import DeepQNetwork
from src.ai.model_paths import mode_model_dir
from settings import CONFIG
from src.ai.shared_tetris_env import SharedTetrisEnv
from src.ai.tetris_env import TetrisEnv


CHECKPOINT_NAME = "latest_checkpoint.pth"


class _NoOpWriter:
    def add_scalar(self, *_args, **_kwargs):
        return

    def close(self):
        return


@dataclass
class EnvSlot:
    env: object
    episode_id: int | None = None
    episode_cap: int = 0
    state: list[float] | None = None
    steps: int = 0
    reward: float = 0.0
    env_reward: float = 0.0
    heur_bonus: float = 0.0
    heur_weight: float = 0.0
    phi_prev: float = 0.0


class PrioritizedReplayBuffer:
    """预分配张量回放池，支持优先经验回放（PER）。"""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        prioritized: bool = True,
        alpha: float = 0.6,
        priority_eps: float = 1e-5,
    ):
        self.capacity = max(1, int(capacity))
        self.state_dim = int(state_dim)
        self.prioritized = bool(prioritized)
        self.alpha = float(alpha)
        self.priority_eps = float(priority_eps)

        self.states = torch.zeros((self.capacity, self.state_dim), dtype=torch.float32)
        self.next_states = torch.zeros((self.capacity, self.state_dim), dtype=torch.float32)
        self.actions = torch.zeros((self.capacity,), dtype=torch.long)
        self.rewards = torch.zeros((self.capacity,), dtype=torch.float32)
        self.dones = torch.zeros((self.capacity,), dtype=torch.float32)
        self.priorities = torch.ones((self.capacity,), dtype=torch.float32)
        self.teacher_actions = torch.full((self.capacity,), -1, dtype=torch.long)

        self.size = 0
        self.position = 0

    def __len__(self) -> int:
        return self.size

    def append(self, state, action: int, reward: float, next_state, done: bool, teacher_action: int = -1):
        idx = self.position
        self.states[idx] = torch.as_tensor(state, dtype=torch.float32)
        self.next_states[idx] = torch.as_tensor(next_state, dtype=torch.float32)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = 1.0 if done else 0.0
        self.teacher_actions[idx] = int(teacher_action)

        if self.size > 0:
            max_priority = float(self.priorities[: self.size].max().item())
        else:
            max_priority = 1.0
        self.priorities[idx] = max_priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= int(batch_size)

    def sample(self, batch_size: int, beta: float):
        batch = int(batch_size)
        if not self.can_sample(batch):
            raise ValueError("经验不足，无法采样")

        if self.prioritized:
            pri = self.priorities[: self.size].clamp(min=self.priority_eps).pow(self.alpha)
            probs = pri / pri.sum()
            replacement = self.size < batch
            indices = torch.multinomial(probs, batch, replacement=replacement)
            weights = (self.size * probs[indices]).pow(-float(beta))
            weights = weights / weights.max().clamp(min=1.0)
        else:
            indices = torch.randint(0, self.size, (batch,))
            weights = torch.ones((batch,), dtype=torch.float32)

        return (
            indices,
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.teacher_actions[indices],
            weights,
        )

    def update_priorities(self, indices: torch.Tensor, td_error: torch.Tensor):
        if not self.prioritized:
            return
        cpu_indices = indices.detach().cpu().long()
        new_priority = td_error.detach().abs().cpu().float() + self.priority_eps
        self.priorities[cpu_indices] = new_priority


def get_args():
    parser = argparse.ArgumentParser(
        """Use imported DQN framework to train current project environment"""
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="TRADITIONAL",
        choices=("CLASSIC", "TRADITIONAL"),
        help="训练模式",
    )
    parser.add_argument("--episodes", type=int, default=5000, help="训练回合数")

    # 课程学习：先短回合，再逐步增长到完整回合长度。
    parser.add_argument("--curriculum_start_steps", type=int, default=1800)
    parser.add_argument("--curriculum_end_steps", type=int, default=6000)
    parser.add_argument("--curriculum_episodes", type=int, default=800)

    parser.add_argument("--batch_size", type=int, default=512, help="每次优化采样数量")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--initial_epsilon", type=float, default=1.0)
    parser.add_argument("--final_epsilon", type=float, default=0.02)
    parser.add_argument("--num_decay_steps", type=int, default=30000)

    parser.add_argument("--warmup_steps", type=int, default=3000)
    parser.add_argument("--train_frequency", type=int, default=1)
    parser.add_argument("--target_sync_interval", type=int, default=3000)

    parser.add_argument("--replay_memory_size", type=int, default=120000)
    parser.add_argument("--save_interval", type=int, default=50)

    parser.add_argument("--eval_interval", type=int, default=25, help="每隔多少回合做一次贪心评估")
    parser.add_argument("--eval_episodes", type=int, default=12, help="每次评估局数")

    parser.add_argument("--hidden_dim1", type=int, default=512)
    parser.add_argument("--hidden_dim2", type=int, default=512)
    parser.add_argument("--ma_window", type=int, default=20, help="日志滑动窗口")

    parser.add_argument("--seed", type=int, default=20260403)
    parser.add_argument("--device", type=str, default=None, choices=("cpu", "cuda"))
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="models")

    parser.add_argument("--num_envs", type=int, default=4, help="并行采样环境数量")
    parser.add_argument("--action_interval_ms", type=int, default=120, help="每步仿真毫秒")

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="开启混合精度训练")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="关闭混合精度训练")
    parser.set_defaults(amp=True)

    replay_group = parser.add_mutually_exclusive_group()
    replay_group.add_argument("--prioritized-replay", dest="prioritized_replay", action="store_true")
    replay_group.add_argument("--uniform-replay", dest="prioritized_replay", action="store_false")
    parser.set_defaults(prioritized_replay=True)
    parser.add_argument("--per_alpha", type=float, default=0.6)
    parser.add_argument("--per_beta_start", type=float, default=0.4)
    parser.add_argument("--per_beta_frames", type=int, default=200000)

    parser.add_argument("--heuristic_scale", type=float, default=0.01, help="启发式塑形奖励缩放系数")
    parser.add_argument("--heuristic_clip", type=float, default=2.0, help="启发式塑形奖励截断绝对值")
    parser.add_argument(
        "--heuristic_decay_episodes",
        type=int,
        default=0,
        help="启发式衰减回合数；<=0时自动取总回合数的一半，衰减结束后自动关闭启发式",
    )

    guidance_group = parser.add_mutually_exclusive_group()
    guidance_group.add_argument(
        "--heuristic-guidance",
        dest="heuristic_guidance",
        action="store_true",
        help="启用启发式引导奖励（默认开启）",
    )
    guidance_group.add_argument(
        "--no-heuristic-guidance",
        dest="heuristic_guidance",
        action="store_false",
        help="关闭启发式引导奖励",
    )
    parser.set_defaults(heuristic_guidance=True)

    teacher_group = parser.add_mutually_exclusive_group()
    teacher_group.add_argument(
        "--teacher-guidance",
        dest="teacher_guidance",
        action="store_true",
        help="启用启发式教师动作指导（探索混合+蒸馏）",
    )
    teacher_group.add_argument(
        "--no-teacher-guidance",
        dest="teacher_guidance",
        action="store_false",
        help="关闭启发式教师动作指导",
    )
    parser.set_defaults(teacher_guidance=True)
    parser.add_argument("--teacher_prob_start", type=float, default=0.65, help="前期采用教师动作的概率")
    parser.add_argument("--teacher_prob_end", type=float, default=0.10, help="后期采用教师动作的概率")
    parser.add_argument(
        "--teacher_decay_episodes",
        type=int,
        default=0,
        help="教师动作概率衰减回合数；<=0时自动取总回合数",
    )
    parser.add_argument("--teacher_label_stride", type=int, default=2, help="每隔多少步打一次教师标签")
    parser.add_argument("--teacher_distill_coef", type=float, default=0.03, help="蒸馏损失权重")

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", help="自动恢复训练")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false", help="从头开始")
    parser.set_defaults(resume=True)

    return parser.parse_args()


def _resolve_device(requested: str | None):
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalized = str(requested or "").strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return default_device


def _epsilon_by_step(step: int, opt) -> float:
    if step >= opt.num_decay_steps:
        return float(opt.final_epsilon)
    ratio = float(step) / float(max(1, opt.num_decay_steps))
    return float(opt.initial_epsilon + (opt.final_epsilon - opt.initial_epsilon) * ratio)


def _per_beta_by_step(step: int, opt) -> float:
    progress = min(1.0, float(step) / float(max(1, opt.per_beta_frames)))
    return float(opt.per_beta_start + (1.0 - opt.per_beta_start) * progress)


def _episode_cap(episode_id: int, opt) -> int:
    if int(opt.curriculum_episodes) <= 0:
        return int(opt.curriculum_end_steps)

    progress = min(1.0, float(max(0, episode_id - 1)) / float(max(1, opt.curriculum_episodes)))
    cap = float(opt.curriculum_start_steps) + (float(opt.curriculum_end_steps) - float(opt.curriculum_start_steps)) * progress
    return int(max(100, round(cap)))


def _resolve_heuristic_decay_episodes(opt) -> int:
    if not bool(opt.heuristic_guidance):
        return 0

    configured = int(getattr(opt, "heuristic_decay_episodes", 0))
    if configured > 0:
        return min(int(opt.episodes), configured)

    return max(1, int(opt.episodes))


def _heuristic_weight_by_episode(episode_id: int, decay_episodes: int) -> float:
    if decay_episodes <= 0:
        return 0.0
    if episode_id > decay_episodes:
        return 0.0
    if decay_episodes == 1:
        return 1.0

    progress = float(max(0, episode_id - 1)) / float(max(1, decay_episodes - 1))
    return float(max(0.0, 1.0 - progress))


def _resolve_teacher_decay_episodes(opt) -> int:
    if not bool(getattr(opt, "teacher_guidance", True)):
        return 0

    configured = int(getattr(opt, "teacher_decay_episodes", 0))
    if configured > 0:
        return min(int(opt.episodes), configured)

    return max(1, int(opt.episodes))


def _teacher_mix_prob_by_episode(episode_id: int, teacher_decay_episodes: int, opt) -> float:
    if teacher_decay_episodes <= 0:
        return 0.0
    if episode_id > teacher_decay_episodes:
        return float(opt.teacher_prob_end)

    if teacher_decay_episodes == 1:
        return float(opt.teacher_prob_end)

    progress = float(max(0, episode_id - 1)) / float(max(1, teacher_decay_episodes - 1))
    start = float(opt.teacher_prob_start)
    end = float(opt.teacher_prob_end)
    value = start + (end - start) * progress
    return float(max(0.0, min(1.0, value)))


def _teacher_action(env, mode: str) -> int | None:
    try:
        normalized = str(mode).strip().upper()
        if normalized == "TRADITIONAL":
            match = getattr(env, "match", None)
            if match is None:
                return None
            player = match._get_entity("player")
            piece = getattr(player, "piece", None)
            grid = match.core.grid
            cfg = match.arena_config
        else:
            player_core = getattr(env, "player_core", None)
            if player_core is None:
                return None
            piece = getattr(player_core, "current_piece", None)
            grid = player_core.grid
            cfg = env.config

        if piece is None:
            return None

        candidates = generate_candidates(grid, piece, cfg)
        if not candidates:
            return None

        selected = max(candidates, key=lambda item: float(item.get("surface_score", 0.0)))
        rotation_steps = int(selected.get("rotation_steps", 0))
        target_x = int(selected.get("target_x", piece.x))

        # 统一动作空间：1左 2右 3转 5硬降
        if rotation_steps > 0:
            return 3
        if piece.x < target_x:
            return 2
        if piece.x > target_x:
            return 1
        return 5
    except Exception:
        return None


def _build_env(opt, seed: int, max_episode_steps: int):
    if opt.mode == "TRADITIONAL":
        return SharedTetrisEnv(
            mode_key=opt.mode,
            config=CONFIG,
            seed=seed,
            action_interval_ms=opt.action_interval_ms,
            max_episode_steps=max_episode_steps,
            init_pygame=False,
        )
    return TetrisEnv(
        config=CONFIG,
        seed=seed,
        max_episode_steps=max_episode_steps,
        opponent_mode="heuristic",
    )


def _save_inference_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)


def _dated_best_model_path(model_dir: Path) -> Path:
    date_tag = datetime.now().strftime("%Y%m%d")
    return model_dir / f"best_{date_tag}.pt"


def _save_training_checkpoint(
    path: Path,
    policy_net,
    target_net,
    optimizer,
    scaler,
    episode: int,
    global_step: int,
    best_eval_reward: float,
    epsilon: float,
    state_dim: int,
    action_dim: int,
):
    payload = {
        "model_state_dict": policy_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "episode": int(episode),
        "global_step": int(global_step),
        "best_eval_reward": float(best_eval_reward),
        "epsilon": float(epsilon),
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_training_checkpoint(path: Path, policy_net, target_net, optimizer, scaler, state_dim: int, action_dim: int):
    if not path.exists():
        return 1, 0, float("-inf"), 0.0

    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[WARN] 加载 checkpoint 失败，将从头训练: {exc}")
        return 1, 0, float("-inf"), 0.0

    ckpt_state_dim = int(checkpoint.get("state_dim", -1))
    ckpt_action_dim = int(checkpoint.get("action_dim", -1))
    if ckpt_state_dim != state_dim or ckpt_action_dim != action_dim:
        print(
            "[WARN] checkpoint 维度不匹配，已忽略恢复: "
            f"ckpt=({ckpt_state_dim},{ckpt_action_dim}) current=({state_dim},{action_dim})"
        )
        return 1, 0, float("-inf"), 0.0

    try:
        policy_net.load_state_dict(checkpoint["model_state_dict"])
        target_net.load_state_dict(checkpoint.get("target_state_dict", checkpoint["model_state_dict"]))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler is not None and scaler_state:
            scaler.load_state_dict(scaler_state)
    except Exception as exc:
        print(f"[WARN] checkpoint 参数恢复失败，将从头训练: {exc}")
        return 1, 0, float("-inf"), 0.0

    start_episode = int(checkpoint.get("episode", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_reward = float(checkpoint.get("best_eval_reward", checkpoint.get("best_reward", float("-inf"))))
    epsilon = float(checkpoint.get("epsilon", 0.0))
    return max(1, start_episode), global_step, best_reward, epsilon


def _to_tensor_batch(states: list[list[float]], device):
    return torch.tensor(states, dtype=torch.float32, device=device)


def _extract_grid_for_heuristic(env, mode: str):
    normalized = str(mode).strip().upper()
    if normalized == "TRADITIONAL" and hasattr(env, "match"):
        core = getattr(env.match, "core", None)
        if core is not None and hasattr(core, "grid"):
            return core.grid

    player_core = getattr(env, "player_core", None)
    if player_core is not None and hasattr(player_core, "grid"):
        return player_core.grid
    return None


def _heuristic_potential(env, mode: str) -> float:
    grid = _extract_grid_for_heuristic(env, mode)
    if grid is None:
        return 0.0

    score = float(evaluate_grid(grid, lines_cleared=0))
    heights, holes = board_profile(grid)
    max_height = max(heights, default=0)
    score -= float(max_height) * 8.0
    score -= float(holes) * 12.0
    return float(score)


def _evaluate_policy(policy_net, eval_env, device, episodes: int, base_seed: int, max_steps: int) -> tuple[float, float]:
    policy_net.eval()
    rewards: list[float] = []
    steps_list: list[float] = []
    with torch.no_grad():
        for idx in range(max(1, int(episodes))):
            state = eval_env.reset(seed=base_seed + idx)
            total_reward = 0.0
            steps = 0
            done = False
            while not done and steps < max_steps:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = int(q_values.argmax(dim=1).item())
                outcome = eval_env.step(action)
                state = outcome.state
                total_reward += float(outcome.reward)
                done = bool(outcome.done)
                steps += 1

            rewards.append(total_reward)
            steps_list.append(float(steps))

    policy_net.train()
    avg_reward = float(sum(rewards) / max(1, len(rewards)))
    avg_steps = float(sum(steps_list) / max(1, len(steps_list)))
    return avg_reward, avg_steps


def _build_grad_scaler(amp_enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def _autocast_ctx(device, amp_enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=amp_enabled)
    return torch.cuda.amp.autocast(enabled=amp_enabled)


def _activate_slot(slot: EnvSlot, slot_idx: int, episode_id: int, opt, heuristic_decay_episodes: int):
    slot.episode_id = int(episode_id)
    slot.episode_cap = _episode_cap(episode_id, opt)
    slot.heur_weight = _heuristic_weight_by_episode(int(episode_id), int(heuristic_decay_episodes))
    if hasattr(slot.env, "max_episode_steps"):
        slot.env.max_episode_steps = int(slot.episode_cap)

    seed = int(opt.seed + episode_id * 97 + slot_idx * 997)
    slot.state = slot.env.reset(seed=seed)
    slot.steps = 0
    slot.reward = 0.0
    slot.env_reward = 0.0
    slot.heur_bonus = 0.0
    slot.phi_prev = _heuristic_potential(slot.env, opt.mode) if slot.heur_weight > 0 else 0.0


def train(opt):
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    device = _resolve_device(opt.device)
    amp_enabled = bool(opt.amp and device.type == "cuda")

    warmup_cap = _episode_cap(1, opt)
    probe_env = _build_env(opt, seed=opt.seed, max_episode_steps=warmup_cap)
    init_state = probe_env.reset(seed=opt.seed)
    state_dim = len(init_state)
    action_dim = int(probe_env.action_size())

    hidden_dims = (int(opt.hidden_dim1), int(opt.hidden_dim2))
    policy_net = DeepQNetwork(input_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims).to(device)
    target_net = DeepQNetwork(input_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=opt.lr)
    scaler = _build_grad_scaler(amp_enabled)

    model_dir = mode_model_dir(opt.saved_path, opt.mode)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / CHECKPOINT_NAME
    best_model_path = model_dir / "best.pt"
    last_model_path = model_dir / "last.pt"
    history_path = model_dir / "training_history_framework.json"

    log_path = Path(opt.log_path) / opt.mode.lower()
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_path)) if SummaryWriter is not None else _NoOpWriter()

    start_episode = 1
    global_step = 0
    best_eval_reward = float("-inf")
    latest_epsilon = float(opt.initial_epsilon)
    if opt.resume:
        start_episode, global_step, best_eval_reward, latest_epsilon = _load_training_checkpoint(
            checkpoint_path,
            policy_net,
            target_net,
            optimizer,
            scaler,
            state_dim,
            action_dim,
        )

    replay = PrioritizedReplayBuffer(
        capacity=opt.replay_memory_size,
        state_dim=state_dim,
        prioritized=opt.prioritized_replay,
        alpha=opt.per_alpha,
    )

    eval_env = _build_env(opt, seed=opt.seed + 123456, max_episode_steps=int(opt.curriculum_end_steps))
    heuristic_decay_episodes = _resolve_heuristic_decay_episodes(opt)
    teacher_decay_episodes = _resolve_teacher_decay_episodes(opt)

    requested_envs = max(1, int(opt.num_envs))
    remaining_episodes = max(0, int(opt.episodes) - int(start_episode) + 1)
    active_env_count = max(1, min(requested_envs, remaining_episodes))
    slots = [EnvSlot(env=_build_env(opt, seed=opt.seed + 2000 + i * 13, max_episode_steps=warmup_cap)) for i in range(active_env_count)]

    next_episode_id = int(start_episode)
    for idx, slot in enumerate(slots):
        if next_episode_id <= opt.episodes:
            _activate_slot(slot, idx, next_episode_id, opt, heuristic_decay_episodes)
            next_episode_id += 1

    mode_label = "模式2(TRADITIONAL)" if opt.mode == "TRADITIONAL" else opt.mode
    print(
        "[CONFIG] "
        f"mode={opt.mode} [{mode_label}] device={device} amp={'on' if amp_enabled else 'off'} "
        f"num_envs={active_env_count} prioritized_replay={'on' if opt.prioritized_replay else 'off'} "
        f"heuristic_guidance={'on' if opt.heuristic_guidance else 'off'} "
        f"heuristic_decay_episodes={heuristic_decay_episodes} "
        f"teacher_guidance={'on' if opt.teacher_guidance else 'off'} "
        f"teacher_distill_coef={float(opt.teacher_distill_coef):.3f}"
    )

    history: list[dict[str, float | int]] = []
    recent_rewards = deque(maxlen=max(5, int(opt.ma_window)))
    recent_losses = deque(maxlen=max(5, int(opt.ma_window)))

    latest_loss = 0.0
    latest_distill_loss = 0.0
    completed_episodes = int(start_episode - 1)
    last_finished_episode = int(start_episode - 1)

    try:
        while completed_episodes < int(opt.episodes):
            active_indices = [idx for idx, slot in enumerate(slots) if slot.episode_id is not None]
            if not active_indices:
                break

            epsilon = _epsilon_by_step(global_step, opt)
            latest_epsilon = float(epsilon)

            chosen_actions: dict[int, int] = {}
            teacher_actions_for_step: dict[int, int] = {}
            greedy_states: list[list[float]] = []
            greedy_slot_indices: list[int] = []

            for idx in active_indices:
                slot = slots[idx]
                if slot.state is None:
                    continue
                if random.random() < epsilon:
                    mixed_teacher_action: int | None = None
                    if bool(opt.teacher_guidance) and slot.episode_id is not None:
                        teacher_prob = _teacher_mix_prob_by_episode(int(slot.episode_id), teacher_decay_episodes, opt)
                        if random.random() < teacher_prob:
                            mixed_teacher_action = _teacher_action(slot.env, opt.mode)

                    if mixed_teacher_action is not None and 0 <= mixed_teacher_action < action_dim:
                        chosen_actions[idx] = int(mixed_teacher_action)
                        teacher_actions_for_step[idx] = int(mixed_teacher_action)
                    if hasattr(slot.env, "sample_action"):
                        chosen_actions.setdefault(idx, int(slot.env.sample_action()))
                    else:
                        chosen_actions.setdefault(idx, random.randrange(action_dim))
                else:
                    greedy_slot_indices.append(idx)
                    greedy_states.append(slot.state)

            if greedy_states:
                state_batch = _to_tensor_batch(greedy_states, device)
                with torch.no_grad():
                    with _autocast_ctx(device, amp_enabled):
                        q_values = policy_net(state_batch)
                    greedy_actions = q_values.argmax(dim=1).tolist()
                for idx, act in zip(greedy_slot_indices, greedy_actions):
                    chosen_actions[idx] = int(act)

            done_indices: list[int] = []
            for idx in active_indices:
                slot = slots[idx]
                if slot.state is None:
                    continue

                action = int(chosen_actions[idx])
                outcome = slot.env.step(action)
                next_state = outcome.state
                env_reward = float(outcome.reward)
                teacher_action = int(teacher_actions_for_step.get(idx, -1))

                should_label = (
                    bool(opt.teacher_guidance)
                    and int(opt.teacher_label_stride) > 0
                    and (slot.steps % int(opt.teacher_label_stride) == 0)
                )
                if teacher_action < 0 and should_label:
                    labeled_action = _teacher_action(slot.env, opt.mode)
                    if labeled_action is not None and 0 <= int(labeled_action) < action_dim:
                        teacher_action = int(labeled_action)

                heuristic_bonus = 0.0
                if slot.heur_weight > 0:
                    phi_next = _heuristic_potential(slot.env, opt.mode)
                    heuristic_bonus = (
                        float(opt.heuristic_scale)
                        * float(slot.heur_weight)
                        * (float(opt.gamma) * phi_next - slot.phi_prev)
                    )
                    clip_value = abs(float(opt.heuristic_clip))
                    if clip_value > 0:
                        heuristic_bonus = max(-clip_value, min(clip_value, heuristic_bonus))
                    slot.phi_prev = phi_next

                reward = env_reward + heuristic_bonus
                slot.steps += 1
                done = bool(outcome.done) or slot.steps >= slot.episode_cap

                replay.append(slot.state, action, reward, next_state, done, teacher_action=teacher_action)

                slot.state = next_state
                slot.reward += float(reward)
                slot.env_reward += float(env_reward)
                slot.heur_bonus += float(heuristic_bonus)

                global_step += 1
                if done:
                    done_indices.append(idx)

            if (
                replay.can_sample(opt.batch_size)
                and global_step >= int(opt.warmup_steps)
                and global_step % int(opt.train_frequency) == 0
            ):
                beta = _per_beta_by_step(global_step, opt)
                updates = max(1, len(active_indices))
                for _ in range(updates):
                    (
                        batch_indices,
                        state_batch_cpu,
                        action_batch_cpu,
                        reward_batch_cpu,
                        next_state_batch_cpu,
                        done_batch_cpu,
                        teacher_action_batch_cpu,
                        weight_batch_cpu,
                    ) = replay.sample(opt.batch_size, beta)

                    state_batch = state_batch_cpu.to(device)
                    action_batch = action_batch_cpu.to(device)
                    reward_batch = reward_batch_cpu.to(device)
                    next_state_batch = next_state_batch_cpu.to(device)
                    done_batch = done_batch_cpu.to(device)
                    teacher_action_batch = teacher_action_batch_cpu.to(device)
                    weight_batch = weight_batch_cpu.to(device)

                    with _autocast_ctx(device, amp_enabled):
                        all_q_values = policy_net(state_batch)
                        q_values = all_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                        with torch.no_grad():
                            next_actions = policy_net(next_state_batch).argmax(dim=1, keepdim=True)
                            next_q = target_net(next_state_batch).gather(1, next_actions).squeeze(1)
                            target = reward_batch + (1.0 - done_batch) * float(opt.gamma) * next_q

                        td_error = target - q_values
                        element_loss = F.smooth_l1_loss(q_values, target, reduction="none")
                        td_loss = (element_loss * weight_batch).mean()

                        distill_loss = torch.tensor(0.0, device=device)
                        if bool(opt.teacher_guidance) and float(opt.teacher_distill_coef) > 0:
                            valid_teacher = (teacher_action_batch >= 0) & (teacher_action_batch < action_dim)
                            if valid_teacher.any():
                                distill_loss = F.cross_entropy(
                                    all_q_values[valid_teacher],
                                    teacher_action_batch[valid_teacher],
                                )

                        loss = td_loss + float(opt.teacher_distill_coef) * distill_loss

                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()

                    latest_loss = float(loss.item())
                    latest_distill_loss = float(distill_loss.item())
                    replay.update_priorities(batch_indices, td_error)

            if global_step % int(opt.target_sync_interval) == 0:
                target_net.load_state_dict(policy_net.state_dict())

            for idx in done_indices:
                slot = slots[idx]
                if slot.episode_id is None:
                    continue

                episode_id = int(slot.episode_id)
                completed_episodes += 1
                last_finished_episode = episode_id

                recent_rewards.append(float(slot.reward))
                recent_losses.append(float(latest_loss))
                ma_reward = sum(recent_rewards) / max(1, len(recent_rewards))
                ma_loss = sum(recent_losses) / max(1, len(recent_losses))

                history_item = {
                    "episode": episode_id,
                    "reward": float(slot.reward),
                    "env_reward": float(slot.env_reward),
                    "heuristic_bonus": float(slot.heur_bonus),
                    "heuristic_weight": float(slot.heur_weight),
                    "loss": float(latest_loss),
                    "steps": int(slot.steps),
                    "episode_cap": int(slot.episode_cap),
                    "epsilon": float(latest_epsilon),
                    "global_step": int(global_step),
                }
                history.append(history_item)

                writer.add_scalar("Train/Reward", slot.reward, episode_id)
                writer.add_scalar("Train/EnvReward", slot.env_reward, episode_id)
                writer.add_scalar("Train/HeuristicBonus", slot.heur_bonus, episode_id)
                writer.add_scalar("Train/HeuristicWeight", slot.heur_weight, episode_id)
                writer.add_scalar("Train/Loss", latest_loss, episode_id)
                writer.add_scalar("Train/DistillLoss", latest_distill_loss, episode_id)
                writer.add_scalar("Train/Epsilon", latest_epsilon, episode_id)
                writer.add_scalar("Train/Steps", slot.steps, episode_id)
                writer.add_scalar("Train/MAReward", ma_reward, episode_id)
                writer.add_scalar("Train/MALoss", ma_loss, episode_id)

                if episode_id == 1 or (opt.eval_interval > 0 and episode_id % int(opt.eval_interval) == 0):
                    eval_reward, eval_steps = _evaluate_policy(
                        policy_net=policy_net,
                        eval_env=eval_env,
                        device=device,
                        episodes=int(opt.eval_episodes),
                        base_seed=int(opt.seed + 100000 + episode_id * 17),
                        max_steps=int(opt.curriculum_end_steps),
                    )
                    writer.add_scalar("Eval/AvgReward", eval_reward, episode_id)
                    writer.add_scalar("Eval/AvgSteps", eval_steps, episode_id)
                    if eval_reward > best_eval_reward:
                        best_eval_reward = float(eval_reward)
                        _save_inference_model(policy_net, best_model_path)
                        _save_inference_model(policy_net, _dated_best_model_path(model_dir))
                    print(
                        f"[EVAL] episode={episode_id} avg_reward={eval_reward:.3f} "
                        f"avg_steps={eval_steps:.1f} best_eval_reward={best_eval_reward:.3f}"
                    )

                _save_inference_model(policy_net, last_model_path)
                _save_training_checkpoint(
                    checkpoint_path,
                    policy_net,
                    target_net,
                    optimizer,
                    scaler,
                    episode=episode_id,
                    global_step=global_step,
                    best_eval_reward=best_eval_reward,
                    epsilon=latest_epsilon,
                    state_dim=state_dim,
                    action_dim=action_dim,
                )

                if int(opt.save_interval) > 0 and episode_id % int(opt.save_interval) == 0:
                    periodic_path = model_dir / f"episode_{episode_id}.pt"
                    _save_inference_model(policy_net, periodic_path)

                print(
                    f"[TRAIN] episode={episode_id}/{opt.episodes} "
                    f"reward={slot.reward:.3f} env_reward={slot.env_reward:.3f} "
                    f"heur_bonus={slot.heur_bonus:.3f} heur_w={slot.heur_weight:.3f} "
                    f"loss={latest_loss:.4f} distill={latest_distill_loss:.4f} "
                    f"ma_reward={ma_reward:.3f} ma_loss={ma_loss:.4f} "
                    f"epsilon={latest_epsilon:.4f} steps={slot.steps} cap={slot.episode_cap} "
                    f"global_step={global_step}"
                )

                if next_episode_id <= int(opt.episodes):
                    _activate_slot(slot, idx, next_episode_id, opt, heuristic_decay_episodes)
                    next_episode_id += 1
                else:
                    slot.episode_id = None
                    slot.state = None

        history_payload = {
            "mode": opt.mode,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "best_eval_reward": best_eval_reward,
            "history": history,
        }
        history_path.write_text(json.dumps(history_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    except KeyboardInterrupt:
        print("\n[INFO] 训练被中断，正在保存当前进度...")
        _save_inference_model(policy_net, last_model_path)
        _save_training_checkpoint(
            checkpoint_path,
            policy_net,
            target_net,
            optimizer,
            scaler,
            episode=max(1, last_finished_episode),
            global_step=global_step,
            best_eval_reward=best_eval_reward,
            epsilon=latest_epsilon,
            state_dim=state_dim,
            action_dim=action_dim,
        )
        history_payload = {
            "mode": opt.mode,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "best_eval_reward": best_eval_reward,
            "history": history,
        }
        history_path.write_text(json.dumps(history_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        writer.close()

    print(f"[DONE] best={best_model_path} last={last_model_path}")


if __name__ == "__main__":
    options = get_args()
    train(options)
