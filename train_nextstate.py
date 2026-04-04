from __future__ import annotations

import argparse
import json
import random
import signal
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    try:
        from tensorboardX import SummaryWriter  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        SummaryWriter = None

from deep_q_network import DeepQNetwork
from next_state_features import next_state_feature_size
from settings import CONFIG
from tetris_env import TetrisEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train the next-state Tetris value model")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="models/next_state")
    parser.add_argument("--checkpoint_name", type=str, default="latest_checkpoint.pth")
    parser.add_argument("--log_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=20260404)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--replay_capacity", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=3000)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_steps", type=int, default=30000)
    parser.add_argument("--target_sync_steps", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=8)
    parser.add_argument("--eval_seed_base", type=int, default=20260501)
    parser.add_argument("--curriculum_episodes", type=int, default=1200)
    parser.add_argument("--opponent_start_interval_ms", type=int, default=420)
    parser.add_argument("--opponent_end_interval_ms", type=int, default=180)
    parser.add_argument("--opponent_start_mistake", type=float, default=0.55)
    parser.add_argument("--opponent_end_mistake", type=float, default=0.18)
    parser.add_argument("--max_episode_steps", type=int, default=6000)
    parser.add_argument("--ma_window", type=int, default=50)
    parser.add_argument("--teacher_prob_start", type=float, default=0.70)
    parser.add_argument("--teacher_prob_end", type=float, default=0.08)
    parser.add_argument("--teacher_decay_episodes", type=int, default=900)
    parser.add_argument("--teacher_value_coef", type=float, default=0.12)
    parser.add_argument("--heuristic_bonus_scale", type=float, default=3.0)
    parser.add_argument("--heuristic_decay_episodes", type=int, default=700)
    parser.add_argument("--per_alpha", type=float, default=0.6)
    parser.add_argument("--per_beta_start", type=float, default=0.4)
    parser.add_argument("--per_beta_steps", type=int, default=120000)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


class _NoOpWriter:
    def add_scalar(self, *_args, **_kwargs):
        return

    def close(self):
        return


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = max(1, int(capacity))
        self.alpha = float(alpha)
        self.data: list[tuple | None] = [None] * self.capacity
        self.priorities = torch.ones(self.capacity, dtype=torch.float32)
        self.size = 0
        self.position = 0

    def __len__(self):
        return self.size

    def append(self, transition):
        self.data[self.position] = transition
        max_priority = float(self.priorities[: self.size].max().item()) if self.size > 0 else 1.0
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        if self.size < int(batch_size):
            raise ValueError("经验不足，无法采样")
        pri = self.priorities[: self.size].clamp(min=1e-5).pow(self.alpha)
        probs = pri / pri.sum()
        indices = torch.multinomial(probs, int(batch_size), replacement=self.size < int(batch_size))
        weights = (self.size * probs[indices]).pow(-float(beta))
        weights = weights / weights.max().clamp(min=1.0)
        batch = [self.data[int(idx)] for idx in indices.tolist()]
        return indices, batch, weights

    def update_priorities(self, indices: torch.Tensor, td_error: torch.Tensor):
        cpu_indices = indices.detach().cpu().long()
        self.priorities[cpu_indices] = td_error.detach().abs().cpu().float() + 1e-5


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()


def _epsilon_by_step(global_step: int, args) -> float:
    if global_step >= args.epsilon_decay_steps:
        return float(args.epsilon_end)
    progress = float(global_step) / float(max(1, args.epsilon_decay_steps))
    return float(args.epsilon_start + (args.epsilon_end - args.epsilon_start) * progress)


def _make_grad_scaler(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def _per_beta_by_step(global_step: int, args) -> float:
    progress = min(1.0, float(global_step) / float(max(1, args.per_beta_steps)))
    return float(args.per_beta_start + (1.0 - args.per_beta_start) * progress)


def _curriculum_progress(episode: int, args) -> float:
    if args.curriculum_episodes <= 1:
        return 1.0
    return max(0.0, min(1.0, float(episode - 1) / float(args.curriculum_episodes - 1)))


def _teacher_prob_by_episode(episode: int, args) -> float:
    decay_episodes = max(1, int(args.teacher_decay_episodes))
    progress = min(1.0, float(max(0, episode - 1)) / float(max(1, decay_episodes - 1)))
    return float(args.teacher_prob_start + (args.teacher_prob_end - args.teacher_prob_start) * progress)


def _heuristic_weight_by_episode(episode: int, args) -> float:
    decay_episodes = max(1, int(args.heuristic_decay_episodes))
    progress = min(1.0, float(max(0, episode - 1)) / float(max(1, decay_episodes - 1)))
    return float(max(0.0, 1.0 - progress))


def _heuristic_target(surface_score: float) -> float:
    return float(torch.tanh(torch.tensor(surface_score / 400.0)).item() * 12.0)


def _apply_curriculum(env: TetrisEnv, episode: int, args, full_trap_chance: float, full_trap_threshold: int):
    progress = _curriculum_progress(episode, args)
    difficulty_progress = progress ** 1.5
    trap_progress = max(0.0, (progress - 0.35) / 0.65)
    env.opponent_action_interval_ms = int(
        round(
            args.opponent_start_interval_ms
            + (args.opponent_end_interval_ms - args.opponent_start_interval_ms) * difficulty_progress
        )
    )
    env.opponent_mistake_chance = float(
        args.opponent_start_mistake
        + (args.opponent_end_mistake - args.opponent_start_mistake) * difficulty_progress
    )
    env.config.versus_trap_ai_use_chance = float(full_trap_chance * trap_progress)
    env.config.versus_trap_ai_use_threshold = int(
        round(9999 + (full_trap_threshold - 9999) * trap_progress)
    )


def _collect_candidate_states(env: TetrisEnv):
    candidates = env.get_next_state_candidates()
    if not candidates:
        return [], [], []
    keys = [item["key"] for item in candidates]
    states = [item["state"] for item in candidates]
    scores = [float(item["surface_score"]) for item in candidates]
    return keys, states, scores


def _teacher_index(scores) -> int | None:
    if not scores:
        return None
    return int(max(range(len(scores)), key=lambda idx: scores[idx]))


def _select_training_action(model, states, scores, epsilon: float, teacher_prob: float, device: torch.device, amp_enabled: bool) -> tuple[int, bool]:
    teacher_idx = _teacher_index(scores)
    if teacher_idx is not None and random.random() < teacher_prob:
        return teacher_idx, True
    return _select_action(model, states, epsilon=epsilon, device=device, amp_enabled=amp_enabled), False


def _score_state_batch(model, states, device: torch.device, amp_enabled: bool):
    if not states:
        return torch.empty(0, device=device)
    tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
    with torch.no_grad():
        with _autocast_context(device, amp_enabled):
            values = model(tensor).squeeze(-1)
    return values


def _select_action(model, states, epsilon: float, device: torch.device, amp_enabled: bool) -> int:
    if len(states) == 1:
        return 0
    if random.random() < epsilon:
        return random.randrange(len(states))
    values = _score_state_batch(model, states, device, amp_enabled)
    return int(torch.argmax(values).item())


def _compute_target_values(batch, policy_net, target_net, gamma: float, device: torch.device, amp_enabled: bool):
    targets = []
    with torch.no_grad():
        for _, reward, next_candidates, done, _teacher_value in batch:
            target = float(reward)
            if not done and next_candidates:
                next_tensor = torch.as_tensor(next_candidates, dtype=torch.float32, device=device)
                with _autocast_context(device, amp_enabled):
                    policy_values = policy_net(next_tensor).squeeze(-1)
                    best_index = int(torch.argmax(policy_values).item())
                    target_values = target_net(next_tensor).squeeze(-1)
                    next_value = float(target_values[best_index].item())
                target += gamma * next_value
            targets.append(target)
    return torch.as_tensor(targets, dtype=torch.float32, device=device)


def _run_evaluation(model, args, device: torch.device, amp_enabled: bool):
    eval_config = replace(CONFIG)
    env = TetrisEnv(
        config=eval_config,
        opponent_mode="heuristic",
        opponent_action_interval_ms=args.opponent_end_interval_ms,
        opponent_mistake_chance=args.opponent_end_mistake,
        max_episode_steps=args.max_episode_steps,
    )
    eval_config.versus_trap_ai_use_chance = CONFIG.versus_trap_ai_use_chance
    eval_config.versus_trap_ai_use_threshold = CONFIG.versus_trap_ai_use_threshold

    total_reward = 0.0
    total_lines = 0.0
    total_steps = 0.0
    clears = 0

    was_training = model.training
    model.eval()

    for idx in range(args.eval_episodes):
        env.reset(seed=args.eval_seed_base + idx)
        done = False
        episode_reward = 0.0

        while not done:
            keys, states, _scores = _collect_candidate_states(env)
            if not keys:
                outcome = env.step(0)
                episode_reward += float(outcome.reward)
                done = bool(outcome.done)
                continue

            action_idx = _select_action(model, states, epsilon=0.0, device=device, amp_enabled=amp_enabled)
            outcome = env.step_next_state(*keys[action_idx])
            episode_reward += float(outcome.reward)
            done = bool(outcome.done)

        total_reward += episode_reward
        total_lines += float(env.player_core.lines_cleared_total)
        total_steps += float(env.episode_step)
        if env.player_core.lines_cleared_total > 0:
            clears += 1

    if was_training:
        model.train()

    episodes = max(1, args.eval_episodes)
    return {
        "avg_reward": total_reward / episodes,
        "avg_lines": total_lines / episodes,
        "avg_steps": total_steps / episodes,
        "clear_rate": clears / episodes,
    }


def _save_history(history_path: Path, history):
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_training_curve(history, output_path: Path):
    if not history:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return str(exc)

    episodes = [int(item["episode"]) for item in history]
    train_reward = [float(item.get("train_reward", 0.0)) for item in history]
    ma_reward = [float(item.get("ma_reward", 0.0)) for item in history]
    train_lines = [float(item.get("train_lines", 0.0)) for item in history]
    ma_lines = [float(item.get("ma_lines", 0.0)) for item in history]
    train_loss = [float(item.get("avg_loss", 0.0)) for item in history]

    eval_episodes = [int(item["episode"]) for item in history if "eval_avg_reward" in item]
    eval_reward = [float(item["eval_avg_reward"]) for item in history if "eval_avg_reward" in item]
    eval_lines = [float(item["eval_avg_lines"]) for item in history if "eval_avg_lines" in item]
    eval_clear = [float(item["eval_clear_rate"]) for item in history if "eval_clear_rate" in item]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(episodes, train_reward, color="#5B8FF9", alpha=0.35, linewidth=1.2, label="Train Reward")
    axes[0, 0].plot(episodes, ma_reward, color="#1D39C4", linewidth=2.0, label="MA Reward")
    if eval_episodes:
        axes[0, 0].plot(eval_episodes, eval_reward, color="#13C2C2", linewidth=2.0, label="Eval Reward")
    axes[0, 0].set_title("Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(episodes, train_lines, color="#73D13D", alpha=0.35, linewidth=1.2, label="Train Lines")
    axes[0, 1].plot(episodes, ma_lines, color="#389E0D", linewidth=2.0, label="MA Lines")
    if eval_episodes:
        axes[0, 1].plot(eval_episodes, eval_lines, color="#FAAD14", linewidth=2.0, label="Eval Lines")
    axes[0, 1].set_title("Lines Cleared")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(episodes, train_loss, color="#722ED1", linewidth=1.8)
    axes[1, 0].set_title("Loss")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].grid(alpha=0.25)

    if eval_episodes:
        axes[1, 1].plot(eval_episodes, eval_clear, color="#EB2F96", linewidth=2.0, label="Eval Clear Rate")
    axes[1, 1].set_title("Eval Clear Rate")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return None


def _best_signature_from_history(history) -> tuple[float, float, float]:
    best = (-float("inf"), -float("inf"), -float("inf"))
    for item in history:
        if "eval_avg_lines" not in item:
            continue
        signature = (
            float(item.get("eval_avg_lines", float("-inf"))),
            float(item.get("eval_avg_reward", float("-inf"))),
            float(item.get("eval_avg_steps", float("-inf"))),
        )
        if signature > best:
            best = signature
    return best


def _save_training_checkpoint(
    checkpoint_path: Path,
    *,
    model,
    target_model,
    optimizer,
    scaler,
    episode: int,
    global_step: int,
    update_step: int,
    best_eval_signature: tuple[float, float, float],
    history,
    state_dim: int,
):
    payload = {
        "model_state_dict": model.state_dict(),
        "target_state_dict": target_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "episode": int(episode),
        "global_step": int(global_step),
        "update_step": int(update_step),
        "best_eval_signature": [float(item) for item in best_eval_signature],
        "history": history,
        "state_dim": int(state_dim),
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def _load_json_history(path: Path):
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _load_state_dict_file(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _try_resume_training(
    checkpoint_path: Path,
    *,
    last_model_path: Path,
    history_path: Path,
    model,
    target_model,
    optimizer,
    scaler,
    state_dim: int,
):
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        ckpt_state_dim = int(checkpoint.get("state_dim", -1))
        if ckpt_state_dim != state_dim:
            print(
                f"[WARN] checkpoint 维度不匹配，跳过恢复: ckpt={ckpt_state_dim} current={state_dim}"
            )
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            target_model.load_state_dict(checkpoint.get("target_state_dict", checkpoint["model_state_dict"]))
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler_state = checkpoint.get("scaler_state_dict")
            if scaler is not None and scaler_state:
                scaler.load_state_dict(scaler_state)

            try:
                random.setstate(checkpoint["python_random_state"])
            except Exception:
                pass
            try:
                torch.set_rng_state(checkpoint["torch_rng_state"])
            except Exception:
                pass
            try:
                cuda_rng_state = checkpoint.get("cuda_rng_state")
                if torch.cuda.is_available() and cuda_rng_state:
                    torch.cuda.set_rng_state_all(cuda_rng_state)
            except Exception:
                pass

            history = checkpoint.get("history", [])
            best_signature_raw = checkpoint.get("best_eval_signature", None)
            if isinstance(best_signature_raw, (list, tuple)) and len(best_signature_raw) == 3:
                best_signature = tuple(float(item) for item in best_signature_raw)
            else:
                best_signature = _best_signature_from_history(history)

            return {
                "resumed": True,
                "mode": "checkpoint",
                "start_episode": int(checkpoint.get("episode", 0)) + 1,
                "global_step": int(checkpoint.get("global_step", 0)),
                "update_step": int(checkpoint.get("update_step", 0)),
                "history": history,
                "best_eval_signature": best_signature,
            }

    if last_model_path.exists():
        try:
            state_dict = _load_state_dict_file(last_model_path)
            model.load_state_dict(state_dict)
            target_model.load_state_dict(state_dict)
            history = _load_json_history(history_path)
            return {
                "resumed": True,
                "mode": "weights_only",
                "start_episode": int(history[-1]["episode"]) + 1 if history else 1,
                "global_step": int(history[-1].get("global_step", 0)) if history else 0,
                "update_step": int(history[-1].get("update_step", 0)) if history else 0,
                "history": history,
                "best_eval_signature": _best_signature_from_history(history),
            }
        except Exception as exc:
            print(f"[WARN] last.pt 恢复失败，将从头训练: {exc}")

    return {
        "resumed": False,
        "mode": "fresh",
        "start_episode": 1,
        "global_step": 0,
        "update_step": 0,
        "history": [],
        "best_eval_signature": (-float("inf"), -float("inf"), -float("inf")),
    }


def _restore_recent_metrics(history, ma_window: int):
    recent_rewards = deque(maxlen=max(1, int(ma_window)))
    recent_lines = deque(maxlen=max(1, int(ma_window)))
    recent_steps = deque(maxlen=max(1, int(ma_window)))
    for item in history[-max(1, int(ma_window)) :]:
        recent_rewards.append(float(item.get("train_reward", 0.0)))
        recent_lines.append(float(item.get("train_lines", 0.0)))
        recent_steps.append(float(item.get("train_steps", 0.0)))
    return recent_rewards, recent_lines, recent_steps


def _move_optimizer_to_device(optimizer, device: torch.device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _install_interrupt_handler():
    state = {"requested": False, "count": 0}
    previous_handler = signal.getsignal(signal.SIGINT)

    def _handle_interrupt(_signum, _frame):
        state["count"] += 1
        if state["count"] == 1:
            state["requested"] = True
            print("\n[INTERRUPT] 收到 Ctrl+C，将在当前回合结束后保存并退出。再次按 Ctrl+C 可立即中断。")
            return
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_interrupt)
    return state, previous_handler


def train():
    args = parse_args()
    _set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best.pt"
    last_model_path = save_dir / "last.pt"
    checkpoint_path = save_dir / args.checkpoint_name
    history_path = save_dir / "history.json"
    summary_path = save_dir / "training_summary.json"
    curve_path = save_dir / "training_curve.png"
    log_path = Path(args.log_path) if args.log_path else (save_dir / "tensorboard")

    device = torch.device(args.device)
    amp_enabled = device.type == "cuda"
    if amp_enabled and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if amp_enabled and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True

    training_config = replace(CONFIG)
    full_trap_chance = float(training_config.versus_trap_ai_use_chance)
    full_trap_threshold = int(training_config.versus_trap_ai_use_threshold)
    training_config.versus_trap_ai_use_chance = 0.0
    training_config.versus_trap_ai_use_threshold = 9999

    env = TetrisEnv(
        config=training_config,
        opponent_mode="heuristic",
        opponent_action_interval_ms=args.opponent_start_interval_ms,
        opponent_mistake_chance=args.opponent_start_mistake,
        max_episode_steps=args.max_episode_steps,
    )

    state_dim = next_state_feature_size(env.config.grid_cols)
    model = DeepQNetwork(input_dim=state_dim).to(device)
    target_model = DeepQNetwork(input_dim=state_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()
    scaler = _make_grad_scaler(device, amp_enabled)
    memory = PrioritizedReplayBuffer(capacity=args.replay_capacity, alpha=args.per_alpha)
    writer = SummaryWriter(str(log_path)) if SummaryWriter is not None else _NoOpWriter()

    resume_state = (
        _try_resume_training(
            checkpoint_path,
            last_model_path=last_model_path,
            history_path=history_path,
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            scaler=scaler,
            state_dim=state_dim,
        )
        if args.resume
        else {
            "resumed": False,
            "mode": "fresh",
            "start_episode": 1,
            "global_step": 0,
            "update_step": 0,
            "history": [],
            "best_eval_signature": (-float("inf"), -float("inf"), -float("inf")),
        }
    )
    _move_optimizer_to_device(optimizer, device)

    history = list(resume_state["history"])
    global_step = int(resume_state["global_step"])
    update_step = int(resume_state["update_step"])
    best_eval_signature = tuple(resume_state["best_eval_signature"])
    start_episode = int(resume_state["start_episode"])
    recent_rewards, recent_lines, recent_steps = _restore_recent_metrics(history, args.ma_window)
    elapsed_before_resume = float(history[-1].get("elapsed_sec", 0.0)) if history else 0.0
    started_at = time.time()
    interrupt_state, previous_sigint_handler = _install_interrupt_handler()
    interrupted = False

    print(
        f"[TRAIN] device={device} state_dim={state_dim} replay={args.replay_capacity} "
        f"warmup={args.warmup_steps} eval_every={args.eval_interval} log_dir={log_path} "
        f"resume={resume_state['mode']} start_episode={start_episode}"
    )

    try:
        for episode in range(start_episode, args.episodes + 1):
            _apply_curriculum(env, episode, args, full_trap_chance, full_trap_threshold)
            env.reset(seed=args.seed + episode)
            done = False
            stop_after_episode = False
            episode_reward = 0.0
            losses = []
            teacher_hits = 0
            teacher_prob = _teacher_prob_by_episode(episode, args)
            heuristic_weight = _heuristic_weight_by_episode(episode, args)

            while not done:
                keys, states, scores = _collect_candidate_states(env)
                if not keys:
                    outcome = env.step(0)
                    episode_reward += float(outcome.reward)
                    done = bool(outcome.done)
                    if interrupt_state["requested"]:
                        done = True
                        stop_after_episode = True
                    continue

                epsilon = _epsilon_by_step(global_step, args)
                action_idx, used_teacher = _select_training_action(
                    model,
                    states,
                    scores,
                    epsilon=epsilon,
                    teacher_prob=teacher_prob,
                    device=device,
                    amp_enabled=amp_enabled,
                )
                if used_teacher:
                    teacher_hits += 1
                chosen_key = keys[action_idx]
                chosen_state = states[action_idx]
                chosen_surface_score = scores[action_idx]

                outcome = env.step_next_state(*chosen_key)
                done = bool(outcome.done)
                heuristic_bonus = float(args.heuristic_bonus_scale) * heuristic_weight * _heuristic_target(chosen_surface_score) / 12.0
                reward = float(outcome.reward) + heuristic_bonus
                episode_reward += reward
                global_step += 1
                if interrupt_state["requested"]:
                    done = True
                    stop_after_episode = True

                _, next_candidate_states, _ = _collect_candidate_states(env)
                teacher_value = _heuristic_target(chosen_surface_score)
                memory.append((chosen_state, reward, next_candidate_states, done, teacher_value))

                if len(memory) >= max(args.batch_size, args.warmup_steps):
                    beta = _per_beta_by_step(global_step, args)
                    batch_indices, batch, batch_weights = memory.sample(args.batch_size, beta)
                    state_batch = torch.as_tensor([item[0] for item in batch], dtype=torch.float32, device=device)
                    target_batch = _compute_target_values(
                        batch,
                        policy_net=model,
                        target_net=target_model,
                        gamma=args.gamma,
                        device=device,
                        amp_enabled=amp_enabled,
                    )
                    teacher_value_batch = torch.as_tensor([item[4] for item in batch], dtype=torch.float32, device=device)
                    weight_batch = batch_weights.to(device)

                    optimizer.zero_grad(set_to_none=True)
                    with _autocast_context(device, amp_enabled):
                        preds = model(state_batch).squeeze(-1)
                        element_td = F.smooth_l1_loss(preds, target_batch, reduction="none")
                        td_error = target_batch - preds
                        td_loss = (element_td * weight_batch).mean()
                        distill_loss = F.smooth_l1_loss(preds, teacher_value_batch, reduction="mean")
                        loss = td_loss + float(args.teacher_value_coef) * heuristic_weight * distill_loss
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    update_step += 1
                    losses.append(float(loss.item()))
                    memory.update_priorities(batch_indices, td_error)

                    if global_step % args.target_sync_steps == 0:
                        target_model.load_state_dict(model.state_dict())
                        target_model.eval()

            torch.save(model.state_dict(), last_model_path)

            avg_loss = sum(losses) / len(losses) if losses else 0.0
            recent_rewards.append(float(episode_reward))
            recent_lines.append(float(env.player_core.lines_cleared_total))
            recent_steps.append(float(env.episode_step))
            ma_reward = sum(recent_rewards) / len(recent_rewards)
            ma_lines = sum(recent_lines) / len(recent_lines)
            ma_steps = sum(recent_steps) / len(recent_steps)
            progress = _curriculum_progress(episode, args)
            current_interval = env.opponent_action_interval_ms
            current_mistake = env.opponent_mistake_chance
            current_trap = env.config.versus_trap_ai_use_chance
            epsilon_now = _epsilon_by_step(global_step, args)

            record = {
                "episode": episode,
                "global_step": global_step,
                "update_step": update_step,
                "epsilon": epsilon_now,
                "avg_loss": avg_loss,
                "train_reward": float(episode_reward),
                "train_lines": float(env.player_core.lines_cleared_total),
                "train_steps": float(env.episode_step),
                "ma_reward": float(ma_reward),
                "ma_lines": float(ma_lines),
                "ma_steps": float(ma_steps),
                "teacher_hits": int(teacher_hits),
                "teacher_prob": float(teacher_prob),
                "heuristic_weight": float(heuristic_weight),
                "opponent_action_interval_ms": int(current_interval),
                "opponent_mistake_chance": float(current_mistake),
                "trap_chance": float(current_trap),
                "replay_size": len(memory),
                "progress": float(progress),
                "elapsed_sec": elapsed_before_resume + (time.time() - started_at),
                "interrupted": bool(stop_after_episode),
            }

            writer.add_scalar("Train/Reward", float(episode_reward), episode)
            writer.add_scalar("Train/Lines", float(env.player_core.lines_cleared_total), episode)
            writer.add_scalar("Train/Steps", float(env.episode_step), episode)
            writer.add_scalar("Train/Loss", float(avg_loss), episode)
            writer.add_scalar("Train/Epsilon", float(epsilon_now), episode)
            writer.add_scalar("Train/TeacherHits", int(teacher_hits), episode)
            writer.add_scalar("Train/TeacherProb", float(teacher_prob), episode)
            writer.add_scalar("Train/HeuristicWeight", float(heuristic_weight), episode)
            writer.add_scalar("Train/MAReward", float(ma_reward), episode)
            writer.add_scalar("Train/MALines", float(ma_lines), episode)
            writer.add_scalar("Train/MASteps", float(ma_steps), episode)

            print(
                f"[EP {episode:04d}/{args.episodes}] reward={episode_reward:8.2f} "
                f"lines={env.player_core.lines_cleared_total:4d} steps={env.episode_step:4d} "
                f"eps={epsilon_now:.3f} loss={avg_loss:.4f} "
                f"ma_reward={ma_reward:8.2f} ma_lines={ma_lines:5.2f} ma_steps={ma_steps:5.1f} "
                f"teacher={teacher_hits:3d} tprob={teacher_prob:.2f} h_w={heuristic_weight:.2f} "
                f"opp_ms={current_interval:3d} opp_mistake={current_mistake:.3f} trap={current_trap:.2f} "
                f"buffer={len(memory):6d} prog={progress:.2f}"
            )

            should_eval = (not stop_after_episode) and (episode % args.eval_interval == 0 or episode == args.episodes)
            if should_eval:
                metrics = _run_evaluation(model, args, device=device, amp_enabled=amp_enabled)
                signature = (
                    float(metrics["avg_lines"]),
                    float(metrics["avg_reward"]),
                    float(metrics["avg_steps"]),
                )
                is_best = signature > best_eval_signature
                if is_best:
                    best_eval_signature = signature
                    torch.save(model.state_dict(), best_model_path)

                record["eval_avg_reward"] = float(metrics["avg_reward"])
                record["eval_avg_lines"] = float(metrics["avg_lines"])
                record["eval_avg_steps"] = float(metrics["avg_steps"])
                record["eval_clear_rate"] = float(metrics["clear_rate"])
                record["is_best"] = bool(is_best)

                writer.add_scalar("Eval/AvgReward", float(metrics["avg_reward"]), episode)
                writer.add_scalar("Eval/AvgLines", float(metrics["avg_lines"]), episode)
                writer.add_scalar("Eval/AvgSteps", float(metrics["avg_steps"]), episode)
                writer.add_scalar("Eval/ClearRate", float(metrics["clear_rate"]), episode)

                summary_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

                print(
                    f"[EVAL] episode={episode} avg_reward={metrics['avg_reward']:.2f} "
                    f"avg_lines={metrics['avg_lines']:.2f} avg_steps={metrics['avg_steps']:.2f} "
                    f"clear_rate={metrics['clear_rate']:.2%} best={is_best}"
                )

            history.append(record)
            _save_history(history_path, history)
            summary_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            _save_training_checkpoint(
                checkpoint_path,
                model=model,
                target_model=target_model,
                optimizer=optimizer,
                scaler=scaler,
                episode=episode,
                global_step=global_step,
                update_step=update_step,
                best_eval_signature=best_eval_signature,
                history=history,
                state_dim=state_dim,
            )

            if stop_after_episode:
                interrupted = True
                print(f"[STOP] 已在 episode {episode} 结束后安全保存，可直接用相同命令继续训练。")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n[INTERRUPT] 训练被立即中断，正在尽量保存当前进度...")
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        target_model.load_state_dict(model.state_dict())
        target_model.eval()
        torch.save(model.state_dict(), last_model_path)

        last_episode = int(history[-1]["episode"]) if history else max(0, start_episode - 1)
        if history and not best_model_path.exists():
            torch.save(model.state_dict(), best_model_path)

        _save_history(history_path, history)
        _save_training_checkpoint(
            checkpoint_path,
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            scaler=scaler,
            episode=last_episode,
            global_step=global_step,
            update_step=update_step,
            best_eval_signature=best_eval_signature,
            history=history,
            state_dim=state_dim,
        )
        plot_error = _build_training_curve(history, curve_path)
        if plot_error is None:
            print(f"[PLOT] saved={curve_path}")
        else:
            print(f"[PLOT] skipped={plot_error}")

        status = "INTERRUPTED" if interrupted else "DONE"
        print(
            f"[{status}] best={best_model_path} last={last_model_path} checkpoint={checkpoint_path} "
            f"history={history_path} curve={curve_path} tensorboard={log_path}"
        )
        writer.close()


if __name__ == "__main__":
    train()
