from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from model_paths import resolve_existing_model_path
from rl_trainer import DQNTrainer
from settings import CONFIG


def _resolve_checkpoint_path(mode: str, checkpoint: str) -> Path:
    path = Path(checkpoint)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return resolve_existing_model_path(CONFIG.rl_model_dir, mode, filename=path.name)


def _load_checkpoint_payload(path: Path, device):
    if torch is None:
        raise RuntimeError("PyTorch 不可用，无法评估模型")

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Tetris DQN checkpoint")
    parser.add_argument(
        "--mode",
        type=str,
        default="TRADITIONAL",
        choices=("CLASSIC", "TRADITIONAL", "THREE_BODY"),
        help="评估目标模式",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best.pt",
        help="checkpoint 文件名或完整路径；默认读取当前模式目录下的 best.pt",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="评估局数",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="hard",
        choices=("curriculum", "hard"),
        help="对手评估档位",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("cpu", "cuda"),
        help="评估设备，默认沿用配置",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if torch is None:
        print("评估失败: PyTorch 不可用")
        sys.exit(1)

    trainer_cfg = DQNTrainer._build_trainer_config_from_game(CONFIG)
    trainer_cfg.game_mode = args.mode
    if args.device is not None:
        trainer_cfg.device = args.device

    trainer = DQNTrainer(game_config=CONFIG, trainer_config=trainer_cfg)
    checkpoint_path = _resolve_checkpoint_path(args.mode, args.checkpoint)
    if not checkpoint_path.exists():
        print(f"评估失败: checkpoint 不存在 {checkpoint_path}")
        sys.exit(1)

    payload = _load_checkpoint_payload(checkpoint_path, trainer.device)
    policy_state_dict = payload.get("policy_state_dict")
    if not isinstance(policy_state_dict, dict):
        print(f"评估失败: checkpoint 格式不支持 {checkpoint_path}")
        sys.exit(1)

    trainer.policy_net.load_state_dict(policy_state_dict)
    trainer.target_net.load_state_dict(payload.get("target_state_dict", policy_state_dict))

    stats = trainer.evaluate(episodes=max(1, int(args.episodes)), profile=args.profile)
    print(f"[EVAL] checkpoint={checkpoint_path}")
    print(f"[EVAL] mode={args.mode} profile={args.profile} episodes={args.episodes}")
    print(
        "[EVAL] "
        f"avg_reward={stats['avg_reward']:.3f} "
        f"win_rate={stats['win_rate']:.3f} "
        f"avg_gap={stats['avg_score_gap']:.2f} "
        f"opp_interval={int(stats['opp_interval'])} "
        f"opp_mistake={stats['opp_mistake']:.3f}"
    )


if __name__ == "__main__":
    main()
