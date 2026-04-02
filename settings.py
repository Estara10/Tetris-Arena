import math
import os
from dataclasses import dataclass, field


@dataclass
class GameConfig:
    """
    全局配置中心：
    - 所有可调参数集中在这里
    - 提供自动计算尺寸与自动调参能力
    """

    # 基础网格参数
    base_cell_size: int = 30
    cell_area_scale: float = 1.6
    grid_cols: int = 15
    grid_rows: int = 25
    side_panel_width: int = 240
    versus_gap: int = 40
    versus_target_panel_width: int = 900
    versus_target_height: int = 1040

    # 帧率控制参数
    fps: int = 60

    # 游戏逻辑参数
    fall_speed_ms: int = 500
    horizontal_move_delay_ms: int = 90
    horizontal_move_repeat_ms: int = 30
    debug_logging: bool = False
    score_mapping: dict[int, int] = field(
        default_factory=lambda: {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
        }
    )

    # 对战规则参数
    versus_attack_mapping: dict[int, int] = field(
        default_factory=lambda: {
            1: 0,
            2: 1,
            3: 2,
            4: 4,
        }
    )
    versus_combo_bonus: tuple[int, ...] = (0, 1, 1, 2, 2, 3, 3, 4)
    versus_b2b_bonus: int = 1
    versus_garbage_cap_per_lock: int = 4
    versus_garbage_apply_on_nonclear: bool = True
    versus_warning_duration_ms: int = 1500

    versus_trap_energy_cost: int = 4
    versus_trap_energy_gain_per_line: int = 1
    versus_trap_cooldown_ms: int = 10000
    versus_trap_forced_lines: int = 3
    versus_trap_warning_ms: int = 1000
    versus_trap_ai_use_threshold: int = 5
    versus_trap_ai_use_chance: float = 0.35

    # AI 运行参数
    ai_controller_mode: str = "heuristic"
    ai_model_path: str = "models/tetris_dqn.pt"
    ai_model_device: str = "cpu"
    ai_model_action_interval_ms: int = 140
    ai_model_reaction_delay_ms: int = 120

    # 强化学习训练参数
    rl_enabled: bool = False
    rl_state_use_hold: bool = True
    rl_state_preview_count: int = 3
    rl_gamma: float = 0.99
    rl_batch_size: int = 64
    rl_replay_capacity: int = 50000
    rl_target_sync_interval: int = 1000
    rl_learning_rate: float = 0.0005
    rl_epsilon_start: float = 1.0
    rl_epsilon_end: float = 0.05
    rl_epsilon_decay_steps: int = 30000
    rl_train_no_render: bool = True
    rl_eval_seed: int = 20260330
    rl_train_episodes: int = 300
    rl_warmup_steps: int = 1200
    rl_train_frequency: int = 1
    rl_max_episode_steps: int = 6000
    rl_eval_interval: int = 20
    rl_eval_episodes: int = 20
    rl_checkpoint_interval: int = 25
    rl_model_dir: str = "models"
    rl_curriculum_episodes: int = 160
    rl_opponent_start_interval_ms: int = 300
    rl_opponent_end_interval_ms: int = 140
    rl_opponent_start_mistake: float = 0.40
    rl_opponent_end_mistake: float = 0.14

    # 同盘竞技模式参数
    shared_arena_grid_cols: int = 33
    shared_arena_grid_rows: int = 25
    shared_arena_fall_speed_ms: int = 500
    shared_arena_control_tick_ms: int = 16
    shared_arena_ai_gravity_offset_ms: int = 90
    shared_arena_duration_ms: int = 180000
    shared_arena_spawns: list[int] = field(default_factory=lambda: [9, 17, 25])
    shared_arena_player_cooldown_ms: int = 200
    shared_arena_ai_cooldown_ms: int = 300
    shared_arena_score_on_lock: bool = True
    shared_arena_score_mapping: dict[int, int] = field(
        default_factory=lambda: {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
        }
    )

    # 背景参数
    bg_image_path: str = "background.png"
    bg_speed: int = 1
    bg_fallback_color: tuple[int, int, int] = (200, 200, 200)

    # 渲染参数
    render_bg_color: tuple[int, int, int] = (20, 20, 24)
    grid_color: tuple[int, int, int] = (185, 185, 185)
    text_color: tuple[int, int, int] = (245, 245, 245)
    panel_color: tuple[int, int, int] = (34, 34, 42)
    overlay_color: tuple[int, int, int, int] = (0, 0, 0, 140)
    ghost_alpha: int = 100

    block_colors: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: {
            "I": (0, 255, 255),
            "J": (0, 90, 255),
            "L": (255, 165, 0),
            "O": (255, 225, 0),
            "S": (0, 210, 0),
            "T": (170, 60, 220),
            "Z": (255, 70, 70),
            "G": (135, 135, 145),
        }
    )

    # 字体参数
    font_name: str = "arial"
    font_size_title: int = 30
    font_size_main: int = 26
    font_size_hint: int = 22

    @property
    def cell_size(self) -> int:
        """
        面积倍率 -> 边长倍率：
        边长 = base_cell_size * sqrt(cell_area_scale)
        """
        value = int(round(self.base_cell_size * math.sqrt(self.cell_area_scale)))
        return max(8, value)

    @property
    def game_width(self) -> int:
        return self.grid_cols * self.cell_size

    @property
    def game_height(self) -> int:
        return self.grid_rows * self.cell_size

    @property
    def screen_width(self) -> int:
        return self.game_width + self.side_panel_width

    @property
    def screen_height(self) -> int:
        return self.game_height

    @property
    def versus_screen_width(self) -> int:
        return self.screen_width * 2 + self.versus_gap

    def set_cell_area_scale(self, scale: float) -> None:
        """手动修改单元格面积缩放倍率。"""
        if scale <= 0:
            raise ValueError("cell_area_scale 必须大于 0")
        self.cell_area_scale = scale

    def auto_tune_cell_size(
        self,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> None:
        """
        自动调参：根据目标窗口尺寸，反推合适的 cell_size，再回写到 cell_area_scale。
        """
        candidates = []
        if target_height is not None:
            candidates.append(target_height // self.grid_rows)
        if target_width is not None:
            panel_adjusted = max(1, target_width - self.side_panel_width)
            candidates.append(panel_adjusted // self.grid_cols)

        if not candidates:
            return

        tuned_cell = max(8, min(candidates))
        self.cell_area_scale = (tuned_cell / self.base_cell_size) ** 2


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw


def load_config() -> GameConfig:
    """
    从环境变量加载配置，便于批量调参：
    - TETRIS_AREA_SCALE
    - TETRIS_FPS
    - TETRIS_BG_SPEED
    - TETRIS_TARGET_WIDTH
    - TETRIS_TARGET_HEIGHT
    """
    config = GameConfig(
        cell_area_scale=_env_float("TETRIS_AREA_SCALE", 1.6),
        fps=_env_int("TETRIS_FPS", 60),
        bg_speed=_env_int("TETRIS_BG_SPEED", 1),
        ai_controller_mode=_env_str("TETRIS_AI_MODE", "heuristic"),
        ai_model_path=_env_str("TETRIS_AI_MODEL_PATH", "models/tetris_dqn.pt"),
        ai_model_device=_env_str("TETRIS_AI_DEVICE", "cpu"),
        rl_enabled=_env_bool("TETRIS_RL_ENABLED", False),
        rl_train_episodes=_env_int("TETRIS_RL_EPISODES", 300),
        rl_warmup_steps=_env_int("TETRIS_RL_WARMUP", 1200),
        rl_batch_size=_env_int("TETRIS_RL_BATCH", 64),
        rl_replay_capacity=_env_int("TETRIS_RL_REPLAY", 50000),
        rl_learning_rate=_env_float("TETRIS_RL_LR", 0.0005),
        rl_epsilon_start=_env_float("TETRIS_RL_EPS_START", 1.0),
        rl_epsilon_end=_env_float("TETRIS_RL_EPS_END", 0.05),
        rl_epsilon_decay_steps=_env_int("TETRIS_RL_EPS_DECAY", 30000),
        rl_max_episode_steps=_env_int("TETRIS_RL_MAX_STEPS", 6000),
        rl_eval_episodes=_env_int("TETRIS_RL_EVAL_EPISODES", 20),
        rl_curriculum_episodes=_env_int("TETRIS_RL_CURRIC_EPISODES", 160),
        rl_opponent_start_interval_ms=_env_int("TETRIS_RL_OPP_START_INTERVAL", 300),
        rl_opponent_end_interval_ms=_env_int("TETRIS_RL_OPP_END_INTERVAL", 140),
        rl_opponent_start_mistake=_env_float("TETRIS_RL_OPP_START_MISTAKE", 0.40),
        rl_opponent_end_mistake=_env_float("TETRIS_RL_OPP_END_MISTAKE", 0.14),
        rl_model_dir=_env_str("TETRIS_RL_MODEL_DIR", "models"),
        shared_arena_fall_speed_ms=_env_int("TETRIS_ARENA_FALL_MS", 500),
        shared_arena_control_tick_ms=_env_int("TETRIS_ARENA_CONTROL_TICK_MS", 16),
        shared_arena_ai_gravity_offset_ms=_env_int("TETRIS_ARENA_AI_GRAVITY_OFFSET_MS", 90),
    )
    target_width = os.getenv("TETRIS_TARGET_WIDTH")
    target_height = os.getenv("TETRIS_TARGET_HEIGHT")
    if target_width is not None or target_height is not None:
        config.auto_tune_cell_size(
            target_width=_env_int("TETRIS_TARGET_WIDTH", config.screen_width),
            target_height=_env_int("TETRIS_TARGET_HEIGHT", config.screen_height),
        )
    return config


CONFIG = load_config()
