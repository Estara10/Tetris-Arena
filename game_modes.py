from dataclasses import dataclass


@dataclass(frozen=True)
class MatchMode:
    key: str
    label: str
    description_lines: tuple[str, str]
    objective_lines: int | None = None
    ai_action_interval_ms: int = 90
    ai_mistake_chance: float = 0.18


@dataclass(frozen=True)
class AILevel:
    key: str
    label: str
    menu_title: str
    description_lines: tuple[str, str]
    action_interval_ms: int
    mistake_chance: float
    fall_speed_ms: int


GAME_MODES = {
    "CLASSIC": MatchMode(
        key="CLASSIC",
        label="经典对战版",
        description_lines=(
            "同步 7-bag 对战，双方获得完全相同的方块序列。",
            "双方结束后比较总分，得分高者获胜。",
        ),
        ai_action_interval_ms=500,
        ai_mistake_chance=0.18,
    ),
    "TRADITIONAL": MatchMode(
        key="TRADITIONAL",
        label="传统版",
        description_lines=(
            "25x25 经典推挤模式，下落位 9、17。",
            "两人同盘，先被撞者才能回撞；消行得分。",
        ),
        ai_action_interval_ms=200,
        ai_mistake_chance=0.1,
    ),
}

AI_LEVELS = {
    "EASY": AILevel(
        key="EASY",
        label="初级",
        menu_title="初级 AI",
        description_lines=(
            "AI 会明显放慢思考和落子，适合先熟悉双屏对战。",
            "如果刚开始打不过 AI，建议先从这一档练习。",
        ),
        action_interval_ms=500,
        mistake_chance=0.42,
        fall_speed_ms=500,
    ),
    "NORMAL": AILevel(
        key="NORMAL",
        label="中级",
        menu_title="中级 AI",
        description_lines=(
            "速度与稳定性较均衡，适合常规对战。",
            "如果想要正常对抗体验，可以先从这一档开始。",
        ),
        action_interval_ms=500,
        mistake_chance=0.18,
        fall_speed_ms=500,
    ),
    "HARD": AILevel(
        key="HARD",
        label="高级",
        menu_title="高级 AI",
        description_lines=(
            "AI 行动更快，落子节奏更强，失误更少。",
            "适合已经熟悉玩法后再来挑战。",
        ),
        action_interval_ms=500,
        mistake_chance=0.08,
        fall_speed_ms=500,
    ),
}
AI_LEVEL_ORDER = ("EASY", "NORMAL", "HARD")

MENU_MODE_SELECT = "MODE_SELECT"
MENU_CLASSIC_LEVEL = "CLASSIC_LEVEL"
