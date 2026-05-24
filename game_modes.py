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
            "同步 7-bag 发牌，比拼最终得分。",
            "左右双棋盘，节奏稳定。",
        ),
        ai_action_interval_ms=500,
        ai_mistake_chance=0.18,
    ),
    "TRADITIONAL": MatchMode(
        key="TRADITIONAL",
        label="对抗模式",
        description_lines=(
            "25x25 同盘推挤，消行得分。",
            "出生位 9、17，正面对抗。",
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
            "节奏最慢，适合先上手。",
            "建议首次游玩选择这一档。",
        ),
        action_interval_ms=500,
        mistake_chance=0.42,
        fall_speed_ms=1000,
    ),
    "NORMAL": AILevel(
        key="NORMAL",
        label="中级",
        menu_title="中级 AI",
        description_lines=(
            "速度稳定，适合常规对抗。",
            "推荐熟悉规则后选择。",
        ),
        action_interval_ms=500,
        mistake_chance=0.18,
        fall_speed_ms=667,
    ),
    "HARD": AILevel(
        key="HARD",
        label="高级",
        menu_title="高级 AI",
        description_lines=(
            "反应更快，压迫感更强。",
            "适合熟练后挑战。",
        ),
        action_interval_ms=500,
        mistake_chance=0.08,
        fall_speed_ms=500,
    ),
}
AI_LEVEL_ORDER = ("EASY", "NORMAL", "HARD")

MENU_MODE_SELECT = "MODE_SELECT"
MENU_CLASSIC_LEVEL = "CLASSIC_LEVEL"
