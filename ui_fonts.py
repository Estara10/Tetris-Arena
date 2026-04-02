import pygame

from settings import GameConfig


def build_ui_font(config: GameConfig, size: int, bold: bool = False):
    """
    优先选择常见中文字体，避免中文界面显示成方块。
    """
    preferred_fonts = (
        "microsoftyahei",
        "simhei",
        "pingfangsc",
        "hiraginosansgb",
        "notosanscjk",
        "notosanscjksc",
        "wenquanyizenhei",
        "sarasa ui sc",
        "arialunicode",
        config.font_name,
    )
    seen = set()

    for font_name in preferred_fonts:
        font_key = font_name.lower()
        if font_key in seen:
            continue
        seen.add(font_key)

        font_path = pygame.font.match_font(font_name)
        if font_path:
            font = pygame.font.Font(font_path, size)
            font.set_bold(bold)
            return font

    return pygame.font.SysFont(None, size, bold=bold)


def build_menu_fonts(config: GameConfig):
    pygame.font.init()

    title_size = max(42, min(54, config.screen_height // 18))
    subtitle_size = max(22, min(26, config.screen_height // 34))
    card_title_size = max(24, min(32, config.screen_height // 30))
    card_text_size = max(18, min(20, config.screen_height // 48))
    hint_size = max(16, min(18, config.screen_height // 58))
    keycap_size = max(22, min(28, config.screen_height // 34))
    small_size = max(14, min(17, config.screen_height // 64))
    stat_size = max(16, min(18, config.screen_height // 54))

    return {
        "title": build_ui_font(config, title_size, bold=True),
        "subtitle": build_ui_font(config, subtitle_size, bold=True),
        "card_title": build_ui_font(config, card_title_size, bold=True),
        "card_text": build_ui_font(config, card_text_size, bold=True),
        "hint": build_ui_font(config, hint_size, bold=True),
        "keycap": build_ui_font(config, keycap_size, bold=True),
        "small": build_ui_font(config, small_size, bold=True),
        "stat": build_ui_font(config, stat_size, bold=True),
    }
