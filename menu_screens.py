import pygame

from game_modes import AI_LEVELS, AI_LEVEL_ORDER, GAME_MODES
from settings import GameConfig


MODE_META = {
    "CLASSIC": {
        "tag": "推荐入门",
        "features": ("同步发牌", "比分决胜"),
        "footer": "适合先熟悉双屏对战和同步发牌节奏。",
    },
    "CHALLENGE": {
        "tag": "动态加压",
        "features": ("目标消行", "难度自适应"),
        "footer": "你的表现越好，AI 的压迫感就会越强。",
    },
    "ARENA": {
        "tag": "同盘对抗",
        "features": ("同盘碰撞", "限时积分"),
        "footer": "玩家与 AI 在同一棋盘同步下落，计时结束比分高者胜。",
    },
}

LEVEL_META = {
    "EASY": {
        "tag": "新手友好",
        "summary": "节奏最慢，适合先上手。",
    },
    "NORMAL": {
        "tag": "均衡推荐",
        "summary": "速度稳定，适合常规对抗。",
    },
    "HARD": {
        "tag": "高手挑战",
        "summary": "反应更快，压迫感最强。",
    },
}


def _lerp_color(color_a, color_b, ratio: float):
    """
    根据给定的比例在两个 RGB 颜色组间插值。

    参数:
    color_a (tuple): 颜色起始点
    color_b (tuple): 颜色终结处
    ratio (float): 取值 (0-1)，接近0偏A，1偏B

    返回:
    tuple: 生成的新颜色RGB组
    """
    return tuple(
        int(left + (right - left) * ratio)
        for left, right in zip(color_a, color_b)
    )


def _wrap_text(font, text: str, max_width: int):
    """
    自动对所供文本实现换行逻辑，使之不超过绘图区划定的宽度。

    参数:
    font (pygame.font.Font): 用于测量文本大小的字体类型
    text (str): 要求显示的文本段落
    max_width (int): 界限的最大宽度(像素数)

    返回:
    list[str]: 被折分出来的各行段落表
    """
    if not text:
        return [""]

    lines = []
    current = ""
    for char in text:
        candidate = current + char
        if current and font.size(candidate)[0] > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate

    if current:
        lines.append(current)
    return lines


def _draw_text_block(
    screen,
    font,
    color,
    x: int,
    y: int,
    max_width: int,
    paragraphs,
    line_gap: int = 8,
    paragraph_gap: int = 10,
    max_lines: int | None = None,
):
    """
    按照预定宽度在屏幕上输出可以多行的连续段落，常用来绘制UI提示文本或游戏内指南。

    参数:
    screen (pygame.Surface): 主渲染目标画布
    font (pygame.font.Font): 决定字形的字体设定
    color (tuple): RGB色彩配置
    x (int): 文本描绘区域左上角的 x 坐标
    y (int): 文本描绘起始 y 坐标
    max_width (int): 允许占用的横向跨度
    paragraphs (list[str]): 要输出的所有行/段列表
    line_gap (int): 普通换行由于紧凑产生的行距增量
    paragraph_gap (int): 不同段落间隔额外跳过的纵向像素
    max_lines (int | None): 超过限定截断的最大允许行数输出

    返回:
    int: 所有文本绘制操作之后光标最终驻留的新 y 坐标
    """
    used_lines = 0

    for paragraph in paragraphs:
        wrapped = _wrap_text(font, paragraph, max_width)
        for line in wrapped:
            if max_lines is not None and used_lines >= max_lines:
                return y

            surface = font.render(line, True, color)
            screen.blit(surface, (x, y))
            y += surface.get_height() + line_gap
            used_lines += 1
        y += paragraph_gap

    return y - paragraph_gap if paragraphs else y


def draw_vertical_gradient(screen, top_color, bottom_color):
    width, height = screen.get_size()
    for y in range(height):
        ratio = y / max(1, height - 1)
        color = _lerp_color(top_color, bottom_color, ratio)
        pygame.draw.line(screen, color, (0, y), (width, y))


def draw_menu_background(screen, config: GameConfig):
    draw_vertical_gradient(screen, (9, 16, 28), (18, 28, 42))

    width, height = screen.get_size()
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)

    for center, radius, color in (
        ((int(width * 0.10), int(height * 0.18)), 180, (55, 132, 215, 70)),
        ((int(width * 0.88), int(height * 0.16)), 190, (232, 126, 88, 58)),
        ((int(width * 0.52), int(height * 0.88)), 230, (34, 176, 160, 40)),
    ):
        pygame.draw.circle(overlay, color, center, radius)

    for x in range(-height, width, 120):
        pygame.draw.line(
            overlay,
            (255, 255, 255, 12),
            (x, 0),
            (x + height, height),
            1,
        )

    pygame.draw.rect(overlay, (8, 12, 20, 86), pygame.Rect(0, 0, width, height))
    screen.blit(overlay, (0, 0))


def draw_panel(screen, rect, fill_color, border_color, shadow_offset=8, border_radius=26):
    shadow_surface = pygame.Surface((rect.width + 24, rect.height + 24), pygame.SRCALPHA)
    pygame.draw.rect(
        shadow_surface,
        (0, 0, 0, 88),
        pygame.Rect(12, 12, rect.width, rect.height),
        border_radius=border_radius,
    )
    screen.blit(shadow_surface, (rect.x - 12, rect.y - 12 + shadow_offset))

    pygame.draw.rect(screen, fill_color, rect, border_radius=border_radius)
    pygame.draw.rect(screen, border_color, rect, 2, border_radius=border_radius)


def draw_chip(screen, rect, font, text, fill_color, text_color, border_color=None):
    pygame.draw.rect(screen, fill_color, rect, border_radius=999)
    if border_color is not None:
        pygame.draw.rect(screen, border_color, rect, 1, border_radius=999)

    if text:
        label = font.render(text, True, text_color)
        screen.blit(
            label,
            (
                rect.centerx - label.get_width() // 2,
                rect.centery - label.get_height() // 2,
            ),
        )


def draw_chip_row(screen, center_x, y, font, chips, fill_color, text_color, border_color=None):
    if not chips:
        return

    spacing = 12
    chip_rects = []
    total_width = 0
    for chip in chips:
        label = font.render(chip, True, text_color)
        rect = pygame.Rect(0, 0, label.get_width() + 24, label.get_height() + 10)
        chip_rects.append((rect, chip))
        total_width += rect.width
    total_width += spacing * max(0, len(chip_rects) - 1)

    cursor_x = center_x - total_width // 2
    for rect, chip in chip_rects:
        rect.x = cursor_x
        rect.y = y
        draw_chip(screen, rect, font, chip, fill_color, text_color, border_color)
        cursor_x += rect.width + spacing


def draw_action_bar(screen, fonts, items):
    width, height = screen.get_size()
    spacing = 14
    rendered = []
    total_width = 0

    for item in items:
        label = fonts["small"].render(item, True, (222, 230, 244))
        rect = pygame.Rect(0, 0, label.get_width() + 24, label.get_height() + 12)
        rendered.append((rect, label))
        total_width += rect.width
    total_width += spacing * max(0, len(rendered) - 1)

    cursor_x = width // 2 - total_width // 2
    y = height - 48
    for rect, label in rendered:
        rect.x = cursor_x
        rect.y = y
        draw_chip(
            screen,
            rect,
            fonts["small"],
            "",
            (16, 22, 34),
            (222, 230, 244),
            (77, 98, 130),
        )
        screen.blit(
            label,
            (
                rect.centerx - label.get_width() // 2,
                rect.centery - label.get_height() // 2,
            ),
        )
        cursor_x += rect.width + spacing


def draw_hero_banner(screen, rect, fonts, title, subtitle, badges):
    draw_panel(screen, rect, (15, 22, 34), (82, 108, 145), shadow_offset=10, border_radius=28)

    eyebrow = fonts["small"].render("双屏 AI 对战", True, (128, 212, 255))
    title_surface = fonts["title"].render(title, True, (247, 249, 252))

    eyebrow_x = rect.x + 34
    eyebrow_y = rect.y + 24
    title_x = rect.x + 34
    title_y = eyebrow_y + eyebrow.get_height() + 18
    subtitle_y = title_y + title_surface.get_height() + 16

    screen.blit(eyebrow, (eyebrow_x, eyebrow_y))
    screen.blit(title_surface, (title_x, title_y))

    subtitle_bottom = _draw_text_block(
        screen,
        fonts["subtitle"],
        (189, 201, 219),
        title_x,
        subtitle_y,
        rect.width - 68,
        [subtitle],
        line_gap=6,
        paragraph_gap=0,
        max_lines=2,
    )

    badges_y = max(subtitle_bottom + 18, rect.bottom - 52)
    draw_chip_row(
        screen,
        rect.centerx,
        badges_y,
        fonts["small"],
        badges,
        (24, 33, 48),
        (226, 232, 243),
        (78, 100, 132),
    )


def draw_mode_card(screen, rect, fonts, accent_color, key_hint, title, lines, tag, features, footer):
    draw_panel(screen, rect, (19, 27, 40), accent_color, shadow_offset=10, border_radius=24)

    header_rect = pygame.Rect(rect.x, rect.y, rect.width, 64)
    pygame.draw.rect(screen, (31, 43, 61), header_rect, border_radius=24)
    pygame.draw.rect(screen, accent_color, header_rect, 2, border_radius=24)

    key_rect = pygame.Rect(rect.x + 22, rect.y + 17, 72, 34)
    draw_chip(screen, key_rect, fonts["keycap"], key_hint, accent_color, (10, 14, 22))

    tag_rect = pygame.Rect(rect.right - 120, rect.y + 17, 96, 32)
    draw_chip(screen, tag_rect, fonts["small"], tag, (27, 37, 54), (233, 238, 245), (86, 108, 141))

    title_surface = fonts["card_title"].render(title, True, (247, 248, 252))
    title_y = rect.y + 84
    screen.blit(title_surface, (rect.x + 24, title_y))

    text_bottom = _draw_text_block(
        screen,
        fonts["card_text"],
        (212, 219, 230),
        rect.x + 24,
        title_y + title_surface.get_height() + 18,
        rect.width - 48,
        lines,
        line_gap=6,
        paragraph_gap=10,
        max_lines=4,
    )

    feature_y = max(text_bottom + 18, rect.bottom - 84)
    draw_chip_row(
        screen,
        rect.centerx,
        feature_y,
        fonts["small"],
        features,
        (25, 34, 50),
        (226, 232, 243),
        (74, 96, 128),
    )

    footer_surface = fonts["small"].render(footer, True, (158, 172, 193))
    screen.blit(footer_surface, (rect.x + 24, rect.bottom - 36))


def draw_level_card(screen, rect, fonts, accent_color, key_hint, level, tag, summary):
    draw_panel(screen, rect, (19, 27, 40), accent_color, shadow_offset=10, border_radius=24)

    key_rect = pygame.Rect(rect.x + 22, rect.y + 18, 68, 34)
    draw_chip(screen, key_rect, fonts["keycap"], key_hint, accent_color, (10, 14, 22))

    tag_rect = pygame.Rect(rect.right - 118, rect.y + 18, 94, 30)
    draw_chip(screen, tag_rect, fonts["small"], tag, (27, 37, 54), (233, 238, 245), (86, 108, 141))

    title_surface = fonts["card_title"].render(level.menu_title, True, (247, 248, 252))
    screen.blit(title_surface, (rect.x + 22, rect.y + 72))

    summary_bottom = _draw_text_block(
        screen,
        fonts["card_text"],
        (212, 219, 230),
        rect.x + 22,
        rect.y + 122,
        rect.width - 44,
        [summary],
        line_gap=6,
        paragraph_gap=0,
        max_lines=2,
    )

    stat_y = summary_bottom + 18
    stat_lines = (
        f"行动节奏  {level.action_interval_ms} ms",
        f"自动下落  {level.fall_speed_ms} ms",
        f"失误概率  {int(level.mistake_chance * 100)}%",
    )
    for stat in stat_lines:
        stat_surface = fonts["stat"].render(stat, True, (228, 234, 243))
        screen.blit(stat_surface, (rect.x + 22, stat_y))
        stat_y += stat_surface.get_height() + 12

    footer_surface = fonts["small"].render(level.description_lines[0], True, (158, 172, 193))
    screen.blit(footer_surface, (rect.x + 22, rect.bottom - 34))


def draw_mode_select_menu(screen, config: GameConfig, fonts):
    draw_menu_background(screen, config)

    width, _height = screen.get_size()
    margin = max(48, width // 32)
    hero_width = min(width - margin * 2, 1180)
    hero_rect = pygame.Rect(width // 2 - hero_width // 2, 44, hero_width, 230)

    draw_hero_banner(
        screen,
        hero_rect,
        fonts,
        "俄罗斯方块 AI 对战",
        "双屏同步发牌，和 AI 在同一局里正面对抗。",
        ("双屏同步", "中文界面", "单步移动"),
    )

    mode_order = ("CLASSIC", "CHALLENGE", "ARENA")
    key_hints = {
        "CLASSIC": "1",
        "CHALLENGE": "2",
        "ARENA": "3",
    }
    accent_colors = {
        "CLASSIC": (97, 190, 246),
        "CHALLENGE": (248, 150, 109),
        "ARENA": (162, 203, 114),
    }

    card_gap = 24
    card_count = len(mode_order)
    card_width = min(420, (width - margin * 2 - card_gap * (card_count - 1)) // card_count)
    card_height = 330
    start_x = width // 2 - (card_width * card_count + card_gap * (card_count - 1)) // 2
    card_y = hero_rect.bottom + 34

    for index, mode_key in enumerate(mode_order):
        rect = pygame.Rect(
            start_x + index * (card_width + card_gap),
            card_y,
            card_width,
            card_height,
        )
        meta = MODE_META[mode_key]
        mode = GAME_MODES[mode_key]
        draw_mode_card(
            screen,
            rect,
            fonts,
            accent_colors[mode_key],
            key_hints[mode_key],
            mode.label,
            mode.description_lines,
            meta["tag"],
            meta["features"],
            meta["footer"],
        )

    draw_action_bar(
        screen,
        fonts,
        ("按 1 进入经典模式", "按 2 进入挑战模式", "按 3 进入同盘竞技"),
    )


def draw_classic_level_menu(screen, config: GameConfig, fonts):
    draw_menu_background(screen, config)

    width, _height = screen.get_size()
    margin = max(48, width // 32)
    hero_width = min(width - margin * 2, 1220)
    hero_rect = pygame.Rect(width // 2 - hero_width // 2, 44, hero_width, 220)

    draw_hero_banner(
        screen,
        hero_rect,
        fonts,
        "选择经典模式 AI 等级",
        "建议先从初级开始，熟悉双屏节奏后再挑战更高等级。",
        ("初级最慢", "中级均衡", "高级最快"),
    )

    card_gap = 22
    card_width = min(350, (width - margin * 2 - card_gap * 2) // 3)
    card_height = 320
    start_x = width // 2 - (card_width * 3 + card_gap * 2) // 2
    card_y = hero_rect.bottom + 34

    accent_colors = {
        "EASY": (104, 198, 255),
        "NORMAL": (255, 196, 92),
        "HARD": (255, 124, 124),
    }

    for index, level_key in enumerate(AI_LEVEL_ORDER, start=1):
        level = AI_LEVELS[level_key]
        meta = LEVEL_META[level_key]
        rect = pygame.Rect(
            start_x + (index - 1) * (card_width + card_gap),
            card_y,
            card_width,
            card_height,
        )
        draw_level_card(
            screen,
            rect,
            fonts,
            accent_colors[level_key],
            str(index),
            level,
            meta["tag"],
            meta["summary"],
        )

    draw_action_bar(
        screen,
        fonts,
        ("按 1/2/3 选择等级", "左右单步移动", "M 返回模式选择"),
    )
