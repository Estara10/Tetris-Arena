import pygame

from game_modes import AI_LEVELS, AI_LEVEL_ORDER, GAME_MODES
from settings import GameConfig


MODE_META = {
    "CLASSIC": {
        "tag": "推荐入门",
        "features": ("同步发牌", "比分决胜"),
        "footer": "双屏同步发牌。",
    },
    "TRADITIONAL": {
        "tag": "对抗压制",
        "features": ("相向推挤", "消行得分"),
        "footer": "同盘推挤对抗。",
    },
}

LEVEL_META = {
    "EASY": {
        "tag": "新手友好",
        "summary": "节奏最慢，适合上手。",
    },
    "NORMAL": {
        "tag": "均衡推荐",
        "summary": "速度稳定，适合常规局。",
    },
    "HARD": {
        "tag": "高手挑战",
        "summary": "反应更快，压迫更强。",
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


def draw_menu_background(screen, config, light=False):
    if light:
        draw_vertical_gradient(screen, (235, 238, 245), (212, 216, 228))
    else:
        draw_vertical_gradient(screen, (7, 10, 18), (13, 18, 29))

    width, height = screen.get_size()
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)

    if light:
        for y in range(0, height, 36):
            alpha = 12 if y % 72 == 0 else 6
            pygame.draw.line(overlay, (140, 148, 168, alpha), (0, y), (width, y), 1)

        for x in range(0, width, 36):
            alpha = 10 if x % 72 == 0 else 5
            pygame.draw.line(overlay, (140, 148, 168, alpha), (x, 0), (x, height), 1)

        for x_col in range(-height, width, 160):
            pygame.draw.line(
                overlay,
                (190, 195, 210, 6),
                (x_col, 0),
                (x_col + height, height),
                1,
            )

        pygame.draw.rect(overlay, (80, 158, 228, 32), pygame.Rect(0, 0, width, 5))
        pygame.draw.rect(overlay, (238, 148, 88, 24), pygame.Rect(0, height - 5, width, 5))
        pygame.draw.rect(overlay, (210, 215, 228, 40), pygame.Rect(0, 0, width, height))
    else:
        for y in range(0, height, 36):
            alpha = 18 if y % 72 == 0 else 10
            pygame.draw.line(overlay, (104, 126, 164, alpha), (0, y), (width, y), 1)

        for x in range(0, width, 36):
            alpha = 16 if x % 72 == 0 else 8
            pygame.draw.line(overlay, (104, 126, 164, alpha), (x, 0), (x, height), 1)

        for x_col in range(-height, width, 160):
            pygame.draw.line(
                overlay,
                (255, 255, 255, 8),
                (x_col, 0),
                (x_col + height, height),
                1,
            )

        pygame.draw.rect(overlay, (32, 96, 156, 36), pygame.Rect(0, 0, width, 5))
        pygame.draw.rect(overlay, (232, 118, 86, 28), pygame.Rect(0, height - 5, width, 5))
        pygame.draw.rect(overlay, (3, 6, 12, 86), pygame.Rect(0, 0, width, height))
    screen.blit(overlay, (0, 0))


def draw_soft_glow(screen, rect, color, spread=18, alpha=38, border_radius=10, light=False):
    actual_alpha = alpha
    if light:
        actual_alpha = max(12, alpha // 2)
        glow_color = (color[0] - 40, color[1] - 20, color[2] - 20)
    else:
        glow_color = color

    glow_surface = pygame.Surface((rect.width + spread * 2, rect.height + spread * 2), pygame.SRCALPHA)
    pygame.draw.rect(
        glow_surface,
        (glow_color[0], glow_color[1], glow_color[2], actual_alpha),
        pygame.Rect(spread, spread, rect.width, rect.height),
        border_radius=border_radius,
    )
    screen.blit(glow_surface, (rect.x - spread, rect.y - spread))


def draw_panel(screen, rect, fill_color, border_color, shadow_offset=8, border_radius=10, light=False):
    draw_soft_glow(screen, rect, border_color, spread=14, alpha=22, border_radius=border_radius, light=light)

    shadow_surface = pygame.Surface((rect.width + 24, rect.height + 24), pygame.SRCALPHA)
    shadow_alpha = 45 if light else 88
    pygame.draw.rect(
        shadow_surface,
        (0, 0, 0, shadow_alpha),
        pygame.Rect(12, 12, rect.width, rect.height),
        border_radius=border_radius,
    )
    screen.blit(shadow_surface, (rect.x - 12, rect.y - 12 + shadow_offset))

    inner_border_color = (200, 208, 225) if light else (45, 60, 84)
    pygame.draw.rect(screen, fill_color, rect, border_radius=border_radius)
    pygame.draw.rect(
        screen,
        inner_border_color,
        rect.inflate(-6, -6),
        1,
        border_radius=max(6, border_radius - 3),
    )
    pygame.draw.rect(screen, border_color, rect, 2, border_radius=border_radius)


def draw_chip(screen, rect, font, text, fill_color, text_color, border_color=None, light=False):
    pygame.draw.rect(screen, fill_color, rect, border_radius=8)
    if border_color is not None:
        pygame.draw.rect(screen, border_color, rect, 1, border_radius=8)

    if text:
        label = font.render(text, True, text_color)
        screen.blit(
            label,
            (
                rect.centerx - label.get_width() // 2,
                rect.centery - label.get_height() // 2,
            ),
        )


def draw_chip_row(screen, center_x, y, font, chips, fill_color, text_color, border_color=None, light=False):
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
        draw_chip(screen, rect, font, chip, fill_color, text_color, border_color, light=light)
        cursor_x += rect.width + spacing


def draw_action_bar(screen, fonts, items, light=False):
    width, height = screen.get_size()
    spacing = 14
    rendered = []
    total_width = 0

    label_color = (65, 70, 92) if light else (222, 230, 244)
    chip_fill = (232, 238, 250) if light else (16, 22, 34)
    chip_border = (168, 182, 210) if light else (77, 98, 130)

    for item in items:
        label = fonts["small"].render(item, True, label_color)
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
            chip_fill,
            label_color,
            chip_border,
            light=light,
        )
        screen.blit(
            label,
            (
                rect.centerx - label.get_width() // 2,
                rect.centery - label.get_height() // 2,
            ),
        )
        cursor_x += rect.width + spacing


def draw_section_label(screen, fonts, x, y, eyebrow, title, accent_color, light=False):
    title_color = (25, 30, 48) if light else (242, 246, 252)
    eyebrow_color = (90, 100, 125) if light else accent_color

    eyebrow_surface = fonts["small"].render(eyebrow, True, eyebrow_color)
    title_surface = fonts["subtitle"].render(title, True, title_color)
    screen.blit(eyebrow_surface, (x, y))
    screen.blit(title_surface, (x, y + eyebrow_surface.get_height() + 8))

    accent_rect = pygame.Rect(x, y + eyebrow_surface.get_height() + title_surface.get_height() + 20, 120, 4)
    pygame.draw.rect(screen, accent_color, accent_rect, border_radius=2)
    return accent_rect.bottom


def draw_mode_preview(screen, rect, accent_color, variant="classic", light=False):
    if light:
        bg = (238, 242, 252)
        border = (175, 188, 215)
        grid_alpha = (205, 212, 228)
        stack_c = (182, 190, 212)
        stack_t = (195, 200, 218)
    else:
        bg = (9, 14, 24)
        border = (72, 94, 128)
        grid_alpha = (40, 56, 78)
        stack_c = (112, 122, 154)
        stack_t = (132, 132, 142)

    pygame.draw.rect(screen, bg, rect, border_radius=8)
    pygame.draw.rect(screen, border, rect, 1, border_radius=8)

    cols = 5
    rows = 6
    cell = min((rect.width - 20) // cols, (rect.height - 20) // rows)
    start_x = rect.x + (rect.width - cols * cell) // 2
    start_y = rect.y + (rect.height - rows * cell) // 2

    for col in range(cols + 1):
        px = start_x + col * cell
        pygame.draw.line(screen, grid_alpha, (px, start_y), (px, start_y + rows * cell), 1)
    for row in range(rows + 1):
        py = start_y + row * cell
        pygame.draw.line(screen, grid_alpha, (start_x, py), (start_x + cols * cell, py), 1)

    stack_height = 1 if variant == "classic" else 2
    stack_color = stack_c if variant == "classic" else stack_t
    for row in range(rows - stack_height, rows):
        for col in range(cols):
            block_rect = pygame.Rect(start_x + col * cell + 1, start_y + row * cell + 1, cell - 2, cell - 2)
            pygame.draw.rect(screen, stack_color, block_rect, border_radius=2)

    pieces = (
        ((1, 2), (2, 2), (2, 1), (3, 1)),
        ((1, 2), (2, 2), (3, 2), (2, 1)),
    )
    cells = pieces[0] if variant == "classic" else pieces[1]
    for col, row in cells:
        block_rect = pygame.Rect(start_x + col * cell + 1, start_y + row * cell + 1, cell - 2, cell - 2)
        pygame.draw.rect(screen, accent_color, block_rect, border_radius=2)
        highlight = (244, 247, 252) if not light else (255, 255, 255)
        pygame.draw.rect(screen, highlight, block_rect, 1, border_radius=2)


def draw_hero_banner(screen, rect, fonts, title, subtitle, badges, light=False):
    if light:
        fill = (252, 253, 255)
        border = (175, 188, 210)
        accent_bar_color = (80, 165, 235)
        eyebrow_color = (65, 145, 225)
        title_color = (22, 26, 42)
        subtitle_color = (95, 102, 122)
        badge_fill = (232, 238, 250)
        badge_text = (55, 60, 82)
        badge_border = (168, 182, 212)
    else:
        fill = (13, 20, 32)
        border = (76, 104, 146)
        accent_bar_color = (102, 188, 245)
        eyebrow_color = (128, 212, 255)
        title_color = (247, 249, 252)
        subtitle_color = (189, 201, 219)
        badge_fill = (24, 33, 48)
        badge_text = (226, 232, 243)
        badge_border = (78, 100, 132)

    draw_panel(screen, rect, fill, border, shadow_offset=8, border_radius=10, light=light)

    accent_bar = pygame.Rect(rect.x + 32, rect.y + 18, rect.width - 64, 5)
    pygame.draw.rect(screen, accent_bar_color, accent_bar, border_radius=2)

    eyebrow = fonts["small"].render("双屏 AI 对战", True, eyebrow_color)
    title_surface = fonts["title"].render(title, True, title_color)

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
        subtitle_color,
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
        badge_fill,
        badge_text,
        badge_border,
        light=light,
    )


def draw_mode_card(screen, rect, fonts, accent_color, key_hint, title, lines, tag, features, footer, light=False):
    if light:
        fill = (252, 253, 255)
        key_text = (255, 255, 255)
        tag_fill = (232, 238, 250)
        tag_text = (60, 65, 85)
        tag_border = (170, 185, 215)
        card_title_color = (22, 26, 42)
        card_text_color = (100, 106, 128)
        feature_fill = (232, 238, 250)
        feature_text = (60, 65, 85)
        feature_border = (170, 185, 215)
    else:
        fill = (15, 23, 36)
        key_text = (10, 14, 22)
        tag_fill = (27, 37, 54)
        tag_text = (233, 238, 245)
        tag_border = (86, 108, 141)
        card_title_color = (247, 248, 252)
        card_text_color = (212, 219, 230)
        feature_fill = (25, 34, 50)
        feature_text = (226, 232, 243)
        feature_border = (74, 96, 128)

    draw_panel(screen, rect, fill, accent_color, shadow_offset=8, border_radius=10, light=light)

    accent_bar = pygame.Rect(rect.x + 20, rect.y + 18, rect.width - 40, 5)
    pygame.draw.rect(screen, accent_color, accent_bar, border_radius=2)

    key_rect = pygame.Rect(rect.x + 22, rect.y + 32, 72, 34)
    draw_chip(screen, key_rect, fonts["keycap"], key_hint, accent_color, key_text, light=light)

    tag_rect = pygame.Rect(rect.right - 120, rect.y + 33, 96, 32)
    draw_chip(screen, tag_rect, fonts["small"], tag, tag_fill, tag_text, tag_border, light=light)

    preview_rect = pygame.Rect(rect.right - 132, rect.y + 84, 108, 122)
    draw_mode_preview(
        screen,
        preview_rect,
        accent_color,
        "classic" if key_hint == "1" else "traditional",
        light=light,
    )

    title_surface = fonts["card_title"].render(title, True, card_title_color)
    title_y = rect.y + 84
    screen.blit(title_surface, (rect.x + 24, title_y))

    text_bottom = _draw_text_block(
        screen,
        fonts["card_text"],
        card_text_color,
        rect.x + 24,
        title_y + title_surface.get_height() + 18,
        rect.width - preview_rect.width - 48,
        (lines[0],),
        line_gap=6,
        paragraph_gap=0,
        max_lines=3,
    )

    feature_y = max(text_bottom + 18, preview_rect.bottom + 16)
    feature_x = rect.x + 24
    for feature in features:
        label = fonts["small"].render(feature, True, feature_text)
        chip_rect = pygame.Rect(feature_x, feature_y, label.get_width() + 22, label.get_height() + 10)
        if chip_rect.right > rect.right - 24:
            break
        draw_chip(screen, chip_rect, fonts["small"], feature, feature_fill, feature_text, feature_border, light=light)
        feature_x += chip_rect.width + 10


def draw_level_card(screen, rect, fonts, accent_color, key_hint, level, tag, summary, light=False):
    if light:
        fill = (252, 253, 255)
        key_text = (255, 255, 255)
        tag_fill = (232, 238, 250)
        tag_text = (60, 65, 85)
        tag_border = (170, 185, 215)
        card_title_color = (22, 26, 42)
        card_text_color = (100, 106, 128)
        stat_color = (110, 115, 135)
    else:
        fill = (15, 23, 36)
        key_text = (10, 14, 22)
        tag_fill = (27, 37, 54)
        tag_text = (233, 238, 245)
        tag_border = (86, 108, 141)
        card_title_color = (247, 248, 252)
        card_text_color = (212, 219, 230)
        stat_color = (228, 234, 243)

    draw_panel(screen, rect, fill, accent_color, shadow_offset=8, border_radius=10, light=light)

    accent_bar = pygame.Rect(rect.x + 20, rect.y + 18, rect.width - 40, 5)
    pygame.draw.rect(screen, accent_color, accent_bar, border_radius=2)

    key_rect = pygame.Rect(rect.x + 22, rect.y + 32, 68, 34)
    draw_chip(screen, key_rect, fonts["keycap"], key_hint, accent_color, key_text, light=light)

    tag_rect = pygame.Rect(rect.right - 118, rect.y + 33, 94, 30)
    draw_chip(screen, tag_rect, fonts["small"], tag, tag_fill, tag_text, tag_border, light=light)

    preview_rect = pygame.Rect(rect.right - 120, rect.y + 88, 98, 112)
    draw_mode_preview(screen, preview_rect, accent_color, "classic", light=light)

    title_surface = fonts["card_title"].render(level.menu_title, True, card_title_color)
    screen.blit(title_surface, (rect.x + 22, rect.y + 84))

    summary_bottom = _draw_text_block(
        screen,
        fonts["card_text"],
        card_text_color,
        rect.x + 22,
        rect.y + 130,
        rect.width - preview_rect.width - 42,
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
        if stat_y + fonts["stat"].get_height() > rect.bottom - 18:
            break
        stat_surface = fonts["stat"].render(stat, True, stat_color)
        screen.blit(stat_surface, (rect.x + 22, stat_y))
        stat_y += stat_surface.get_height() + 12


def draw_mode_select_menu(screen, config, fonts):
    draw_menu_background(screen, config, light=True)

    width, _height = screen.get_size()
    margin = max(48, width // 32)
    hero_width = min(width - margin * 2, 1180)
    hero_rect = pygame.Rect(width // 2 - hero_width // 2, 44, hero_width, 230)

    draw_hero_banner(
        screen,
        hero_rect,
        fonts,
        "俄罗斯方块 AI 对战",
        "选择模式，开始与 AI 对战。",
        ("双屏同步", "竞技对抗", "中文界面"),
        light=True,
    )

    section_bottom = draw_section_label(
        screen,
        fonts,
        width // 2 - hero_width // 2,
        hero_rect.bottom + 24,
        "MODE SELECT",
        "选择对战规则",
        (82, 165, 230),
        light=True,
    )

    mode_order = ("CLASSIC", "TRADITIONAL")
    key_hints = {
        "CLASSIC": "1",
        "TRADITIONAL": "2",
    }
    accent_colors = {
        "CLASSIC": (78, 170, 230),
        "TRADITIONAL": (240, 142, 96),
    }

    card_gap = 24
    card_count = len(mode_order)
    card_width = min(420, (width - margin * 2 - card_gap * (card_count - 1)) // card_count)
    card_height = 300
    start_x = width // 2 - (card_width * card_count + card_gap * (card_count - 1)) // 2
    card_y = section_bottom + 26

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
            light=True,
        )

    draw_action_bar(
        screen,
        fonts,
        ("按 1 进入经典模式", "按 2 进入对抗模式", "数字键可直接进入"),
        light=True,
    )


def draw_classic_level_menu(screen, config, fonts):
    draw_menu_background(screen, config, light=True)

    width, _height = screen.get_size()
    margin = max(48, width // 32)
    hero_width = min(width - margin * 2, 1220)
    hero_rect = pygame.Rect(width // 2 - hero_width // 2, 44, hero_width, 220)

    draw_hero_banner(
        screen,
        hero_rect,
        fonts,
        "选择 AI 等级",
        "建议先从初级开始。",
        ("初级最慢", "中级均衡", "高级最快"),
        light=True,
    )

    section_bottom = draw_section_label(
        screen,
        fonts,
        width // 2 - hero_width // 2,
        hero_rect.bottom + 24,
        "AI LEVEL",
        "选择 AI 强度",
        (82, 165, 230),
        light=True,
    )

    card_gap = 22
    card_width = min(350, (width - margin * 2 - card_gap * 2) // 3)
    card_height = 300
    start_x = width // 2 - (card_width * 3 + card_gap * 2) // 2
    card_y = section_bottom + 26

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
            light=True,
        )

    draw_action_bar(
        screen,
        fonts,
        ("按 1/2/3 选择等级", "左右单步移动", "M 返回模式选择"),
        light=True,
    )
