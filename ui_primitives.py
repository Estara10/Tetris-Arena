"""Shared rendering primitives used across all UI layers."""

import pygame


# ---- pure color utilities -----------------------------------------------


def lerp_color(color_a, color_b, ratio: float):
    """Linear interpolation between two RGB tuples."""
    return tuple(
        int(left + (right - left) * ratio)
        for left, right in zip(color_a, color_b)
    )


def tint(color, amount: float):
    """Lighten towards white."""
    return tuple(
        max(0, min(255, int(channel + (255 - channel) * amount)))
        for channel in color
    )


def shade(color, amount: float):
    """Darken towards black."""
    return tuple(
        max(0, min(255, int(channel * (1.0 - amount))))
        for channel in color
    )


# ---- drawing primitives ------------------------------------------------


def draw_vertical_gradient(screen, rect, top_color, bottom_color):
    """Fill a rect with a vertical linear gradient."""
    for offset_y in range(rect.height):
        ratio = offset_y / max(1, rect.height - 1)
        color = lerp_color(top_color, bottom_color, ratio)
        pygame.draw.line(
            screen, color,
            (rect.x, rect.y + offset_y),
            (rect.right, rect.y + offset_y),
        )


def draw_soft_glow(screen, rect, color, spread=16, alpha=32, border_radius=8):
    """Draw a soft translucent glow behind a rect."""
    gw = rect.width + spread * 2
    gh = rect.height + spread * 2
    glow = pygame.Surface((gw, gh), pygame.SRCALPHA)
    glow_rect = pygame.Rect(spread, spread, rect.width, rect.height)
    pygame.draw.rect(
        glow,
        (color[0], color[1], color[2], alpha),
        glow_rect,
        border_radius=border_radius,
    )
    screen.blit(glow, (rect.x - spread, rect.y - spread))


def draw_card(
    screen, rect, fill_color, border_color,
    glow_color=None, border_radius=8,
    shadow_offset=(14, 10), shadow_alpha=84,
    inner_border_color=(42, 56, 78), inner_border_width=1,
    border_width=2,
):
    """Draw a card with shadow, optional glow, fill, inner border, and outer border."""
    sox, soy = shadow_offset
    sw = rect.width + abs(sox) * 2 + 8
    sh = rect.height + abs(soy) * 2 + 8
    shadow = pygame.Surface((sw, sh), pygame.SRCALPHA)
    pygame.draw.rect(
        shadow,
        (0, 0, 0, shadow_alpha),
        pygame.Rect(abs(sox), abs(soy), rect.width, rect.height),
        border_radius=border_radius,
    )
    screen.blit(shadow, (rect.x - abs(sox), rect.y - abs(soy)))

    if glow_color is not None:
        draw_soft_glow(screen, rect, glow_color, spread=14, alpha=34, border_radius=border_radius)

    pygame.draw.rect(screen, fill_color, rect, border_radius=border_radius)

    if inner_border_width > 0 and inner_border_color is not None:
        pygame.draw.rect(
            screen, inner_border_color,
            rect.inflate(-4, -4),
            inner_border_width,
            border_radius=max(4, border_radius - 2),
        )

    pygame.draw.rect(screen, border_color, rect, border_width, border_radius=border_radius)
