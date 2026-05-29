"""Visual effects: particles, floating text, screen shake, lock flash."""

import math
import random

import pygame


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "color", "alpha", "lifetime", "elapsed", "size")

    def __init__(self, x, y, vx, vy, color, lifetime, size=3):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.color = color
        self.alpha = 255
        self.lifetime = lifetime
        self.elapsed = 0
        self.size = size


class VisualEffects:
    def __init__(self):
        self.particles: list[Particle] = []
        self.floating_texts: list[dict] = []
        self.lock_flashes: list[dict] = []
        self._shake_intensity = 0.0
        self._shake_duration = 0
        self._shake_elapsed = 0

    # ---- spawn helpers --------------------------------------------------

    def spawn_line_clear_particles(self, x, y, width, color, count=30):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(40, 220)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - random.uniform(60, 200)
            p = Particle(
                x + random.uniform(0, width),
                y + random.uniform(-6, 6),
                vx,
                vy,
                color,
                random.randint(350, 750),
                random.randint(2, 5),
            )
            self.particles.append(p)

    def spawn_floating_text(self, text, x, y, color=(255, 255, 255), duration=1400):
        self.floating_texts.append(
            {
                "text": text,
                "x": x,
                "y": y,
                "color": color,
                "duration": duration,
                "elapsed": 0,
            }
        )

    def spawn_lock_flash(self, cells, cell_size, board_x, board_y, duration=180, color=(255, 255, 255)):
        self.lock_flashes.append(
            {
                "cells": list(cells),
                "cell_size": cell_size,
                "board_x": board_x,
                "board_y": board_y,
                "duration": duration,
                "elapsed": 0,
                "color": color,
            }
        )

    def trigger_shake(self, intensity=6.0, duration=350):
        if intensity > self._shake_intensity or self._shake_elapsed >= self._shake_duration:
            self._shake_intensity = intensity
            self._shake_duration = duration
            self._shake_elapsed = 0

    # ---- update ---------------------------------------------------------

    def update(self, dt: int):
        for p in self.particles:
            p.elapsed += dt
            p.x += p.vx * dt / 1000.0
            p.y += p.vy * dt / 1000.0
            p.vy += 420 * dt / 1000.0
            progress = min(1.0, p.elapsed / max(1, p.lifetime))
            p.alpha = max(0, int(255 * (1.0 - progress * progress)))
        self.particles = [p for p in self.particles if p.elapsed < p.lifetime]

        for ft in self.floating_texts:
            ft["elapsed"] += dt
            ft["y"] -= 50 * dt / 1000.0
        self.floating_texts = [
            ft for ft in self.floating_texts if ft["elapsed"] < ft["duration"]
        ]

        for lf in self.lock_flashes:
            lf["elapsed"] += dt
        self.lock_flashes = [
            lf for lf in self.lock_flashes if lf["elapsed"] < lf["duration"]
        ]

        if self._shake_elapsed < self._shake_duration:
            self._shake_elapsed += dt

    @property
    def is_shaking(self):
        return self._shake_elapsed < self._shake_duration

    def get_shake_offset(self):
        if self._shake_elapsed >= self._shake_duration:
            return (0, 0)
        progress = self._shake_elapsed / max(1, self._shake_duration)
        decay = (1.0 - progress) ** 2
        intensity = self._shake_intensity * decay
        return (
            int(random.uniform(-intensity, intensity)),
            int(random.uniform(-intensity, intensity)),
        )

    # ---- draw -----------------------------------------------------------

    def draw(self, screen, font=None):
        for p in self.particles:
            if p.alpha <= 0:
                continue
            r = max(1, p.size)
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p.color, p.alpha), (r, r), r)
            screen.blit(s, (int(p.x - r), int(p.y - r)))

        if font is not None:
            for ft in self.floating_texts:
                progress = min(1.0, ft["elapsed"] / max(1, ft["duration"]))
                alpha = max(0, int(255 * (1.0 - progress)))
                if alpha <= 0:
                    continue
                text_surf = font.render(ft["text"], True, ft["color"])
                text_surf.set_alpha(alpha)
                screen.blit(
                    text_surf,
                    (int(ft["x"] - text_surf.get_width() // 2), int(ft["y"])),
                )

        for lf in self.lock_flashes:
            progress = min(1.0, lf["elapsed"] / max(1, lf["duration"]))
            alpha = max(0, int(200 * (1.0 - progress)))
            if alpha <= 0:
                continue
            for col, row in lf["cells"]:
                px = lf["board_x"] + col * lf["cell_size"]
                py = lf["board_y"] + row * lf["cell_size"]
                fs = pygame.Surface((lf["cell_size"], lf["cell_size"]), pygame.SRCALPHA)
                c = lf.get("color", (255, 255, 255))
                fs.fill((c[0], c[1], c[2], alpha))
                screen.blit(fs, (px, py))
