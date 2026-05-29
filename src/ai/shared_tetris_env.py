from dataclasses import dataclass
import random
import pygame

from src.ai.ai_controller import board_profile
from src.game.game_modes import GAME_MODES
from settings import CONFIG, GameConfig
from src.app.shared_arena_match import SharedArenaMatch
from src.ai.shared_arena_model import (
    ACTION_HARD_DROP,
    ACTION_LEFT,
    ACTION_MEANING,
    ACTION_RIGHT,
    ACTION_ROTATE,
    ACTION_SOFT_DROP,
    build_shared_arena_state,
)

@dataclass
class StepOutcome:
    state: list[float]
    reward: float
    done: bool
    info: dict

class MockScreen:
    def get_size(self):
        return 800, 600
    def fill(self, *args, **kwargs): pass

class SharedTetrisEnv:
    action_meaning = ACTION_MEANING
    
    def __init__(
        self,
        mode_key="TRADITIONAL",
        config=None,
        seed=None,
        action_interval_ms: int = 100,
        max_episode_steps: int = 6000,
        init_pygame: bool = False,
    ):
        self.config = config or CONFIG
        self.mode_key = mode_key
        self.seed = seed
        self.rng = random.Random(seed)
        self.match = None
        self._prev_score = 0
        self._prev_lines = 0
        self.episode_step = 0
        
        self.action_interval_ms = max(20, int(action_interval_ms))
        self.max_episode_steps = max(100, int(max_episode_steps))
        
        if init_pygame:
            if not pygame.get_init():
                pygame.init()
        else:
            # 训练加速模式下避免完整 pygame.init，但字体模块仍需可用。
            if not pygame.font.get_init():
                pygame.font.init()
        self.screen = MockScreen()
        self.reset(seed)
        
    def reset(self, seed=None) -> list[float]:
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)
        self.match = SharedArenaMatch(self.screen, GAME_MODES[self.mode_key], self.config)
        self.match.cooldown_ms["player"] = 0
        self._prev_score = 0
        self._prev_lines = 0
        self.episode_step = 0
        
        # Initialize grid profile baseline
        player = self.match._get_entity("player")
        heights, holes = board_profile(self.match.core.grid)
        self._prev_holes = holes
        self._prev_height = sum(heights)
        
        return self._get_state_vector()
        
    def step(self, action: int) -> StepOutcome:
        if self.match.finished:
            return StepOutcome(self._get_state_vector(), 0.0, True, {})
            
        self.episode_step += 1
        
        # Reset held keys
        self.match.handle_keyup(pygame.K_a)
        self.match.handle_keyup(pygame.K_d)
        self.match.handle_keyup(pygame.K_s)
        
        if action == ACTION_LEFT:
            self.match.handle_keydown(pygame.K_a)
        elif action == ACTION_RIGHT:
            self.match.handle_keydown(pygame.K_d)
        elif action == ACTION_ROTATE:
            self.match.handle_keydown(pygame.K_l)
        elif action == ACTION_SOFT_DROP:
            self.match.handle_keydown(pygame.K_s)
        elif action == ACTION_HARD_DROP:
            self.match.handle_keydown(pygame.K_w)
            
        # Step simulation by interval
        self.match.update(self.action_interval_ms, [])
        
        player = self.match._get_entity("player")
        
        # Reward from clearing lines natively, not just score
        lines_diff = player.lines_cleared_total - self._prev_lines
        reward = lines_diff * 50.0  # massive reward for line clears
        self._prev_lines = player.lines_cleared_total
        
        # Optional: very small reward for piece lock to encourage placing pieces?
        # But `score` is +1 on lock when score_on_lock=True
        score_diff = player.score - self._prev_score
        reward += score_diff * 1.0
        self._prev_score = player.score
        
        # Dense shaping reward
        heights, holes = board_profile(self.match.core.grid)
        curr_height = sum(heights)
        curr_holes = holes
        
        # Penalty for increasing holes
        delta_holes = curr_holes - self._prev_holes
        if delta_holes > 0:
            reward -= delta_holes * 0.5
            
        # Penalty for aggressive height growth
        delta_height = curr_height - self._prev_height
        if delta_height > 0:
            reward -= delta_height * 0.05
            
        self._prev_holes = curr_holes
        self._prev_height = curr_height
        
        # Small survival reward
        reward += 0.1
        
        done = self.match.finished
        
        # Penalize for dying / losing the game
        if done and self.match.winner_id != "player":
            reward -= 50.0
        
        # Max steps logic
        if self.episode_step >= self.max_episode_steps:
            done = True
            
        return StepOutcome(self._get_state_vector(), float(reward), done, {})

    def sample_action(self):
        return self.rng.choice(list(self.action_meaning.keys()))
        
    def state_size(self):
        return len(self._get_state_vector())
        
    def action_size(self):
        return len(self.action_meaning)
        
    def _get_state_vector(self) -> list[float]:
        return build_shared_arena_state(self.match, controlled_entity_id="player")
        
    def set_opponent_profile(self, action_interval_ms: int, mistake_chance: float):
        # Traditional shared arena manages opponent through AI controller logic 
        # or fixed parameters, but we allow RL trainer to send dynamic changes seamlessly
        pass
        
    @property
    def player_core(self):
        return self.match._get_entity("player")
        
    @property
    def ai_core(self):
        return self.match._get_entity("ai1")

    @property
    def opponent_entities(self):
        return [ent for ent in self.match.entities if not ent.is_player]

    def competitive_score_gap(self) -> float:
        player_score = float(self.player_core.score)
        opponent_best = max((float(ent.score) for ent in self.opponent_entities), default=0.0)
        return player_score - opponent_best

    def competitive_line_gap(self) -> float:
        player_lines = float(self.player_core.lines_cleared_total)
        opponent_best = max(
            (float(ent.lines_cleared_total) for ent in self.opponent_entities),
            default=0.0,
        )
        return player_lines - opponent_best

    def did_player_win(self) -> bool:
        if self.match.finished:
            return self.match.winner_id == "player"
        return self.competitive_score_gap() > 0
