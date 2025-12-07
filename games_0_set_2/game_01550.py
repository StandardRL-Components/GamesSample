
# Generated: 2025-08-27T17:29:54.607946
# Source Brief: brief_01550.md
# Brief Index: 1550

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ← to move left, → to move right. Catch the fruit!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you catch falling fruit in a basket. "
        "Catch 50 to win, but miss 10 and you lose!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GROUND_HEIGHT = 50
        self.WIN_SCORE = 50
        self.LOSE_MISSES = 10
        self.MAX_STEPS = 3000  # 100 seconds at 30fps

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self._init_colors()
        self._init_fonts()
        
        # Game state attributes (initialized in reset)
        self.rng = None
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.catcher_pos = None
        self.catcher_width = 80
        self.catcher_height = 20
        self.catcher_speed = 12
        self.fruits = []
        self.particles = []
        self.fruit_spawn_timer = 0
        self.fruit_spawn_rate = 50
        self.base_fruit_acceleration = 0.03
        
        # Initialize state for the first time
        self.reset()
    
    def _init_colors(self):
        """Initialize color constants for visual clarity."""
        self.COLOR_BG = (135, 206, 235)  # Sky Blue
        self.COLOR_GROUND = (34, 139, 34)  # Forest Green
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_SHADOW = (50, 50, 50)
        self.COLOR_BASKET = (139, 69, 19)  # Saddle Brown
        self.COLOR_BASKET_RIM = (160, 82, 45) # Sienna
        self.FRUIT_COLORS = {
            'apple': (220, 20, 60), # Crimson
            'orange': (255, 140, 0), # DarkOrange
            'banana': (255, 255, 0) # Yellow
        }
        self.FRUIT_STEM_COLOR = (139, 69, 19)

    def _init_fonts(self):
        """Initialize fonts, with a fallback to the default font."""
        try:
            self.font_large = pygame.font.SysFont('Arial', 36, bold=True)
            self.font_medium = pygame.font.SysFont('Arial', 24, bold=True)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.rng is None or seed is not None:
            self.rng = random.Random(seed)
        
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        
        self.catcher_pos = [self.WIDTH / 2, self.HEIGHT - self.GROUND_HEIGHT]
        self.fruits = []
        self.particles = []
        self.fruit_spawn_timer = self.fruit_spawn_rate
        self.base_fruit_acceleration = 0.03
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        
        # 1. Handle Player Input
        if movement == 3:  # Left
            self.catcher_pos[0] -= self.catcher_speed
        elif movement == 4:  # Right
            self.catcher_pos[0] += self.catcher_speed
        
        self.catcher_pos[0] = max(
            self.catcher_width / 2, 
            min(self.WIDTH - self.catcher_width / 2, self.catcher_pos[0])
        )

        # 2. Update Game Logic
        self._update_fruits()
        reward = self._check_collisions()
        self._update_particles()
        self._spawn_fruit_if_needed()
        
        # 3. Check Termination Conditions
        self.steps += 1
        terminated = False
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += 100
            self.game_over = True
        elif self.misses >= self.LOSE_MISSES:
            terminated = True
            reward -= 100
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_fruit_if_needed(self):
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            fruit_type = self.rng.choice(list(self.FRUIT_COLORS.keys()))
            fruit = {
                'pos': [self.rng.uniform(20, self.WIDTH - 20), -20],
                'vy': self.rng.uniform(1.0, 2.5),
                'size': self.rng.randint(12, 16),
                'type': fruit_type,
                'color': self.FRUIT_COLORS[fruit_type],
            }
            self.fruits.append(fruit)
            # Spawn rate gets faster as game progresses
            spawn_rate = max(15, 50 - self.score)
            self.fruit_spawn_timer = self.rng.randint(int(spawn_rate * 0.8), int(spawn_rate * 1.2))

    def _update_fruits(self):
        # Difficulty scaling: acceleration increases every 5 points
        current_accel = self.base_fruit_acceleration + (self.score // 5) * 0.005
        for fruit in self.fruits:
            fruit['vy'] += current_accel
            fruit['pos'][1] += fruit['vy']

    def _check_collisions(self):
        reward = 0
        catcher_rect = pygame.Rect(
            self.catcher_pos[0] - self.catcher_width / 2,
            self.catcher_pos[1] - self.catcher_height,
            self.catcher_width,
            self.catcher_height
        )

        for fruit in self.fruits[:]:  # Iterate on a copy for safe removal
            fruit_rect = pygame.Rect(
                fruit['pos'][0] - fruit['size'], fruit['pos'][1] - fruit['size'],
                fruit['size'] * 2, fruit['size'] * 2
            )
            # Check for catch
            if catcher_rect.colliderect(fruit_rect):
                self.score += 1
                reward += 1
                self.fruits.remove(fruit)
                # Sound effect: catch.wav
                self._create_particles(fruit['pos'], fruit['color'])
                continue

            # Check for miss
            if fruit['pos'][1] - fruit['size'] > self.HEIGHT - self.GROUND_HEIGHT:
                self.misses += 1
                reward -= 1
                self.fruits.remove(fruit)
                # Sound effect: miss.wav
        
        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            particle = {
                'pos': list(pos),
                'vel': [self.rng.uniform(-2, 2), self.rng.uniform(-3, -1)],
                'size': self.rng.uniform(2, 5),
                'color': color,
                'lifetime': self.rng.randint(15, 30)
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['lifetime'] -= 1
            p['size'] -= 0.1
            if p['lifetime'] <= 0 or p['size'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.HEIGHT - self.GROUND_HEIGHT, self.WIDTH, self.GROUND_HEIGHT))
        for fruit in self.fruits: self._draw_fruit(fruit)
        for p in self.particles: pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), max(0, int(p['size'])))
        self._draw_catcher()

    def _draw_catcher(self):
        x, y = self.catcher_pos
        w, h = self.catcher_width, self.catcher_height
        points = [(x - w / 2, y - h), (x + w / 2, y - h), (x + w / 2 - 10, y), (x - w / 2 + 10, y)]
        pygame.draw.polygon(self.screen, self.COLOR_BASKET, points)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_RIM, (x - w / 2, y - h, w, 5))

    def _draw_fruit(self, fruit):
        x, y = int(fruit['pos'][0]), int(fruit['pos'][1])
        size, color = int(fruit['size']), fruit['color']
        
        if fruit['type'] in ['apple', 'orange']:
            pygame.gfxdraw.aacircle(self.screen, x, y, size, color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, size, color)
            if fruit['type'] == 'apple':
                pygame.draw.line(self.screen, self.FRUIT_STEM_COLOR, (x, y - size), (x+2, y - size - 5), 2)
            else: # orange
                pygame.gfxdraw.filled_circle(self.screen, x, y - size + 2, 2, self.COLOR_GROUND)
        elif fruit['type'] == 'banana':
            rect = pygame.Rect(x - size, y - size, size * 2, size * 2)
            pygame.draw.arc(self.screen, color, rect, math.radians(20), math.radians(160), width=max(1, int(size * 0.6)))

    def _render_ui(self):
        def draw_text(text, font, color, pos):
            shadow_pos = (pos[0] + 2, pos[1] + 2)
            self.screen.blit(font.render(text, True, self.COLOR_UI_SHADOW), shadow_pos)
            self.screen.blit(font.render(text, True, color), pos)

        draw_text(f"Score: {self.score}", self.font_medium, self.COLOR_UI_TEXT, (10, 10))
        miss_text = f"Misses: {self.misses}/{self.LOSE_MISSES}"
        text_w, _ = self.font_medium.size(miss_text)
        draw_text(miss_text, self.font_medium, self.COLOR_UI_TEXT, (self.WIDTH - text_w - 10, 10))

        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg, color = "YOU WIN!", (100, 255, 100)
            elif self.misses >= self.LOSE_MISSES:
                msg, color = "GAME OVER", (255, 100, 100)
            else:
                msg, color = "TIME'S UP!", (255, 255, 100)
            
            text_w, text_h = self.font_large.size(msg)
            draw_text(msg, self.font_large, color, (self.WIDTH / 2 - text_w / 2, self.HEIGHT / 2 - text_h / 2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "misses": self.misses}

    def close(self):
        pygame.quit()