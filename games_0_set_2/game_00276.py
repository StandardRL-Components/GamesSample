import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → arrow keys to move the basket. "
        "Catch the falling fruit and avoid the bombs!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you catch falling fruit. "
        "Reach the target score of 100 before the 60-second timer runs out. "
        "Bombs will cost you points and time!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    WIN_SCORE = 100

    # Colors
    COLOR_BG_TOP = (15, 23, 42)
    COLOR_BG_BOTTOM = (51, 65, 85)
    COLOR_BASKET = (34, 211, 238)
    COLOR_BASKET_ACCENT = (14, 165, 233)
    COLOR_APPLE = (220, 38, 38)
    COLOR_BANANA = (234, 179, 8)
    COLOR_GRAPE = (139, 92, 246)
    COLOR_GOLDEN_APPLE = (252, 211, 77)
    COLOR_BOMB = (30, 41, 59)
    COLOR_BOMB_SKULL = (203, 213, 225)
    COLOR_TEXT = (248, 250, 252)
    COLOR_TIMER_BAR_FULL = (74, 222, 128)
    COLOR_TIMER_BAR_MID = (250, 204, 21)
    COLOR_TIMER_BAR_LOW = (239, 68, 68)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("sans-serif", 48, bold=True)
        self.font_medium = pygame.font.SysFont("sans-serif", 24, bold=True)

        # Game state variables
        self.basket_width = 80
        self.basket_height = 20
        self.basket_speed = 12

        # Difficulty scaling constants
        self.base_fruit_speed = 2.0
        self.base_bomb_spawn_interval = 150  # frames
        self.fruit_spawn_interval = 25  # frames

        # The following are initialized here to make the object valid
        # before the first reset(), and are re-initialized in reset().
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.time_remaining_frames = self.GAME_DURATION_SECONDS * self.FPS
        self.basket_x = self.SCREEN_WIDTH // 2 - self.basket_width // 2
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.fruit_speed = self.base_fruit_speed
        self.bomb_spawn_interval = self.base_bomb_spawn_interval
        self.fruit_spawn_timer = 0
        self.bomb_spawn_timer = 0  # Will be randomized in reset

        # Reward structure
        self.REWARD_FRUIT = 1
        self.REWARD_GOLDEN_FRUIT = 5
        self.REWARD_BOMB = -10
        self.REWARD_WIN = 100
        self.REWARD_LOSE = -50

        # This is called to check for API conformance and initialization errors
        # We need a valid state for it to run without errors.
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False

        self.time_remaining_frames = self.GAME_DURATION_SECONDS * self.FPS
        self.basket_x = self.SCREEN_WIDTH // 2 - self.basket_width // 2

        self.fruits = []
        self.bombs = []
        self.particles = []

        self.fruit_speed = self.base_fruit_speed
        self.bomb_spawn_interval = self.base_bomb_spawn_interval
        self.fruit_spawn_timer = 0
        self.bomb_spawn_timer = self.np_random.integers(0, self.bomb_spawn_interval // 2)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 3=left, 4=right

        step_reward = 0

        # Update game logic
        self._update_basket(movement)
        self._update_objects()

        reward_from_collisions = self._handle_collisions()
        step_reward += reward_from_collisions

        self._update_particles()
        self._update_difficulty()
        self._spawn_objects()

        # Update time and step counters
        self.time_remaining_frames -= 1
        self.steps += 1

        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.win_condition:
                step_reward += self.REWARD_WIN
            else:
                step_reward += self.REWARD_LOSE

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_basket(self, movement):
        if movement == 3:  # Left
            self.basket_x -= self.basket_speed
        elif movement == 4:  # Right
            self.basket_x += self.basket_speed

        # Clamp basket position to screen bounds
        self.basket_x = max(0, min(self.SCREEN_WIDTH - self.basket_width, self.basket_x))

    def _update_objects(self):
        for fruit in self.fruits:
            fruit['y'] += self.fruit_speed * fruit['speed_mod']
        for bomb in self.bombs:
            bomb['y'] += self.fruit_speed * bomb['speed_mod']

        # Remove objects that are off-screen
        self.fruits = [f for f in self.fruits if f['y'] < self.SCREEN_HEIGHT + f['radius']]
        self.bombs = [b for b in self.bombs if b['y'] < self.SCREEN_HEIGHT + b['radius']]

    def _handle_collisions(self):
        reward = 0
        basket_rect = pygame.Rect(self.basket_x, self.SCREEN_HEIGHT - self.basket_height - 10, self.basket_width,
                                  self.basket_height)

        # Fruit collisions
        for fruit in self.fruits[:]:
            fruit_rect = pygame.Rect(fruit['x'] - fruit['radius'], fruit['y'] - fruit['radius'], fruit['radius'] * 2,
                                     fruit['radius'] * 2)
            if basket_rect.colliderect(fruit_rect):
                if fruit['type'] == 'golden_apple':
                    self.score += 5
                    reward += self.REWARD_GOLDEN_FRUIT
                else:
                    self.score += 1
                    reward += self.REWARD_FRUIT

                self._create_particles(fruit['x'], fruit['y'], fruit['color'])
                self.fruits.remove(fruit)

        # Bomb collisions
        for bomb in self.bombs[:]:
            bomb_rect = pygame.Rect(bomb['x'] - bomb['radius'], bomb['y'] - bomb['radius'], bomb['radius'] * 2,
                                    bomb['radius'] * 2)
            if basket_rect.colliderect(bomb_rect):
                self.score = max(0, self.score - 10)
                self.time_remaining_frames = max(0, self.time_remaining_frames - 5 * self.FPS)
                reward += self.REWARD_BOMB

                self._create_particles(bomb['x'], bomb['y'], self.COLOR_BOMB_SKULL, 30)
                self.bombs.remove(bomb)

        return reward

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 100 == 0:
            self.fruit_speed = min(8.0, self.fruit_speed + 0.1)
            self.bomb_spawn_interval = max(60, self.bomb_spawn_interval - 2)

    def _spawn_objects(self):
        # Spawn fruits
        self.fruit_spawn_timer += 1
        if self.fruit_spawn_timer >= self.fruit_spawn_interval:
            self.fruit_spawn_timer = 0
            x_pos = self.np_random.integers(20, self.SCREEN_WIDTH - 20)
            speed_mod = self.np_random.uniform(0.8, 1.2)

            rand_val = self.np_random.random()
            if rand_val < 0.05:  # 5% chance for golden apple
                fruit_type = 'golden_apple'
                color = self.COLOR_GOLDEN_APPLE
                radius = 14
            elif rand_val < 0.35:
                fruit_type = 'apple'
                color = self.COLOR_APPLE
                radius = 12
            elif rand_val < 0.70:
                fruit_type = 'banana'
                color = self.COLOR_BANANA
                radius = 12  # will be drawn as arc
            else:
                fruit_type = 'grape'
                color = self.COLOR_GRAPE
                radius = 12  # will be drawn as cluster

            self.fruits.append(
                {'x': x_pos, 'y': -radius, 'type': fruit_type, 'color': color, 'radius': radius, 'speed_mod': speed_mod})

        # Spawn bombs
        self.bomb_spawn_timer += 1
        if self.bomb_spawn_timer >= self.bomb_spawn_interval:
            self.bomb_spawn_timer = 0
            x_pos = self.np_random.integers(20, self.SCREEN_WIDTH - 20)
            speed_mod = self.np_random.uniform(0.9, 1.3)
            self.bombs.append({'x': x_pos, 'y': -15, 'radius': 15, 'speed_mod': speed_mod})

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.win_condition = True
            return True
        if self.time_remaining_frames <= 0:
            self.win_condition = False
            return True
        return False

    def _get_observation(self):
        # Clear screen with a gradient background
        for y in range(self.SCREEN_HEIGHT):
            r = self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * y / self.SCREEN_HEIGHT
            g = self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * y / self.SCREEN_HEIGHT
            b = self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * y / self.SCREEN_HEIGHT
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.SCREEN_WIDTH, y))

        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_particles()
        for fruit in self.fruits:
            self._draw_fruit(fruit)
        for bomb in self.bombs:
            self._draw_bomb(bomb)
        self._draw_basket()

    def _render_ui(self):
        # Score display
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer bar
        timer_ratio = self.time_remaining_frames / (self.GAME_DURATION_SECONDS * self.FPS)
        bar_width = (self.SCREEN_WIDTH - 20) * timer_ratio
        bar_color = self.COLOR_TIMER_BAR_LOW
        if timer_ratio > 0.66:
            bar_color = self.COLOR_TIMER_BAR_FULL
        elif timer_ratio > 0.33:
            bar_color = self.COLOR_TIMER_BAR_MID

        pygame.draw.rect(self.screen, self.COLOR_BOMB, (10, 40, self.SCREEN_WIDTH - 20, 15))
        if bar_width > 0:
            pygame.draw.rect(self.screen, bar_color, (10, 40, bar_width, 15))

        # Timer text
        time_left_sec = math.ceil(self.time_remaining_frames / self.FPS)
        timer_text = self.font_medium.render(f"TIME: {time_left_sec}", True, self.COLOR_TEXT)
        text_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_text, text_rect)

        # Game over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            message = "YOU WIN!" if self.win_condition else "TIME'S UP!"
            color = self.COLOR_GOLDEN_APPLE if self.win_condition else self.COLOR_APPLE

            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": math.ceil(self.time_remaining_frames / self.FPS)
        }

    # --- Drawing Helpers ---
    def _draw_basket(self):
        basket_rect = pygame.Rect(self.basket_x, self.SCREEN_HEIGHT - self.basket_height - 10, self.basket_width,
                                  self.basket_height)
        pygame.draw.rect(self.screen, self.COLOR_BASKET, basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_ACCENT, basket_rect, width=3, border_radius=5)

    def _draw_fruit(self, fruit):
        x, y = int(fruit['x']), int(fruit['y'])
        r = fruit['radius']
        if fruit['type'] == 'apple':
            pygame.gfxdraw.aacircle(self.screen, x, y, r, fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, fruit['color'])
        elif fruit['type'] == 'golden_apple':
            # Glow effect
            glow_r = int(r * 1.8)
            glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*fruit['color'], 50), (glow_r, glow_r), glow_r)
            self.screen.blit(glow_surf, (x - glow_r, y - glow_r), special_flags=pygame.BLEND_RGBA_ADD)
            pygame.gfxdraw.aacircle(self.screen, x, y, r, fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, fruit['color'])
        elif fruit['type'] == 'banana':
            arc_rect = pygame.Rect(x - r, y - r, r * 2, r * 2)
            pygame.draw.arc(self.screen, fruit['color'], arc_rect, math.pi * 0.2, math.pi * 0.8, 5)
        elif fruit['type'] == 'grape':
            offsets = [(-r * 0.4, -r * 0.2), (r * 0.4, -r * 0.2), (0, r * 0.4), (-r * 0.2, r * 0.1), (r * 0.2, r * 0.1)]
            for ox, oy in offsets:
                pygame.gfxdraw.aacircle(self.screen, int(x + ox), int(y + oy), int(r * 0.5), fruit['color'])
                pygame.gfxdraw.filled_circle(self.screen, int(x + ox), int(y + oy), int(r * 0.5), fruit['color'])

    def _draw_bomb(self, bomb):
        x, y = int(bomb['x']), int(bomb['y'])
        r = bomb['radius']
        pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_BOMB)
        pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_BOMB)
        # Skull
        eye_r = int(r * 0.2)
        pygame.draw.circle(self.screen, self.COLOR_BOMB_SKULL, (x - int(r * 0.4), y - int(r * 0.2)), eye_r)
        pygame.draw.circle(self.screen, self.COLOR_BOMB_SKULL, (x + int(r * 0.4), y - int(r * 0.2)), eye_r)
        pygame.draw.rect(self.screen, self.COLOR_BOMB_SKULL,
                         (x - int(r * 0.15), y + int(r * 0.1), int(r * 0.3), int(r * 0.5)))

    # --- Particle System ---
    def _create_particles(self, x, y, color, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append(
                {'x': x, 'y': y, 'vx': vx, 'vy': vy, 'lifetime': lifetime, 'max_life': lifetime, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vx'] *= 0.95  # friction
            p['vy'] *= 0.95
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _draw_particles(self):
        for p in self.particles:
            life_ratio = p['lifetime'] / p['max_life']
            radius = int(life_ratio * 4)
            if radius > 0:
                # Use alpha blending for fade-out effect
                color = (*p['color'], int(life_ratio * 255))
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (int(p['x'] - radius), int(p['y'] - radius)))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

        # print("✓ Implementation validated successfully")