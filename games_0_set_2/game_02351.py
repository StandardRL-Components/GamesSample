
# Generated: 2025-08-28T04:32:11.582330
# Source Brief: brief_02351.md
# Brief Index: 2351

        
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
        "Controls: ←→ to move the catcher. Catch the falling fruit to score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Catch falling fruit in a vibrant, fast-paced arcade game. Reach 50 points to win, but be careful! Missing 10 fruits ends the game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and game constants
        self.W, self.H = 640, 400
        self.FPS = 30
        self.WIN_SCORE = 50
        self.LOSE_MISSES = 10
        self.MAX_STEPS = 1000

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        self.font_tiny = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG_TOP = (135, 206, 250)  # Light Sky Blue
        self.COLOR_BG_BOTTOM = (144, 238, 144)  # Light Green
        self.COLOR_CATCHER = (255, 140, 0)  # DarkOrange
        self.COLOR_CATCHER_OUTLINE = (205, 102, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        self.COLOR_MISS_ICON = (255, 0, 0)
        self.FRUIT_COLORS = [
            (255, 0, 0),    # Red (Apple)
            (255, 255, 0),  # Yellow (Banana)
            (0, 128, 0),    # Green (Pear)
            (128, 0, 128),  # Purple (Grape)
            (255, 165, 0),  # Orange (Orange)
        ]
        
        # Pre-render background for performance
        self.bg_surface = self._create_gradient_background()

        # Initialize state variables
        self.catcher_pos_x = 0
        self.score = 0
        self.missed_fruits = 0
        self.steps = 0
        self.game_over = False
        self.fruits = []
        self.particles = []
        self.base_fruit_speed = 0
        self.current_fruit_speed = 0
        self.base_spawn_rate = 0
        self.current_spawn_rate = 0
        self.spawn_timer = 0
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.missed_fruits = 0
        self.game_over = False
        self.fruits = []
        self.particles = []

        self.catcher_pos_x = self.W / 2
        self.catcher_width = 100
        self.catcher_height = 20
        self.catcher_speed = 10

        self.base_fruit_speed = 2.0
        self.current_fruit_speed = self.base_fruit_speed
        self.base_spawn_rate = 1.0  # fruits per second
        self.current_spawn_rate = self.base_spawn_rate
        self.spawn_timer = 1.0 / self.current_spawn_rate

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 3=left, 4=right

            # --- 1. Calculate pre-move reward ---
            old_catcher_x = self.catcher_pos_x
            if self.fruits:
                # Find nearest fruit horizontally
                nearest_fruit = min(self.fruits, key=lambda f: abs(f['pos'].x - self.catcher_pos_x))
                dist_before = abs(nearest_fruit['pos'].x - self.catcher_pos_x)
            else:
                dist_before = 0

            # --- 2. Update Catcher Position ---
            if movement == 3:  # Left
                self.catcher_pos_x -= self.catcher_speed
            elif movement == 4:  # Right
                self.catcher_pos_x += self.catcher_speed
            
            # Clamp catcher to screen
            self.catcher_pos_x = max(self.catcher_width / 2, min(self.W - self.catcher_width / 2, self.catcher_pos_x))

            # --- 3. Calculate post-move reward ---
            if self.fruits and old_catcher_x != self.catcher_pos_x:
                dist_after = abs(nearest_fruit['pos'].x - self.catcher_pos_x)
                if dist_after < dist_before:
                    reward += 1.0  # Reward for moving towards fruit
                else:
                    reward -= 0.1  # Penalty for moving away
            
            # --- 4. Update Game Logic ---
            self._update_difficulty()
            self._spawn_fruits()
            
            catch_reward, miss_penalty = self._update_fruits()
            reward += catch_reward + miss_penalty

            self._update_particles()
        
        # --- 5. Check Termination ---
        self.steps += 1
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += 100  # Goal-oriented win reward
            self.game_over = True
        elif self.missed_fruits >= self.LOSE_MISSES:
            terminated = True
            reward -= 100  # Goal-oriented lose penalty
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _update_difficulty(self):
        self.current_fruit_speed = self.base_fruit_speed + (self.score // 5) * 0.1
        self.current_spawn_rate = self.base_spawn_rate + (self.score // 10) * 0.01

    def _spawn_fruits(self):
        self.spawn_timer -= 1.0 / self.FPS
        if self.spawn_timer <= 0:
            self.spawn_timer += 1.0 / self.current_spawn_rate
            new_fruit = {
                'pos': pygame.Vector2(self.np_random.uniform(20, self.W - 20), -20),
                'color': random.choice(self.FRUIT_COLORS),
                'radius': self.np_random.integers(10, 16),
                'speed': self.current_fruit_speed * self.np_random.uniform(0.8, 1.2)
            }
            self.fruits.append(new_fruit)

    def _update_fruits(self):
        catch_reward = 0
        miss_penalty = 0
        catcher_rect = pygame.Rect(
            self.catcher_pos_x - self.catcher_width / 2,
            self.H - self.catcher_height - 10,
            self.catcher_width,
            self.catcher_height
        )

        for fruit in self.fruits[:]:
            fruit['pos'].y += fruit['speed']

            fruit_rect = pygame.Rect(
                fruit['pos'].x - fruit['radius'],
                fruit['pos'].y - fruit['radius'],
                fruit['radius'] * 2,
                fruit['radius'] * 2
            )

            if catcher_rect.colliderect(fruit_rect):
                # SFX: Catch sound
                self.score += 1
                catch_reward += 5
                self._create_particles(fruit['pos'], (255, 255, 255), 20)
                self.fruits.remove(fruit)
            elif fruit['pos'].y > self.H + fruit['radius']:
                # SFX: Miss sound
                self.missed_fruits += 1
                self._create_particles(pygame.Vector2(fruit['pos'].x, self.H - 5), (255, 0, 0), 10)
                self.fruits.remove(fruit)
        
        return catch_reward, miss_penalty

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.uniform(0.5, 1.0) * self.FPS,
                'radius': self.np_random.uniform(2, 5),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self.screen.blit(self.bg_surface, (0, 0))

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render fruits
        for fruit in self.fruits:
            pos = (int(fruit['pos'].x), int(fruit['pos'].y))
            radius = int(fruit['radius'])
            color = fruit['color']
            outline_color = tuple(max(0, c - 50) for c in color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, outline_color)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / (self.FPS * 1.0)))
            if alpha > 0:
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (int(p['radius']), int(p['radius'])), int(p['radius']))
                self.screen.blit(temp_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

        # Render catcher
        catcher_rect = pygame.Rect(
            self.catcher_pos_x - self.catcher_width / 2,
            self.H - self.catcher_height - 10,
            self.catcher_width,
            self.catcher_height
        )
        pygame.draw.rect(self.screen, self.COLOR_CATCHER, catcher_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_CATCHER_OUTLINE, catcher_rect, width=3, border_radius=5)
        
    def _render_ui(self):
        # Helper to draw shadowed text
        def draw_text(text, font, color, pos, shadow_color):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Score display
        score_text = f"Score: {self.score}"
        draw_text(score_text, self.font_small, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)

        # Misses display
        miss_text = "Misses:"
        draw_text(miss_text, self.font_tiny, self.COLOR_TEXT, (10, 45), self.COLOR_TEXT_SHADOW)
        for i in range(self.LOSE_MISSES):
            pos = (90 + i * 20, 55)
            if i < self.missed_fruits:
                color = self.COLOR_MISS_ICON
            else:
                color = (100, 100, 100)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 7, (50, 50, 50))

        # Game Over message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
            elif self.missed_fruits >= self.LOSE_MISSES:
                msg = "GAME OVER"
            else:
                msg = "TIME'S UP!"
            
            draw_text(msg, self.font_large, self.COLOR_TEXT, (self.W/2 - self.font_large.size(msg)[0]/2, self.H/2 - 50), self.COLOR_TEXT_SHADOW)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_fruits": self.missed_fruits,
        }

    def _create_gradient_background(self):
        bg = pygame.Surface((self.W, self.H))
        for y in range(self.H):
            r = self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * y / self.H
            g = self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * y / self.H
            b = self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * y / self.H
            pygame.draw.line(bg, (int(r), int(g), int(b)), (0, y), (self.W, y))
        return bg

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")