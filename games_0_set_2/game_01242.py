
# Generated: 2025-08-27T16:29:47.297103
# Source Brief: brief_01242.md
# Brief Index: 1242

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: ← to move the basket left, → to move the basket right. Catch the fruit!"
    )

    # Short, user-facing description of the game
    game_description = (
        "Catch falling fruit in a basket. Catch 20 to win, but miss 5 and you lose."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 20
        self.LOSE_MISSES = 5

        # Colors
        self.COLOR_BG = (135, 206, 235)  # Light Sky Blue
        self.COLOR_BASKET = (139, 69, 19)  # Saddle Brown
        self.COLOR_BASKET_RIM = (101, 52, 16)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        self.COLOR_MISS = (255, 0, 0, 150)
        self.FRUIT_COLORS = [
            (255, 0, 0),    # Red
            (255, 255, 0),  # Yellow
            (0, 255, 0),    # Green
            (255, 165, 0),  # Orange
        ]

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)

        # Game State Variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.basket_rect = None
        self.fruits = []
        self.particles = []
        self.miss_effects = []
        self.base_fruit_speed = 0.0
        self.current_fruit_speed = 0.0
        self.spawn_timer = 0
        self.spawn_interval = 0
        self.catches_for_spawn_rate_increase = 0
        
        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False

        # Basket
        self.BASKET_WIDTH, self.BASKET_HEIGHT = 80, 20
        self.BASKET_SPEED = 10
        basket_y = self.HEIGHT - self.BASKET_HEIGHT - 10
        self.basket_rect = pygame.Rect(
            (self.WIDTH - self.BASKET_WIDTH) / 2, basket_y, self.BASKET_WIDTH, self.BASKET_HEIGHT
        )

        # Fruits and difficulty
        self.fruits = []
        self.base_fruit_speed = 2.0
        self.current_fruit_speed = self.base_fruit_speed
        self.spawn_interval = 120  # Initial frames between spawns
        self.spawn_timer = self.spawn_interval - 30 # Spawn first fruit quickly
        self.catches_for_spawn_rate_increase = 0

        # Effects
        self.particles = []
        self.miss_effects = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # 1. Handle Player Input
        movement = action[0]
        if movement == 3:  # Left
            self.basket_rect.x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_rect.x += self.BASKET_SPEED
        
        # Clamp basket to screen
        self.basket_rect.x = np.clip(self.basket_rect.x, 0, self.WIDTH - self.BASKET_WIDTH)

        # 2. Update Game State
        
        # Update difficulty
        self.current_fruit_speed = self.base_fruit_speed + (self.steps // 100) * 0.01

        # Update fruits
        fruits_to_remove = []
        for fruit in self.fruits:
            fruit['pos'][0] += fruit['vel'][0]
            fruit['pos'][1] += fruit['vel'][1]
            fruit['rect'].center = fruit['pos']

            # Wall bounce for horizontal movement
            if fruit['rect'].left < 0 or fruit['rect'].right > self.WIDTH:
                fruit['vel'][0] *= -1

            # Check for catch
            if self.basket_rect.colliderect(fruit['rect']):
                # SFX: catch_fruit
                self.score += 1
                self.catches_for_spawn_rate_increase += 1
                if fruit['rect'].centerx < 20 or fruit['rect'].centerx > self.WIDTH - 20:
                    reward += 2.0  # Risky catch
                else:
                    reward += 1.0  # Normal catch
                self._spawn_particles(fruit['rect'].center, fruit['color'])
                fruits_to_remove.append(fruit)
                continue

            # Check for miss
            if fruit['rect'].top > self.HEIGHT:
                # SFX: miss_fruit
                self.misses += 1
                reward -= 1.0
                self._spawn_miss_effect(fruit['rect'].midbottom)
                fruits_to_remove.append(fruit)

        # Remove caught/missed fruits
        self.fruits = [f for f in self.fruits if f not in fruits_to_remove]

        # Continuous proximity reward
        for fruit in self.fruits:
            if abs(fruit['rect'].centerx - self.basket_rect.centerx) < 5:
                reward += 0.1

        # Update spawn logic
        self.spawn_timer += 1
        max_fruits = min(3, 1 + self.catches_for_spawn_rate_increase // 5)
        spawn_rate_multiplier = 0.99 ** (self.catches_for_spawn_rate_increase // 5)
        current_spawn_interval = self.spawn_interval * spawn_rate_multiplier
        
        if self.spawn_timer >= current_spawn_interval and len(self.fruits) < max_fruits:
            self._spawn_fruit()
            self.spawn_timer = 0
            
        # Update effects
        self._update_particles()
        self._update_miss_effects()

        # 3. Check for Termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100
            elif self.misses >= self.LOSE_MISSES:
                reward -= 100
            elif self.steps >= self.MAX_STEPS:
                reward -= 10

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _spawn_fruit(self):
        radius = 12
        x_pos = self.np_random.integers(radius, self.WIDTH - radius)
        # Use np.random.Generator.uniform for float values
        x_vel = self.np_random.uniform(-1.0, 1.0)
        fruit = {
            'pos': [float(x_pos), float(-radius)],
            'vel': [x_vel, self.current_fruit_speed],
            'rect': pygame.Rect(x_pos - radius, -radius * 2, radius * 2, radius * 2),
            'color': random.choice(self.FRUIT_COLORS),
            'radius': radius
        }
        self.fruits.append(fruit)

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _spawn_miss_effect(self, pos):
        self.miss_effects.append({'pos': pos, 'life': 20, 'radius': 25})

    def _update_miss_effects(self):
        for effect in self.miss_effects:
            effect['life'] -= 1
        self.miss_effects = [e for e in self.miss_effects if e['life'] > 0]

    def _check_termination(self):
        return (
            self.score >= self.WIN_SCORE
            or self.misses >= self.LOSE_MISSES
            or self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw miss effects (flashing red circles)
        for effect in self.miss_effects:
            alpha = int(150 * (effect['life'] / 20) * (math.sin(effect['life'] * 0.8) * 0.5 + 0.5))
            if alpha > 0:
                s = pygame.Surface((effect['radius']*2, effect['radius']*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (255, 0, 0, alpha), (effect['radius'], effect['radius']), effect['radius'])
                self.screen.blit(s, (int(effect['pos'][0] - effect['radius']), self.HEIGHT - effect['radius']*2))

        # Draw fruits
        for fruit in self.fruits:
            pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], fruit['radius'], fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], fruit['radius'], fruit['color'])

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

        # Draw basket
        pygame.draw.rect(self.screen, self.COLOR_BASKET, self.basket_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASKET_RIM, self.basket_rect, width=3, border_radius=5)

    def _render_text(self, text, font, pos, color=None, shadow_color=None):
        color = color or self.COLOR_TEXT
        shadow_color = shadow_color or self.COLOR_TEXT_SHADOW
        
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(topleft=pos)

        shadow_surf = font.render(text, True, shadow_color)
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)
        
    def _render_ui(self):
        self._render_text(f"Score: {self.score}", self.font_small, (10, 10))
        self._render_text(f"Misses: {self.misses}", self.font_small, (self.WIDTH - 150, 10))
        
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = (0, 255, 0)
            else:
                msg = "GAME OVER"
                color = (255, 0, 0)
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            shadow_surf = self.font_large.render(msg, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surf, (text_rect.x + 3, text_rect.y + 3))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
        }

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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    # Set up the environment with rendering
    env = GameEnv(render_mode="rgb_array")
    
    # Use a real screen for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Catcher")
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    while not done:
        # Action defaults
        movement = 0  # no-op
        space = 0
        shift = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Key presses for human control
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
        
        if keys[pygame.K_r]: # Press R to reset
            obs, info = env.reset()

        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(30)
        
    print(f"Game Over! Final Info: {info}")
    env.close()