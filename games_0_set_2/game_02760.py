
# Generated: 2025-08-27T21:20:48.656296
# Source Brief: brief_02760.md
# Brief Index: 2760

        
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
    user_guide = "Controls: Use ← and → to move the catcher. Catch the falling fruit!"

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced arcade game. Catch falling fruit in your basket to score points. Miss five and it's game over!"

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.GRID_SQUARE_SIZE = 400
        self.CELL_SIZE = self.GRID_SQUARE_SIZE // self.GRID_SIZE
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_SQUARE_SIZE) // 2
        self.GRID_Y_OFFSET = 0
        self.CATCH_LINE_Y = self.GRID_SQUARE_SIZE - self.CELL_SIZE
        self.WIN_SCORE = 50
        self.MAX_MISSES = 5
        self.MAX_STEPS = 1000
        self.BASE_FALL_SPEED = 3.0
        self.SPEED_INCREMENT = 0.5

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 60, 80)
        self.COLOR_CATCHER = (139, 69, 19)
        self.COLOR_CATCHER_RIM = (160, 82, 45)
        self.COLOR_SCORE = (255, 255, 255)
        self.COLOR_MISS = (255, 50, 50)
        self.FRUIT_COLORS = {
            "apple": (220, 20, 60),
            "banana": (255, 223, 0),
            "grape": (128, 0, 128),
        }
        self.FRUIT_TYPES = list(self.FRUIT_COLORS.keys())

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
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.catcher_pos = 0
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.fruit_spawn_timer = 0
        self.game_over = False
        self.win_message = ""
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.catcher_pos = self.GRID_SIZE // 2 - 1
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.fruit_spawn_timer = 20  # Spawn first fruit quickly
        self.game_over = False
        self.win_message = ""
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # 1. Update Catcher Position & Calculate Continuous Reward
        prev_pos = self.catcher_pos
        if movement == 3:  # Left
            self.catcher_pos = max(0, self.catcher_pos - 1)
        elif movement == 4:  # Right
            self.catcher_pos = min(self.GRID_SIZE - 1, self.catcher_pos + 1)
        
        moved = self.catcher_pos != prev_pos
        if moved:
            # Find the lowest fruit to calculate directional reward
            lowest_fruit = None
            if self.fruits:
                lowest_fruit = max(self.fruits, key=lambda f: f['y'])
            
            if lowest_fruit:
                target_col = lowest_fruit['col']
                # Moving towards the fruit
                if (movement == 3 and self.catcher_pos < target_col) or \
                   (movement == 4 and self.catcher_pos > target_col):
                    reward += 0.1
                # Moving away from the fruit
                else:
                    reward -= 0.1

        # 2. Update Game Logic
        self.steps += 1
        
        # Update particles
        self._update_particles()
        
        # Update fruits
        self._update_fruits()
        reward += self._check_collisions()
        
        # Spawn new fruits
        self._spawn_fruit()

        # 3. Check Termination
        terminated = False
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += 10
            self.win_message = "YOU WIN!"
            self.game_over = True
            # sfx: win_sound
        elif self.misses >= self.MAX_MISSES:
            terminated = True
            reward -= 10
            self.win_message = "GAME OVER"
            self.game_over = True
            # sfx: lose_sound
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 10 # Penalty for running out of time
            self.win_message = "TIME UP!"
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_fruits(self):
        fall_speed = self.BASE_FALL_SPEED + (self.score // 10) * self.SPEED_INCREMENT
        for fruit in self.fruits:
            fruit['y'] += fall_speed

    def _check_collisions(self):
        event_reward = 0
        for fruit in self.fruits[:]:
            if fruit['y'] >= self.CATCH_LINE_Y:
                fruit_center_x = self.GRID_X_OFFSET + fruit['col'] * self.CELL_SIZE + self.CELL_SIZE // 2
                if fruit['col'] == self.catcher_pos:
                    # Catch
                    self.score += 1
                    event_reward += 1
                    self._create_particles(fruit_center_x, fruit['y'] + self.CELL_SIZE // 2, "catch")
                    # sfx: catch_sound
                else:
                    # Miss
                    self.misses += 1
                    event_reward -= 1
                    self._create_particles(fruit_center_x, self.GRID_SQUARE_SIZE, "miss")
                    # sfx: miss_sound
                self.fruits.remove(fruit)
        return event_reward

    def _spawn_fruit(self):
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            col = self.np_random.integers(0, self.GRID_SIZE)
            fruit_type = self.np_random.choice(self.FRUIT_TYPES)
            self.fruits.append({'col': col, 'y': -self.CELL_SIZE, 'type': fruit_type})
            self.fruit_spawn_timer = self.np_random.integers(45, 75)

    def _create_particles(self, x, y, particle_type):
        if particle_type == "catch":
            for _ in range(20):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                self.particles.append({
                    'x': x, 'y': y,
                    'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                    'life': self.np_random.integers(15, 30),
                    'color': (255, 255, self.np_random.integers(100, 255)),
                    'size': self.np_random.integers(2, 5)
                })
        elif particle_type == "miss":
            for _ in range(15):
                angle = self.np_random.uniform(math.pi, 2 * math.pi) # Downward splash
                speed = self.np_random.uniform(0.5, 2)
                self.particles.append({
                    'x': x, 'y': y,
                    'vx': math.cos(angle) * speed, 'vy': -abs(math.sin(angle) * speed),
                    'life': self.np_random.integers(20, 40),
                    'color': (100, 100, 120),
                    'size': self.np_random.integers(2, 4)
                })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_fruits()
        self._render_catcher()
        self._render_particles()

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end_pos = (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_SQUARE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + i * self.CELL_SIZE)
            end_pos = (self.GRID_X_OFFSET + self.GRID_SQUARE_SIZE, self.GRID_Y_OFFSET + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            
    def _render_catcher(self):
        catcher_x = self.GRID_X_OFFSET + self.catcher_pos * self.CELL_SIZE
        catcher_y = self.CATCH_LINE_Y
        rect = pygame.Rect(catcher_x, catcher_y, self.CELL_SIZE, self.CELL_SIZE)
        
        # Draw the basket body
        pygame.draw.rect(self.screen, self.COLOR_CATCHER, rect.inflate(-10, -20).move(0, 10))
        # Draw the rim
        pygame.draw.arc(self.screen, self.COLOR_CATCHER_RIM, rect.inflate(-10, 0), 0, math.pi, 5)

    def _render_fruits(self):
        for fruit in self.fruits:
            center_x = self.GRID_X_OFFSET + fruit['col'] * self.CELL_SIZE + self.CELL_SIZE // 2
            center_y = int(fruit['y'] + self.CELL_SIZE // 2)
            self._draw_fruit(self.screen, fruit['type'], center_x, center_y, self.CELL_SIZE // 2 - 5)

    def _draw_fruit(self, surface, fruit_type, x, y, size):
        if fruit_type == "apple":
            pygame.gfxdraw.aacircle(surface, x, y, size, self.FRUIT_COLORS[fruit_type])
            pygame.gfxdraw.filled_circle(surface, x, y, size, self.FRUIT_COLORS[fruit_type])
            pygame.draw.line(surface, (34, 139, 34), (x, y - size), (x + 2, y - size - 5), 3) # Stem
        elif fruit_type == "banana":
            rect = pygame.Rect(x - size, y - size // 2, size * 2, size)
            pygame.draw.arc(surface, self.FRUIT_COLORS[fruit_type], rect, math.pi / 4, math.pi * 3 / 4, size // 2)
        elif fruit_type == "grape":
            offsets = [(-size//2, 0), (size//2, 0), (0, -size//2), (-size//4, size//3), (size//4, size//3)]
            for dx, dy in offsets:
                pygame.gfxdraw.aacircle(surface, x + dx, y + dy, size // 2, self.FRUIT_COLORS[fruit_type])
                pygame.gfxdraw.filled_circle(surface, x + dx, y + dy, size // 2, self.FRUIT_COLORS[fruit_type])

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 10))
        
        # Misses
        miss_text = self.font_medium.render("MISSES:", True, self.COLOR_SCORE)
        self.screen.blit(miss_text, (self.SCREEN_WIDTH - 180, 10))
        for i in range(self.MAX_MISSES):
            color = self.COLOR_MISS if i < self.misses else self.COLOR_GRID
            x = self.SCREEN_WIDTH - 65 + i * 12
            pygame.draw.line(self.screen, color, (x, 15), (x + 8, 25), 3)
            pygame.draw.line(self.screen, color, (x, 25), (x + 8, 15), 3)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_SCORE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    pygame.display.set_caption("Fruit Catcher")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op by default
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        # The MultiDiscrete action is [movement, space, shift]
        action = [movement, 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Reset after a delay
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS
        
    pygame.quit()