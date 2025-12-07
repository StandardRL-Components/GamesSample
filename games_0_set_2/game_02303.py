
# Generated: 2025-08-28T04:23:05.292727
# Source Brief: brief_02303.md
# Brief Index: 2303

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the ninja. Collect numbers to reach the target sum."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A grid-based arcade puzzle. Control a ninja to collect numbers and reach a target sum of 100 before time runs out. Positive numbers are green, negative numbers are red."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    GRID_MARGIN_X = 40
    GRID_MARGIN_Y = 60
    CELL_WIDTH = (SCREEN_WIDTH - 2 * GRID_MARGIN_X) // GRID_COLS
    CELL_HEIGHT = (SCREEN_HEIGHT - GRID_MARGIN_Y - 20) // GRID_ROWS
    
    FPS = 30
    GAME_DURATION_SECONDS = 60
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_NINJA = (70, 140, 255)
    COLOR_NINJA_EYE = (255, 255, 255)
    COLOR_POSITIVE = (100, 255, 150)
    COLOR_NEGATIVE = (255, 100, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TARGET = (255, 220, 100)

    MOVE_DURATION_FRAMES = 5  # How many frames a move takes

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        self.font_huge = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_sum = 0
        self.target_sum = 100
        self.time_limit_frames = self.GAME_DURATION_SECONDS * self.FPS
        self.time_remaining = 0
        self.difficulty_tier = 0
        
        self.ninja_grid_pos = [0, 0]
        self.ninja_pixel_pos = [0.0, 0.0]
        self.ninja_move_progress = 0
        self.ninja_move_origin = [0, 0]
        self.ninja_move_target = [0, 0]
        
        self.numbers = []
        self.particles = []
        self.number_spawn_timer = 0
        self.last_reward_info = {"type": "none", "value": 0, "pos": (0,0), "alpha": 0}

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_sum = 0
        self.time_remaining = self.time_limit_frames
        self.difficulty_tier = 0
        
        self.ninja_grid_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.ninja_pixel_pos = self._grid_to_pixel(self.ninja_grid_pos)
        self.ninja_move_progress = 0
        
        self.numbers = []
        self.particles = []
        
        # Initial number spawn
        self.number_spawn_timer = 1
        for _ in range(5):
             self._spawn_number()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # --- Update Timers and State ---
        self.steps += 1
        self.time_remaining -= 1
        
        self._update_difficulty()
        self._handle_input(action)
        self._update_ninja_movement()
        self._update_spawners()
        
        # Check for collection after movement is complete
        collection_reward = self._check_collection()
        reward += collection_reward

        # --- Calculate Reward ---
        # Continuous rewards are handled in _check_collection
        # Event-based rewards
        if 0 < (self.target_sum - self.current_sum) <= 10 and self.current_sum < self.target_sum:
             reward += 5

        # --- Check Termination ---
        terminated = False
        if self.current_sum == self.target_sum:
            reward += 100 # Goal-oriented reward
            terminated = True
            # Sound: win_game.wav
        elif self.time_remaining <= 0:
            reward -= 50 # Penalty for time out
            terminated = True
            # Sound: lose_game.wav
        elif self.steps >= 2000: # Max step limit
            terminated = True

        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        if self.ninja_move_progress > 0:
            return # Don't accept input while moving

        movement = action[0]  # 0-4: none/up/down/left/right
        target_pos = self.ninja_grid_pos[:]

        if movement == 1: # Up
            target_pos[1] -= 1
        elif movement == 2: # Down
            target_pos[1] += 1
        elif movement == 3: # Left
            target_pos[0] -= 1
        elif movement == 4: # Right
            target_pos[0] += 1
        
        if target_pos != self.ninja_grid_pos and self._is_in_bounds(target_pos):
            self.ninja_move_origin = self._grid_to_pixel(self.ninja_grid_pos)
            self.ninja_move_target = self._grid_to_pixel(target_pos)
            self.ninja_move_progress = self.MOVE_DURATION_FRAMES
            self.ninja_grid_pos = target_pos # Update logical position immediately
            # Sound: ninja_swoosh.wav

    def _update_ninja_movement(self):
        if self.ninja_move_progress > 0:
            self.ninja_move_progress -= 1
            t = 1.0 - (self.ninja_move_progress / self.MOVE_DURATION_FRAMES)
            # Ease-out curve for smoother stop
            t = 1 - (1 - t) * (1 - t)
            self.ninja_pixel_pos[0] = self.ninja_move_origin[0] + (self.ninja_move_target[0] - self.ninja_move_origin[0]) * t
            self.ninja_pixel_pos[1] = self.ninja_move_origin[1] + (self.ninja_move_target[1] - self.ninja_move_origin[1]) * t
        else:
            self.ninja_pixel_pos = self._grid_to_pixel(self.ninja_grid_pos)

    def _update_spawners(self):
        self.number_spawn_timer -= 1
        if self.number_spawn_timer <= 0:
            self._spawn_number()
            # Cooldown decreases as time goes on, increasing spawn rate
            base_cooldown = 2 * self.FPS
            time_progress = 1 - (self.time_remaining / self.time_limit_frames)
            self.number_spawn_timer = int(base_cooldown * (1 - 0.75 * time_progress))

    def _spawn_number(self):
        if len(self.numbers) >= self.GRID_COLS * self.GRID_ROWS - 1:
            return

        min_val = -10 - self.difficulty_tier
        max_val = 20 + self.difficulty_tier
        value = self.np_random.integers(min_val, max_val + 1)
        if value == 0: value = 1 # Avoid zero-value numbers

        pos = [self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS)]
        
        # Ensure spawn location is not occupied
        occupied_pos = [n['pos'] for n in self.numbers] + [self.ninja_grid_pos]
        attempts = 0
        while pos in occupied_pos and attempts < 50:
            pos = [self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS)]
            attempts += 1
        
        if pos not in occupied_pos:
            self.numbers.append({'pos': pos, 'value': value, 'spawn_time': self.steps})

    def _update_difficulty(self):
        time_elapsed_seconds = (self.time_limit_frames - self.time_remaining) / self.FPS
        if time_elapsed_seconds > 45:
            self.difficulty_tier = 3
        elif time_elapsed_seconds > 30:
            self.difficulty_tier = 2
        elif time_elapsed_seconds > 15:
            self.difficulty_tier = 1
        else:
            self.difficulty_tier = 0

    def _check_collection(self):
        reward = 0
        collected_index = -1
        for i, num in enumerate(self.numbers):
            if num['pos'] == self.ninja_grid_pos:
                collected_index = i
                break
        
        if collected_index != -1:
            collected_num = self.numbers.pop(collected_index)
            self.current_sum += collected_num['value']
            
            # Continuous feedback rewards
            if collected_num['value'] > 0:
                reward += 1.0
            else:
                reward -= 0.5
            
            # Sound: collect_number.wav
            self._create_particles(self._grid_to_pixel(collected_num['pos']), collected_num['value'])
            
        return reward

    def _create_particles(self, pos, value):
        color = self.COLOR_POSITIVE if value > 0 else self.COLOR_NEGATIVE
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifetime': lifetime, 'max_lifetime': lifetime, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Drag
            p['vel'][1] *= 0.95
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_MARGIN_Y + r * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_MARGIN_X, y), (self.SCREEN_WIDTH - self.GRID_MARGIN_X, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_MARGIN_X + c * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_MARGIN_Y), (x, self.SCREEN_HEIGHT - 20), 1)

        # Draw numbers
        for num in self.numbers:
            px, py = self._grid_to_pixel(num['pos'])
            color = self.COLOR_POSITIVE if num['value'] > 0 else self.COLOR_NEGATIVE
            
            # Pulse effect for numbers
            pulse = math.sin((self.steps - num['spawn_time']) * 0.1) * 2
            rect = pygame.Rect(px - self.CELL_WIDTH//2, py - self.CELL_HEIGHT//2, self.CELL_WIDTH, self.CELL_HEIGHT)
            
            text_surf = self.font_large.render(str(num['value']), True, color)
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)

        # Update and draw particles
        self._update_particles()
        for p in self.particles:
            alpha = 255 * (p['lifetime'] / p['max_lifetime'])
            size = 5 * (p['lifetime'] / p['max_lifetime'])
            color_with_alpha = (*p['color'], alpha)
            
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(surf, color_with_alpha, surf.get_rect())
            self.screen.blit(surf, (int(p['pos'][0] - size/2), int(p['pos'][1] - size/2)))

        # Draw ninja
        px, py = self.ninja_pixel_pos
        
        # Bobbing animation for idle ninja
        if self.ninja_move_progress == 0:
            py += math.sin(self.steps * 0.2) * 2

        ninja_rect = pygame.Rect(px - 12, py - 12, 24, 24)
        pygame.draw.rect(self.screen, self.COLOR_NINJA, ninja_rect, border_radius=4)
        pygame.draw.circle(self.screen, self.COLOR_NINJA_EYE, (int(px), int(py - 2)), 3)

    def _render_ui(self):
        # Top bar background
        pygame.draw.rect(self.screen, (30, 35, 55), (0, 0, self.SCREEN_WIDTH, self.GRID_MARGIN_Y - 10))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.GRID_MARGIN_Y - 10), (self.SCREEN_WIDTH, self.GRID_MARGIN_Y - 10))

        # Time
        time_text = f"TIME: {self.time_remaining // self.FPS:02d}"
        time_surf = self.font_huge.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH // 2 - time_surf.get_width() // 2, 10))

        # Current Sum
        sum_text = f"SUM: {self.current_sum}"
        sum_surf = self.font_large.render(sum_text, True, self.COLOR_TEXT)
        self.screen.blit(sum_surf, (20, 15))

        # Target Sum
        target_text = f"TARGET: {self.target_sum}"
        target_surf = self.font_large.render(target_text, True, self.COLOR_TARGET)
        self.screen.blit(target_surf, (self.SCREEN_WIDTH - target_surf.get_width() - 20, 15))
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.current_sum == self.target_sum:
                end_text = "TARGET REACHED!"
                color = self.COLOR_POSITIVE
            else:
                end_text = "TIME UP!"
                color = self.COLOR_NEGATIVE
            
            end_surf = self.font_huge.render(end_text, True, color)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            overlay.blit(end_surf, end_rect)
            self.screen.blit(overlay, (0,0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "current_sum": self.current_sum,
            "time_remaining_seconds": self.time_remaining // self.FPS,
        }

    def _grid_to_pixel(self, grid_pos):
        px = self.GRID_MARGIN_X + grid_pos[0] * self.CELL_WIDTH + self.CELL_WIDTH // 2
        py = self.GRID_MARGIN_Y + grid_pos[1] * self.CELL_HEIGHT + self.CELL_HEIGHT // 2
        return [px, py]

    def _is_in_bounds(self, grid_pos):
        return 0 <= grid_pos[0] < self.GRID_COLS and 0 <= grid_pos[1] < self.GRID_ROWS

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:", info)
    
    # Run for a few steps with random actions
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if reward != 0:
            print(f"Step {i}: Action={action}, Reward={reward:.2f}, Info={info}")
        if terminated:
            print("Episode finished.")
            obs, info = env.reset()
    
    env.close()
    print("Environment closed.")