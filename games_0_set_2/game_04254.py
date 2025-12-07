
# Generated: 2025-08-28T01:52:50.565006
# Source Brief: brief_04254.md
# Brief Index: 4254

        
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
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move cursor. Hold Shift to cycle color. Press Space to place pixel."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fill a 10x10 grid with color. Enclose regions of other colors to capture them and score big!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_STEPS = 1000

        # Visual constants
        self.CELL_SIZE = 32
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (50, 60, 80)
        self.COLOR_EMPTY = (35, 40, 60)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_FLASH = (255, 255, 255)
        self.COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
        ]
        
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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Etc...        
        self.grid = None
        self.cursor_pos = None
        self.selected_color_idx = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.flash_effects = None
        self.np_random = None
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_color_idx = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.flash_effects = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        reward = 0

        # --- Process Inputs ---
        # 1. Movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)

        # 2. Cycle Color (on press)
        if shift_held and not self.prev_shift_held:
            self.selected_color_idx = (self.selected_color_idx + 1) % len(self.COLORS)
            # SFX: color_cycle.wav

        # 3. Place Pixel (on press)
        if space_held and not self.prev_space_held:
            reward = self._place_pixel()
            # SFX: place_pixel.wav
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # Update game logic
        self.score += reward
        self.steps += 1
        
        self._update_flash_effects()
        
        # Check Termination
        grid_full = np.all(self.grid > 0)
        time_up = self.steps >= self.MAX_STEPS
        
        terminated = grid_full or time_up
        if terminated and not self.game_over:
            if grid_full:
                reward += 100 # Victory bonus
                self.score += 100
                # SFX: victory.wav
            else:
                # SFX: game_over.wav
                pass
            self.game_over = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _place_pixel(self):
        x, y = self.cursor_pos
        if self.grid[y, x] != 0:
            return 0 # Invalid move, no reward

        placed_color_val = self.selected_color_idx + 1
        self.grid[y, x] = placed_color_val
        
        # Reward for expanding a region vs. starting a new one
        placement_reward = -0.2
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                if self.grid[ny, nx] == placed_color_val:
                    placement_reward = 1.0
                    break
        
        # Check for enclosed regions
        enclosure_reward = 0
        enclosed_regions = self._check_enclosures((x, y), placed_color_val)
        for region in enclosed_regions:
            enclosure_reward += 5 * len(region)
            # SFX: region_fill.wav
            self._add_flash_effect(region)
            for r_x, r_y in region:
                self.grid[r_y, r_x] = placed_color_val
        
        return placement_reward + enclosure_reward

    def _check_enclosures(self, placed_pos, placed_color_val):
        enclosed_regions = []
        checked_neighbors = set()

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            start_x, start_y = placed_pos[0] + dx, placed_pos[1] + dy

            if not (0 <= start_x < self.GRID_SIZE and 0 <= start_y < self.GRID_SIZE):
                continue
            
            start_color = self.grid[start_y, start_x]
            if start_color == 0 or start_color == placed_color_val or (start_x, start_y) in checked_neighbors:
                continue

            q = deque([(start_x, start_y)])
            visited = set([(start_x, start_y)])
            region_cells = []
            is_enclosed = True

            while q:
                curr_x, curr_y = q.popleft()
                region_cells.append((curr_x, curr_y))
                checked_neighbors.add((curr_x, curr_y))

                for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    next_x, next_y = curr_x + ddx, curr_y + ddy
                    
                    if not (0 <= next_x < self.GRID_SIZE and 0 <= next_y < self.GRID_SIZE):
                        is_enclosed = False
                        break 

                    if (next_x, next_y) in visited:
                        continue
                    
                    neighbor_color = self.grid[next_y, next_x]
                    if neighbor_color == start_color:
                        visited.add((next_x, next_y))
                        q.append((next_x, next_y))
                
                if not is_enclosed:
                    break
            
            if is_enclosed:
                enclosed_regions.append(region_cells)

        return enclosed_regions
    
    def _add_flash_effect(self, region_cells):
        self.flash_effects.append({"cells": region_cells, "timer": 15}) # 0.5 seconds at 30fps

    def _update_flash_effects(self):
        for effect in self.flash_effects:
            effect["timer"] -= 1
        self.flash_effects = [e for e in self.flash_effects if e["timer"] > 0]
        
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid cells
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_val = self.grid[y, x]
                color = self.COLOR_EMPTY if color_val == 0 else self.COLORS[color_val - 1]
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, color, rect)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_WIDTH, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            
        # Draw flash effects
        for effect in self.flash_effects:
            alpha = int(255 * (effect['timer'] / 15.0))
            flash_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_FLASH, alpha))
            for x, y in effect['cells']:
                self.screen.blit(flash_surface, (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE))

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score):06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Selected Color
        next_text = self.font_small.render("NEXT:", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.WIDTH - 150, 18))
        pygame.draw.rect(self.screen, self.COLORS[self.selected_color_idx], (self.WIDTH - 80, 15, 30, 30))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.WIDTH - 80, 15, 30, 30), 2)
        
        # User Guide
        guide_text = self.font_small.render(self.user_guide, True, self.COLOR_TEXT)
        guide_rect = guide_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 20))
        self.screen.blit(guide_text, guide_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
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

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for interactive testing
    pygame.display.set_caption("Pixel Fill")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action defaults
        movement = 0 # none
        space_held = 0
        shift_held = 0
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Map keyboard state to actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()
            pygame.time.wait(2000) # Pause for 2 seconds before restart
            
        clock.tick(30) # Limit to 30 FPS for interactive mode
        
    env.close()