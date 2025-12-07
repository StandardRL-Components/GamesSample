
# Generated: 2025-08-27T18:08:14.525060
# Source Brief: brief_01738.md
# Brief Index: 1738

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select a tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A memory matching game. Find all pairs of numbers on the 4x4 grid before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION = 60 # seconds
        self.MAX_STEPS = self.GAME_DURATION * self.FPS

        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID_BG = (50, 60, 70)
        self.COLOR_TILE_HIDDEN = (70, 80, 90)
        self.COLOR_TILE_REVEALED = (180, 190, 200)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_MATCH = (80, 220, 100)
        self.COLOR_MISMATCH = (220, 80, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_NUMBER = (10, 10, 10)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_ui = pygame.font.Font(None, 36)
        self.font_tile = pygame.font.Font(None, 60)
        
        # Grid layout
        self.GRID_DIM = 4
        self.TILE_SIZE = 80
        self.GAP_SIZE = 10
        grid_total_size = self.GRID_DIM * self.TILE_SIZE + (self.GRID_DIM + 1) * self.GAP_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - grid_total_size) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - grid_total_size) // 2 + 20

        # Initialize state variables
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_tiles = []
        self.matched_values = set()
        self.mismatch_cooldown = 0
        self.time_remaining = 0.0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.last_action = np.array([0, 0, 0])
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_remaining = float(self.GAME_DURATION)
        self.cursor_pos = [0, 0]
        self.selected_tiles = []
        self.mismatch_cooldown = 0
        self.matched_values = set()
        self.last_action = np.array([0, 0, 0])

        # Create and shuffle grid
        num_pairs = (self.GRID_DIM * self.GRID_DIM) // 2
        numbers = list(range(1, num_pairs + 1)) * 2
        self.np_random.shuffle(numbers)
        
        self.grid = []
        for r in range(self.GRID_DIM):
            row = []
            for c in range(self.GRID_DIM):
                tile = {
                    'value': numbers.pop(),
                    'state': 'hidden', # 'hidden', 'revealed', 'matched'
                    'effect_timer': 0, # for match/mismatch flashes
                }
                row.append(tile)
            self.grid.append(row)
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        self.game_over = False

        # --- 1. Update Timers ---
        self.time_remaining -= 1 / self.FPS
        if self.mismatch_cooldown > 0:
            self.mismatch_cooldown -= 1
            if self.mismatch_cooldown == 0:
                # Flip mismatched tiles back
                for tile_pos in self.selected_tiles:
                    self.grid[tile_pos[1]][tile_pos[0]]['state'] = 'hidden'
                self.selected_tiles = []
        
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                if self.grid[r][c]['effect_timer'] > 0:
                    self.grid[r][c]['effect_timer'] -= 1

        # --- 2. Handle Input and Game Logic ---
        movement, space_held_int, _ = action
        space_held = space_held_int == 1
        
        # Debounced movement (triggers on new direction)
        if movement != 0 and movement != self.last_action[0]:
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_DIM - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_DIM - 1, self.cursor_pos[0] + 1)
        
        # Debounced selection (triggers on press)
        space_pressed = space_held and not (self.last_action[1] == 1)
        if space_pressed and self.mismatch_cooldown == 0 and len(self.selected_tiles) < 2:
            x, y = self.cursor_pos
            tile = self.grid[y][x]
            
            if tile['state'] == 'hidden':
                # sound: Tile flip
                tile['state'] = 'revealed'
                self.selected_tiles.append((x, y))
                reward += 0.1

                if len(self.selected_tiles) == 2:
                    pos1, pos2 = self.selected_tiles
                    tile1 = self.grid[pos1[1]][pos1[0]]
                    tile2 = self.grid[pos2[1]][pos2[0]]

                    if tile1['value'] == tile2['value']:
                        # --- MATCH ---
                        # sound: Correct match
                        tile1['state'] = 'matched'
                        tile2['state'] = 'matched'
                        tile1['effect_timer'] = self.FPS // 2 # 0.5s flash
                        tile2['effect_timer'] = self.FPS // 2
                        self.matched_values.add(tile1['value'])
                        self.selected_tiles = []
                        reward += 10.0
                    else:
                        # --- MISMATCH ---
                        # sound: Incorrect match
                        self.mismatch_cooldown = int(0.75 * self.FPS)
                        tile1['effect_timer'] = self.mismatch_cooldown
                        tile2['effect_timer'] = self.mismatch_cooldown
                        reward -= 1.0
            else:
                reward -= 0.01

        self.last_action = action
        self.steps += 1
        self.score += reward

        # --- 3. Check Termination ---
        if len(self.matched_values) == (self.GRID_DIM * self.GRID_DIM // 2):
            self.game_over = True
            final_reward = 100.0
            reward += final_reward
            self.score += final_reward
        
        if self.time_remaining <= 0:
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
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
        # Draw grid background
        grid_bg_rect = pygame.Rect(
            self.GRID_OFFSET_X, self.GRID_OFFSET_Y, 
            self.GRID_DIM * self.TILE_SIZE + (self.GRID_DIM + 1) * self.GAP_SIZE,
            self.GRID_DIM * self.TILE_SIZE + (self.GRID_DIM + 1) * self.GAP_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_bg_rect, border_radius=5)

        # Draw tiles
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                tile = self.grid[r][c]
                tile_rect = pygame.Rect(
                    self.GRID_OFFSET_X + self.GAP_SIZE + c * (self.TILE_SIZE + self.GAP_SIZE),
                    self.GRID_OFFSET_Y + self.GAP_SIZE + r * (self.TILE_SIZE + self.GAP_SIZE),
                    self.TILE_SIZE,
                    self.TILE_SIZE
                )

                if tile['state'] == 'matched':
                    if tile['effect_timer'] > 0: # Fade out effect
                        alpha = int(255 * (tile['effect_timer'] / (self.FPS // 2)))
                        s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                        pygame.draw.rect(s, (*self.COLOR_MATCH, alpha), s.get_rect(), border_radius=5)
                        self.screen.blit(s, tile_rect.topleft)
                    continue

                color = self.COLOR_TILE_HIDDEN if tile['state'] == 'hidden' else self.COLOR_TILE_REVEALED
                pygame.draw.rect(self.screen, color, tile_rect, border_radius=5)

                if tile['state'] == 'revealed':
                    num_surf = self.font_tile.render(str(tile['value']), True, self.COLOR_NUMBER)
                    num_rect = num_surf.get_rect(center=tile_rect.center)
                    self.screen.blit(num_surf, num_rect)

                if tile['effect_timer'] > 0 and tile['state'] != 'matched':
                    s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                    alpha = int(128 * (tile['effect_timer'] / (0.75 * self.FPS)))
                    s.fill((*self.COLOR_MISMATCH, alpha))
                    self.screen.blit(s, tile_rect.topleft)

        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + self.GAP_SIZE + cursor_x * (self.TILE_SIZE + self.GAP_SIZE) - 4,
            self.GRID_OFFSET_Y + self.GAP_SIZE + cursor_y * (self.TILE_SIZE + self.GAP_SIZE) - 4,
            self.TILE_SIZE + 8,
            self.TILE_SIZE + 8
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=8)
    
    def _render_ui(self):
        # Time remaining
        time_text = f"Time: {max(0, self.time_remaining):.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (20, 15))

        # Matched pairs
        pairs_text = f"Pairs: {len(self.matched_values)} / {self.GRID_DIM**2 // 2}"
        pairs_surf = self.font_ui.render(pairs_text, True, self.COLOR_TEXT)
        pairs_rect = pairs_surf.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(pairs_surf, pairs_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "matched_pairs": len(self.matched_values),
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")