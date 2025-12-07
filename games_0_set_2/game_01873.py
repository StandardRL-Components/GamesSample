
# Generated: 2025-08-28T03:01:49.225687
# Source Brief: brief_01873.md
# Brief Index: 1873

        
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
        "Controls: Use Space to cycle selection. Use arrow keys to move the selected tile. "
        "Combine tiles of the same number to increase their value and your score."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist number puzzle. Combine tiles to reach the target number "
        "before you run out of moves. Plan your moves carefully to maximize your score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Game constants
        self.GRID_SIZE = 4
        self.TARGET_NUMBER = 128
        self.MAX_STEPS = 500
        
        self.GRID_CELL_SIZE = 80
        self.GRID_MARGIN = 10
        self.TILE_BORDER_RADIUS = 8
        
        grid_total_size = (self.GRID_SIZE * self.GRID_CELL_SIZE) + ((self.GRID_SIZE + 1) * self.GRID_MARGIN)
        self.GRID_TOP_LEFT_X = (self.screen_width - grid_total_size) // 2
        self.GRID_TOP_LEFT_Y = (self.screen_height - grid_total_size) // 2

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_BG = (40, 50, 60)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_DARK_TEXT = (20, 30, 40)
        self.COLOR_SELECT = pygame.Color(255, 255, 0)
        self.COLOR_TILE_START = pygame.Color(100, 150, 255) # For '2'
        self.COLOR_TILE_END = pygame.Color(150, 50, 255) # For '2048'

        # Fonts
        self.font_tile = pygame.font.SysFont("sans-serif", 36, bold=True)
        self.font_ui = pygame.font.SysFont("sans-serif", 22, bold=True)
        self.font_game_over = pygame.font.SysFont("sans-serif", 50, bold=True)
        
        # State variables (initialized in reset)
        self.grid = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.selected_tile_coords = None
        self.previous_space_state = None
        self.particles = None
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int64)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.selected_tile_coords = None
        self.previous_space_state = False
        self.particles = []
        
        self._spawn_tile()
        self._spawn_tile()
        
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
        action_taken = False
        
        self._update_particles()

        space_pressed = space_held and not self.previous_space_state
        if space_pressed:
            self._cycle_selection()
        self.previous_space_state = space_held

        if movement != 0 and self.selected_tile_coords is not None:
            move_result, move_reward = self._move_selected_tile(movement)
            reward += move_reward
            if move_result:
                action_taken = True
        elif movement != 0: # Penalty for attempting a move with nothing selected
            reward -= 0.01

        if action_taken:
            self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.win:
                reward += 10  # Win reward
                if self.steps < 200:
                    reward += 5  # Speed bonus
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _cycle_selection(self):
        tiles = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] > 0:
                    tiles.append((r, c))
        
        if not tiles:
            self.selected_tile_coords = None
            return

        if self.selected_tile_coords is None:
            self.selected_tile_coords = tiles[0]
        else:
            try:
                current_index = tiles.index(self.selected_tile_coords)
                next_index = (current_index + 1) % len(tiles)
                self.selected_tile_coords = tiles[next_index]
            except ValueError:
                self.selected_tile_coords = tiles[0]

    def _move_selected_tile(self, move_direction):
        r, c = self.selected_tile_coords
        dr, dc = [(0,0), (-1,0), (1,0), (0,-1), (0,1)][move_direction]
        nr, nc = r + dr, c + dc

        if not (0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE):
            # sfx_bump
            return False, -0.01

        current_val = self.grid[r, c]
        target_val = self.grid[nr, nc]

        if target_val == 0:
            self.grid[nr, nc] = current_val
            self.grid[r, c] = 0
            self.selected_tile_coords = (nr, nc)
            self._spawn_tile()
            return True, -0.01
        elif target_val == current_val:
            new_val = current_val * 2
            self.grid[nr, nc] = new_val
            self.grid[r, c] = 0
            self.score += new_val
            self.selected_tile_coords = None
            self._spawn_tile()
            self._create_merge_effect(nr, nc, new_val)
            # sfx_combine
            return True, 0.1
        else:
            # sfx_bump
            return False, -0.01

    def _spawn_tile(self):
        empty_cells = np.argwhere(self.grid == 0)
        if len(empty_cells) > 0:
            cell_idx = self.np_random.integers(len(empty_cells))
            r, c = empty_cells[cell_idx]
            self.grid[r, c] = 4 if self.np_random.random() < 0.1 else 2

    def _check_termination(self):
        if np.any(self.grid >= self.TARGET_NUMBER):
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        
        has_empty_cell = np.any(self.grid == 0)
        if has_empty_cell:
            return False
            
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                val = self.grid[r, c]
                if c < self.GRID_SIZE - 1 and self.grid[r, c+1] == val: return False
                if r < self.GRID_SIZE - 1 and self.grid[r+1, c] == val: return False
        
        return True

    def _get_tile_color(self, value):
        if value == 0: return self.COLOR_GRID_BG
        log_val = math.log2(value) if value > 0 else 0
        max_log_val = 11  # Corresponds to 2048
        lerp_factor = min((log_val - 1) / (max_log_val - 1), 1.0) if max_log_val > 1 else 0
        return self.COLOR_TILE_START.lerp(self.COLOR_TILE_END, lerp_factor)
    
    def _create_merge_effect(self, r, c, value):
        px, py = self._get_pixel_coords(r, c)
        center_x = px + self.GRID_CELL_SIZE // 2
        center_y = py + self.GRID_CELL_SIZE // 2
        color = self._get_tile_color(value)
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append([center_x, center_y, vx, vy, life, color])

    def _update_particles(self):
        self.particles = [
            [p[0] + p[2], p[1] + p[3], p[2]*0.95, p[3]*0.95, p[4] - 1, p[5]]
            for p in self.particles if p[4] > 0
        ]

    def _get_pixel_coords(self, r, c):
        x = self.GRID_TOP_LEFT_X + self.GRID_MARGIN * (c + 1) + self.GRID_CELL_SIZE * c
        y = self.GRID_TOP_LEFT_Y + self.GRID_MARGIN * (r + 1) + self.GRID_CELL_SIZE * r
        return x, y

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_TOP_LEFT_X, self.GRID_TOP_LEFT_Y,
                                (self.GRID_CELL_SIZE + self.GRID_MARGIN) * self.GRID_SIZE + self.GRID_MARGIN,
                                (self.GRID_CELL_SIZE + self.GRID_MARGIN) * self.GRID_SIZE + self.GRID_MARGIN)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=self.TILE_BORDER_RADIUS)

        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                px, py = self._get_pixel_coords(r, c)
                value = self.grid[r, c]
                color = self._get_tile_color(value)
                
                tile_rect = pygame.Rect(px, py, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE)
                pygame.draw.rect(self.screen, color, tile_rect, border_radius=self.TILE_BORDER_RADIUS)

                if value > 0:
                    text_color = self.COLOR_TEXT if color.hsla[2] < 60 else self.COLOR_DARK_TEXT
                    text_surf = self.font_tile.render(str(value), True, text_color)
                    text_rect = text_surf.get_rect(center=tile_rect.center)
                    self.screen.blit(text_surf, text_rect)

        # Draw selection highlight
        if self.selected_tile_coords:
            r, c = self.selected_tile_coords
            px, py = self._get_pixel_coords(r, c)
            highlight_rect = pygame.Rect(px - 3, py - 3, self.GRID_CELL_SIZE + 6, self.GRID_CELL_SIZE + 6)
            pygame.draw.rect(self.screen, self.COLOR_SELECT, highlight_rect, width=4, border_radius=self.TILE_BORDER_RADIUS + 3)
            
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * p[4]/30)))
            color_tuple = (*p[5][:3], alpha)
            radius = max(0, int(p[4]/5))
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), radius, color_tuple)
    
    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Steps
        steps_left = max(0, self.MAX_STEPS - self.steps)
        steps_surf = self.font_ui.render(f"Moves Left: {steps_left}", True, self.COLOR_TEXT)
        steps_rect = steps_surf.get_rect(topright=(self.screen_width - 20, 10))
        self.screen.blit(steps_surf, steps_rect)

        # Target
        target_surf = self.font_ui.render(f"Target: {self.TARGET_NUMBER}", True, self.COLOR_TEXT)
        target_rect = target_surf.get_rect(midbottom=(self.screen_width / 2, self.screen_height - 10))
        self.screen.blit(target_surf, target_rect)
        
        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (150, 255, 150) if self.win else (255, 100, 100)
            
            end_surf = self.font_game_over.render(msg, True, color)
            end_rect = end_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_surf, end_rect)


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win": self.win,
            "max_tile": np.max(self.grid) if self.grid.size > 0 else 0
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