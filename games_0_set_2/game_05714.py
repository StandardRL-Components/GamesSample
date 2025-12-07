
# Generated: 2025-08-28T05:54:05.454007
# Source Brief: brief_05714.md
# Brief Index: 5714

        
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
        "Controls: ←→ to move, ↑ to rotate clockwise, ↓ for soft drop. Hold Shift to rotate counter-clockwise. Press Space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced falling block puzzle. Strategically place pieces to clear lines and score points before the stack reaches the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.MAX_STEPS = 10000
        self.WIN_SCORE = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_GHOST = (255, 255, 255, 60)
        self.COLOR_FLASH = (255, 255, 255)
        self.TETROMINO_COLORS = [
            (0, 0, 0),      # 0: Empty
            (0, 240, 240),  # I: Cyan
            (240, 240, 0),  # O: Yellow
            (160, 0, 240),  # T: Purple
            (0, 240, 0),    # S: Green
            (240, 0, 0),    # Z: Red
            (0, 0, 240),    # J: Blue
            (240, 160, 0),  # L: Orange
        ]

        # --- Tetromino Shapes (pivot at (0,0)) ---
        self.TETROMINOS = {
            1: [[(-1, 0), (0, 0), (1, 0), (2, 0)], [(0, -1), (0, 0), (0, 1), (0, 2)]],  # I
            2: [[(0, 0), (1, 0), (0, 1), (1, 1)]],  # O
            3: [[(-1, 0), (0, 0), (1, 0), (0, -1)], [(0, -1), (0, 0), (0, 1), (-1, 0)], [(-1, 0), (0, 0), (1, 0), (0, 1)], [(0, -1), (0, 0), (0, 1), (1, 0)]],  # T
            4: [[(-1, 0), (0, 0), (0, -1), (1, -1)], [(0, -1), (0, 0), (1, 0), (1, 1)]],  # S
            5: [[(0, 0), (1, 0), (-1, -1), (0, -1)], [(-1, 0), (-1, 1), (0, 0), (0, -1)]],  # Z
            6: [[(-1, -1), (-1, 0), (0, 0), (1, 0)], [(0, -1), (1, -1), (0, 0), (0, 1)], [(-1, 0), (0, 0), (1, 0), (1, 1)], [(0, -1), (0, 0), (0, 1), (-1, 1)]],  # J
            7: [[(-1, 0), (0, 0), (1, 0), (1, -1)], [(0, -1), (0, 0), (0, 1), (1, 1)], [(-1, 1), (-1, 0), (0, 0), (1, 0)], [(0, -1), (0, 0), (0, 1), (-1, -1)]],  # L
        }

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Etc...        
        self.grid = None
        self.current_piece = None
        self.next_piece_shape = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_progress = 0
        self.fall_rate = 0
        self.lines_to_clear = []
        self.clear_animation_timer = 0
        self.playfield_x = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.playfield_y = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fall_progress = 0
        self.lines_to_clear = []
        self.clear_animation_timer = 0
        self._update_fall_rate()

        self.next_piece_shape = self.np_random.integers(1, len(self.TETROMINOS) + 1)
        self._spawn_new_piece()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False

        # Handle line clear animation pause
        if self.clear_animation_timer > 0:
            self.clear_animation_timer -= 1
            if self.clear_animation_timer == 0:
                self._collapse_cleared_lines()
                self._spawn_new_piece()
                if self._check_collision(self.current_piece['shape'], self.current_piece['rotation'], self.current_piece['x'], self.current_piece['y']):
                    self.game_over = True
            
            terminated = self.game_over or self.steps >= self.MAX_STEPS
            if self.game_over:
                reward = -50
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        action_taken = False

        # 1. Hard Drop (takes priority)
        if space_held:
            # sfx: hard_drop.wav
            dy = 0
            while not self._check_collision(self.current_piece['shape'], self.current_piece['rotation'], self.current_piece['x'], self.current_piece['y'] + dy + 1):
                dy += 1
            self.current_piece['y'] += dy
            self.fall_progress = self.fall_rate # Force lock
            action_taken = True
        
        # 2. Rotation
        if not action_taken:
            if shift_held: # CCW
                if self._rotate_piece(clockwise=False):
                    reward -= 0.02
                action_taken = True
            elif movement == 1: # Up for CW
                if self._rotate_piece(clockwise=True):
                    reward -= 0.02
                action_taken = True

        # 3. Horizontal Movement
        if not action_taken:
            if movement == 3: # Left
                if self._move_piece(-1, 0):
                    reward -= 0.02
                action_taken = True
            elif movement == 4: # Right
                if self._move_piece(1, 0):
                    reward -= 0.02
                action_taken = True
        
        # 4. Soft Drop
        if movement == 2: # Down
            if self._move_piece(0, 1):
                reward += 0.1
                self.fall_progress = 0 # Reset gravity
        
        # 5. Auto-Fall (Gravity)
        self.fall_progress += 1
        if self.fall_progress >= self.fall_rate:
            self.fall_progress = 0
            if not self._move_piece(0, 1): # Move failed, so lock it
                self._lock_piece()
                lines_cleared = self._check_for_line_clears()
                if lines_cleared > 0:
                    # sfx: line_clear.wav
                    reward += {1: 1, 2: 3, 3: 7, 4: 15}.get(lines_cleared, 15)
                    self.score += {1: 10, 2: 30, 3: 60, 4: 100}.get(lines_cleared, 100)
                    self.clear_animation_timer = 10 # frames of animation
                    self._update_fall_rate()
                else:
                    # sfx: piece_lock.wav
                    self._spawn_new_piece()
                    if self._check_collision(self.current_piece['shape'], self.current_piece['rotation'], self.current_piece['x'], self.current_piece['y']):
                        self.game_over = True

        terminated = self.game_over or self.steps >= self.MAX_STEPS or self.score >= self.WIN_SCORE
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100
            elif self.game_over:
                reward += -50
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Helper Methods ---

    def _spawn_new_piece(self):
        self.current_piece = {
            'shape': self.next_piece_shape,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - 1,
            'y': 0,
        }
        self.next_piece_shape = self.np_random.integers(1, len(self.TETROMINOS) + 1)
        # sfx: piece_spawn.wav

    def _update_fall_rate(self):
        level = self.score // 200
        fall_speed_seconds = max(0.1, 1.0 - level * 0.05)
        self.fall_rate = int(fall_speed_seconds * 30) # Assuming 30 FPS

    def _get_piece_coords(self, shape_id, rotation, x, y):
        shape = self.TETROMINOS[shape_id][rotation % len(self.TETROMINOS[shape_id])]
        return [(px + x, py + y) for px, py in shape]

    def _check_collision(self, shape_id, rotation, x, y):
        for px, py in self._get_piece_coords(shape_id, rotation, x, y):
            if not (0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT):
                return True # Out of bounds
            if self.grid[py, px] != 0:
                return True # Collides with locked piece
        return False

    def _move_piece(self, dx, dy):
        target_x = self.current_piece['x'] + dx
        target_y = self.current_piece['y'] + dy
        if not self._check_collision(self.current_piece['shape'], self.current_piece['rotation'], target_x, target_y):
            self.current_piece['x'] = target_x
            self.current_piece['y'] = target_y
            if dx != 0: # sfx: move.wav
                pass
            return True
        return False

    def _rotate_piece(self, clockwise=True):
        current_rotation = self.current_piece['rotation']
        num_rotations = len(self.TETROMINOS[self.current_piece['shape']])
        next_rotation = (current_rotation + (1 if clockwise else -1) + num_rotations) % num_rotations

        # Basic wall kick tests
        kick_offsets = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1)]
        for dx, dy in kick_offsets:
            if not self._check_collision(self.current_piece['shape'], next_rotation, self.current_piece['x'] + dx, self.current_piece['y'] + dy):
                self.current_piece['rotation'] = next_rotation
                self.current_piece['x'] += dx
                self.current_piece['y'] += dy
                # sfx: rotate.wav
                return True
        return False

    def _lock_piece(self):
        for px, py in self._get_piece_coords(self.current_piece['shape'], self.current_piece['rotation'], self.current_piece['x'], self.current_piece['y']):
            if 0 <= py < self.GRID_HEIGHT:
                self.grid[py, px] = self.current_piece['shape']

    def _check_for_line_clears(self):
        self.lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                self.lines_to_clear.append(r)
        return len(self.lines_to_clear)

    def _collapse_cleared_lines(self):
        if not self.lines_to_clear:
            return
        
        new_grid = np.zeros_like(self.grid)
        new_row = self.GRID_HEIGHT - 1
        for r in range(self.GRID_HEIGHT - 1, -1, -1):
            if r not in self.lines_to_clear:
                new_grid[new_row] = self.grid[r]
                new_row -= 1
        self.grid = new_grid
        self.lines_to_clear = []

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            y = self.playfield_y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.playfield_x, y), (self.playfield_x + self.GRID_WIDTH * self.CELL_SIZE, y))
        for c in range(self.GRID_WIDTH + 1):
            x = self.playfield_x + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.playfield_y), (x, self.playfield_y + self.GRID_HEIGHT * self.CELL_SIZE))

        # Draw locked pieces
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    self._draw_block(c, r, self.grid[r, c])

        # Draw line clear animation
        if self.clear_animation_timer > 0:
            flash_color = self.COLOR_FLASH if (self.clear_animation_timer // 2) % 2 == 0 else self.COLOR_GRID
            for r in self.lines_to_clear:
                rect = pygame.Rect(self.playfield_x, self.playfield_y + r * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, flash_color, rect)
            return

        if self.game_over:
            return

        # Draw ghost piece
        ghost_y_offset = 0
        while not self._check_collision(self.current_piece['shape'], self.current_piece['rotation'], self.current_piece['x'], self.current_piece['y'] + ghost_y_offset + 1):
            ghost_y_offset += 1
        
        ghost_coords = self._get_piece_coords(self.current_piece['shape'], self.current_piece['rotation'], self.current_piece['x'], self.current_piece['y'] + ghost_y_offset)
        for c, r in ghost_coords:
            if r >= 0:
                self._draw_block(c, r, self.current_piece['shape'], is_ghost=True)

        # Draw current piece
        piece_coords = self._get_piece_coords(self.current_piece['shape'], self.current_piece['rotation'], self.current_piece['x'], self.current_piece['y'])
        for c, r in piece_coords:
             if r >= 0:
                self._draw_block(c, r, self.current_piece['shape'])

    def _draw_block(self, c, r, color_index, is_ghost=False):
        x = self.playfield_x + c * self.CELL_SIZE
        y = self.playfield_y + r * self.CELL_SIZE
        
        if is_ghost:
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, self.COLOR_GHOST, s.get_rect(), border_radius=3)
            self.screen.blit(s, (x, y))
            return

        main_color = self.TETROMINO_COLORS[color_index]
        light_color = tuple(min(255, val + 50) for val in main_color)
        dark_color = tuple(max(0, val - 50) for val in main_color)
        
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, dark_color, rect, border_radius=3)
        inner_rect = pygame.Rect(x + 2, y + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
        pygame.draw.rect(self.screen, main_color, inner_rect, border_radius=2)
        
        pygame.gfxdraw.line(self.screen, x + 2, y + 2, x + self.CELL_SIZE - 3, y + 2, light_color)
        pygame.gfxdraw.line(self.screen, x + 2, y + 2, x + 2, y + self.CELL_SIZE - 3, light_color)

    def _render_ui(self):
        ui_x = self.playfield_x + self.GRID_WIDTH * self.CELL_SIZE + 40
        
        score_text = self.font_main.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_x, 50))
        score_val = self.font_main.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_val, (ui_x, 80))

        level = self.score // 200
        level_text = self.font_main.render("LEVEL", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (ui_x, 130))
        level_val = self.font_main.render(f"{level}", True, self.COLOR_TEXT)
        self.screen.blit(level_val, (ui_x, 160))
        
        next_text = self.font_main.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (ui_x, 210))
        
        next_piece_coords = self.TETROMINOS[self.next_piece_shape][0]
        min_x = min(c[0] for c in next_piece_coords)
        max_x = max(c[0] for c in next_piece_coords)
        min_y = min(c[1] for c in next_piece_coords)
        max_y = max(c[1] for c in next_piece_coords)
        
        offset_x = ui_x + (4 * self.CELL_SIZE - (max_x - min_x + 1) * self.CELL_SIZE) / 2
        offset_y = 240 + (4 * self.CELL_SIZE - (max_y - min_y + 1) * self.CELL_SIZE) / 2

        for c, r in next_piece_coords:
            draw_x = offset_x + (c - min_x) * self.CELL_SIZE
            draw_y = offset_y + (r - min_y) * self.CELL_SIZE
            
            rect = pygame.Rect(draw_x, draw_y, self.CELL_SIZE, self.CELL_SIZE)
            main_color = self.TETROMINO_COLORS[self.next_piece_shape]
            light_color = tuple(min(255, val + 50) for val in main_color)
            dark_color = tuple(max(0, val - 50) for val in main_color)
            pygame.draw.rect(self.screen, dark_color, rect, border_radius=3)
            inner_rect = pygame.Rect(draw_x + 2, draw_y + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
            pygame.draw.rect(self.screen, main_color, inner_rect, border_radius=2)
            pygame.gfxdraw.line(self.screen, int(draw_x) + 2, int(draw_y) + 2, int(draw_x) + self.CELL_SIZE - 3, int(draw_y) + 2, light_color)
            pygame.gfxdraw.line(self.screen, int(draw_x) + 2, int(draw_y) + 2, int(draw_x) + 2, int(draw_y) + self.CELL_SIZE - 3, light_color)

    def validate_implementation(self):
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

    def close(self):
        pygame.font.quit()
        pygame.quit()