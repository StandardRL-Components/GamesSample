
# Generated: 2025-08-28T02:31:50.949273
# Source Brief: brief_01729.md
# Brief Index: 1729

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ←→ to move, ↑ to rotate, ↓ for soft drop. Press space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced falling block puzzle. Clear lines by filling them with blocks, but don't let the stack reach the top!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.GRID_X = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.WIN_CONDITION_LINES = 10
        
        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_DANGER = (60, 30, 30)
        self.COLOR_FLASH = (255, 255, 255)
        self.COLOR_GHOST = (128, 128, 128, 100) # Semi-transparent
        
        # Tetromino shapes and colors
        self.TETROMINOS = {
            'I': {'shape': [[1, 1, 1, 1]], 'color': (0, 240, 240)},
            'J': {'shape': [[1, 0, 0], [1, 1, 1]], 'color': (0, 0, 240)},
            'L': {'shape': [[0, 0, 1], [1, 1, 1]], 'color': (240, 160, 0)},
            'O': {'shape': [[1, 1], [1, 1]], 'color': (240, 240, 0)},
            'S': {'shape': [[0, 1, 1], [1, 1, 0]], 'color': (0, 240, 0)},
            'T': {'shape': [[0, 1, 0], [1, 1, 1]], 'color': (160, 0, 240)},
            'Z': {'shape': [[1, 1, 0], [0, 1, 1]], 'color': (240, 0, 0)}
        }
        self.TETROMINO_KEYS = list(self.TETROMINOS.keys())
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 24)
        except pygame.error:
            self.font_main = pygame.font.SysFont("monospace", 36)
            self.font_small = pygame.font.SysFont("monospace", 24)
            
        # Initialize state variables
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = 0.0
        self.fall_counter = 0.0
        self.prev_space_held = False
        self.lines_since_speedup = 0
        self.line_clear_animation = {'timer': 0, 'rows': []}
        
        self.reset()
    
    def _get_new_piece(self):
        shape_key = self.np_random.choice(self.TETROMINO_KEYS)
        piece_data = self.TETROMINOS[shape_key]
        return {
            'shape': piece_data['shape'],
            'color': piece_data['color'],
            'x': self.GRID_WIDTH // 2 - len(piece_data['shape'][0]) // 2,
            'y': 0
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = [[(0,0,0) for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        
        self.current_piece = self._get_new_piece()
        self.next_piece = self._get_new_piece()
        
        self.fall_speed = 1.0  # Cells per second
        self.fall_counter = 0.0
        self.prev_space_held = False
        self.lines_since_speedup = 0
        self.line_clear_animation = {'timer': 0, 'rows': []}

        if not self._is_valid_position(self.current_piece):
            self.game_over = True

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed
        terminated = False
        
        self.clock.tick(self.FPS)
        dt = self.clock.get_time() / 1000.0 # Time in seconds

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        if not self.game_over:
            # --- Player Input ---
            space_pressed = space_held and not self.prev_space_held
            self.prev_space_held = space_held

            if movement == 1: # Up: Rotate
                self._rotate_piece()
            elif movement == 3: # Left: Move Left
                self._move(-1, 0)
            elif movement == 4: # Right: Move Right
                self._move(1, 0)

            if space_pressed: # Space: Hard Drop
                reward += self._hard_drop()
                # Sound effect placeholder: # play_hard_drop_sound()
            
            # --- Game Logic (Auto-fall) ---
            soft_drop_multiplier = 4.0 if movement == 2 else 1.0 # Down: Soft Drop
            self.fall_counter += self.fall_speed * dt * soft_drop_multiplier
            
            if self.fall_counter >= 1.0:
                self.fall_counter = 0.0
                if not self._move(0, 1): # If move down fails, lock piece
                    reward += self._lock_piece()
                    # Sound effect placeholder: # play_lock_sound()

        # Update line clear animation
        if self.line_clear_animation['timer'] > 0:
            self.line_clear_animation['timer'] -= 1

        # Check for termination conditions
        if self.game_over:
            reward += -100.0
            terminated = True
        elif self.lines_cleared >= self.WIN_CONDITION_LINES:
            reward += 100.0
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        self.steps += 1
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _is_valid_position(self, piece, offset_x=0, offset_y=0):
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    x = piece['x'] + c + offset_x
                    y = piece['y'] + r + offset_y
                    if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                        return False # Out of bounds
                    if y >= 0 and self.grid[y][x] != (0,0,0):
                        return False # Collision with locked block
        return True

    def _move(self, dx, dy):
        if self._is_valid_position(self.current_piece, dx, dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            return True
        return False

    def _rotate_piece(self):
        original_shape = self.current_piece['shape']
        rotated_shape = [list(row) for row in zip(*self.current_piece['shape'][::-1])]
        
        self.current_piece['shape'] = rotated_shape
        if not self._is_valid_position(self.current_piece):
            # Try wall kicks
            if self._is_valid_position(self.current_piece, 1, 0): self.current_piece['x'] += 1
            elif self._is_valid_position(self.current_piece, -1, 0): self.current_piece['x'] -= 1
            elif self._is_valid_position(self.current_piece, 2, 0): self.current_piece['x'] += 2
            elif self._is_valid_position(self.current_piece, -2, 0): self.current_piece['x'] -= 2
            else: self.current_piece['shape'] = original_shape # Revert if all fail
                
    def _hard_drop(self):
        while self._is_valid_position(self.current_piece, 0, 1):
            self.current_piece['y'] += 1
        return self._lock_piece()

    def _lock_piece(self):
        for r, row in enumerate(self.current_piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    x = self.current_piece['x'] + c
                    y = self.current_piece['y'] + r
                    if y >= 0: self.grid[y][x] = self.current_piece['color']
        
        if self.current_piece['y'] < 1: self.game_over = True
        
        _, clear_reward = self._clear_lines()
        
        self.current_piece = self.next_piece
        self.next_piece = self._get_new_piece()
        
        if not self._is_valid_position(self.current_piece): self.game_over = True
        
        return -0.2 + clear_reward

    def _clear_lines(self):
        lines_to_clear = [r for r, row in enumerate(self.grid) if all(cell != (0,0,0) for cell in row)]
        
        if lines_to_clear:
            # Sound effect placeholder: # play_line_clear_sound()
            for r in lines_to_clear:
                self.grid.pop(r)
                self.grid.insert(0, [(0,0,0) for _ in range(self.GRID_WIDTH)])
            
            cleared_count = len(lines_to_clear)
            self.lines_cleared += cleared_count
            self.lines_since_speedup += cleared_count
            
            if self.lines_since_speedup >= 2:
                self.fall_speed += 0.02 * (self.lines_since_speedup // 2)
                self.lines_since_speedup %= 2
                
            self.line_clear_animation = {'timer': 5, 'rows': lines_to_clear}
            
            reward_map = {1: 1, 2: 3, 3: 7, 4: 15}
            reward = reward_map.get(cleared_count, 0)
            self.score += reward
            return cleared_count, reward
            
        return 0, 0

    def _get_ghost_y(self):
        ghost_y = self.current_piece['y']
        while self._is_valid_position(self.current_piece, 0, ghost_y - self.current_piece['y'] + 1):
            ghost_y += 1
        return ghost_y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return { "score": self.score, "steps": self.steps, "lines_cleared": self.lines_cleared }

    def _draw_block(self, surface, x, y, color):
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(surface, color, rect)
        
        light_color = tuple(min(255, c + 40) for c in color)
        dark_color = tuple(max(0, c - 40) for c in color)
        
        pygame.draw.line(surface, light_color, rect.topleft, rect.topright, 1)
        pygame.draw.line(surface, light_color, rect.topleft, rect.bottomleft, 1)
        pygame.draw.line(surface, dark_color, rect.bottomright, rect.topright, 1)
        pygame.draw.line(surface, dark_color, rect.bottomright, rect.bottomleft, 1)

    def _render_game(self):
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        danger_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE * 2)
        pygame.draw.rect(self.screen, self.COLOR_DANGER, danger_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, grid_rect, 1)

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] != (0,0,0):
                    self._draw_block(self.screen, self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.grid[r][c])

        if not self.game_over and self.current_piece:
            ghost_y = self._get_ghost_y()
            ghost_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            for r_off, row in enumerate(self.current_piece['shape']):
                for c_off, cell in enumerate(row):
                    if cell:
                        x = self.GRID_X + (self.current_piece['x'] + c_off) * self.CELL_SIZE
                        y = self.GRID_Y + (ghost_y + r_off) * self.CELL_SIZE
                        ghost_surf.fill(self.COLOR_GHOST)
                        self.screen.blit(ghost_surf, (x, y))

            for r_off, row in enumerate(self.current_piece['shape']):
                for c_off, cell in enumerate(row):
                    if cell:
                        x = self.GRID_X + (self.current_piece['x'] + c_off) * self.CELL_SIZE
                        y = self.GRID_Y + (self.current_piece['y'] + r_off) * self.CELL_SIZE
                        self._draw_block(self.screen, x, y, self.current_piece['color'])
                        
        if self.line_clear_animation['timer'] > 0:
            alpha = int(255 * (self.line_clear_animation['timer'] / 5.0))
            flash_surf = pygame.Surface((self.CELL_SIZE * self.GRID_WIDTH, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surf.fill((*self.COLOR_FLASH, alpha))
            for r in self.line_clear_animation['rows']:
                self.screen.blit(flash_surf, (self.GRID_X, self.GRID_Y + r * self.CELL_SIZE))

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        lines_text = self.font_small.render(f"LINES: {self.lines_cleared}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (20, 60))

        next_text = self.font_small.render("NEXT:", True, self.COLOR_TEXT)
        next_box_x = self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20
        next_box_y = self.GRID_Y
        self.screen.blit(next_text, (next_box_x, next_box_y))
        
        preview_area = pygame.Rect(next_box_x, next_box_y + 30, 4.5 * self.CELL_SIZE, 4 * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_area)

        if self.next_piece:
            shape = self.next_piece['shape']
            start_x = preview_area.centerx - (len(shape[0]) * self.CELL_SIZE) / 2
            start_y = preview_area.centery - (len(shape) * self.CELL_SIZE) / 2
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.screen, int(start_x + c * self.CELL_SIZE), int(start_y + r * self.CELL_SIZE), self.next_piece['color'])
    
    def close(self):
        pygame.font.quit()
        pygame.quit()