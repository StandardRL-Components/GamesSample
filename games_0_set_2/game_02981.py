
# Generated: 2025-08-28T06:36:48.066658
# Source Brief: brief_02981.md
# Brief Index: 2981

        
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

    # User-facing strings updated for the puzzle game
    user_guide = (
        "Controls: ←→ to move the block. Press space to drop it."
    )
    game_description = (
        "Fast-paced puzzle game. Position falling blocks to fill and clear rows. Clear 3 rows to win!"
    )

    # Frames auto-advance for real-time timers
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30

    # Grid and Block properties
    GRID_COLS, GRID_ROWS = 10, 5
    CELL_SIZE = 32
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 30

    # Colors
    COLOR_BG = (20, 20, 40)
    COLOR_GRID = (60, 60, 80)
    COLOR_WHITE = (255, 255, 255)
    COLOR_UI_TEXT = (240, 240, 220)
    BLOCK_COLORS = [
        (50, 200, 255),  # Cyan
        (255, 220, 50),   # Yellow
        (255, 50, 200),   # Magenta
        (50, 255, 100),   # Green
        (255, 100, 50),   # Orange
    ]

    # Game Mechanics
    GAME_DURATION_FRAMES = 60 * FPS
    BLOCK_AUTODROP_FRAMES = 2.5 * FPS
    INITIAL_MOVES = 40
    WIN_CONDITION_ROWS = 3
    MAX_STEPS = 1000
    MOVE_COOLDOWN_FRAMES = 4
    CLEAR_ANIMATION_FRAMES = 15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Simple pixel font data
        self._init_pixel_font()

        # Initialize state variables
        self.grid = None
        self.current_block = None
        self.score = 0
        self.rows_cleared = 0
        self.moves_left = 0
        self.game_timer = 0
        self.block_drop_timer = 0
        self.move_cooldown = 0
        self.steps = 0
        self.game_over = False
        self.win_status = False
        self.clear_animation_timer = 0
        self.rows_to_clear = []

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.score = 0
        self.rows_cleared = 0
        self.moves_left = self.INITIAL_MOVES
        self.game_timer = self.GAME_DURATION_FRAMES
        self.steps = 0
        self.game_over = False
        self.win_status = False
        self.clear_animation_timer = 0
        self.rows_to_clear = []
        
        self._spawn_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        self.steps += 1
        self.game_timer -= 1
        
        # --- Handle Clear Animation ---
        if self.clear_animation_timer > 0:
            self.clear_animation_timer -= 1
            if self.clear_animation_timer == 0:
                self._execute_clear()
        else:
            # --- Handle Input and Game Logic ---
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            # Update movement cooldown
            if self.move_cooldown > 0:
                self.move_cooldown -= 1

            # Handle player actions
            if self.move_cooldown == 0:
                if movement == 1:  # Left
                    self.current_block['x'] = max(0, self.current_block['x'] - 1)
                    self.move_cooldown = self.MOVE_COOLDOWN_FRAMES
                elif movement == 2:  # Right
                    self.current_block['x'] = min(self.GRID_COLS - self.current_block['width'], self.current_block['x'] + 1)
                    self.move_cooldown = self.MOVE_COOLDOWN_FRAMES
            
            # Update block auto-drop timer
            self.block_drop_timer -= 1

            # Check for drop action (space press or timer expiry)
            if space_held or self.block_drop_timer <= 0:
                reward += self._place_block()
                
                # Check for row clears after placing a block
                clear_reward = self._check_for_clears()
                if clear_reward > 0:
                    reward += clear_reward
                    self.clear_animation_timer = self.CLEAR_ANIMATION_FRAMES
                    # sound: row_clear_start.wav
                else:
                    # Only spawn next block if not waiting for clear animation
                    if not self._is_spawn_blocked():
                        self._spawn_block()
                    else:
                        # Game over if spawn is blocked
                        self.game_over = True

        # --- Check Termination Conditions ---
        terminal_reward = 0
        if not terminated:
            if self.rows_cleared >= self.WIN_CONDITION_ROWS:
                terminated = True
                self.win_status = True
                self.game_over = True
                terminal_reward = 100
                # sound: win.wav
            elif self.moves_left <= 0 or self.game_timer <= 0 or self._is_spawn_blocked():
                terminated = True
                self.game_over = True
                terminal_reward = -100
                # sound: lose.wav
            elif self.steps >= self.MAX_STEPS:
                terminated = True
                self.game_over = True
        
        reward += terminal_reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_block(self):
        self.moves_left -= 1
        if self.moves_left < 0:
             self.game_over = True
             return
             
        width = self.np_random.integers(1, 4) # Block width of 1, 2, or 3
        color_index = self.np_random.integers(0, len(self.BLOCK_COLORS))
        
        self.current_block = {
            'width': width,
            'color_idx': color_index + 1, # 0 is empty
            'x': self.np_random.integers(0, self.GRID_COLS - width + 1),
        }
        self.block_drop_timer = self.BLOCK_AUTODROP_FRAMES
        self.move_cooldown = self.MOVE_COOLDOWN_FRAMES # Cooldown after spawn
        
    def _is_spawn_blocked(self):
        # Check if top row is occupied, preventing new blocks
        return np.any(self.grid[0, :] != 0)

    def _place_block(self):
        # Find lowest available row for the block
        target_row = -1
        for r in range(self.GRID_ROWS - 1, -1, -1):
            is_row_free = True
            for c in range(self.current_block['width']):
                if self.grid[r, self.current_block['x'] + c] != 0:
                    is_row_free = False
                    break
            if is_row_free:
                target_row = r
                break
        
        if target_row != -1:
            # Place block in grid
            for c in range(self.current_block['width']):
                self.grid[target_row, self.current_block['x'] + c] = self.current_block['color_idx']
            self.current_block = None
            # sound: block_place.wav
            return 0.1 # Reward for placing a block
        else:
            # This case means the column is full, which is a loss condition
            self.game_over = True
            return 0
    
    def _check_for_clears(self):
        self.rows_to_clear = []
        for r in range(self.GRID_ROWS):
            if np.all(self.grid[r, :] != 0):
                self.rows_to_clear.append(r)
        
        if self.rows_to_clear:
            cleared_count = len(self.rows_to_clear)
            self.rows_cleared += cleared_count
            self.score += (10 * cleared_count) * cleared_count # Bonus for multi-clears
            return 10.0 * cleared_count
        return 0

    def _execute_clear(self):
        if not self.rows_to_clear:
            return

        # Create a new grid and copy non-cleared rows
        new_grid = np.zeros_like(self.grid)
        new_row_idx = self.GRID_ROWS - 1
        for r in range(self.GRID_ROWS - 1, -1, -1):
            if r not in self.rows_to_clear:
                new_grid[new_row_idx] = self.grid[r]
                new_row_idx -= 1
        
        self.grid = new_grid
        self.rows_to_clear = []
        
        # After clearing, spawn the next block if game is not over
        if not self.game_over:
             if not self._is_spawn_blocked():
                self._spawn_block()
             else:
                self.game_over = True
        # sound: row_cleared.wav

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
            "rows_cleared": self.rows_cleared,
            "moves_left": self.moves_left,
            "time_left_seconds": self.game_timer / self.FPS,
        }

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, (0,0,0), grid_rect)

        # Draw placed blocks
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_idx = self.grid[r, c]
                if color_idx > 0:
                    color = self.BLOCK_COLORS[color_idx - 1]
                    rect = pygame.Rect(self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, tuple(min(255, x+30) for x in color), rect.inflate(-6, -6))


        # Draw clear animation
        if self.clear_animation_timer > 0:
            flash_alpha = 150 * (math.sin(self.clear_animation_timer * math.pi / self.CLEAR_ANIMATION_FRAMES * 2) + 1)
            flash_color = (*self.COLOR_WHITE, flash_alpha)
            flash_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill(flash_color)
            for r in self.rows_to_clear:
                for c in range(self.GRID_COLS):
                    self.screen.blit(flash_surface, (self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE))

        # Draw falling block
        if self.current_block and self.clear_animation_timer == 0:
            color = self.BLOCK_COLORS[self.current_block['color_idx'] - 1]
            block_width_px = self.current_block['width'] * self.CELL_SIZE
            x_pos = self.GRID_X + self.current_block['x'] * self.CELL_SIZE
            y_pos = self.GRID_Y - self.CELL_SIZE # Position above the grid
            
            rect = pygame.Rect(x_pos, y_pos, block_width_px, self.CELL_SIZE)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, tuple(min(255, x+50) for x in color), rect.inflate(-6, -6))

        # Draw grid lines
        for i in range(self.GRID_ROWS + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y + i * self.CELL_SIZE), (self.GRID_X + self.GRID_WIDTH, self.GRID_Y + i * self.CELL_SIZE))
        for i in range(self.GRID_COLS + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y), (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT))

    def _render_ui(self):
        # Render Score, Moves, Time, Rows
        self._render_pixel_text(f"SCORE {self.score}", 20, 20, 3)
        self._render_pixel_text(f"MOVES {self.moves_left}", 240, 20, 3)
        time_sec = max(0, int(self.game_timer / self.FPS))
        self._render_pixel_text(f"TIME {time_sec}", 440, 20, 3)
        self._render_pixel_text(f"GOAL {self.rows_cleared}/{self.WIN_CONDITION_ROWS}", self.GRID_X, self.GRID_Y + self.GRID_HEIGHT + 20, 3)

        # Render Game Over/Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win_status else "GAME OVER"
            text_width = len(msg) * 6 * 6
            self._render_pixel_text(msg, (self.SCREEN_WIDTH - text_width) // 2, 180, 6)
            
    def _init_pixel_font(self):
        self.PIXEL_FONT = {
            'A': [0,1,1,0,0, 1,0,0,1,0, 1,1,1,1,0, 1,0,0,1,0, 1,0,0,1,0],
            'B': [1,1,1,0,0, 1,0,0,1,0, 1,1,1,0,0, 1,0,0,1,0, 1,1,1,0,0],
            'C': [0,1,1,1,0, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 0,1,1,1,0],
            'D': [1,1,1,0,0, 1,0,0,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,1,1,0,0],
            'E': [1,1,1,1,0, 1,0,0,0,0, 1,1,1,0,0, 1,0,0,0,0, 1,1,1,1,0],
            'F': [1,1,1,1,0, 1,0,0,0,0, 1,1,1,0,0, 1,0,0,0,0, 1,0,0,0,0],
            'G': [0,1,1,1,0, 1,0,0,0,0, 1,0,1,1,0, 1,0,0,1,0, 0,1,1,1,0],
            'H': [1,0,0,1,0, 1,0,0,1,0, 1,1,1,1,0, 1,0,0,1,0, 1,0,0,1,0],
            'I': [1,1,1,0,0, 0,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0, 1,1,1,0,0],
            'L': [1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,0],
            'M': [1,0,0,0,1, 1,1,0,1,1, 1,0,1,0,1, 1,0,0,0,1, 1,0,0,0,1],
            'N': [1,0,0,1,0, 1,1,0,1,0, 1,0,1,1,0, 1,0,0,1,0, 1,0,0,1,0],
            'O': [0,1,1,0,0, 1,0,0,1,0, 1,0,0,1,0, 1,0,0,1,0, 0,1,1,0,0],
            'P': [1,1,1,0,0, 1,0,0,1,0, 1,1,1,0,0, 1,0,0,0,0, 1,0,0,0,0],
            'R': [1,1,1,0,0, 1,0,0,1,0, 1,1,1,0,0, 1,0,1,0,0, 1,0,0,1,0],
            'S': [0,1,1,1,0, 1,0,0,0,0, 0,1,1,0,0, 0,0,0,1,0, 1,1,1,0,0],
            'T': [1,1,1,1,1, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0],
            'U': [1,0,0,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,0,0,1,0, 0,1,1,0,0],
            'V': [1,0,0,0,1, 1,0,0,0,1, 0,1,0,1,0, 0,1,0,1,0, 0,0,1,0,0],
            'W': [1,0,0,0,1, 1,0,0,0,1, 1,0,1,0,1, 1,1,0,1,1, 1,0,0,0,1],
            'Y': [1,0,1,0,0, 0,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0],
            ' ': [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0],
            '0': [0,1,1,0,0, 1,0,0,1,0, 1,0,1,1,0, 1,1,0,1,0, 0,1,1,0,0],
            '1': [0,1,0,0,0, 1,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0, 1,1,1,0,0],
            '2': [1,1,1,0,0, 0,0,1,0,0, 0,1,0,0,0, 1,0,0,0,0, 1,1,1,1,0],
            '3': [1,1,1,0,0, 0,0,1,0,0, 0,1,1,0,0, 0,0,1,0,0, 1,1,1,0,0],
            '4': [1,0,1,0,0, 1,0,1,0,0, 1,1,1,1,0, 0,0,1,0,0, 0,0,1,0,0],
            '5': [1,1,1,1,0, 1,0,0,0,0, 1,1,1,0,0, 0,0,0,1,0, 1,1,1,0,0],
            '6': [0,1,1,0,0, 1,0,0,0,0, 1,1,1,0,0, 1,0,0,1,0, 0,1,1,0,0],
            '7': [1,1,1,1,0, 0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0, 1,0,0,0,0],
            '8': [0,1,1,0,0, 1,0,0,1,0, 0,1,1,0,0, 1,0,0,1,0, 0,1,1,0,0],
            '9': [0,1,1,0,0, 1,0,0,1,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,0,0],
            '!': [0,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0, 0,0,0,0,0, 0,1,0,0,0],
            '/': [0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0, 1,0,0,0,0, 1,0,0,0,0],
        }

    def _render_pixel_text(self, text, x, y, size):
        for char in text.upper():
            if char in self.PIXEL_FONT:
                pattern = self.PIXEL_FONT[char]
                for i, pixel in enumerate(pattern):
                    if pixel:
                        px = x + (i % 5) * size
                        py = y + (i // 5) * size
                        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (px, py, size, size))
            x += 6 * size
            
    def close(self):
        pygame.quit()