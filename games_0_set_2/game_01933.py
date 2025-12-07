# Generated: 2025-08-28T03:08:58.564311
# Source Brief: brief_01933.md
# Brief Index: 1933

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the falling block. Each move costs 1 turn. "
        "Fill the grid completely to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A spatial puzzle game. Place falling blocks to fill a 10x10 grid "
        "before you run out of moves. Clearing rows earns points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 10, 10
    CELL_SIZE = 32
    MAX_MOVES = 50

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID_LINES = (40, 50, 70)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 35, 50)
    
    # Block colors (Vibrant) - 0 is empty
    COLORS_BLOCK = [
        (0, 0, 0),
        (0, 255, 255),  # I-Block (Cyan)
        (255, 255, 0),  # O-Block (Yellow)
        (128, 0, 128),  # T-Block (Purple)
        (0, 0, 255),    # J-Block (Blue)
        (255, 165, 0),  # L-Block (Orange)
        (0, 255, 0),    # S-Block (Green)
        (255, 0, 0),    # Z-Block (Red)
    ]
    # Locked block colors (Darker/Desaturated)
    COLORS_LOCKED = [
        (0, 0, 0),
        (0, 139, 139),
        (139, 139, 0),
        (75, 0, 130),
        (0, 0, 139),
        (139, 90, 0),
        (0, 100, 0),
        (139, 0, 0),
    ]

    # --- Block Shapes (Tetrominos) ---
    BLOCK_SHAPES = [
        [[0, 0], [0, 1], [0, 2], [0, 3]],  # I
        [[0, 0], [1, 0], [0, 1], [1, 1]],  # O
        [[0, 1], [1, 0], [1, 1], [2, 1]],  # T
        [[0, 0], [0, 1], [0, 2], [ -1, 2]], # J
        [[0, 0], [0, 1], [0, 2], [1, 2]],  # L
        [[0, 1], [1, 1], [1, 0], [2, 0]],  # S
        [[0, 0], [1, 0], [1, 1], [2, 1]],  # Z
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        self.grid_pixel_width = self.GRID_COLS * self.CELL_SIZE
        self.grid_pixel_height = self.GRID_ROWS * self.CELL_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_pixel_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_pixel_height) // 2 + 20
        
        self.np_random = None
        self.grid = None
        self.current_block = None
        self.steps = 0
        self.score = 0
        self.remaining_moves = 0
        self.game_over = False

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            # Fallback for older gym versions or no seed
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.steps = 0
        self.score = 0
        self.remaining_moves = self.MAX_MOVES
        self.game_over = False
        
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.2  # Cost for taking a move
        self.steps += 1
        self.remaining_moves -= 1
        
        movement = action[0]
        
        # --- 1. Handle Horizontal Movement ---
        new_x = self.current_block['x']
        if movement == 3:  # Left
            new_x -= 1
        elif movement == 4:  # Right
            new_x += 1
        
        if not self._check_collision(self.current_block['shape'], (new_x, self.current_block['y'])):
            self.current_block['x'] = new_x
        
        # --- 2. Handle Vertical Movement (Gravity) ---
        new_y = self.current_block['y'] + 1
        
        if self._check_collision(self.current_block['shape'], (self.current_block['x'], new_y)):
            # Collision detected below, lock the block
            self._lock_block()
            
            cleared_rows = self._clear_rows()
            if cleared_rows > 0:
                reward += cleared_rows  # +1 per row
            if cleared_rows > 1:
                reward += 5  # Bonus for multi-row clear
            
            if np.all(self.grid != 0):
                self.game_over = True
                reward += 100  # Win bonus
            else:
                self._spawn_new_block()
                # Check for game over on spawn (stack too high)
                if self._check_collision(self.current_block['shape'], (self.current_block['x'], self.current_block['y'])):
                    self.game_over = True
        else:
            # No collision, continue falling
            self.current_block['y'] = new_y
            
        # --- 3. Check Termination Conditions ---
        terminated = self.game_over
        if self.remaining_moves <= 0 and not self.game_over:
            terminated = True
            reward -= 50  # Loss penalty

        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_new_block(self):
        shape_idx = self.np_random.integers(0, len(self.BLOCK_SHAPES))
        color_idx = self.np_random.integers(1, len(self.COLORS_BLOCK))
        self.current_block = {
            'shape': self.BLOCK_SHAPES[shape_idx],
            'color_idx': color_idx,
            'x': self.GRID_COLS // 2 - 1,
            'y': -2 # Start above the visible grid
        }

    def _check_collision(self, shape, pos):
        x, y = pos
        for dx, dy in shape:
            check_x, check_y = x + dx, y + dy
            if not (0 <= check_x < self.GRID_COLS):
                return True # Wall collision
            if check_y >= self.GRID_ROWS:
                return True # Floor collision
            if check_y >= 0 and self.grid[check_y, check_x] != 0:
                return True # Block collision
        return False

    def _lock_block(self):
        shape = self.current_block['shape']
        pos = (self.current_block['x'], self.current_block['y'])
        color_idx = self.current_block['color_idx']
        for dx, dy in shape:
            grid_x, grid_y = pos[0] + dx, pos[1] + dy
            if 0 <= grid_y < self.GRID_ROWS and 0 <= grid_x < self.GRID_COLS:
                self.grid[grid_y, grid_x] = color_idx

    def _clear_rows(self):
        full_rows = [r for r in range(self.GRID_ROWS) if np.all(self.grid[r, :] != 0)]
        if not full_rows:
            return 0
        
        # Create a new grid and copy non-full rows down
        new_grid = np.zeros_like(self.grid)
        new_row_idx = self.GRID_ROWS - 1
        for r in range(self.GRID_ROWS - 1, -1, -1):
            if r not in full_rows:
                new_grid[new_row_idx, :] = self.grid[r, :]
                new_row_idx -= 1
        self.grid = new_grid
        return len(full_rows)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (
            self.grid_offset_x, self.grid_offset_y,
            self.grid_pixel_width, self.grid_pixel_height
        ))
        
        # Draw grid lines
        for x in range(self.GRID_COLS + 1):
            start_pos = (self.grid_offset_x + x * self.CELL_SIZE, self.grid_offset_y)
            end_pos = (self.grid_offset_x + x * self.CELL_SIZE, self.grid_offset_y + self.grid_pixel_height)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos)
        for y in range(self.GRID_ROWS + 1):
            start_pos = (self.grid_offset_x, self.grid_offset_y + y * self.CELL_SIZE)
            end_pos = (self.grid_offset_x + self.grid_pixel_width, self.grid_offset_y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos)

        # Draw locked blocks
        if self.grid is not None:
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    color_idx = self.grid[r, c]
                    if color_idx != 0:
                        self._draw_cell(c, r, self.COLORS_LOCKED[color_idx])

        # Draw current falling block
        if self.current_block and not self.game_over:
            shape = self.current_block['shape']
            pos = (self.current_block['x'], self.current_block['y'])
            color = self.COLORS_BLOCK[self.current_block['color_idx']]
            for dx, dy in shape:
                grid_x, grid_y = pos[0] + dx, pos[1] + dy
                if grid_y >= 0: # Only draw if inside or below the top of the grid
                    self._draw_cell(grid_x, grid_y, color, is_active=True)

    def _draw_cell(self, grid_x, grid_y, color, is_active=False):
        px = self.grid_offset_x + grid_x * self.CELL_SIZE
        py = self.grid_offset_y + grid_y * self.CELL_SIZE
        
        cell_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        
        # Main cell color
        pygame.draw.rect(self.screen, color, cell_rect.inflate(-2, -2))
        
        # Border effect for polish
        border_color = tuple(min(255, c + 40) for c in color) if is_active else tuple(max(0, c - 20) for c in color)
        pygame.draw.rect(self.screen, border_color, cell_rect, 1)

    def _render_ui(self):
        # Moves display
        moves_text = self.font_main.render(f"Moves: {self.remaining_moves}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        # Score display
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)

        # Game Over display
        if self.game_over:
            is_win = np.all(self.grid != 0)
            message = "GRID FILLED!" if is_win else "OUT OF MOVES"
            color = (100, 255, 100) if is_win else (255, 100, 100)
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_moves": self.remaining_moves,
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset first, which initializes the environment state
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Now that the env is reset, test observation space getter
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Due to the headless requirement, this will not open a window.
    # It will run but you won't see anything. To play visually,
    # comment out the `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` line.
    
    # Re-initialize pygame with the default video driver for interactive play
    pygame.quit()
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.init()
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Filler")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # Game loop
    while not terminated:
        action = np.array([0, 0, 0])  # Default to no-op
        
        should_step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                should_step = True
                if event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
                    should_step = False
                    continue
        
        if should_step:
            # A key was pressed, so we step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")

        # Render the environment state to the screen
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate
        
    print("Game Over!")
    env.close()