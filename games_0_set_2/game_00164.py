
# Generated: 2025-08-27T12:48:13.849743
# Source Brief: brief_00164.md
# Brief Index: 164

        
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
        "Controls: ←→ to move the falling block. Goal is to clear 5 rows before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game where you place falling blocks to fill and clear rows on a 10x10 grid."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.CELL_SIZE = 32
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2
        self.MAX_MOVES = 50
        self.WIN_CONDITION_ROWS = 5
        self.CLEAR_ANIMATION_FRAMES = 10 # frames to show clear animation

        # Colors
        self.COLOR_BG = (20, 20, 40)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_FLASH = (255, 255, 255, 180) # Semi-transparent white
        self.BLOCK_COLORS = [
            (50, 205, 205),   # Cyan
            (255, 215, 0),    # Gold
            (218, 112, 214),  # Orchid
            (124, 252, 0),    # Lawn Green
            (255, 69, 0),     # OrangeRed
            (138, 43, 226)    # BlueViolet
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
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        
        # Internal state variables are initialized in reset()
        self.grid = None
        self.current_block_x = 0
        self.current_block_y = 0
        self.current_block_color_idx = 0
        self.score = 0
        self.moves_left = 0
        self.rows_cleared_total = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.clear_animation_timer = 0
        self.cleared_rows_info = []

        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.rows_cleared_total = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.clear_animation_timer = 0
        self.cleared_rows_info = []
        
        self._spawn_block()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _spawn_block(self):
        self.current_block_x = self.GRID_SIZE // 2
        self.current_block_y = 0
        # The block color index is 1-based, 0 is empty
        self.current_block_color_idx = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False
        self.steps += 1

        # Handle clear animation state
        if self.clear_animation_timer > 0:
            self.clear_animation_timer -= 1
            if self.clear_animation_timer == 0:
                self._clear_and_shift_rows()
                if not self.game_over:
                    self._spawn_block()
            # Return observation during animation without changing state
            return self._get_observation(), 0, False, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 1=left, 2=right
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused
        
        # Decrement moves regardless of action
        self.moves_left -= 1
        
        # Apply horizontal movement
        if movement == 1: # Move left
            if self.current_block_x > 0:
                self.current_block_x -= 1
        elif movement == 2: # Move right
            if self.current_block_x < self.GRID_SIZE - 1:
                self.current_block_x += 1
        
        # Apply vertical movement (gravity)
        can_fall = (self.current_block_y < self.GRID_SIZE - 1 and
                    self.grid[self.current_block_y + 1][self.current_block_x] == 0)

        if can_fall:
            self.current_block_y += 1
        else: # Block has landed
            # Place block in grid
            self.grid[self.current_block_y][self.current_block_x] = self.current_block_color_idx
            # sfx: block_land.wav

            # Calculate partial row reward
            num_partially_filled = 0
            for r in range(self.GRID_SIZE):
                row_sum = np.sum(self.grid[r, :] > 0)
                if 0 < row_sum < self.GRID_SIZE:
                    num_partially_filled += 1
            reward += num_partially_filled * 0.1

            # Check for completed rows
            cleared_rows_indices = []
            for r in range(self.GRID_SIZE):
                if np.all(self.grid[r, :] > 0):
                    cleared_rows_indices.append(r)
            
            if cleared_rows_indices:
                # sfx: row_clear.wav
                self.rows_cleared_total += len(cleared_rows_indices)
                reward += len(cleared_rows_indices) * 10
                self.cleared_rows_info = cleared_rows_indices
                self.clear_animation_timer = self.CLEAR_ANIMATION_FRAMES
            else:
                # No rows cleared, check for termination before spawning next block
                if self.moves_left <= 0 and self.rows_cleared_total < self.WIN_CONDITION_ROWS:
                    self.game_over = True
                else:
                    self._spawn_block()

        # Check for termination conditions
        if self.rows_cleared_total >= self.WIN_CONDITION_ROWS:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100
            # sfx: game_win.wav
        elif self.moves_left <= 0:
            self.game_over = True
            terminated = True
            # Only apply loss penalty if not already a win
            if not self.win:
                reward -= 100
                # sfx: game_lose.wav

        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _clear_and_shift_rows(self):
        # Create a new grid and copy over non-cleared rows
        new_grid = np.zeros_like(self.grid)
        new_row_idx = self.GRID_SIZE - 1
        for r in range(self.GRID_SIZE - 1, -1, -1):
            if r not in self.cleared_rows_info:
                if new_row_idx >= 0:
                    new_grid[new_row_idx] = self.grid[r]
                    new_row_idx -= 1
        self.grid = new_grid
        self.cleared_rows_info = []

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid()
        self._render_blocks()
        if not self.game_over:
            self._render_falling_block()
        if self.clear_animation_timer > 0:
            self._render_clear_animation()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            pygame.draw.line(self.screen, self.COLOR_GRID,
                             (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET),
                             (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT), 1)
            # Horizontal lines
            pygame.draw.line(self.screen, self.COLOR_GRID,
                             (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + i * self.CELL_SIZE),
                             (self.GRID_X_OFFSET + self.GRID_WIDTH, self.GRID_Y_OFFSET + i * self.CELL_SIZE), 1)

    def _render_blocks(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r][c] > 0:
                    color_idx = self.grid[r][c] - 1
                    self._draw_block(c, r, self.BLOCK_COLORS[color_idx])

    def _render_falling_block(self):
        if self.current_block_color_idx > 0:
            color = self.BLOCK_COLORS[self.current_block_color_idx - 1]
            # Glow effect
            glow_color = (*color, 60) # RGBA with low alpha
            glow_surf = pygame.Surface((self.CELL_SIZE * 1.5, self.CELL_SIZE * 1.5), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color, (glow_surf.get_width()//2, glow_surf.get_height()//2), self.CELL_SIZE * 0.7)
            
            x_pos = self.GRID_X_OFFSET + self.current_block_x * self.CELL_SIZE
            y_pos = self.GRID_Y_OFFSET + self.current_block_y * self.CELL_SIZE
            self.screen.blit(glow_surf, (x_pos - self.CELL_SIZE*0.25, y_pos - self.CELL_SIZE*0.25))

            self._draw_block(self.current_block_x, self.current_block_y, color)

    def _draw_block(self, c, r, color):
        x = self.GRID_X_OFFSET + c * self.CELL_SIZE
        y = self.GRID_Y_OFFSET + r * self.CELL_SIZE
        
        main_rect = pygame.Rect(x + 2, y + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
        
        # Simple 3D effect
        light_color = tuple(min(255, val + 40) for val in color)
        dark_color = tuple(max(0, val - 40) for val in color)
        
        pygame.draw.rect(self.screen, color, main_rect)
        
        pygame.draw.line(self.screen, light_color, (main_rect.left, main_rect.top), (main_rect.right - 1, main_rect.top), 2)
        pygame.draw.line(self.screen, light_color, (main_rect.left, main_rect.top), (main_rect.left, main_rect.bottom - 1), 2)
        pygame.draw.line(self.screen, dark_color, (main_rect.right - 1, main_rect.top + 1), (main_rect.right - 1, main_rect.bottom - 1), 2)
        pygame.draw.line(self.screen, dark_color, (main_rect.left + 1, main_rect.bottom - 1), (main_rect.right - 1, main_rect.bottom - 1), 2)

    def _render_clear_animation(self):
        flash_surface = pygame.Surface((self.GRID_WIDTH, self.CELL_SIZE), pygame.SRCALPHA)
        
        alpha = self.COLOR_FLASH[3] * (self.clear_animation_timer / self.CLEAR_ANIMATION_FRAMES)
        flash_surface.fill((*self.COLOR_FLASH[:3], alpha))
        
        for r in self.cleared_rows_info:
            y = self.GRID_Y_OFFSET + r * self.CELL_SIZE
            self.screen.blit(flash_surface, (self.GRID_X_OFFSET, y))

    def _render_ui(self):
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        moves_text = self.font_medium.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 20))

        rows_text = self.font_medium.render(f"Rows: {self.rows_cleared_total} / {self.WIN_CONDITION_ROWS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(rows_text, (20, 50))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                end_text = self.font_large.render("YOU WIN!", True, (100, 255, 100))
            else:
                end_text = self.font_large.render("GAME OVER", True, (255, 100, 100))
            
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "rows_cleared": self.rows_cleared_total,
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
        
        # Test game-specific assertions from brief
        self.reset()
        assert self.grid.shape == (10, 10)
        assert self.moves_left == self.MAX_MOVES
        
        initial_moves = self.moves_left
        self.step(self.action_space.sample())
        assert self.moves_left == initial_moves - 1
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows for manual play and testing of the environment.
    env = GameEnv(render_mode="rgb_array")
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Drop")
    
    obs, info = env.reset()
    terminated = False
    running = True
    
    print("--- Pixel Drop ---")
    print(env.user_guide)
    print("Press 'R' to reset, 'ESC' to quit.")
    
    while running:
        action_taken = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    action_taken = True
                
                if not terminated:
                    action = [0, 0, 0] # Default no-op
                    if event.key == pygame.K_LEFT:
                        action = [1, 0, 0]
                    elif event.key == pygame.K_RIGHT:
                        action = [2, 0, 0]
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
                    action_taken = True
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        env.clock.tick(15)

    env.close()