
# Generated: 2025-08-27T14:56:35.120277
# Source Brief: brief_00840.md
# Brief Index: 840

        
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
        "Controls: ←→ to move the falling block. Press space to drop it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro puzzle game. Drop blocks to complete horizontal or vertical lines and clear them from the grid."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.CELL_SIZE = 32
        self.GRID_BORDER = 2
        self.MAX_STEPS = 1000
        self.WIN_CONDITION_LINES = 5
        self.ANIMATION_FRAMES = 8 # How many steps the clear animation takes

        # Centering the grid
        self.GRID_PX_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.GRID_PX_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_PX_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_PX_HEIGHT) - 10

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_WHITE = (255, 255, 255)
        self.BLOCK_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.grid = None
        self.falling_block_pos = None
        self.falling_block_color_idx = None
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.clear_animation_timer = 0
        self.cleared_lines_info = {}
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.clear_animation_timer = 0
        self.cleared_lines_info = {}

        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Animation Phase ---
        # If an animation is playing, we don't process player input.
        # We just advance the animation timer.
        if self.clear_animation_timer > 0:
            self.clear_animation_timer -= 1
            if self.clear_animation_timer == 0:
                self._apply_gravity()
                # Check for win/loss after gravity shift
                if self.lines_cleared >= self.WIN_CONDITION_LINES:
                    self.game_over = True
                    reward += 100
                else:
                    self._spawn_new_block()
                    if self.game_over: # Loss on spawn
                        reward -= 100
            
            self.score += reward
            terminated = self.game_over or self.steps >= self.MAX_STEPS
            return self._get_observation(), reward, terminated, False, self._get_info()

        # --- Action Phase ---
        self.steps += 1
        
        movement = action[0]  # 3=left, 4=right
        space_held = action[1] == 1
        
        # Handle horizontal movement
        if movement == 3: # Left
            self.falling_block_pos = max(0, self.falling_block_pos - 1)
        elif movement == 4: # Right
            self.falling_block_pos = min(self.GRID_WIDTH - 1, self.falling_block_pos + 1)

        # Handle drop
        if space_held:
            # Find where the block will land
            landing_row = -1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, self.falling_block_pos] == 0:
                    landing_row = r
                    break
            
            if landing_row == -1: # Column is full
                self.game_over = True
                reward = -100
            else:
                # Place block and get placement reward
                self.grid[landing_row, self.falling_block_pos] = self.falling_block_color_idx
                reward = 0.1
                
                # Check for line clears
                cleared_rows, cleared_cols = self._find_completed_lines()
                num_cleared = len(cleared_rows) + len(cleared_cols)

                if num_cleared > 0:
                    reward += num_cleared
                    self.lines_cleared += num_cleared
                    self.cleared_lines_info = {'rows': cleared_rows, 'cols': cleared_cols}
                    self.clear_animation_timer = self.ANIMATION_FRAMES
                    # Sound: Line clear chime
                else:
                    # No lines cleared, check for win and spawn next block
                    if self.lines_cleared >= self.WIN_CONDITION_LINES:
                        self.game_over = True
                        reward += 100
                    else:
                        self._spawn_new_block()
                        if self.game_over: # Loss on spawn
                            reward -= 100
        
        self.score += reward
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_new_block(self):
        self.falling_block_pos = self.GRID_WIDTH // 2
        self.falling_block_color_idx = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1)
        
        # Check for game over condition on spawn
        if self.grid[0, self.falling_block_pos] != 0:
            self.game_over = True

    def _find_completed_lines(self):
        cleared_rows = [r for r in range(self.GRID_HEIGHT) if np.all(self.grid[r, :] > 0)]
        cleared_cols = [c for c in range(self.GRID_WIDTH) if np.all(self.grid[:, c] > 0)]
        return cleared_rows, cleared_cols

    def _apply_gravity(self):
        rows_to_clear = self.cleared_lines_info.get('rows', [])
        cols_to_clear = self.cleared_lines_info.get('cols', [])

        # Create a mask of cells to keep
        keep_mask = np.ones_like(self.grid, dtype=bool)
        if rows_to_clear:
            keep_mask[rows_to_clear, :] = False
        if cols_to_clear:
            keep_mask[:, cols_to_clear] = False

        new_grid = np.zeros_like(self.grid)
        for c in range(self.GRID_WIDTH):
            kept_cells = self.grid[:, c][keep_mask[:, c]]
            new_grid[self.GRID_HEIGHT - len(kept_cells):, c] = kept_cells
        
        self.grid = new_grid
        self.cleared_lines_info = {} # Reset animation info

    def _render_block(self, surface, x, y, color_idx, size_mod=0):
        color = self.BLOCK_COLORS[color_idx - 1]
        dark_color = tuple(max(0, c - 50) for c in color)
        
        block_rect = pygame.Rect(
            x + size_mod, y + size_mod, 
            self.CELL_SIZE - 2 * size_mod, self.CELL_SIZE - 2 * size_mod
        )
        
        pygame.draw.rect(surface, dark_color, block_rect)
        inner_rect = block_rect.inflate(-self.GRID_BORDER*2, -self.GRID_BORDER*2)
        pygame.draw.rect(surface, color, inner_rect)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_PX_WIDTH, self.GRID_PX_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw placed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] > 0:
                    self._render_block(
                        self.screen,
                        self.GRID_X_OFFSET + c * self.CELL_SIZE,
                        self.GRID_Y_OFFSET + r * self.CELL_SIZE,
                        self.grid[r, c]
                    )
        
        # Draw falling block
        if not self.game_over and self.clear_animation_timer == 0:
            preview_y = self.GRID_Y_OFFSET - self.CELL_SIZE - 5
            preview_x = self.GRID_X_OFFSET + self.falling_block_pos * self.CELL_SIZE
            self._render_block(self.screen, preview_x, preview_y, self.falling_block_color_idx)
            
            # Draw a faint drop line
            for r in range(self.GRID_HEIGHT):
                if self.grid[r, self.falling_block_pos] == 0:
                     pygame.draw.rect(self.screen, self.COLOR_GRID, 
                                      (preview_x + self.CELL_SIZE // 2 - 1, 
                                       self.GRID_Y_OFFSET + r * self.CELL_SIZE, 2, self.CELL_SIZE))

        # Draw clear animation
        if self.clear_animation_timer > 0:
            # Sound: Animation sparkle
            alpha = 255 * (math.sin(self.clear_animation_timer / self.ANIMATION_FRAMES * math.pi))
            flash_color = (255, 255, 255, int(alpha))
            flash_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill(flash_color)

            for r in self.cleared_lines_info.get('rows', []):
                for c in range(self.GRID_WIDTH):
                    self.screen.blit(flash_surface, (self.GRID_X_OFFSET + c * self.CELL_SIZE, self.GRID_Y_OFFSET + r * self.CELL_SIZE))
            
            for c in self.cleared_lines_info.get('cols', []):
                for r in range(self.GRID_HEIGHT):
                    self.screen.blit(flash_surface, (self.GRID_X_OFFSET + c * self.CELL_SIZE, self.GRID_Y_OFFSET + r * self.CELL_SIZE))

        # Draw grid lines on top
        for i in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, 
                             (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET),
                             (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_PX_HEIGHT))
        for i in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_BG,
                             (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + i * self.CELL_SIZE),
                             (self.GRID_X_OFFSET + self.GRID_PX_WIDTH, self.GRID_Y_OFFSET + i * self.CELL_SIZE))

    def _render_ui(self):
        # Score and Lines
        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        lines_text = self.font_medium.render(f"LINES: {self.lines_cleared} / {self.WIN_CONDITION_LINES}", True, self.COLOR_UI_TEXT)
        
        self.screen.blit(score_text, (self.GRID_X_OFFSET, 20))
        self.screen.blit(lines_text, (self.SCREEN_WIDTH - self.GRID_X_OFFSET - lines_text.get_width(), 20))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                end_text_str = "YOU WIN!"
                # Sound: Win fanfare
            else:
                end_text_str = "GAME OVER"
                # Sound: Loss stinger
            
            end_text = self.font_large.render(end_text_str, True, self.COLOR_WHITE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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
            "lines_cleared": self.lines_cleared,
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override Pygame screen for direct rendering
    pygame.display.set_caption("Pixel Drop")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0])  # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False
                
        if not terminated:
            keys = pygame.key.get_pressed()
            
            # This is a one-shot action mapping, more suitable for human play
            # We check for a key press event to trigger a single action
            move_action = 0
            space_action = 0

            for event in pygame.event.get(pygame.KEYDOWN):
                 if event.key == pygame.K_LEFT:
                     move_action = 3
                 elif event.key == pygame.K_RIGHT:
                     move_action = 4
                 elif event.key == pygame.K_SPACE:
                     space_action = 1
            
            action = np.array([move_action, space_action, 0])
            
            # Only step if an action was taken
            if np.any(action):
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

        # Render the environment's observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human play
        
    env.close()