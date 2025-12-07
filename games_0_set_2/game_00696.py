
# Generated: 2025-08-27T14:28:53.391962
# Source Brief: brief_00696.md
# Brief Index: 696

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    """
    Pixel Perfect is a puzzle game where the player must recreate a target
    pixel art image within a time limit. Players move a cursor, select a
    color, and fill in grid squares to match the reference image, earning
    points for correct placements and bonuses for completing rows or columns.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to fill a square. Shift to cycle color."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target pixel art image before time runs out. Match colors to score points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 10
        self.MAX_STEPS = 600 # 60 seconds at 10 steps/sec

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("consolas", 14, bold=True)
        
        # --- Colors and Palette ---
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_GRID_LINE = (60, 60, 70)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.PALETTE = [
            (50, 50, 60),     # 0: Empty/BG color of grid
            (230, 25, 75),    # 1: Red
            (60, 180, 75),    # 2: Green
            (0, 130, 200),    # 3: Blue
            (245, 130, 48),   # 4: Orange
            (145, 30, 180),   # 5: Purple
            (70, 240, 240),   # 6: Cyan
            (240, 50, 230),   # 7: Magenta
            (255, 255, 25),   # 8: Yellow
        ]

        # --- Layout ---
        self.CELL_SIZE = 32
        self.GRID_WIDTH = self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2 + 20

        # --- State Variables (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.target_image = None
        self.player_grid = None
        self.cursor_pos = (0, 0)
        self.selected_color_idx = 1
        self.completed_rows = set()
        self.completed_cols = set()
        self.last_fill_info = None # For visual feedback

        # Initialize state
        self.reset()

        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize RNG
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        # Initialize game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        # Generate a new random target image (using colors 1 through N)
        self.target_image = self.np_random.integers(1, len(self.PALETTE), size=(self.GRID_SIZE, self.GRID_SIZE))
        self.player_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        self.cursor_pos = (0, 0)
        self.selected_color_idx = 1
        self.completed_rows = set()
        self.completed_cols = set()
        self.last_fill_info = None

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        self.steps += 1
        self.last_fill_info = None # Reset visual feedback each step

        # Unpack factorized action
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        # Action 1: Cycle color (Shift)
        if shift_pressed:
            # Cycle through selectable colors (indices 1 to len-1)
            self.selected_color_idx = (self.selected_color_idx % (len(self.PALETTE) - 1)) + 1
            # sfx: color_swap.wav

        # Action 2: Move cursor (Arrows)
        cx, cy = self.cursor_pos
        if movement == 1: cy -= 1  # Up
        elif movement == 2: cy += 1  # Down
        elif movement == 3: cx -= 1  # Left
        elif movement == 4: cx += 1  # Right
        # Wrap cursor around grid
        self.cursor_pos = (cx % self.GRID_SIZE, cy % self.GRID_SIZE)

        # Action 3: Fill square (Space)
        if space_pressed:
            fill_x, fill_y = self.cursor_pos
            target_color = self.target_image[fill_y, fill_x]
            current_color = self.player_grid[fill_y, fill_x]
            
            # Only apply changes and rewards if the color is actually changing
            if current_color != self.selected_color_idx:
                self.player_grid[fill_y, fill_x] = self.selected_color_idx
                
                is_correct = self.selected_color_idx == target_color
                self.last_fill_info = {'pos': (fill_x, fill_y), 'correct': is_correct}

                if is_correct:
                    reward += 1.0
                    # sfx: place_correct.wav
                else:
                    reward -= 0.2
                    # sfx: place_wrong.wav

                # Check for row completion bonus
                if fill_y not in self.completed_rows:
                    if np.array_equal(self.player_grid[fill_y, :], self.target_image[fill_y, :]):
                        reward += 5.0
                        self.completed_rows.add(fill_y)
                        # sfx: row_complete.wav
                
                # Check for column completion bonus
                if fill_x not in self.completed_cols:
                    if np.array_equal(self.player_grid[:, fill_x], self.target_image[:, fill_x]):
                        reward += 5.0
                        self.completed_cols.add(fill_x)
                        # sfx: col_complete.wav
        
        self.score += reward
        
        # --- Check Termination Conditions ---
        is_complete = np.array_equal(self.player_grid, self.target_image)
        time_up = self.steps >= self.MAX_STEPS
        
        terminated = is_complete or time_up
        
        if is_complete and not self.game_over:
            reward += 100.0
            self.score += 100.0
            # sfx: level_win.wav

        if time_up and not self.game_over:
            # sfx: time_up.wav
            pass

        self.game_over = terminated
        
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
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw player grid
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.GRID_X + x * self.CELL_SIZE,
                    self.GRID_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                color_idx = self.player_grid[y, x]
                pygame.draw.rect(self.screen, self.PALETTE[color_idx], rect)

        # Draw fill feedback flash
        if self.last_fill_info:
            pos = self.last_fill_info['pos']
            is_correct = self.last_fill_info['correct']
            flash_color = (255, 255, 255, 100) if is_correct else (255, 0, 0, 100)
            flash_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill(flash_color)
            self.screen.blit(flash_surface, (self.GRID_X + pos[0] * self.CELL_SIZE, self.GRID_Y + pos[1] * self.CELL_SIZE))

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y)
            end_pos = (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.GRID_X, self.GRID_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_X + self.GRID_WIDTH, self.GRID_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos, 1)

        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_X + cursor_x * self.CELL_SIZE,
            self.GRID_Y + cursor_y * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

        # Draw color palette
        palette_title_surf = self.font_title.render("PALETTE", True, self.COLOR_TEXT)
        self.screen.blit(palette_title_surf, (20, self.SCREEN_HEIGHT - 40))

        palette_cell_size = 28
        for i, color in enumerate(self.PALETTE):
            if i == 0: continue # Skip empty color
            rect = pygame.Rect(
                20 + (i - 1) * (palette_cell_size + 5),
                self.SCREEN_HEIGHT - 28,
                palette_cell_size,
                palette_cell_size
            )
            pygame.draw.rect(self.screen, color, rect)
            if i == self.selected_color_idx:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3)

    def _render_ui(self):
        # Draw target image
        target_title_surf = self.font_title.render("TARGET", True, self.COLOR_TEXT)
        self.screen.blit(target_title_surf, (20, 10))
        target_cell_size = 4
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    20 + x * target_cell_size,
                    30 + y * target_cell_size,
                    target_cell_size,
                    target_cell_size
                )
                color_idx = self.target_image[y, x]
                pygame.draw.rect(self.screen, self.PALETTE[color_idx], rect)

        # Draw score
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 20, 10))

        # Draw time
        time_remaining = (self.MAX_STEPS - self.steps) / 10.0
        time_text = f"TIME: {max(0, time_remaining):.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 20, 35))

        # Draw game over message
        if self.game_over:
            is_win = np.array_equal(self.player_grid, self.target_image)
            message = "PERFECT!" if is_win else "TIME'S UP!"
            color = (100, 255, 100) if is_win else (255, 100, 100)
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_font = pygame.font.SysFont("consolas", 60, bold=True)
            end_surf = end_font.render(message, True, color)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_surf, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "selected_color": self.selected_color_idx,
            "match_percentage": np.mean(self.player_grid == self.target_image) * 100
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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
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
    # Set this to "dummy" to run headlessly
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- To run with a manual human player ---
    # This requires a visible pygame window, so don't use the dummy driver
    
    # Re-initialize pygame for display
    pygame.display.init()
    pygame.display.set_caption("Pixel Perfect")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        # Map keyboard inputs to MultiDiscrete action
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_pressed = 1 if keys[pygame.K_SPACE] else 0
        shift_pressed = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_pressed, shift_pressed]
        
        # Poll for events
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                # Any key press triggers a step in this turn-based game
                action_taken = True
                if event.key == pygame.K_r: # Press 'r' to reset
                    obs, info = env.reset()

        # Only step if an action key was pressed
        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for human play

    print(f"Game Over! Final Score: {info['score']:.1f}")
    env.close()