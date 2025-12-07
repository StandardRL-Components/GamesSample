
# Generated: 2025-08-28T06:09:32.687641
# Source Brief: brief_02851.md
# Brief Index: 2851

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to swap the selected pixel with an adjacent one. "
        "Space and Shift cycle through the selectable pixels on the grid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Rearrange the pixels on the main grid to match the target "
        "pattern shown in the top-right corner. You have a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.NUM_COLORS = 4
        self.MAX_MOVES = 30

        # --- Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Visuals ---
        self.FONT_UI = pygame.font.Font(None, 28)
        self.FONT_TITLE = pygame.font.Font(None, 36)
        self.FONT_GAMEOVER = pygame.font.Font(None, 72)

        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_BG = (30, 35, 55)
        self.COLOR_GRID_LINE = (50, 60, 80)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_SELECTOR = (255, 255, 255)
        self.PALETTE = [
            (239, 71, 111),  # Pink
            (255, 209, 102), # Yellow
            (6, 214, 160),   # Green
            (17, 138, 178),  # Blue
        ]
        
        # --- Grid Layout ---
        self.CELL_SIZE = 32
        self.GRID_WIDTH = self.GRID_HEIGHT = self.CELL_SIZE * self.GRID_SIZE
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20

        # --- Target Grid Layout ---
        self.T_CELL_SIZE = 8
        self.T_GRID_WIDTH = self.T_GRID_HEIGHT = self.T_CELL_SIZE * self.GRID_SIZE
        self.T_GRID_X = self.WIDTH - self.T_GRID_WIDTH - 20
        self.T_GRID_Y = 20

        # --- State Variables ---
        self.grid = None
        self.target_grid = None
        self.selected_index = 0
        self.moves_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.np_random = None

        # --- Final Validation ---
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.game_over_message = ""
        
        # Generate target pattern
        self.target_grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE))

        # Generate initial grid by shuffling the target to guarantee solvability
        flat_target = self.target_grid.flatten()
        
        # Keep shuffling until the grid is not the same as the target
        while True:
            self.np_random.shuffle(flat_target)
            self.grid = flat_target.reshape((self.GRID_SIZE, self.GRID_SIZE))
            if not np.array_equal(self.grid, self.target_grid):
                break

        self.selected_index = 0
        self.score = self._calculate_correct_pixels()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        reward = 0
        
        # Handle selection changes (free actions)
        if space_pressed:
            # Sound: UI_Select_Next.wav
            self.selected_index = (self.selected_index + 1) % (self.GRID_SIZE * self.GRID_SIZE)
        if shift_pressed:
            # Sound: UI_Select_Prev.wav
            self.selected_index = (self.selected_index - 1 + (self.GRID_SIZE * self.GRID_SIZE)) % (self.GRID_SIZE * self.GRID_SIZE)

        # Handle pixel swaps (consumes a move)
        if movement > 0 and self.moves_remaining > 0:
            self.moves_remaining -= 1
            
            r1, c1 = self._index_to_coords(self.selected_index)
            r2, c2 = r1, c1

            if movement == 1: r2 = (r1 - 1 + self.GRID_SIZE) % self.GRID_SIZE # Up
            elif movement == 2: r2 = (r1 + 1) % self.GRID_SIZE # Down
            elif movement == 3: c2 = (c1 - 1 + self.GRID_SIZE) % self.GRID_SIZE # Left
            elif movement == 4: c2 = (c1 + 1) % self.GRID_SIZE # Right
            
            # Swap pixels
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            # Sound: Pixel_Swap.wav
        
        # Calculate score and reward
        self.score = self._calculate_correct_pixels()
        reward = self.score

        # Check for termination conditions
        is_win = self.score == self.GRID_SIZE * self.GRID_SIZE
        is_loss = self.moves_remaining <= 0 and not is_win
        terminated = is_win or is_loss

        if terminated:
            self.game_over = True
            if is_win:
                reward += 100  # Win bonus
                self.game_over_message = "COMPLETE!"
                # Sound: Win_Jingle.wav
            else:
                self.game_over_message = "OUT OF MOVES"
                # Sound: Loss_Sound.wav

        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,
            self._get_info()
        )

    def _calculate_correct_pixels(self):
        if self.grid is None or self.target_grid is None:
            return 0
        return np.sum(self.grid == self.target_grid)

    def _index_to_coords(self, index):
        return index // self.GRID_SIZE, index % self.GRID_SIZE

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw main grid background
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # Draw main grid pixels and lines
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(
                    self.GRID_X + c * self.CELL_SIZE,
                    self.GRID_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                color_index = self.grid[r, c]
                pygame.draw.rect(self.screen, self.PALETTE[color_index], cell_rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, cell_rect, 1)

        # Draw pulsating selector
        sel_r, sel_c = self._index_to_coords(self.selected_index)
        selector_rect = pygame.Rect(
            self.GRID_X + sel_c * self.CELL_SIZE,
            self.GRID_Y + sel_r * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE,
        )
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        width = int(2 + pulse * 3)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, width, border_radius=2)
        
        # Draw target grid
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(
                    self.T_GRID_X + c * self.T_CELL_SIZE,
                    self.T_GRID_Y + r * self.T_CELL_SIZE,
                    self.T_CELL_SIZE,
                    self.T_CELL_SIZE,
                )
                color_index = self.target_grid[r, c]
                pygame.draw.rect(self.screen, self.PALETTE[color_index], cell_rect)

    def _render_ui(self):
        # --- Render Moves Remaining ---
        moves_text = self.FONT_TITLE.render("MOVES", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (20, 20))
        moves_val_text = self.FONT_GAMEOVER.render(f"{self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_val_text, (20, 45))

        # --- Render Target Title and Match Score ---
        target_title = self.FONT_UI.render("TARGET", True, self.COLOR_UI_TEXT)
        self.screen.blit(target_title, (self.T_GRID_X, self.T_GRID_Y - 22))
        
        match_text = self.FONT_UI.render("MATCH", True, self.COLOR_UI_TEXT)
        match_rect = match_text.get_rect(topleft=(self.T_GRID_X, self.T_GRID_Y + self.T_GRID_HEIGHT + 10))
        self.screen.blit(match_text, match_rect)
        
        score_text = self.FONT_TITLE.render(f"{self.score}/{self.GRID_SIZE**2}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topleft=(match_rect.left, match_rect.bottom + 5))
        self.screen.blit(score_text, score_rect)

        # --- Render Game Over Text ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            game_over_surf = self.FONT_GAMEOVER.render(self.game_over_message, True, self.COLOR_SELECTOR)
            game_over_rect = game_over_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(game_over_surf, game_over_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "correct_pixels": self.score,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to get an observation
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test reset return types
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a dummy screen for display if running as a script
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Pattern")
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # no-op default
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                elif event.key == pygame.K_r: # Reset button
                    obs, info = env.reset()
                    done = False
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not done:
            # Only step if an action was taken (or reset)
            if action.any() or 'event' in locals() and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # Update the display
        frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(display_screen, frame)
        pygame.display.flip()

        env.clock.tick(30) # Limit frame rate
        
    env.close()