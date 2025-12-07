import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import os
import os
import pygame


# Set the SDL_VIDEODRIVER to "dummy" for headless execution
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to shift all pixels. Try to match the target image in the top right."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A pixel-pushing puzzle. Shift the entire grid of pixels to recreate the target image before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    MAX_MOVES = 50
    MAX_STEPS = 500  # Safety break

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID_BG = (30, 35, 50)
    COLOR_GRID_LINE = (50, 55, 70)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_ACCENT = (100, 255, 200)
    COLOR_FLASH = (255, 255, 255)

    # Limited 8-color palette for the puzzle itself
    # 0: BG, 1: Outline, 2: Fill, 3: Eyes, 4-7: Unused
    PALETTE = [
        (40, 40, 60),  # 0: Background
        (255, 220, 100),  # 1: Face Outline (Yellow)
        (240, 200, 80),  # 2: Face Fill (Slightly darker yellow)
        (20, 25, 40),  # 3: Eyes/Mouth (Same as BG)
        (255, 100, 100),  # 4: Red
        (100, 255, 100),  # 5: Green
        (100, 100, 255),  # 6: Blue
        (255, 100, 255),  # 7: Magenta
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Fonts ---
        try:
            self.font_large = pygame.font.SysFont("Consolas", 30)
            self.font_medium = pygame.font.SysFont("Consolas", 20)
            self.font_small = pygame.font.SysFont("Consolas", 14)
        except pygame.error:
            self.font_large = pygame.font.SysFont(None, 40)
            self.font_medium = pygame.font.SysFont(None, 28)
            self.font_small = pygame.font.SysFont(None, 20)

        # --- Game State (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.moves_left = None
        self.target_grid = None
        self.current_grid = None
        self.completed_rows = None
        self.completed_cols = None
        self.show_flash = None
        self.last_action_was_move = None

        # Initialize state variables for the first time by calling reset
        # self.reset() is called here to ensure all attributes are set before validation
        # We will call it again in the validation function to ensure a clean state for tests
        obs, info = self.reset()

        # Validate implementation at the end of __init__
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.show_flash = False
        self.last_action_was_move = False

        # --- Create Target Image (Smiley Face) ---
        self.target_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        # Outline
        self.target_grid[1:9, 1:9] = 1
        # Fill
        self.target_grid[2:8, 2:8] = 2
        # Eyes
        self.target_grid[3, 3] = 3
        self.target_grid[3, 6] = 3
        # Mouth
        self.target_grid[6, 3:7] = 3
        self.target_grid[5, 4:6] = 3

        # --- Create Shuffled Grid ---
        # Flatten, shuffle, and reshape to ensure the puzzle is solvable
        flat_target = self.target_grid.flatten()
        self.np_random.shuffle(flat_target)
        self.current_grid = flat_target.reshape((self.GRID_SIZE, self.GRID_SIZE))

        # Ensure the starting grid isn't already the solution
        if np.array_equal(self.current_grid, self.target_grid):
            # If by chance it is, perform a random roll
            self.current_grid = np.roll(self.current_grid, shift=1, axis=0)

        self.completed_rows = np.array([np.array_equal(self.current_grid[i, :], self.target_grid[i, :]) for i in range(self.GRID_SIZE)])
        self.completed_cols = np.array([np.array_equal(self.current_grid[:, i], self.target_grid[:, i]) for i in range(self.GRID_SIZE)])

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]

        self.steps += 1
        reward = 0
        # self.game_over is determined by conditions below, not reset each step
        self.show_flash = False

        old_grid = self.current_grid.copy()

        # --- Apply Action ---
        if 1 <= movement <= 4:  # A move was made
            self.last_action_was_move = True
            self.show_flash = True
            self.moves_left -= 1

            if movement == 1:  # Up
                self.current_grid = np.roll(self.current_grid, -1, axis=0)
            elif movement == 2:  # Down
                self.current_grid = np.roll(self.current_grid, 1, axis=0)
            elif movement == 3:  # Left
                self.current_grid = np.roll(self.current_grid, -1, axis=1)
            elif movement == 4:  # Right
                self.current_grid = np.roll(self.current_grid, 1, axis=1)
        else:  # No-op
            self.last_action_was_move = False

        # --- Calculate Reward ---
        if self.last_action_was_move:
            # Continuous feedback for pixel placement
            correct_before = (old_grid == self.target_grid)
            correct_after = (self.current_grid == self.target_grid)

            newly_correct = np.sum(correct_after & ~correct_before)
            newly_incorrect = np.sum(~correct_after & correct_before)

            reward += newly_correct * 0.1
            reward -= newly_incorrect * 0.02

            # Event-based reward for completing rows/columns
            for i in range(self.GRID_SIZE):
                # Rows
                is_row_complete = np.array_equal(self.current_grid[i, :], self.target_grid[i, :])
                if is_row_complete and not self.completed_rows[i]:
                    reward += 5
                    self.completed_rows[i] = True
                # Columns
                is_col_complete = np.array_equal(self.current_grid[:, i], self.target_grid[:, i])
                if is_col_complete and not self.completed_cols[i]:
                    reward += 5
                    self.completed_cols[i] = True

        self.score += reward

        # --- Check Termination Conditions ---
        puzzle_solved = np.array_equal(self.current_grid, self.target_grid)
        out_of_moves = self.moves_left <= 0

        terminated = False
        if puzzle_solved:
            reward += 100
            self.score += 100
            terminated = True
        elif out_of_moves:
            reward -= 10
            self.score -= 10
            terminated = True

        truncated = self.steps >= self.MAX_STEPS
        
        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Render Target Preview ---
        preview_size = 12
        preview_pad = 2
        preview_area_size = self.GRID_SIZE * (preview_size + preview_pad)
        preview_x = self.SCREEN_WIDTH - preview_area_size - 20
        preview_y = 40

        # Draw background for preview
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (preview_x - 10, preview_y - 10, preview_area_size + 20, preview_area_size + 20), border_radius=5)

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = self.target_grid[r, c]
                color = self.PALETTE[color_index]
                rect = pygame.Rect(
                    preview_x + c * (preview_size + preview_pad),
                    preview_y + r * (preview_size + preview_pad),
                    preview_size,
                    preview_size
                )
                pygame.draw.rect(self.screen, color, rect, border_radius=2)

        # --- Render Main Grid ---
        cell_size = 32
        cell_pad = 4
        grid_area_size = self.GRID_SIZE * (cell_size + cell_pad)
        grid_x = (self.SCREEN_WIDTH - grid_area_size) // 2
        grid_y = (self.SCREEN_HEIGHT - grid_area_size) // 2 + 20

        # Draw background for main grid
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (grid_x - 10, grid_y - 10, grid_area_size + 14, grid_area_size + 14), border_radius=5)

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = self.current_grid[r, c]
                color = self.PALETTE[color_index]
                rect = pygame.Rect(
                    grid_x + c * (cell_size + cell_pad),
                    grid_y + r * (cell_size + cell_pad),
                    cell_size,
                    cell_size
                )

                # Flash effect on move
                if self.show_flash:
                    pygame.draw.rect(self.screen, self.COLOR_FLASH, rect, border_radius=4)
                else:
                    pygame.draw.rect(self.screen, color, rect, border_radius=4)

        # The flash should only last for one frame render
        if self.show_flash:
            self.show_flash = False

    def _render_ui(self):
        # --- Moves Left ---
        moves_text = self.font_large.render(f"{self.moves_left}", True, self.COLOR_TEXT_ACCENT)
        moves_label = self.font_medium.render("Moves Left", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (30, 40))
        self.screen.blit(moves_label, (30, 80))

        # --- Score ---
        # Ensure score is not None before rendering
        score_val = self.score if self.score is not None else 0.0
        score_text = self.font_large.render(f"{score_val:.1f}", True, self.COLOR_TEXT_ACCENT)
        score_label = self.font_medium.render("Score", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (30, self.SCREEN_HEIGHT - 90))
        self.screen.blit(score_label, (30, self.SCREEN_HEIGHT - 50))

        # --- Target Label ---
        target_label = self.font_medium.render("Target", True, self.COLOR_TEXT)
        self.screen.blit(target_label, (self.SCREEN_WIDTH - 120, 10))

        # --- Game Over Message ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            puzzle_solved = np.array_equal(self.current_grid, self.target_grid)
            if puzzle_solved:
                msg = "PUZZLE SOLVED!"
                color = self.COLOR_TEXT_ACCENT
            else:
                msg = "OUT OF MOVES"
                color = (255, 100, 100)

            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "puzzle_solved": np.array_equal(self.current_grid, self.target_grid)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        # Custom Assertions from Brief
        # Assert color counts remain constant after a move
        self.reset()
        initial_colors, initial_counts = np.unique(self.current_grid, return_counts=True)
        self.step(self.action_space.sample())  # make a move
        moved_colors, moved_counts = np.unique(self.current_grid, return_counts=True)
        assert np.array_equal(initial_colors, moved_colors) and np.array_equal(initial_counts, moved_counts)

        # Assert episode terminates after MAX_MOVES
        self.reset()
        # After MAX_MOVES - 1 moves, the game should not be terminated.
        for _ in range(self.MAX_MOVES - 1):
            _, _, term, _, _ = self.step([1, 0, 0])  # Move up
            assert not term, f"Game terminated early on move number {_ + 1}"
        
        # The MAX_MOVES-th move should terminate the game.
        _, _, term, _, _ = self.step([1, 0, 0])  # Final move
        assert term, "Game did not terminate after MAX_MOVES"

        # print("✓ Implementation validated successfully")