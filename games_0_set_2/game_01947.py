
# Generated: 2025-08-28T03:13:14.067416
# Source Brief: brief_01947.md
# Brief Index: 1947

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a pixel-matching puzzle game.

    The player controls a highlighted pixel on a grid and must swap it with
    adjacent pixels to match a target pattern within a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ↑↓←→ to swap the highlighted pixel with an adjacent one."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target pattern by swapping colored pixels. You have a limited number of moves!"
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
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_ROWS = 5
        self.GRID_COLS = 8
        self.MAX_MOVES = 20
        self.SHUFFLE_MOVES = 10

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_LINE = (50, 60, 80)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HIGHLIGHT = (255, 255, 0)
        self.PIXEL_COLORS = [
            (255, 80, 80),   # Bright Red
            (80, 255, 80),   # Bright Green
            (80, 80, 255),   # Bright Blue
            (255, 255, 80),  # Bright Yellow
            (255, 80, 255),  # Bright Magenta
            (80, 255, 255),  # Bright Cyan
        ]

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_title = pygame.font.Font(None, 28)
        self.font_gameover = pygame.font.Font(None, 72)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.moves_remaining = 0
        self.target_grid = None
        self.current_grid = None
        self.player_pos = None
        self.win_state = ""
        
        # Rendering variables
        self.cell_size = 0
        self.grid_width = 0
        self.grid_height = 0
        self.main_grid_pos = (0, 0)
        self.target_grid_pos = (0, 0)
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.win_state = ""
        
        self._generate_puzzle()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        """Creates a new solvable puzzle."""
        # Create a solved target grid with random colors
        self.target_grid = self.np_random.integers(
            0, len(self.PIXEL_COLORS), size=(self.GRID_ROWS, self.GRID_COLS)
        )
        
        self.current_grid = np.copy(self.target_grid)
        
        # Select a random starting player position
        start_row = self.np_random.integers(0, self.GRID_ROWS)
        start_col = self.np_random.integers(0, self.GRID_COLS)
        self.player_pos = [start_row, start_col]
        
        # Shuffle the grid by making random moves, guaranteeing solvability
        for _ in range(self.SHUFFLE_MOVES):
            move = self.np_random.integers(1, 5) # 1-4 for up, down, left, right
            
            r1, c1 = self.player_pos
            r2, c2 = r1, c1

            if move == 1: r2 = (r1 - 1 + self.GRID_ROWS) % self.GRID_ROWS
            elif move == 2: r2 = (r1 + 1) % self.GRID_ROWS
            elif move == 3: c2 = (c1 - 1 + self.GRID_COLS) % self.GRID_COLS
            elif move == 4: c2 = (c1 + 1) % self.GRID_COLS
            
            # Swap pixels
            self.current_grid[r1, c1], self.current_grid[r2, c2] = \
                self.current_grid[r2, c2], self.current_grid[r1, c1]
            
            # Update player position to follow the swapped pixel
            self.player_pos = [r2, c2]

    def step(self, action):
        # If the game is over, do nothing and just return the final state
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]
        # space_held = action[1] == 1 # Not used
        # shift_held = action[2] == 1 # Not used
        
        reward = 0.0
        terminated = False
        
        # Process a move if one was made (movement != 0)
        if movement != 0:
            self.moves_remaining -= 1
            
            # --- Calculate Rewards based on state change ---
            correct_rows_cols_before = self._count_correct_rows_cols()
            
            pos1_r, pos1_c = self.player_pos
            color_p = self.current_grid[pos1_r, pos1_c]

            pos2_r, pos2_c = pos1_r, pos1_c
            if movement == 1: pos2_r = (pos1_r - 1 + self.GRID_ROWS) % self.GRID_ROWS
            elif movement == 2: pos2_r = (pos1_r + 1) % self.GRID_ROWS
            elif movement == 3: pos2_c = (pos1_c - 1 + self.GRID_COLS) % self.GRID_COLS
            elif movement == 4: pos2_c = (pos1_c + 1) % self.GRID_COLS
            color_q = self.current_grid[pos2_r, pos2_c]

            p_was_correct = (color_p == self.target_grid[pos1_r, pos1_c])
            q_was_correct = (color_q == self.target_grid[pos2_r, pos2_c])

            # Apply the move (swap pixels and update player position)
            self.current_grid[pos1_r, pos1_c], self.current_grid[pos2_r, pos2_c] = color_q, color_p
            self.player_pos = [pos2_r, pos2_c]

            # Check correctness after swap
            p_is_correct = (color_p == self.target_grid[pos2_r, pos2_c])
            q_is_correct = (color_q == self.target_grid[pos1_r, pos1_c])

            # Reward for moving pixel P to/from correct spot
            if not p_was_correct and p_is_correct: reward += 1.0
            elif p_was_correct and not p_is_correct: reward -= 0.1

            # Reward for moving pixel Q to/from correct spot
            if not q_was_correct and q_is_correct: reward += 1.0
            elif q_was_correct and not q_is_correct: reward -= 0.1

            # Event-based reward for completing rows/columns
            correct_rows_cols_after = self._count_correct_rows_cols()
            reward += (correct_rows_cols_after - correct_rows_cols_before) * 5.0

        # --- Check for Termination Conditions ---
        is_solved = np.array_equal(self.current_grid, self.target_grid)
        
        if is_solved:
            reward += 100  # Goal-oriented reward for winning
            terminated = True
            self.game_over = True
            self.win_state = "WIN"
        elif self.moves_remaining <= 0 and not is_solved:
            reward += -50  # Goal-oriented reward for losing
            terminated = True
            self.game_over = True
            self.win_state = "LOSE"
            
        self.score += reward
        self.steps += 1
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _count_correct_rows_cols(self):
        """Counts how many rows and columns are fully correct."""
        count = 0
        for r in range(self.GRID_ROWS):
            if np.array_equal(self.current_grid[r, :], self.target_grid[r, :]):
                count += 1
        for c in range(self.GRID_COLS):
            if np.array_equal(self.current_grid[:, c], self.target_grid[:, c]):
                count += 1
        return count

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the main game grids."""
        # Calculate grid dimensions and positions for centering
        self.cell_size = 40
        self.grid_width = self.GRID_COLS * self.cell_size
        self.grid_height = self.GRID_ROWS * self.cell_size
        
        total_width = self.grid_width * 2 + 80 # Two grids + padding
        start_x = (self.SCREEN_WIDTH - total_width) // 2
        
        self.main_grid_pos = (start_x, (self.SCREEN_HEIGHT - self.grid_height) // 2 + 20)
        self.target_grid_pos = (start_x + self.grid_width + 80, self.main_grid_pos[1])

        # Draw grids
        self._render_grid(self.current_grid, self.main_grid_pos, "Current State")
        self._render_grid(self.target_grid, self.target_grid_pos, "Target Pattern")

        # Highlight player pixel on the main grid
        if not self.game_over:
            player_r, player_c = self.player_pos
            highlight_rect = pygame.Rect(
                int(self.main_grid_pos[0] + player_c * self.cell_size),
                int(self.main_grid_pos[1] + player_r * self.cell_size),
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, highlight_rect, 4)

    def _render_grid(self, grid_data, top_left, title):
        """Helper function to draw a grid of colored pixels."""
        tx, ty = int(top_left[0]), int(top_left[1])
        
        title_surf = self.font_title.render(title, True, self.COLOR_TEXT)
        title_rect = title_surf.get_rect(center=(tx + self.grid_width / 2, ty - 25))
        self.screen.blit(title_surf, title_rect)
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_index = grid_data[r, c]
                color = self.PIXEL_COLORS[color_index]
                rect = pygame.Rect(
                    tx + c * self.cell_size,
                    ty + r * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)

    def _render_ui(self):
        """Renders the score, moves, and game over messages."""
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 20))

        moves_text = f"Moves: {self.moves_remaining}"
        moves_surf = self.font_main.render(moves_text, True, self.COLOR_TEXT)
        moves_rect = moves_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(moves_surf, moves_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg_text = "YOU WIN!" if self.win_state == "WIN" else "OUT OF MOVES"
            msg_color = (100, 255, 100) if self.win_state == "WIN" else (255, 100, 100)

            msg_surf = self.font_gameover.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "is_solved": np.array_equal(self.current_grid, self.target_grid),
        }
    
    def close(self):
        """Clean up pygame resources."""
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