
# Generated: 2025-08-27T13:10:25.145057
# Source Brief: brief_00288.md
# Brief Index: 288

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    """
    A puzzle game where the player flips tiles on a 4x4 grid to match a target pattern.
    Each flip action inverts the selected tile and its orthogonal neighbors. The player
    has a limited number of moves to solve the puzzle.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to flip the selected tile and its neighbors. "
        "Press shift to reset the selector to the top-left corner."
    )

    # User-facing description of the game
    game_description = (
        "A strategic tile-flipping puzzle. Flip tiles to match the target pattern on the right "
        "before you run out of moves. Each flip affects a cross-shaped area."
    )

    # The game is turn-based, so it should only advance on action.
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Visuals & Fonts ---
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_LINES = (50, 60, 80)
        self.COLOR_TILE_0 = (70, 80, 100)
        self.COLOR_TILE_1 = (230, 230, 240)
        self.COLOR_CURSOR = (255, 180, 0)
        self.COLOR_FLASH = (255, 255, 150, 150) # RGBA for transparency
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_ACCENT = (255, 180, 0)

        # --- Game Constants ---
        self.GRID_SIZE = 4
        self.MAX_MOVES = 10
        
        # --- Game State (initialized in reset) ---
        self.grid = None
        self.target_grid = None
        self.cursor_pos = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.last_flipped_tiles = None
        self.last_action_was_flip = None
        self.rng = None

        # --- Layout ---
        self.main_grid_rect = pygame.Rect(40, 40, 320, 320)
        self.tile_size = self.main_grid_rect.width // self.GRID_SIZE
        
        self.target_grid_rect = pygame.Rect(420, 140, 160, 160)
        self.target_tile_size = self.target_grid_rect.width // self.GRID_SIZE

        # Initialize state variables
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        # Generate a solvable puzzle
        # Start with a random solved state (the target)
        self.target_grid = self.rng.integers(0, 2, size=(self.GRID_SIZE, self.GRID_SIZE))
        self.grid = np.copy(self.target_grid)
        
        # Apply a number of random flips to create the starting puzzle
        # This guarantees the puzzle is solvable
        num_scramble_flips = self.rng.integers(3, 8)
        for _ in range(num_scramble_flips):
            r, c = self.rng.integers(0, self.GRID_SIZE, size=2)
            self._perform_flip(r, c) # Use internal flip that doesn't track for rendering

        # Ensure the starting grid is not already the solution
        if np.array_equal(self.grid, self.target_grid):
            # If by chance we ended up with the solution, flip one more time
            self._perform_flip(0, 0)
            
        # Initialize game state
        self.cursor_pos = [0, 0] # [row, col]
        self.moves_left = self.MAX_MOVES
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_flipped_tiles = []
        self.last_action_was_flip = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Reset one-frame visual effects
        self.last_flipped_tiles = []
        self.last_action_was_flip = False

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1
        
        # --- Handle Input and Update Game Logic ---
        self._handle_input(movement, space_pressed, shift_pressed)
        
        # Update step counter
        self.steps += 1
        
        # Calculate reward and termination state
        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        self.game_over = terminated
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Handle cursor movement
        if movement == 1:  # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_SIZE
        elif movement == 2:  # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE
        elif movement == 3:  # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_SIZE
        elif movement == 4:  # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE

        # Handle cursor reset
        if shift_pressed:
            self.cursor_pos = [0, 0]

        # Handle tile flip
        if space_pressed and self.moves_left > 0:
            # SFX: Tile Flip Sound
            r, c = self.cursor_pos
            self._perform_flip(r, c)
            self.last_flipped_tiles.append((r, c))
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                    self.last_flipped_tiles.append((nr, nc))
            
            self.moves_left -= 1
            self.last_action_was_flip = True

    def _perform_flip(self, r, c):
        """Flips the tile at (r, c) and its orthogonal neighbors."""
        for dr, dc in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                self.grid[nr, nc] = 1 - self.grid[nr, nc]
    
    def _calculate_reward(self):
        # No reward for just moving the cursor
        if not self.last_action_was_flip:
            return 0
            
        is_win = np.array_equal(self.grid, self.target_grid)
        
        if is_win:
            # SFX: Puzzle Solved Chime
            return 100.0  # Large positive reward for winning
        
        # Calculate continuous feedback reward
        matches = np.sum(self.grid == self.target_grid)
        total_tiles = self.GRID_SIZE * self.GRID_SIZE
        mismatches = total_tiles - matches
        
        # Reward is proportional to the number of correct tiles
        # Scaled to be between -1.6 and +1.6 for a 4x4 grid
        reward = (matches * 0.1) - (mismatches * 0.1)
        
        return reward

    def _check_termination(self):
        is_win = np.array_equal(self.grid, self.target_grid)
        is_loss = self.moves_left <= 0 and not is_win
        return is_win or is_loss

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "is_solved": np.array_equal(self.grid, self.target_grid)
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Main Grid ---
        self._draw_grid(
            self.screen, 
            self.grid, 
            self.main_grid_rect, 
            self.tile_size, 
            draw_cursor=True
        )
        
        # --- Draw Target Grid ---
        self._draw_grid(
            self.screen, 
            self.target_grid, 
            self.target_grid_rect, 
            self.target_tile_size, 
            draw_cursor=False
        )

    def _draw_grid(self, surface, grid_data, grid_rect, tile_size, draw_cursor):
        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile_rect = pygame.Rect(
                    grid_rect.left + c * tile_size,
                    grid_rect.top + r * tile_size,
                    tile_size,
                    tile_size
                )
                color = self.COLOR_TILE_1 if grid_data[r, c] == 1 else self.COLOR_TILE_0
                pygame.draw.rect(surface, color, tile_rect)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (grid_rect.left + i * tile_size, grid_rect.top)
            end_pos = (grid_rect.left + i * tile_size, grid_rect.bottom)
            pygame.draw.line(surface, self.COLOR_GRID_LINES, start_pos, end_pos, 2)
            # Horizontal
            start_pos = (grid_rect.left, grid_rect.top + i * tile_size)
            end_pos = (grid_rect.right, grid_rect.top + i * tile_size)
            pygame.draw.line(surface, self.COLOR_GRID_LINES, start_pos, end_pos, 2)
            
        # Draw flash effect for recently flipped tiles
        if draw_cursor and self.last_flipped_tiles:
            flash_surface = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
            flash_surface.fill(self.COLOR_FLASH)
            for r, c in self.last_flipped_tiles:
                pos = (grid_rect.left + c * tile_size, grid_rect.top + r * tile_size)
                surface.blit(flash_surface, pos)

        # Draw cursor
        if draw_cursor:
            r, c = self.cursor_pos
            cursor_rect = pygame.Rect(
                grid_rect.left + c * tile_size,
                grid_rect.top + r * tile_size,
                tile_size,
                tile_size
            )
            pygame.draw.rect(surface, self.COLOR_CURSOR, cursor_rect, 4) # 4px thick border

    def _render_ui(self):
        # --- Moves Left Display ---
        moves_text = self.font_large.render("MOVES", True, self.COLOR_TEXT)
        moves_val = self.font_large.render(f"{self.moves_left:02d}", True, self.COLOR_TEXT_ACCENT)
        self.screen.blit(moves_text, (420, 40))
        self.screen.blit(moves_val, (520, 40))

        # --- Target Pattern Label ---
        target_text = self.font_large.render("TARGET", True, self.COLOR_TEXT)
        self.screen.blit(target_text, (420, 100))
        
        # --- Game Over / Win Message ---
        if self.game_over:
            is_win = np.array_equal(self.grid, self.target_grid)
            message = "PUZZLE SOLVED!" if is_win else "OUT OF MOVES"
            color = (150, 255, 150) if is_win else (255, 100, 100)
            
            overlay = pygame.Surface((self.main_grid_rect.width, 80), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            msg_render = self.font_large.render(message, True, color)
            msg_rect = msg_render.get_rect(center=(overlay.get_width() / 2, 40))
            overlay.blit(msg_render, msg_rect)
            
            self.screen.blit(overlay, (self.main_grid_rect.left, self.main_grid_rect.centery - 40))


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

# Example of how to run the environment
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless for this test
    
    env = GameEnv(render_mode="rgb_array")
    
    # Test reset
    obs, info = env.reset()
    print("Reset successful.")
    print("Initial Info:", info)
    
    # Test a few steps with random actions
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print("Episode finished.")
            break
    
    env.close()

    # Example of how to run with live rendering for human play
    # Note: Requires a display. Comment out the os.environ line above.
    # from gymnasium.utils.play import play
    # env = GameEnv(render_mode="rgb_array")
    # play(env, fps=10, keys_to_action={
    #     "w": np.array([1, 0, 0]),
    #     "s": np.array([2, 0, 0]),
    #     "a": np.array([3, 0, 0]),
    #     "d": np.array([4, 0, 0]),
    #     " ": np.array([0, 1, 0]),
    #     pygame.K_LSHIFT: np.array([0, 0, 1]),
    #     pygame.K_RSHIFT: np.array([0, 0, 1]),
    # })