
# Generated: 2025-08-28T02:48:11.418510
# Source Brief: brief_01819.md
# Brief Index: 1819

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move selection. Press space to flip. Hold shift to reset selection to top-left."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Flip adjacent tiles in a 3x3 grid to match the target pattern. You have a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 3
        self.MAX_MOVES = 15

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (60, 65, 80)
        self.COLOR_TILE_0 = (45, 50, 65)
        self.COLOR_TILE_1 = (255, 198, 0)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_SELECTOR = (0, 221, 255)
        self.COLOR_FLASH = (255, 255, 255)

        # Grid rendering properties
        self.grid_area_size = 300
        self.tile_size = self.grid_area_size // self.GRID_SIZE
        self.grid_start_x = (self.SCREEN_WIDTH - self.grid_area_size) // 2
        self.grid_start_y = (self.SCREEN_HEIGHT - self.grid_area_size) // 2
        
        # Initialize state variables
        self.current_grid = None
        self.target_grid = None
        self.selected_tile = None
        self.moves_remaining = None
        self.score = None
        self.game_over = None
        self.win_state = None
        self.last_flipped_indices = []
        self.steps = None
        
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.moves_remaining = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_state = False
        self.selected_tile = (0, 0)
        self.last_flipped_indices = []

        # Generate a solvable puzzle
        self.target_grid = self.np_random.integers(0, 2, size=(self.GRID_SIZE, self.GRID_SIZE))
        self.current_grid = np.copy(self.target_grid)

        # Scramble the grid by making a few random moves, ensuring it's not already solved
        num_scrambles = self.np_random.integers(4, 9)
        for _ in range(num_scrambles):
            rand_x, rand_y = self.np_random.integers(0, self.GRID_SIZE, size=2)
            self._apply_flip(rand_x, rand_y, self.current_grid)
        
        # Ensure the starting grid is not the target grid
        if np.array_equal(self.current_grid, self.target_grid):
            # If by chance it is, flip one more time
            self._apply_flip(0, 0, self.current_grid)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing and return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.last_flipped_indices = [] # Clear flash effect from previous step

        # Unpack factorized action
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        terminated = False
        
        # 1. Handle selection reset (Shift)
        if shift_pressed:
            self.selected_tile = (0, 0)
        
        # 2. Handle selection movement (Arrows)
        sel_x, sel_y = self.selected_tile
        if movement == 1: # Up
            sel_y = (sel_y - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 2: # Down
            sel_y = (sel_y + 1) % self.GRID_SIZE
        elif movement == 3: # Left
            sel_x = (sel_x - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 4: # Right
            sel_x = (sel_x + 1) % self.GRID_SIZE
        self.selected_tile = (sel_x, sel_y)

        # 3. Handle flip action (Space)
        if space_pressed:
            # This is a "move"
            self.moves_remaining -= 1
            
            # Apply the flip logic
            self.last_flipped_indices = self._get_flip_indices(sel_x, sel_y)
            for x, y in self.last_flipped_indices:
                self.current_grid[y, x] = 1 - self.current_grid[y, x]
            # sfx: tile_flip.wav

            # Check for win condition
            if np.array_equal(self.current_grid, self.target_grid):
                reward = 100.0
                self.score += reward
                terminated = True
                self.win_state = True
                # sfx: win_jingle.wav
            else:
                # Calculate continuous reward based on new state
                matches = np.sum(self.current_grid == self.target_grid)
                mismatches = (self.GRID_SIZE ** 2) - matches
                reward = (matches * 1.0) - (mismatches * 0.1)
                self.score += reward
                
                # Check for loss condition (out of moves)
                if self.moves_remaining <= 0:
                    reward = -10.0 # Override continuous reward with loss penalty
                    self.score += reward # Adjust score
                    terminated = True
                    self.win_state = False
                    # sfx: lose_buzzer.wav
        
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _apply_flip(self, x, y, grid):
        """Helper to apply a flip to a given grid, used for puzzle generation."""
        indices_to_flip = self._get_flip_indices(x, y)
        for fx, fy in indices_to_flip:
            grid[fy, fx] = 1 - grid[fy, fx]

    def _get_flip_indices(self, x, y):
        """Returns a list of (x, y) coordinates to be flipped for a given selection."""
        indices = [(x, y)]
        if x > 0: indices.append((x - 1, y))
        if x < self.GRID_SIZE - 1: indices.append((x + 1, y))
        if y > 0: indices.append((x, y - 1))
        if y < self.GRID_SIZE - 1: indices.append((x, y + 1))
        return indices
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.MAX_MOVES - self.moves_remaining,
            "moves_remaining": self.moves_remaining,
            "selected_tile": self.selected_tile,
        }

    def _render_game(self):
        # Render main grid
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.grid_start_x + x * self.tile_size,
                    self.grid_start_y + y * self.tile_size,
                    self.tile_size,
                    self.tile_size
                )
                
                # Draw flash effect if tile was just flipped
                if (x, y) in self.last_flipped_indices:
                    flash_rect = rect.inflate(10, 10)
                    pygame.draw.rect(self.screen, self.COLOR_FLASH, flash_rect, border_radius=8)

                # Draw tile
                color = self.COLOR_TILE_1 if self.current_grid[y, x] == 1 else self.COLOR_TILE_0
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, width=2, border_radius=5)
        
        # Render selector
        sel_x, sel_y = self.selected_tile
        selector_rect = pygame.Rect(
            self.grid_start_x + sel_x * self.tile_size,
            self.grid_start_y + sel_y * self.tile_size,
            self.tile_size,
            self.tile_size
        )
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, width=4, border_radius=7)

    def _render_ui(self):
        # Render moves remaining
        moves_text = self.font_medium.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        # Render score
        score_text = self.font_medium.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, self.SCREEN_HEIGHT - 40))

        # Render target pattern
        target_title = self.font_small.render("Target", True, self.COLOR_TEXT)
        self.screen.blit(target_title, (self.SCREEN_WIDTH - 110, 20))
        
        target_tile_size = 30
        target_start_x = self.SCREEN_WIDTH - 115
        target_start_y = 50
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    target_start_x + x * target_tile_size,
                    target_start_y + y * target_tile_size,
                    target_tile_size,
                    target_tile_size
                )
                color = self.COLOR_TILE_1 if self.target_grid[y, x] == 1 else self.COLOR_TILE_0
                pygame.draw.rect(self.screen, color, rect, border_radius=3)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, width=1, border_radius=3)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.win_state:
                end_text = self.font_large.render("YOU WIN!", True, self.COLOR_TILE_1)
            else:
                end_text = self.font_large.render("OUT OF MOVES", True, (200, 50, 50))
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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
    # Set this to run in a headless environment
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # To play manually, you would need a different setup to capture keyboard events
    # This example demonstrates the programmatic interface
    
    obs, info = env.reset()
    print("Initial State:")
    print(f"  Score: {info['score']}, Moves Taken: {info['steps']}")

    terminated = False
    total_reward = 0
    
    # Run for a few random steps
    for i in range(20):
        if terminated:
            print("\nEpisode finished.")
            break
            
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        action_str = f"Move: {action[0]}, Flip: {action[1]}, Reset: {action[2]}"
        print(
            f"Step {i+1}: Action: [{action_str}], "
            f"Reward: {reward:.2f}, "
            f"Total Reward: {total_reward:.2f}, "
            f"Moves Taken: {info['steps']}, "
            f"Terminated: {terminated}"
        )

    env.close()