
# Generated: 2025-08-28T02:23:27.585912
# Source Brief: brief_04436.md
# Brief Index: 4436

        
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
    user_guide = "Controls: Use arrow keys (↑, ↓, ←, →) to slide the tiles into the empty space."

    # Must be a short, user-facing description of the game:
    game_description = "Slide numbered tiles in a grid to arrange them in ascending order within a limited number of moves."

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 200
        self.GRID_DIM = 3

        # Colors
        self.COLOR_BG = (210, 210, 210)
        self.COLOR_GRID = (130, 130, 130)
        self.COLOR_TILE = (255, 255, 255)
        self.COLOR_TEXT = (20, 20, 20)
        self.COLOR_UI_TEXT = (50, 50, 50)
        self.COLOR_SUCCESS = (100, 200, 100)
        self.COLOR_FAILURE = (200, 100, 100)

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
        self.tile_font = pygame.font.Font(None, 60)
        self.ui_font = pygame.font.Font(None, 32)
        self.end_font = pygame.font.Font(None, 80)
        
        # Game state variables (initialized in reset)
        self.grid = None
        self.empty_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.target_grid = np.append(np.arange(1, self.GRID_DIM**2), 0).reshape((self.GRID_DIM, self.GRID_DIM))
        
        # Initialize state variables
        self.reset()

        # Self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._shuffle_grid()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _shuffle_grid(self):
        # Start with the solved state
        self.grid = np.copy(self.target_grid)
        self.empty_pos = (self.GRID_DIM - 1, self.GRID_DIM - 1)

        # Perform a large number of random valid moves to shuffle
        # This guarantees a solvable puzzle
        shuffles = 150
        for _ in range(shuffles):
            valid_moves = []
            r, c = self.empty_pos
            if r > 0: valid_moves.append(1) # Up
            if r < self.GRID_DIM - 1: valid_moves.append(2) # Down
            if c > 0: valid_moves.append(3) # Left
            if c < self.GRID_DIM - 1: valid_moves.append(4) # Right

            if not valid_moves: continue
            
            # Choose a random move and apply it without counting steps or rewards
            move = self.np_random.choice(valid_moves)
            self._apply_move(move, count_step=False)

    def _apply_move(self, movement, count_step=True):
        """Applies a move to the grid. Returns True if move was valid, False otherwise."""
        if movement == 0: # No-op
            return False

        r, c = self.empty_pos
        dr, dc = 0, 0

        # The action moves the EMPTY space.
        # action 'up' moves empty space up, sliding a tile down into the void.
        if movement == 1: dr = -1 # Up
        elif movement == 2: dr = 1 # Down
        elif movement == 3: dc = -1 # Left
        elif movement == 4: dc = 1 # Right

        new_r, new_c = r + dr, c + dc

        # Check if the move is within bounds
        if 0 <= new_r < self.GRID_DIM and 0 <= new_c < self.GRID_DIM:
            # Swap tile with empty space
            self.grid[r, c], self.grid[new_r, new_c] = self.grid[new_r, new_c], self.grid[r, c]
            self.empty_pos = (new_r, new_c)
            if count_step:
                self.steps += 1
            return True
        
        return False
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        correct_before = self._count_correct_tiles()

        # Update game logic
        move_was_valid = self._apply_move(movement)

        if move_was_valid:
            # Penalty for making a move
            reward -= 1

            # Reward for placing a tile correctly
            correct_after = self._count_correct_tiles()
            if correct_after > correct_before:
                reward += 5 * (correct_after - correct_before)
        
        # Check termination conditions
        terminated = False
        is_solved = self._check_solved()
        limit_reached = self.steps >= self.MAX_STEPS

        if is_solved:
            reward += 100
            terminated = True
        elif limit_reached:
            reward -= 50
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _count_correct_tiles(self):
        # Exclude the empty spot from the count of correct tiles
        return np.sum(self.grid == self.target_grid) - (1 if self.grid[self.GRID_DIM-1, self.GRID_DIM-1] == 0 else 0)

    def _check_solved(self):
        return np.array_equal(self.grid, self.target_grid)

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
        grid_pixel_size = 330
        cell_size = grid_pixel_size // self.GRID_DIM
        padding = 6

        start_x = (self.WIDTH - grid_pixel_size) // 2
        start_y = (self.HEIGHT - grid_pixel_size) // 2

        # Draw grid background
        pygame.draw.rect(
            self.screen, self.COLOR_GRID,
            (start_x - padding, start_y - padding, grid_pixel_size + padding * 2, grid_pixel_size + padding * 2),
            border_radius=8
        )

        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                tile_val = self.grid[r, c]
                if tile_val == 0:
                    continue

                tile_x = start_x + c * cell_size + padding // 2
                tile_y = start_y + r * cell_size + padding // 2
                
                tile_rect = pygame.Rect(tile_x, tile_y, cell_size - padding, cell_size - padding)

                # Draw tile
                pygame.draw.rect(self.screen, self.COLOR_TILE, tile_rect, border_radius=5)
                
                # Draw number
                text_surf = self.tile_font.render(str(tile_val), True, self.COLOR_TEXT)
                text_rect = text_surf.get_rect(center=tile_rect.center)
                self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Moves remaining
        moves_text = f"Moves Left: {self.MAX_STEPS - self.steps}"
        text_surf = self.ui_font.render(moves_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (20, 20))

        # Score
        score_text = f"Score: {self.score}"
        text_surf = self.ui_font.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = text_surf.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(text_surf, score_rect)
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            if self._check_solved():
                msg = "SOLVED!"
                color = self.COLOR_SUCCESS
            else:
                msg = "GAME OVER"
                color = self.COLOR_FAILURE
            
            text_surf = self.end_font.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "is_solved": self._check_solved(),
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
        
        print("✓ Implementation validated successfully")