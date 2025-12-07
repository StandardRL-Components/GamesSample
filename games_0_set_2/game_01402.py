
# Generated: 2025-08-27T17:01:52.931905
# Source Brief: brief_01402.md
# Brief Index: 1402

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to push all pixels Up, Down, Left, or Right. Try to match the target image."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Shift rows and columns of pixels to recreate the target image before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 10
        self.MAX_MOVES = 20
        self.MAX_COLORS = 6
        self.STARTING_LEVEL = 1

        # Colors
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_GRID_BG = (50, 50, 60)
        self.COLOR_BORDER = (80, 80, 90)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_TEXT_ACCENT = (100, 200, 255)
        self.PIXEL_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
        ]

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 18)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Game state
        self.level = self.STARTING_LEVEL
        self.np_random = None
        
        # Initialize state variables
        self.current_grid = None
        self.target_grid = None
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.last_push_direction = 0  # For visual feedback

        # Call reset to initialize the state for the first time
        # Note: Seeding is done here, so np_random is available after this
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.last_push_direction = 0

        num_colors = min(self.MAX_COLORS, self.level + 1)
        
        # Generate target grid
        self.target_grid = self.np_random.integers(
            0, num_colors, size=(self.GRID_SIZE, self.GRID_SIZE)
        )

        # Generate scrambled current grid by applying random moves to the target
        self.current_grid = self.target_grid.copy()
        num_scramble_moves = self.np_random.integers(15, 25)
        for _ in range(num_scramble_moves):
            move = self.np_random.integers(1, 5) # 1-4 for up/down/left/right
            self._apply_push(move)
        
        # Ensure it's not solved by accident after scrambling
        if np.array_equal(self.current_grid, self.target_grid):
            self._apply_push(self.np_random.integers(1, 5))

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _apply_push(self, move_direction):
        # Sound effect placeholder: # pygame.mixer.Sound.play(push_sound)
        if move_direction == 1: # Up
            self.current_grid = np.roll(self.current_grid, -1, axis=0)
        elif move_direction == 2: # Down
            self.current_grid = np.roll(self.current_grid, 1, axis=0)
        elif move_direction == 3: # Left
            self.current_grid = np.roll(self.current_grid, -1, axis=1)
        elif move_direction == 4: # Right
            self.current_grid = np.roll(self.current_grid, 1, axis=1)
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held and shift_held have no effect in this game
        
        self.steps += 1
        self.last_push_direction = 0
        reward = 0.0
        terminated = False

        if movement in [1, 2, 3, 4]: # A valid push action
            self.moves_left -= 1
            self.last_push_direction = movement
            
            old_correct_pixels = np.sum(self.current_grid == self.target_grid)
            
            self._apply_push(movement)

            new_correct_pixels = np.sum(self.current_grid == self.target_grid)

            # Continuous feedback reward for improving the board state
            reward = (new_correct_pixels - old_correct_pixels) * 0.1
        
        # Check for win condition
        is_win = np.array_equal(self.current_grid, self.target_grid)
        if is_win:
            # Sound effect placeholder: # pygame.mixer.Sound.play(win_sound)
            reward += 100.0  # Large goal-oriented reward
            self.game_over = True
            terminated = True
            self.level += 1 # Increase difficulty for the next game
        
        # Check for loss condition (out of moves)
        is_loss = self.moves_left <= 0 and not is_win
        if is_loss:
            # Sound effect placeholder: # pygame.mixer.Sound.play(loss_sound)
            self.game_over = True
            terminated = True
        
        # Safeguard termination
        if self.steps >= 1000:
            terminated = True

        # MUST return exactly this 5-tuple
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

        # After rendering the effect for one frame, reset it
        self.last_push_direction = 0
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        # Score is the number of correctly placed pixels
        score = int(np.sum(self.current_grid == self.target_grid))
        return {
            "score": score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "level": self.level,
        }

    def _render_game(self):
        # Calculate grid dimensions and positions
        grid_pixel_size = 280
        cell_size = grid_pixel_size // self.GRID_SIZE
        padding = 30
        
        # Target Grid (left)
        target_x = (self.SCREEN_WIDTH // 2) - grid_pixel_size - padding
        target_y = (self.SCREEN_HEIGHT - grid_pixel_size) // 2 + 20
        self._draw_grid(self.target_grid, target_x, target_y, cell_size, "TARGET")

        # Playable Grid (right)
        play_x = (self.SCREEN_WIDTH // 2) + padding
        play_y = target_y
        self._draw_grid(self.current_grid, play_x, play_y, cell_size, "YOUR GRID")

        # Draw push feedback on the playable grid
        if self.last_push_direction != 0:
            self._draw_push_effect(play_x, play_y, cell_size)

    def _draw_grid(self, grid, start_x, start_y, cell_size, title):
        grid_pixel_size = cell_size * self.GRID_SIZE
        # Draw background and border
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (start_x, start_y, grid_pixel_size, grid_pixel_size))
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (start_x, start_y, grid_pixel_size, grid_pixel_size), 2)

        # Draw cells
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = grid[r, c]
                color = self.PIXEL_COLORS[color_index]
                # Draw with a small gap to create a border effect
                rect = (start_x + c * cell_size + 1, start_y + r * cell_size + 1, cell_size - 2, cell_size - 2)
                pygame.draw.rect(self.screen, color, rect)
        
        # Draw title
        title_surf = self.font_large.render(title, True, self.COLOR_TEXT)
        title_rect = title_surf.get_rect(center=(start_x + grid_pixel_size / 2, start_y - 25))
        self.screen.blit(title_surf, title_rect)

    def _draw_push_effect(self, grid_start_x, grid_start_y, cell_size):
        highlight_color = (255, 255, 255, 60) # Semi-transparent white
        grid_pixel_size = self.GRID_SIZE * cell_size
        
        # Create a surface that covers the entire grid for the highlight
        surface = pygame.Surface((grid_pixel_size, grid_pixel_size), pygame.SRCALPHA)
        surface.fill(highlight_color)
        self.screen.blit(surface, (grid_start_x, grid_start_y))

    def _render_ui(self):
        # Moves left display
        moves_text = f"MOVES: {self.moves_left}"
        moves_surf = self.font_medium.render(moves_text, True, self.COLOR_TEXT_ACCENT)
        self.screen.blit(moves_surf, (20, 20))

        # Match display
        correct_pixels = int(np.sum(self.current_grid == self.target_grid))
        total_pixels = self.GRID_SIZE * self.GRID_SIZE
        match_text = f"MATCH: {correct_pixels} / {total_pixels}"
        match_surf = self.font_medium.render(match_text, True, self.COLOR_TEXT)
        self.screen.blit(match_surf, (20, 50))
        
        # Game Over / Win message
        if self.game_over:
            is_win = np.array_equal(self.current_grid, self.target_grid)
            message = "COMPLETE!" if is_win else "OUT OF MOVES"
            color = (100, 255, 100) if is_win else (255, 100, 100)
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_surf = self.font_large.render(message, True, color)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_surf, end_rect)

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