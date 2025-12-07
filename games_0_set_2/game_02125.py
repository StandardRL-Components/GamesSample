
# Generated: 2025-08-28T03:47:34.226894
# Source Brief: brief_02125.md
# Brief Index: 2125

        
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
        "Controls: Arrow keys to move cursor. Press space to change the color of the selected square and its neighbors."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Your goal is to make all squares the same color by clicking them. Each click changes the color of the selected square and its neighbors, but you only have a limited number of moves."
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
        
        # Pygame setup
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18)

        # Game constants
        self.GRID_ROWS, self.GRID_COLS = 5, 5
        self.MAX_MOVES = 10
        self.FLASH_DURATION = 5 # in steps

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_LINE = (50, 60, 70)
        self.COLOR_PALETTE = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 80, 255),   # Blue
            (255, 255, 80),  # Yellow
        ]
        self.NUM_COLORS = len(self.COLOR_PALETTE)
        self.COLOR_CURSOR = (220, 220, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_FLASH = (255, 255, 255)

        # Grid rendering properties
        self.grid_size = 350
        self.cell_size = self.grid_size // self.GRID_ROWS
        self.grid_top = (self.height - self.grid_size) // 2
        self.grid_left = (self.width - self.grid_size) // 2

        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.moves_remaining = 0
        self.score = 0
        self.steps = 0
        self.flash_effects = []
        
        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.flash_effects = []
        
        # Center the cursor
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        
        # Generate a random, non-solved grid
        while True:
            self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_ROWS, self.GRID_COLS))
            if len(np.unique(self.grid)) > 1:
                break
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        
        reward = 0
        terminated = False
        self.steps += 1
        
        # Update flash effect timers
        self.flash_effects = [fx for fx in self.flash_effects if fx['timer'] > 1]
        for fx in self.flash_effects:
            fx['timer'] -= 1

        # 1. Handle cursor movement
        if movement == 1:  # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_ROWS) % self.GRID_ROWS
        elif movement == 2:  # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_ROWS
        elif movement == 3:  # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_COLS) % self.GRID_COLS
        elif movement == 4:  # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_COLS

        # 2. Handle main game action ("click") if space is pressed
        if space_pressed:
            # Click sound placeholder: # pygame.mixer.Sound('click.wav').play()
            self.moves_remaining -= 1
            reward = -0.1  # Cost per move

            # Identify affected squares (target and orthogonal neighbors)
            r, c = self.cursor_pos
            affected_squares = [(r, c)]
            if r > 0: affected_squares.append((r - 1, c))
            if r < self.GRID_ROWS - 1: affected_squares.append((r + 1, c))
            if c > 0: affected_squares.append((r, c - 1))
            if c < self.GRID_COLS - 1: affected_squares.append((r, c + 1))
            
            # Change colors and add flash effects for visual feedback
            for sr, sc in affected_squares:
                self.grid[sr, sc] = (self.grid[sr, sc] + 1) % self.NUM_COLORS
                # Add/reset flash effect for this square
                self.flash_effects = [fx for fx in self.flash_effects if fx['pos'] != (sr, sc)]
                self.flash_effects.append({'pos': (sr, sc), 'timer': self.FLASH_DURATION})

            # Calculate majority color reward
            colors, counts = np.unique(self.grid, return_counts=True)
            if len(counts) > 0:
                reward += np.max(counts)

            # Check for win condition
            if len(colors) == 1:
                terminated = True
                reward += 100
                # Win sound placeholder: # pygame.mixer.Sound('win.wav').play()

            # Check for loss condition
            if self.moves_remaining <= 0 and not terminated:
                terminated = True
                reward += -10
                # Lose sound placeholder: # pygame.mixer.Sound('lose.wav').play()

        self.score += reward
        
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
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid squares
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_index = self.grid[r, c]
                color = self.COLOR_PALETTE[color_index]
                rect = pygame.Rect(
                    self.grid_left + c * self.cell_size,
                    self.grid_top + r * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)

        # Draw grid lines for a clean, defined look
        for i in range(self.GRID_ROWS + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE,
                             (self.grid_left, self.grid_top + i * self.cell_size),
                             (self.grid_left + self.grid_size, self.grid_top + i * self.cell_size), 2)
        for i in range(self.GRID_COLS + 1):
             pygame.draw.line(self.screen, self.COLOR_GRID_LINE,
                             (self.grid_left + i * self.cell_size, self.grid_top),
                             (self.grid_left + i * self.cell_size, self.grid_top + self.grid_size), 2)
        
        # Draw flash effects for action feedback
        for fx in self.flash_effects:
            r, c = fx['pos']
            alpha = int(200 * (fx['timer'] / self.FLASH_DURATION)) # Fade out effect
            flash_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_FLASH, alpha))
            self.screen.blit(flash_surface, (self.grid_left + c * self.cell_size, self.grid_top + r * self.cell_size))

        # Draw cursor as a bright, thick border
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_left + cursor_c * self.cell_size,
            self.grid_top + cursor_r * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 5)

    def _render_ui(self):
        # Render Moves Remaining in top-left
        moves_text_surf = self.font_large.render(str(self.moves_remaining), True, self.COLOR_TEXT)
        moves_label_surf = self.font_small.render("MOVES", True, self.COLOR_TEXT)
        self.screen.blit(moves_text_surf, (30, 20))
        self.screen.blit(moves_label_surf, (30, 60))

        # Render Score in top-right
        score_text_surf = self.font_large.render(f"{int(self.score)}", True, self.COLOR_TEXT)
        score_label_surf = self.font_small.render("SCORE", True, self.COLOR_TEXT)
        score_text_rect = score_text_surf.get_rect(topright=(self.width - 30, 20))
        score_label_rect = score_label_surf.get_rect(topright=(self.width - 30, 60))
        self.screen.blit(score_text_surf, score_text_rect)
        self.screen.blit(score_label_surf, score_label_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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