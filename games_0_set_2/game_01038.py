
# Generated: 2025-08-27T15:38:36.994813
# Source Brief: brief_01038.md
# Brief Index: 1038

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select a square."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fill the entire 10x10 grid with a single color. You have 15 moves. "
        "Selecting a square will color it and its empty neighbors."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_MOVES = 15
        self.CELL_SIZE = 36
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_ORIGIN_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_ORIGIN_Y = (self.HEIGHT - self.GRID_WIDTH) // 2

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 60, bold=True)

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID_LINES = (40, 45, 55)
        self.COLOR_EMPTY_CELL = (30, 35, 45)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.VIBRANT_COLORS = [
            (255, 87, 34),   # Deep Orange
            (3, 169, 244),   # Light Blue
            (76, 175, 80),   # Green
            (233, 30, 99),   # Pink
            (156, 39, 176),  # Purple
            (0, 188, 212),   # Cyan
        ]

        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.remaining_moves = None
        self.score = None
        self.game_over = None
        self.win = None
        self.initial_color_set = None
        self.fill_color = None
        self.steps = None
        self.last_space_held = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.remaining_moves = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.win = False
        self.initial_color_set = False
        self.fill_color = self.COLOR_EMPTY_CELL
        self.steps = 0
        self.last_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return (
                self._get_observation(),
                0,
                True,
                False,
                self._get_info(),
            )

        movement = action[0]
        space_held = action[1] == 1
        
        self._handle_movement(movement)

        reward = 0
        space_pressed = space_held and not self.last_space_held
        if space_pressed:
            reward = self._handle_click()
        
        self.last_space_held = space_held
        self.steps += 1
        self.score += reward

        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100
                # sfx: win_sound
            else:
                reward -= 100
                # sfx: lose_sound
            self.score += reward
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_movement(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: # Up
            self.cursor_pos[1] -= 1
        elif movement == 2: # Down
            self.cursor_pos[1] += 1
        elif movement == 3: # Left
            self.cursor_pos[0] -= 1
        elif movement == 4: # Right
            self.cursor_pos[0] += 1
        
        if movement > 0:
            # sfx: cursor_move
            pass

        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)

    def _handle_click(self):
        click_reward = 0
        cx, cy = self.cursor_pos
        
        # Only allow clicks on empty squares
        if self.grid[cy, cx] == 0:
            # sfx: click_success
            if not self.initial_color_set:
                self.fill_color = self.VIBRANT_COLORS[self.np_random.integers(0, len(self.VIBRANT_COLORS))]
                self.initial_color_set = True
            
            squares_to_fill = [(cx, cy)]
            # Add orthogonal neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                    squares_to_fill.append((nx, ny))
            
            newly_filled_count = 0
            for x, y in squares_to_fill:
                if self.grid[y, x] == 0:
                    self.grid[y, x] = 1
                    newly_filled_count += 1
            
            # Reward is +1 for each newly filled square
            click_reward = newly_filled_count
            self.remaining_moves -= 1
        else:
            # sfx: click_fail
            pass
            
        return click_reward
    
    def _check_termination(self):
        if np.all(self.grid == 1):
            self.win = True
            return True
        if self.remaining_moves <= 0:
            self.win = False
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid cells
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.GRID_ORIGIN_X + x * self.CELL_SIZE,
                    self.GRID_ORIGIN_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                color = self.fill_color if self.grid[y, x] == 1 else self.COLOR_EMPTY_CELL
                pygame.draw.rect(self.screen, color, rect)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.GRID_ORIGIN_X + i * self.CELL_SIZE, self.GRID_ORIGIN_Y)
            end_pos = (self.GRID_ORIGIN_X + i * self.CELL_SIZE, self.GRID_ORIGIN_Y + self.GRID_WIDTH)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos, 2)
            # Horizontal lines
            start_pos = (self.GRID_ORIGIN_X, self.GRID_ORIGIN_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_ORIGIN_X + self.GRID_WIDTH, self.GRID_ORIGIN_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos, 2)

        # Draw pulsing cursor
        cursor_rect = pygame.Rect(
            self.GRID_ORIGIN_X + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_ORIGIN_Y + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE,
        )
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # Varies between 0 and 1
        thickness = 2 + int(pulse * 3)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, thickness)

    def _render_ui(self):
        # Display remaining moves
        moves_text = self.font_main.render(f"Moves: {self.remaining_moves}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        # Display game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_moves": self.remaining_moves,
            "win": self.win,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")