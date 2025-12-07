
# Generated: 2025-08-27T20:41:58.436307
# Source Brief: brief_02546.md
# Brief Index: 2546

        
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
        "Controls: Arrow keys to move selector. Space to flip tiles. Clear the board before you run out of moves!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Flipping a tile also flips its neighbors. Match all tile colors to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_COLORS = 3
        self.MAX_MOVES = 25
        self.ANIMATION_FRAMES = 8

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.TILE_COLORS = [
            (220, 50, 50),   # Red
            (50, 220, 50),   # Green
            (50, 100, 220)   # Blue
        ]
        self.COLOR_SELECTOR = (255, 255, 0) # Yellow
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_FLASH = (255, 255, 255)

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
        try:
            self.font_main = pygame.font.SysFont('Consolas', 24, bold=True)
            self.font_large = pygame.font.SysFont('Consolas', 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 30)
            self.font_large = pygame.font.Font(None, 60)


        # Calculate grid rendering properties
        board_area_size = min(self.WIDTH, self.HEIGHT) - 40
        self.tile_size = board_area_size // self.GRID_SIZE
        self.grid_line_width = max(1, self.tile_size // 12)
        self.board_pixel_size = self.GRID_SIZE * self.tile_size
        self.board_offset_x = (self.WIDTH - self.board_pixel_size) // 2
        self.board_offset_y = (self.HEIGHT - self.board_pixel_size) // 2

        # State variables are initialized in reset()
        self.grid = None
        self.selector_pos = None
        self.moves_left = None
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.last_space_press = False
        self.animation_state = {}
        self.steps = 0

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.selector_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.last_space_press = False
        self.animation_state = {}
        self.steps = 0

        # Ensure the starting board is not already solved
        if self._check_win_condition():
            rand_r, rand_c = self.np_random.integers(0, self.GRID_SIZE, size=2)
            self._perform_flip(rand_r, rand_c, animate=False)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        self.steps += 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1 # Unused

        space_pressed_this_frame = space_held and not self.last_space_press
        self.last_space_press = space_held

        # 1. Handle Selector Movement
        if movement == 1:  # Up
            self.selector_pos[0] = (self.selector_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 2:  # Down
            self.selector_pos[0] = (self.selector_pos[0] + 1) % self.GRID_SIZE
        elif movement == 3:  # Left
            self.selector_pos[1] = (self.selector_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 4:  # Right
            self.selector_pos[1] = (self.selector_pos[1] + 1) % self.GRID_SIZE
        
        # 2. Handle Flip Action
        if space_pressed_this_frame:
            # sfx: flip_sound()
            self.moves_left -= 1
            reward = -0.1  # Cost of making a move

            self._perform_flip(self.selector_pos[0], self.selector_pos[1], animate=True)

            # Calculate reward based on board uniformity
            majority_color = self._get_majority_color()
            if majority_color is not None:
                matches = np.sum(self.grid == majority_color)
                reward += matches
            
            # 3. Check for Termination
            if self._check_win_condition():
                # sfx: win_jingle()
                terminated = True
                self.game_over = True
                self.win_state = True
                reward += 100
            elif self.moves_left <= 0:
                # sfx: lose_sound()
                terminated = True
                self.game_over = True
                self.win_state = False
                reward += -10
        
        self.score += reward
        self._update_animations()

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _perform_flip(self, r, c, animate=True):
        tiles_to_flip = [(r, c)]
        if r > 0: tiles_to_flip.append((r - 1, c))
        if r < self.GRID_SIZE - 1: tiles_to_flip.append((r + 1, c))
        if c > 0: tiles_to_flip.append((r, c - 1))
        if c < self.GRID_SIZE - 1: tiles_to_flip.append((r, c + 1))

        for row, col in tiles_to_flip:
            self.grid[row, col] = (self.grid[row, col] + 1) % self.NUM_COLORS
            if animate:
                self.animation_state[(row, col)] = self.ANIMATION_FRAMES

    def _get_majority_color(self):
        colors, counts = np.unique(self.grid, return_counts=True)
        return colors[np.argmax(counts)] if len(counts) > 0 else None

    def _check_win_condition(self):
        return np.all(self.grid == self.grid[0, 0])

    def _update_animations(self):
        finished_anims = [pos for pos, timer in self.animation_state.items() if timer <= 1]
        for pos in finished_anims:
            del self.animation_state[pos]
        for pos in self.animation_state:
            self.animation_state[pos] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = self.grid[r, c]
                tile_color = self.TILE_COLORS[color_index]
                rect = pygame.Rect(
                    self.board_offset_x + c * self.tile_size,
                    self.board_offset_y + r * self.tile_size,
                    self.tile_size, self.tile_size
                )
                pygame.draw.rect(self.screen, tile_color, rect)

                # Draw animation flash
                if (r, c) in self.animation_state:
                    progress = self.animation_state[(r, c)] / self.ANIMATION_FRAMES
                    flash_surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
                    alpha = int(255 * (progress ** 1.5))  # Ease-out effect
                    flash_surface.fill((*self.COLOR_FLASH, alpha))
                    self.screen.blit(flash_surface, rect.topleft)
        
        # Draw grid lines over tiles
        for i in range(self.GRID_SIZE + 1):
            start_x = self.board_offset_x + i * self.tile_size - self.grid_line_width // 2
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, self.board_offset_y), (start_x, self.board_offset_y + self.board_pixel_size), self.grid_line_width)
            start_y = self.board_offset_y + i * self.tile_size - self.grid_line_width // 2
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.board_offset_x, start_y), (self.board_offset_x + self.board_pixel_size, start_y), self.grid_line_width)

        # Draw selector
        sel_r, sel_c = self.selector_pos
        selector_rect = pygame.Rect(
            self.board_offset_x + sel_c * self.tile_size,
            self.board_offset_y + sel_r * self.tile_size,
            self.tile_size, self.tile_size
        )
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, width=max(2, self.grid_line_width))

    def _render_ui(self):
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 180))
            self.screen.blit(overlay, (0, 0))

            end_text_str = "YOU WIN!" if self.win_state else "GAME OVER"
            end_text_color = self.TILE_COLORS[1] if self.win_state else self.TILE_COLORS[0]
            
            end_text = self.font_large.render(end_text_str, True, end_text_color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "is_win": self.win_state if self.game_over else False
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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