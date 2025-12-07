
# Generated: 2025-08-28T03:16:30.017392
# Source Brief: brief_01979.md
# Brief Index: 1979

        
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
    user_guide = "Controls: Use arrow keys to push all pixels in the chosen direction."

    # Must be a short, user-facing description of the game:
    game_description = "Recreate the target image by pushing pixels around the grid. Each push costs one move."

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # Constants
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    MAX_MOVES = 50
    MAX_STEPS = 500

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (50, 50, 70)
    COLOR_TEXT = (220, 220, 240)
    PALETTE = {
        1: (255, 220, 0),    # Bright Yellow
        2: (40, 40, 40),     # Dark Gray/Black
    }
    DESATURATED_PALETTE = {
        k: tuple(int(c * 0.8) for c in v) for k, v in PALETTE.items()
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans-serif", 24)
        self.font_title = pygame.font.SysFont("sans-serif", 18, bold=True)

        self.grid_rect = self._calculate_grid_rect()
        self.cell_size = self.grid_rect.width // self.GRID_SIZE

        self.target_image = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.target_image[2:8, 2:8] = 1
        self.target_image[2,2]=0; self.target_image[2,7]=0; self.target_image[7,2]=0; self.target_image[7,7]=0
        self.target_image[3, 3] = 2; self.target_image[3, 6] = 2
        self.target_image[4, 4:6] = 1
        self.target_image[6, 3:7] = 2
        self.target_image[5, 2] = 0; self.target_image[5, 7] = 0

        self.target_pixels = [
            pixel_id for pixel_id in self.target_image.flatten() if pixel_id != 0
        ]

        self.grid = None
        self.moves_left = 0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.last_action_feedback = 0

        self.reset()
        self.validate_implementation()

    def _calculate_grid_rect(self):
        grid_dim = min(self.SCREEN_WIDTH, self.SCREEN_HEIGHT) * 0.85
        grid_dim = int(grid_dim - (grid_dim % self.GRID_SIZE))
        left = (self.SCREEN_WIDTH - grid_dim) / 2
        top = (self.SCREEN_HEIGHT - grid_dim) / 2
        return pygame.Rect(left, top, grid_dim, grid_dim)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.moves_left = self.MAX_MOVES

        self.grid = [[[] for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        
        shuffled_pixels = list(self.target_pixels)
        self.np_random.shuffle(shuffled_pixels)

        for pixel_id in shuffled_pixels:
            r = self.np_random.integers(0, self.GRID_SIZE)
            c = self.np_random.integers(0, self.GRID_SIZE)
            self.grid[r][c].append(pixel_id)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        
        self.steps += 1

        if 1 <= movement <= 4:
            self.moves_left -= 1
            self.last_action_feedback = 10 # Countdown for visual effect
            # sfx: push_sound

            new_grid = [[[] for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
            
            if movement == 1: # UP
                for r in range(self.GRID_SIZE):
                    for c in range(self.GRID_SIZE):
                        if self.grid[r][c]:
                            new_r = (r - 1 + self.GRID_SIZE) % self.GRID_SIZE
                            new_grid[new_r][c].extend(self.grid[r][c])
            elif movement == 2: # DOWN
                for r in range(self.GRID_SIZE - 1, -1, -1):
                    for c in range(self.GRID_SIZE):
                        if self.grid[r][c]:
                            new_r = (r + 1) % self.GRID_SIZE
                            new_grid[new_r][c].extend(self.grid[r][c])
            elif movement == 3: # LEFT
                for c in range(self.GRID_SIZE):
                    for r in range(self.GRID_SIZE):
                        if self.grid[r][c]:
                            new_c = (c - 1 + self.GRID_SIZE) % self.GRID_SIZE
                            new_grid[r][new_c].extend(self.grid[r][c])
            elif movement == 4: # RIGHT
                for c in range(self.GRID_SIZE - 1, -1, -1):
                    for r in range(self.GRID_SIZE):
                        if self.grid[r][c]:
                            new_c = (c + 1) % self.GRID_SIZE
                            new_grid[r][new_c].extend(self.grid[r][c])
            
            self.grid = new_grid
            reward = self._calculate_reward()
            self.score += reward

        terminated = self._check_termination()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_topmost_pixel(self, r, c):
        return self.grid[r][c][-1] if self.grid[r][c] else 0

    def _is_grid_solved(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self._get_topmost_pixel(r, c) != self.target_image[r, c]:
                    return False
        return True

    def _calculate_reward(self):
        if self._is_grid_solved():
            # sfx: victory_fanfare
            return 100.0

        reward = 0.0
        
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self._get_topmost_pixel(r, c) == self.target_image[r, c] and self.target_image[r,c] != 0:
                    reward += 0.1

        for r in range(self.GRID_SIZE):
            if all(self._get_topmost_pixel(r, c) == self.target_image[r, c] for c in range(self.GRID_SIZE)):
                reward += 5.0
        
        for c in range(self.GRID_SIZE):
            if all(self._get_topmost_pixel(r, c) == self.target_image[r, c] for r in range(self.GRID_SIZE)):
                reward += 5.0

        return reward

    def _check_termination(self):
        if self._is_grid_solved():
            self.game_over = True
            return True

        if self.moves_left <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            # sfx: failure_sound
            return True

        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        flash_intensity = max(0, self.last_action_feedback * 20)
        grid_color = tuple(min(255, c + flash_intensity) for c in self.COLOR_GRID)
        if self.last_action_feedback > 0: self.last_action_feedback -= 1

        for i in range(self.GRID_SIZE + 1):
            x = self.grid_rect.left + i * self.cell_size
            pygame.draw.line(self.screen, grid_color, (x, self.grid_rect.top), (x, self.grid_rect.bottom), 1)
            y = self.grid_rect.top + i * self.cell_size
            pygame.draw.line(self.screen, grid_color, (self.grid_rect.left, y), (self.grid_rect.right, y), 1)

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                pixels_in_cell = self.grid[r][c]
                stack_size = len(pixels_in_cell)
                if stack_size > 0:
                    for i, pixel_id in enumerate(pixels_in_cell):
                        base_size = self.cell_size * 0.8
                        size_reduction = (stack_size - 1 - i) * (self.cell_size * 0.15)
                        pixel_size = max(4, base_size - size_reduction)
                        offset = (self.cell_size - pixel_size) / 2
                        
                        px = self.grid_rect.left + c * self.cell_size + offset
                        py = self.grid_rect.top + r * self.cell_size + offset
                        
                        color = self.DESATURATED_PALETTE.get(pixel_id, (255, 0, 255))
                        
                        rect = pygame.Rect(int(px), int(py), int(pixel_size), int(pixel_size))
                        pygame.draw.rect(self.screen, color, rect, border_radius=int(pixel_size*0.25))

    def _render_ui(self):
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}/{self.MAX_MOVES}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, self.SCREEN_HEIGHT - 40))

        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(right=self.SCREEN_WIDTH - 20, top=self.SCREEN_HEIGHT - 40)
        self.screen.blit(score_text, score_rect)

        preview_size = 100
        preview_rect = pygame.Rect(self.SCREEN_WIDTH - preview_size - 20, 20, preview_size, preview_size)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_rect, 2, border_radius=5)
        
        title_text = self.font_title.render("Target", True, self.COLOR_TEXT)
        title_rect = title_text.get_rect(centerx=preview_rect.centerx, bottom=preview_rect.top - 5)
        self.screen.blit(title_text, title_rect)

        preview_cell_size = preview_size / self.GRID_SIZE
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                pixel_id = self.target_image[r, c]
                if pixel_id != 0:
                    color = self.PALETTE.get(pixel_id)
                    cell_rect = pygame.Rect(
                        preview_rect.left + c * preview_cell_size,
                        preview_rect.top + r * preview_cell_size,
                        math.ceil(preview_cell_size),
                        math.ceil(preview_cell_size)
                    )
                    pygame.draw.rect(self.screen, color, cell_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            is_won = self._is_grid_solved()
            msg = "YOU WIN!" if is_won else "OUT OF MOVES"
            color = (100, 255, 100) if is_won else (255, 100, 100)
            
            end_font = pygame.font.SysFont("sans-serif", 60, bold=True)
            end_text = end_font.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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