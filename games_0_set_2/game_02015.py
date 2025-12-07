import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to reveal a square. Shift to place a flag."
    )

    # Must be a user-facing description of the game:
    game_description = (
        "A classic mine-sweeping puzzle game. Reveal all safe squares without detonating any mines."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Grid Dimensions
        self.width, self.height = 640, 400
        self.rows, self.cols = 10, 16
        self.cell_size = 40  # 10*40=400, 16*40=640
        self.num_mines = 20

        # Colors
        self.COLOR_BG = (192, 192, 192)
        self.COLOR_GRID = (128, 128, 128)
        self.COLOR_HIDDEN = (224, 224, 224)
        self.COLOR_REVEALED = (192, 192, 192)
        self.COLOR_FLAG = (255, 0, 0)
        self.COLOR_MINE = (0, 0, 0)
        self.COLOR_CURSOR = (0, 0, 255)
        self.COLOR_NUMBERS = {
            1: (0, 0, 255), 2: (0, 128, 0), 3: (255, 0, 0),
            4: (0, 0, 128), 5: (128, 0, 0), 6: (0, 128, 128),
            7: (0, 0, 0), 8: (128, 128, 128)
        }
        self.COLOR_TEXT = (0, 0, 0)

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", int(self.cell_size * 0.6))
        self.game_over_font = pygame.font.SysFont("Arial", 50)

        # Game State (initialized in reset)
        self.grid = None
        self.revealed = None
        self.flags = None
        self.cursor_pos = None
        self.game_over = False
        self.win = False
        self.steps = 0
        self.score = 0
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # Reset game state
        self.game_over = False
        self.win = False
        self.steps = 0
        self.score = 0
        self.cursor_pos = [self.rows // 2, self.cols // 2]
        
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.revealed = np.zeros((self.rows, self.cols), dtype=bool)
        self.flags = np.zeros((self.rows, self.cols), dtype=bool)

        # Place mines
        mine_indices = self.np_random.choice(self.rows * self.cols, self.num_mines, replace=False)
        mine_coords = np.unravel_index(mine_indices, (self.rows, self.cols))
        self.grid[mine_coords] = -1  # -1 represents a mine

        # Calculate neighbor counts
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == -1:
                    continue
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] == -1:
                            count += 1
                self.grid[r, c] = count

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, reveal_action, flag_action = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # 1. Update cursor position
        if movement == 1:  # Up
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[0] = min(self.rows - 1, self.cursor_pos[0] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[1] = min(self.cols - 1, self.cursor_pos[1] + 1)

        r, c = self.cursor_pos[0], self.cursor_pos[1]

        # 2. Handle actions
        if flag_action:
            if not self.revealed[r, c]:
                self.flags[r, c] = not self.flags[r, c]
        elif reveal_action:
            if not self.revealed[r, c] and not self.flags[r, c]:
                if self.grid[r, c] == -1: # Hit a mine
                    self.game_over = True
                    self.win = False
                    reward = -1.0
                else: # Revealed a safe cell
                    revealed_before = np.sum(self.revealed)
                    self._reveal_cell(r, c)
                    revealed_after = np.sum(self.revealed)
                    reward = 0.01 * (revealed_after - revealed_before)

        # 3. Check for win condition
        num_safe_cells = self.rows * self.cols - self.num_mines
        num_revealed_safe = np.sum(self.revealed) # Since mines can't be revealed
        if not self.game_over and num_revealed_safe == num_safe_cells:
            self.game_over = True
            self.win = True
            self.score = 1
            reward = 1.0

        self.steps += 1
        terminated = self.game_over

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _reveal_cell(self, r, c):
        # Recursive function to reveal cells
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return
        if self.revealed[r, c] or self.flags[r, c]:
            return
        
        self.revealed[r, c] = True

        if self.grid[r, c] == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    self._reveal_cell(r + dr, c + dc)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        for r in range(self.rows):
            for c in range(self.cols):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                is_mine = self.grid[r, c] == -1
                
                if self.revealed[r, c] or (self.game_over and is_mine):
                    pygame.draw.rect(self.screen, self.COLOR_REVEALED, rect)
                    if is_mine:
                        pygame.draw.circle(self.screen, self.COLOR_MINE, rect.center, self.cell_size // 4)
                    elif self.grid[r, c] > 0:
                        num = self.grid[r, c]
                        text_surf = self.font.render(str(num), True, self.COLOR_NUMBERS.get(num, self.COLOR_TEXT))
                        text_rect = text_surf.get_rect(center=rect.center)
                        self.screen.blit(text_surf, text_rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_HIDDEN, rect)
                    if self.flags[r, c]:
                        pygame.draw.polygon(self.screen, self.COLOR_FLAG, [
                            (rect.left + self.cell_size // 4, rect.top + self.cell_size // 4),
                            (rect.left + self.cell_size // 4, rect.bottom - self.cell_size // 4),
                            (rect.right - self.cell_size // 4, rect.centery)
                        ])
                
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        cursor_rect = pygame.Rect(self.cursor_pos[1] * self.cell_size, self.cursor_pos[0] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

        if self.game_over:
            msg = "You Win!" if self.win else "Game Over"
            color = (0, 128, 0) if self.win else (255, 0, 0)
            text_surf = self.game_over_font.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.width // 2, self.height // 2))
            bg_rect = text_rect.inflate(20, 20)
            pygame.draw.rect(self.screen, (255, 255, 255, 180), bg_rect, border_radius=10)
            self.screen.blit(text_surf, text_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()