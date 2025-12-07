
# Generated: 2025-08-28T03:39:43.013026
# Source Brief: brief_04998.md
# Brief Index: 4998

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to reveal a square. Shift to flag a square."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic puzzle game. Reveal all the safe squares on the grid without hitting a mine. "
        "Numbers on revealed squares indicate how many mines are adjacent."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 9
    NUM_MINES = 10
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID_LINES = (50, 60, 70)
    COLOR_UNREVEALED = (70, 80, 90)
    COLOR_REVEALED_SAFE = (100, 110, 120)
    COLOR_CURSOR = (70, 150, 255, 100)  # Semi-transparent
    COLOR_EXPLOSION = (255, 80, 80)
    COLOR_FLAG = (255, 200, 80)
    COLOR_TEXT = (255, 255, 255)
    COLOR_REVEAL_FLASH = (255, 255, 255, 150)
    NUMBER_COLORS = [
        None, (80, 160, 255), (80, 200, 80), (255, 80, 80),
        (150, 80, 255), (255, 120, 0), (80, 200, 200),
        (200, 80, 200), (200, 200, 80)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Game state variables are initialized in reset()
        self.grid_width = 0
        self.grid_height = 0
        self.cell_size = 0
        self.grid_offset_x = 0
        self.grid_offset_y = 0
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_revealed_pos = [] # For flash effect

        self._generate_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        reward = 0
        self.last_revealed_pos = [] # Clear flash effect from previous step

        # --- Handle player input ---
        self._handle_movement(movement)

        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        if not self.game_over and not self.win:
            if space_press:
                # sound_placeholder = "reveal_sound.wav"
                reward += self._reveal_square(self.cursor_pos[0], self.cursor_pos[1])
            elif shift_press:
                # sound_placeholder = "flag_sound.wav"
                self._toggle_flag(self.cursor_pos[0], self.cursor_pos[1])
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        self.steps += 1
        
        # --- Check termination conditions ---
        terminated = self.game_over or self.win or self.steps >= self.MAX_STEPS

        if not self.game_over and self._check_win():
            self.win = True
            terminated = True
            reward += 100 # Win bonus
            # sound_placeholder = "win_sound.wav"

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "mines_left": self.NUM_MINES - np.sum(self.flagged_grid)
        }

    # --- Helper Methods for Game Logic ---

    def _generate_board(self):
        self.mine_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.revealed_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.flagged_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.number_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)

        # Place mines
        mine_indices = self.np_random.choice(self.GRID_SIZE * self.GRID_SIZE, self.NUM_MINES, replace=False)
        for idx in mine_indices:
            x, y = idx % self.GRID_SIZE, idx // self.GRID_SIZE
            self.mine_grid[y, x] = True

        # Calculate numbers
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if not self.mine_grid[y, x]:
                    count = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.mine_grid[ny, nx]:
                                count += 1
                    self.number_grid[y, x] = count

    def _handle_movement(self, movement):
        x, y = self.cursor_pos
        if movement == 1: y -= 1  # Up
        elif movement == 2: y += 1  # Down
        elif movement == 3: x -= 1  # Left
        elif movement == 4: x += 1  # Right
        
        # Wrap around logic
        self.cursor_pos[0] = x % self.GRID_SIZE
        self.cursor_pos[1] = y % self.GRID_SIZE

    def _reveal_square(self, x, y):
        if self.revealed_grid[y, x] or self.flagged_grid[y, x]:
            return 0 # No action, no reward

        if self.mine_grid[y, x]:
            self.revealed_grid[y, x] = True
            self.last_revealed_pos.append((x, y))
            self.game_over = True
            # sound_placeholder = "explosion_sound.wav"
            return -100 # Mine penalty

        # Flood fill for empty squares
        reward = 0
        q = deque([(x, y)])
        visited = set([(x, y)])

        while q:
            cx, cy = q.popleft()
            
            if self.revealed_grid[cy, cx]: continue
            
            self.revealed_grid[cy, cx] = True
            self.last_revealed_pos.append((cx, cy))
            
            # Reward for revealing a safe square
            reward += 1.0
            num = self.number_grid[cy, cx]
            if num > 0:
                reward -= 0.1 # Small penalty for risky reveal
            else:
                reward -= 0.01 # Tiny penalty for safe reveal

            # If it's an empty square (0), add neighbors to the queue
            if num == 0:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in visited:
                            if not self.flagged_grid[ny, nx]:
                                q.append((nx, ny))
                                visited.add((nx, ny))
        return reward

    def _toggle_flag(self, x, y):
        if not self.revealed_grid[y, x]:
            self.flagged_grid[y, x] = not self.flagged_grid[y, x]

    def _check_win(self):
        return np.sum(self.revealed_grid) == (self.GRID_SIZE * self.GRID_SIZE - self.NUM_MINES)

    # --- Helper Methods for Rendering ---

    def _calculate_grid_layout(self):
        # Calculate cell size and offsets to center the grid
        self.grid_height = self.SCREEN_HEIGHT - 80
        self.grid_width = self.grid_height
        self.cell_size = self.grid_width // self.GRID_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2 + 20

    def _render_game(self):
        self._calculate_grid_layout()
        
        # Draw grid squares
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                
                # Draw main square
                if self.revealed_grid[y, x]:
                    if self.mine_grid[y, x]:
                        pygame.draw.rect(self.screen, self.COLOR_EXPLOSION, rect)
                    else:
                        pygame.draw.rect(self.screen, self.COLOR_REVEALED_SAFE, rect)
                        num = self.number_grid[y, x]
                        if num > 0:
                            self._draw_text(str(num), self.font_large, self.NUMBER_COLORS[num], rect.center)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, rect)
                    if self.flagged_grid[y, x]:
                        self._draw_flag(rect)

        # Draw flash effect for recently revealed squares
        flash_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        flash_surface.fill(self.COLOR_REVEAL_FLASH)
        for x, y in self.last_revealed_pos:
            pos = (self.grid_offset_x + x * self.cell_size, self.grid_offset_y + y * self.cell_size)
            self.screen.blit(flash_surface, pos)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y + self.grid_height)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos, 2)
            # Horizontal lines
            start_pos = (self.grid_offset_x, self.grid_offset_y + i * self.cell_size)
            end_pos = (self.grid_offset_x + self.grid_width, self.grid_offset_y + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos, 2)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.cell_size,
            self.grid_offset_y + self.cursor_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        cursor_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        cursor_surface.fill(self.COLOR_CURSOR)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _draw_text(self, text, font, color, center_pos):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=center_pos)
        self.screen.blit(text_surf, text_rect)

    def _draw_flag(self, rect):
        center_x, center_y = rect.center
        pole_x = center_x - self.cell_size // 6
        flag_points = [
            (pole_x, center_y - self.cell_size // 3),
            (pole_x + self.cell_size // 3, center_y - self.cell_size // 4),
            (pole_x, center_y - self.cell_size // 6)
        ]
        pygame.draw.line(self.screen, self.COLOR_FLAG, (pole_x, center_y - self.cell_size // 3), (pole_x, center_y + self.cell_size // 3), 3)
        pygame.gfxdraw.aapolygon(self.screen, flag_points, self.COLOR_FLAG)
        pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_FLAG)
        
    def _render_ui(self):
        # Score
        self._draw_text(f"SCORE: {int(self.score)}", self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - 80, 20))
        # Steps
        self._draw_text(f"STEPS: {self.steps}/{self.MAX_STEPS}", self.font_small, self.COLOR_TEXT, (80, 20))
        
        # Game Over / Win message
        if self.game_over:
            self._draw_text("GAME OVER", self.font_large, self.COLOR_EXPLOSION, (self.SCREEN_WIDTH // 2, 25))
        elif self.win:
            self._draw_text("YOU WIN!", self.font_large, (80, 255, 80), (self.SCREEN_WIDTH // 2, 25))

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
        assert np.sum(self.mine_grid) == self.NUM_MINES
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

        # Test reward conditions
        self.reset()
        found_mine = False
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.mine_grid[y,x]:
                    self.cursor_pos = [x, y]
                    _, reward, terminated, _, _ = self.step(np.array([0, 1, 0])) # Reveal mine
                    assert reward == -100
                    assert terminated == True
                    found_mine = True
                    break
            if found_mine: break
        
        print("✓ Implementation validated successfully")