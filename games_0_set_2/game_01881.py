
# Generated: 2025-08-28T03:00:17.618384
# Source Brief: brief_01881.md
# Brief Index: 1881

        
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
        "Controls: Use arrow keys to move the selector. Press space to swap the selected "
        "block with the one in the direction of your last move."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic match-3 puzzle game. Swap adjacent blocks to create lines of three or "
        "more of the same color. Create combos and chain reactions to maximize your score. "
        "The game ends when you reach 500 points or run out of valid moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_COLORS = 5
    BLOCK_SIZE = 40
    GRID_LINE_WIDTH = 2
    
    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_TEXT = (220, 220, 240)
    BLOCK_COLORS = [
        (220, 50, 50),   # Red
        (50, 220, 50),   # Green
        (50, 100, 220),  # Blue
        (220, 220, 50),  # Yellow
        (160, 50, 220),  # Purple
    ]
    COLOR_CURSOR = (255, 255, 255)
    COLOR_FLASH = (255, 255, 255)

    # --- Game settings ---
    TARGET_SCORE = 500
    MAX_STEPS = 1000
    REWARD_MATCH_PER_BLOCK = 1
    REWARD_BONUS_4 = 5
    REWARD_BONUS_5_PLUS = 10
    REWARD_WIN = 100
    REWARD_LOSS_NO_MOVES = -10
    PENALTY_INVALID_SWAP = -0.1

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Etc...
        self.grid_render_pos = (
            (self.SCREEN_WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2,
            (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2
        )
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.last_move_dir = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.blocks_to_flash = set()
        self.animation_state = "IDLE" # Can be IDLE, FLASHING
        
        # Initialize state variables for first call
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.last_move_dir = [1, 0] # Default to right
        self.animation_state = "IDLE"
        self.blocks_to_flash.clear()

        # Generate a valid starting board
        while True:
            self._fill_grid()
            # Ensure no matches on start by repeatedly clearing and refilling
            while self._find_and_clear_matches(commit_and_score=False)[0] > 0:
                 self._fill_grid()
            
            if self._has_valid_moves():
                break
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # Handle flashing animation from previous step
        if self.animation_state == "FLASHING":
            self._resolve_cleared_blocks()
            
            # Check for chain reactions
            while True:
                chain_reward, cleared_blocks = self._find_and_clear_matches()
                if not cleared_blocks:
                    break
                reward += chain_reward
                self.blocks_to_flash.update(cleared_blocks)
                self._resolve_cleared_blocks()

            self.animation_state = "IDLE"
            self.blocks_to_flash.clear()
        
        # Unpack and process new action
        movement, space_held, _ = action
        
        # 1. Handle cursor movement
        self._move_cursor(movement)

        # 2. Handle swap action
        if space_held:
            swap_reward, did_swap = self._attempt_swap()
            reward += swap_reward
            if did_swap:
                match_reward, cleared_blocks = self._find_and_clear_matches()
                reward += match_reward
                if cleared_blocks:
                    self.blocks_to_flash = cleared_blocks
                    self.animation_state = "FLASHING"

        # 3. Check for termination
        terminated = False
        if not self.game_over:
            if self.score >= self.TARGET_SCORE:
                reward += self.REWARD_WIN
                terminated = True
            elif self.steps >= self.MAX_STEPS:
                terminated = True
            elif self.animation_state == "IDLE" and not self._has_valid_moves():
                reward += self.REWARD_LOSS_NO_MOVES
                terminated = True
            
        self.game_over = terminated
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _fill_grid(self):
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))

    def _move_cursor(self, movement):
        # movement: 0=none, 1=up, 2=down, 3=left, 4=right
        dx, dy = 0, 0
        if movement == 1: dy = -1
        elif movement == 2: dy = 1
        elif movement == 3: dx = -1
        elif movement == 4: dx = 1
        
        if dx != 0 or dy != 0:
            self.last_move_dir = [dx, dy]

        self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_WIDTH
        self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_HEIGHT

    def _attempt_swap(self):
        cx, cy = self.cursor_pos
        dx, dy = self.last_move_dir
        nx, ny = cx + dx, cy + dy

        if not (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT):
            return self.PENALTY_INVALID_SWAP, False

        self.grid[cy, cx], self.grid[ny, nx] = self.grid[ny, nx], self.grid[cy, cx]
        
        matches = self._find_matches_on_grid(self.grid)

        if not matches:
            self.grid[cy, cx], self.grid[ny, nx] = self.grid[ny, nx], self.grid[cy, cx]
            return self.PENALTY_INVALID_SWAP, False
        
        return 0, True

    def _find_matches_on_grid(self, grid):
        matches = set()
        # Check horizontal
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if grid[r, c] != 0 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    for i in range(3): matches.add((r, c+i))
        
        # Check vertical
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if grid[r, c] != 0 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    for i in range(3): matches.add((r+i, c))
        return matches

    def _find_and_clear_matches(self, commit_and_score=True):
        all_matched_coords = self._find_matches_on_grid(self.grid)
        if not all_matched_coords:
            return 0, set()

        reward = 0
        processed_coords = set()
        
        def calculate_line_reward(line_len):
            r = line_len * self.REWARD_MATCH_PER_BLOCK
            if line_len == 4: r += self.REWARD_BONUS_4
            if line_len >= 5: r += self.REWARD_BONUS_5_PLUS
            return r

        # Horizontal lines
        for r in range(self.GRID_HEIGHT):
            c = 0
            while c < self.GRID_WIDTH:
                if (r, c) in all_matched_coords and (r,c) not in processed_coords:
                    color = self.grid[r, c]
                    line_len = 0
                    while c + line_len < self.GRID_WIDTH and self.grid[r, c + line_len] == color:
                        line_len += 1
                    
                    if line_len >= 3:
                        reward += calculate_line_reward(line_len)
                        for i in range(line_len): processed_coords.add((r, c + i))
                    c += line_len
                else:
                    c += 1

        # Vertical lines
        for c in range(self.GRID_WIDTH):
            r = 0
            while r < self.GRID_HEIGHT:
                if (r, c) in all_matched_coords and (r,c) not in processed_coords:
                    color = self.grid[r, c]
                    line_len = 0
                    while r + line_len < self.GRID_HEIGHT and self.grid[r + line_len, c] == color:
                        line_len += 1
                    
                    if line_len >= 3:
                        reward += calculate_line_reward(line_len)
                        for i in range(line_len): processed_coords.add((r + i, c))
                    r += line_len
                else:
                    r += 1
        
        if commit_and_score:
            self.score += reward
        return reward, all_matched_coords

    def _resolve_cleared_blocks(self):
        if not self.blocks_to_flash:
            return
            
        for r, c in self.blocks_to_flash:
            self.grid[r, c] = 0
        
        # Apply gravity
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if empty_row != r:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1
        
        # Refill top
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _has_valid_moves(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Swap right
                if c < self.GRID_WIDTH - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_matches_on_grid(self.grid):
                        self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                
                # Swap down
                if r < self.GRID_HEIGHT - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_matches_on_grid(self.grid):
                        self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_x, grid_y = self.grid_render_pos
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (grid_x, grid_y, self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE))
        
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_idx = self.grid[r, c]
                if color_idx > 0:
                    block_rect = pygame.Rect(
                        grid_x + c * self.BLOCK_SIZE,
                        grid_y + r * self.BLOCK_SIZE,
                        self.BLOCK_SIZE,
                        self.BLOCK_SIZE
                    )
                    
                    if self.animation_state == "FLASHING" and (r, c) in self.blocks_to_flash:
                        color = self.COLOR_FLASH
                    else:
                        color = self.BLOCK_COLORS[color_idx - 1]
                    
                    pygame.draw.rect(self.screen, color, block_rect.inflate(-self.GRID_LINE_WIDTH, -self.GRID_LINE_WIDTH), border_radius=4)
        
        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            grid_x + cx * self.BLOCK_SIZE,
            grid_y + cy * self.BLOCK_SIZE,
            self.BLOCK_SIZE,
            self.BLOCK_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=6)
        
        # Draw swap target indicator
        dx, dy = self.last_move_dir
        nx, ny = cx + dx, cy + dy
        if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
            target_rect = pygame.Rect(
                grid_x + nx * self.BLOCK_SIZE + 3,
                grid_y + ny * self.BLOCK_SIZE + 3,
                self.BLOCK_SIZE - 6,
                self.BLOCK_SIZE - 6
            )
            # Draw a dashed rectangle
            step = 6
            for i in range(0, self.BLOCK_SIZE - 6, step * 2):
                pygame.draw.line(self.screen, self.COLOR_CURSOR, (target_rect.left + i, target_rect.top), (target_rect.left + i + step, target_rect.top))
                pygame.draw.line(self.screen, self.COLOR_CURSOR, (target_rect.left + i, target_rect.bottom-1), (target_rect.left + i + step, target_rect.bottom-1))
                pygame.draw.line(self.screen, self.COLOR_CURSOR, (target_rect.left, target_rect.top + i), (target_rect.left, target_rect.top + i + step))
                pygame.draw.line(self.screen, self.COLOR_CURSOR, (target_rect.right-1, target_rect.top + i), (target_rect.right-1, target_rect.top + i + step))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Target Score
        target_text = self.font_small.render(f"TARGET: {self.TARGET_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(target_text, (20, 40))
        
        # Steps
        steps_text = self.font_large.render(f"MOVES: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 20, 10))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.score >= self.TARGET_SCORE else "GAME OVER"
                
            end_text = self.font_large.render(end_text_str, True, self.COLOR_CURSOR)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.font.quit()
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
        # Can't get obs before reset, so we reset first
        print("Testing reset()...")
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space again after reset
        print("Testing _get_observation()...")
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        print("Testing step()...")
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")