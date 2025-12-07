
# Generated: 2025-08-28T05:30:05.661320
# Source Brief: brief_02643.md
# Brief Index: 2643

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to swap the selected "
        "gem with the one in the direction you last moved."
    )

    game_description = (
        "Swap adjacent gems to create lines of 3 or more. Achieve a 5-gem match to win! "
        "You only have 15 moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 8
        self.SQUARE_SIZE = 40
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.SQUARE_SIZE) // 2
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.SQUARE_SIZE) // 2
        self.NUM_COLORS = 6
        self.MAX_MOVES = 15
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_LINES = (40, 50, 80)
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLORS = [
            (220, 50, 50),    # Red
            (50, 150, 220),   # Blue
            (50, 220, 100),   # Green
            (230, 230, 80),   # Yellow
            (180, 80, 230),   # Purple
            (240, 140, 50),   # Orange
        ]
        self.COLOR_FLASH = (255, 255, 255, 150) # Semi-transparent white

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # --- Game State ---
        self.grid = None
        self.cursor_pos = None
        self.last_move_dir = None
        self.prev_space_held = None
        self.moves_left = None
        self.score = None
        self.game_over = None
        self.steps = None
        self.flashing_cells = None

        if render_mode == "rgb_array":
            self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        while True:
            self._init_grid()
            if not self._find_all_matches():
                break

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.last_move_dir = (0, 0)
        self.prev_space_held = False
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.flashing_cells = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.game_over = False
        self.flashing_cells = []
        
        self._move_cursor(movement)

        if space_held and not self.prev_space_held:
            swap_reward, win_condition_met = self._handle_swap()
            reward += swap_reward
            if win_condition_met:
                self.game_over = True
                reward += 50 # Goal-oriented reward for winning
        
        self.prev_space_held = space_held
        self.steps += 1
        
        terminated = self.game_over or self.moves_left <= 0 or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _init_grid(self):
        self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_WIDTH, self.GRID_HEIGHT))

    def _move_cursor(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            self.last_move_dir = (dx, dy)
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_WIDTH
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_HEIGHT

    def _handle_swap(self):
        dx, dy = self.last_move_dir
        if dx == 0 and dy == 0:
            return 0, False # No last move direction, so no swap

        x1, y1 = self.cursor_pos
        x2, y2 = x1 + dx, y1 + dy

        if not (0 <= x2 < self.GRID_WIDTH and 0 <= y2 < self.GRID_HEIGHT):
            return 0, False # Swap is out of bounds

        self.moves_left -= 1
        
        # Perform swap
        self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]

        total_reward = 0
        win_condition_met = False
        matches_found_this_swap = False

        # Cascade loop
        while True:
            matched_cells = self._find_all_matches()
            if not matched_cells:
                break
            
            matches_found_this_swap = True
            self.flashing_cells.extend(list(matched_cells))

            # Calculate reward and check for win condition
            match_reward, new_win = self._calculate_match_reward(matched_cells)
            total_reward += match_reward
            if new_win:
                win_condition_met = True

            self._apply_gravity_and_refill(matched_cells)

        if not matches_found_this_swap:
            # Revert swap if it resulted in no matches, but keep move cost
            # self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]
            # Brief implies non-matching swaps are permanent and penalized
            return -0.1, False

        return total_reward, win_condition_met
    
    def _find_all_matches(self):
        all_matched_cells = set()
        # Horizontal matches
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if self.grid[x, y] == self.grid[x + 1, y] == self.grid[x + 2, y]:
                    color = self.grid[x, y]
                    match_len = 3
                    while x + match_len < self.GRID_WIDTH and self.grid[x + match_len, y] == color:
                        match_len += 1
                    for i in range(match_len):
                        all_matched_cells.add((x + i, y))
        # Vertical matches
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if self.grid[x, y] == self.grid[x, y + 1] == self.grid[x, y + 2]:
                    color = self.grid[x, y]
                    match_len = 3
                    while y + match_len < self.GRID_HEIGHT and self.grid[x, y + match_len] == color:
                        match_len += 1
                    for i in range(match_len):
                        all_matched_cells.add((x, y + i))
        return all_matched_cells

    def _calculate_match_reward(self, matched_cells):
        reward = 0
        win = False
        
        # This is a simplification; it rewards per cell, which approximates rewarding per-match-length
        # A more complex group-finding algorithm would be needed for exact per-match rewards
        num_matched = len(matched_cells)
        if num_matched == 3: reward += 1
        elif num_matched == 4: reward += 5
        elif num_matched >= 5: 
            reward += 10
            win = True
        
        self.score += int(reward * 10) # Scale reward for score display
        return reward, win

    def _apply_gravity_and_refill(self, matched_cells):
        for x in range(self.GRID_WIDTH):
            column = [self.grid[x, y] for y in range(self.GRID_HEIGHT) if (x, y) not in matched_cells]
            num_removed = self.GRID_HEIGHT - len(column)
            if num_removed > 0:
                new_gems = self.np_random.integers(0, self.NUM_COLORS, size=num_removed).tolist()
                new_column = new_gems + column
                for y in range(self.GRID_HEIGHT):
                    self.grid[x, y] = new_column[y]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            start = (self.GRID_X_OFFSET + x * self.SQUARE_SIZE, self.GRID_Y_OFFSET)
            end = (self.GRID_X_OFFSET + x * self.SQUARE_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.SQUARE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start, end, 1)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + y * self.SQUARE_SIZE)
            end = (self.GRID_X_OFFSET + self.GRID_WIDTH * self.SQUARE_SIZE, self.GRID_Y_OFFSET + y * self.SQUARE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start, end, 1)

        # Draw gems
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                color_index = self.grid[x, y]
                gem_color = self.COLORS[color_index]
                
                rect = pygame.Rect(
                    self.GRID_X_OFFSET + x * self.SQUARE_SIZE,
                    self.GRID_Y_OFFSET + y * self.SQUARE_SIZE,
                    self.SQUARE_SIZE,
                    self.SQUARE_SIZE
                )
                
                # Draw gem with a 3D effect
                padding = 4
                inner_rect = rect.inflate(-padding*2, -padding*2)
                
                # Use gfxdraw for anti-aliased shapes
                pygame.gfxdraw.box(self.screen, inner_rect, gem_color)
                pygame.gfxdraw.aacircle(self.screen, inner_rect.centerx, inner_rect.centery, inner_rect.width // 2 - 2, gem_color)
                
                # Highlight
                highlight_color = tuple(min(255, c + 60) for c in gem_color)
                pygame.draw.line(self.screen, highlight_color, inner_rect.topleft, inner_rect.topright, 2)
                pygame.draw.line(self.screen, highlight_color, inner_rect.topleft, inner_rect.bottomleft, 2)
                
                # Shadow
                shadow_color = tuple(max(0, c - 60) for c in gem_color)
                pygame.draw.line(self.screen, shadow_color, inner_rect.bottomright, inner_rect.topright, 2)
                pygame.draw.line(self.screen, shadow_color, inner_rect.bottomright, inner_rect.bottomleft, 2)
        
        # Draw flash effect on matched cells
        if self.flashing_cells:
            flash_surface = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE), pygame.SRCALPHA)
            flash_surface.fill(self.COLOR_FLASH)
            for x, y in self.flashing_cells:
                pos = (self.GRID_X_OFFSET + x * self.SQUARE_SIZE, self.GRID_Y_OFFSET + y * self.SQUARE_SIZE)
                self.screen.blit(flash_surface, pos)

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        rect = pygame.Rect(
            self.GRID_X_OFFSET + cx * self.SQUARE_SIZE,
            self.GRID_Y_OFFSET + cy * self.SQUARE_SIZE,
            self.SQUARE_SIZE,
            self.SQUARE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 4)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WON!" if self.score > 0 else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_CURSOR)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Swap")
    
    terminated = False
    clock = pygame.time.Clock()
    
    # Game loop for human play
    while not terminated:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # In a human-playable loop, we only step when an action occurs
        # or on a timer to allow for continuous key presses to register.
        # Since auto_advance is False, we control the stepping.
        # We step once per frame to make it playable.
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Wait 3 seconds before closing

        clock.tick(15) # Limit frame rate for human play
        
    env.close()