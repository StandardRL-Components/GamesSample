
# Generated: 2025-08-28T05:30:36.974894
# Source Brief: brief_05590.md
# Brief Index: 5590

        
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
        "Controls: Use arrow keys to slide all blocks in a direction. "
        "Match 3 or more of the same color to clear them."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Slide blocks to create matches and clear the board before time runs out. "
        "Create combos for a higher score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    NUM_COLORS = 5
    MAX_STEPS = 2000

    COLOR_BG = (25, 35, 45)
    COLOR_GRID_LINES = (50, 60, 70)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_FLASH = (255, 255, 255)

    BLOCK_COLORS = [
        (0, 0, 0),  # 0: Empty
        (255, 80, 80),   # 1: Red
        (80, 255, 80),   # 2: Green
        (80, 120, 255),  # 3: Blue
        (255, 240, 80),  # 4: Yellow
        (200, 80, 255),  # 5: Purple
    ]
    BLOCK_SHADOW_COLORS = [
        (0, 0, 0),
        (180, 50, 50),
        (50, 180, 50),
        (50, 80, 180),
        (180, 170, 50),
        (140, 50, 180),
    ]

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
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Calculate grid rendering properties
        self.grid_render_size = 360
        self.block_size = self.grid_render_size // self.GRID_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_render_size) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_render_size) // 2

        self.grid = None
        self.matched_blocks = set()
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        
        # Ensure no initial matches to give the player a clean start
        while True:
            matches = self._find_matches()
            if not matches:
                break
            for r, c in matches:
                self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)
        
        self.matched_blocks = set()
        self.steps = 0
        self.score = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Phase 1: Clear previous matches and apply gravity ---
        if self.matched_blocks:
            for r, c in self.matched_blocks:
                self.grid[r, c] = 0
            # sfx: block clear sound
            self._apply_gravity_and_refill()
            self.matched_blocks.clear()

        movement = action[0]
        reward = 0
        
        moved = False
        if movement in [1, 2, 3, 4]: # up, down, left, right
            moved = self._slide_blocks(direction=movement)
            # sfx: whoosh sound
        
        # --- Phase 2: Find new matches ---
        new_matches = self._find_matches()
        if new_matches:
            self.matched_blocks = new_matches
            # sfx: match found chime

        # --- Phase 3: Calculate reward ---
        cleared_count = len(self.matched_blocks)
        if cleared_count > 0:
            reward += cleared_count
            if cleared_count > 5:
                reward += 5 # Bonus for large combo
        elif moved: # Moved but no matches
            reward -= 0.2
        
        self.score += cleared_count
        self.steps += 1
        
        # --- Phase 4: Check for termination ---
        terminated = False
        if self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 10 # Penalty for running out of time
            # sfx: game over sad sound
        
        board_empty = not np.any(self.grid)
        if board_empty:
            terminated = True
            reward += 100 # Huge bonus for clearing the board
            # sfx: victory fanfare
            
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _slide_blocks(self, direction):
        # Create a copy to compare against later to see if anything moved
        original_grid = self.grid.copy()

        if direction == 1: # Up
            for c in range(self.GRID_SIZE):
                col = self.grid[:, c]
                non_empty = col[col != 0]
                new_col = np.zeros(self.GRID_SIZE, dtype=int)
                if len(non_empty) > 0:
                    new_col[:len(non_empty)] = non_empty
                self.grid[:, c] = new_col
        elif direction == 2: # Down
            for c in range(self.GRID_SIZE):
                col = self.grid[:, c]
                non_empty = col[col != 0]
                new_col = np.zeros(self.GRID_SIZE, dtype=int)
                if len(non_empty) > 0:
                    new_col[self.GRID_SIZE - len(non_empty):] = non_empty
                self.grid[:, c] = new_col
        elif direction == 3: # Left
            for r in range(self.GRID_SIZE):
                row = self.grid[r, :]
                non_empty = row[row != 0]
                new_row = np.zeros(self.GRID_SIZE, dtype=int)
                if len(non_empty) > 0:
                    new_row[:len(non_empty)] = non_empty
                self.grid[r, :] = new_row
        elif direction == 4: # Right
            for r in range(self.GRID_SIZE):
                row = self.grid[r, :]
                non_empty = row[row != 0]
                new_row = np.zeros(self.GRID_SIZE, dtype=int)
                if len(non_empty) > 0:
                    new_row[self.GRID_SIZE - len(non_empty):] = non_empty
                self.grid[r, :] = new_row
        
        moved = not np.array_equal(original_grid, self.grid)
        return moved
        
    def _find_matches(self):
        matches = set()
        # Horizontal matches
        for r in range(self.GRID_SIZE):
            c = 0
            while c < self.GRID_SIZE - 2:
                color = self.grid[r, c]
                if color == 0:
                    c += 1
                    continue
                if color == self.grid[r, c+1] and color == self.grid[r, c+2]:
                    match_len = 3
                    while c + match_len < self.GRID_SIZE and self.grid[r, c + match_len] == color:
                        match_len += 1
                    for i in range(match_len):
                        matches.add((r, c + i))
                    c += match_len
                else:
                    c += 1
        
        # Vertical matches
        for c in range(self.GRID_SIZE):
            r = 0
            while r < self.GRID_SIZE - 2:
                color = self.grid[r, c]
                if color == 0:
                    r += 1
                    continue
                if color == self.grid[r+1, c] and color == self.grid[r+2, c]:
                    match_len = 3
                    while r + match_len < self.GRID_SIZE and self.grid[r + match_len, c] == color:
                        match_len += 1
                    for i in range(match_len):
                        matches.add((r + i, c))
                    r += match_len
                else:
                    r += 1
        return matches

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_SIZE):
            col = self.grid[:, c]
            non_empty = col[col != 0]
            num_empty = self.GRID_SIZE - len(non_empty)
            new_col = np.zeros(self.GRID_SIZE, dtype=int)
            
            if len(non_empty) > 0:
                new_col[num_empty:] = non_empty
            
            if num_empty > 0:
                new_blocks = self.np_random.integers(1, self.NUM_COLORS + 1, size=num_empty)
                new_col[:num_empty] = new_blocks
                
            self.grid[:, c] = new_col

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines for background
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, 
                (self.grid_offset_x + i * self.block_size, self.grid_offset_y),
                (self.grid_offset_x + i * self.block_size, self.grid_offset_y + self.grid_render_size), 1)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES,
                (self.grid_offset_x, self.grid_offset_y + i * self.block_size),
                (self.grid_offset_x + self.grid_render_size, self.grid_offset_y + i * self.block_size), 1)
        
        # Draw blocks
        bevel = self.block_size // 8
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = self.grid[r, c]
                if color_index == 0:
                    continue

                is_matched = (r, c) in self.matched_blocks
                x = self.grid_offset_x + c * self.block_size
                y = self.grid_offset_y + r * self.block_size
                rect = pygame.Rect(x, y, self.block_size, self.block_size)

                if is_matched:
                    # Flash effect for matched blocks
                    pygame.draw.rect(self.screen, self.COLOR_FLASH, rect, border_radius=bevel)
                    inner_rect = rect.inflate(-self.block_size * 0.5, -self.block_size * 0.5)
                    pygame.draw.rect(self.screen, self.BLOCK_COLORS[color_index], inner_rect, border_radius=bevel//2)
                else:
                    # Normal block with 3D bevel effect
                    shadow_color = self.BLOCK_SHADOW_COLORS[color_index]
                    main_color = self.BLOCK_COLORS[color_index]
                    pygame.draw.rect(self.screen, shadow_color, rect, border_radius=bevel)
                    inner_rect = rect.inflate(-bevel, -bevel)
                    pygame.draw.rect(self.screen, main_color, inner_rect, border_radius=bevel)

    def _render_ui(self):
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            main = font.render(text, True, color)
            self.screen.blit(main, pos)
            
        score_text = f"SCORE: {self.score}"
        draw_text(score_text, self.font_small, self.COLOR_TEXT, (20, 20))
        
        time_left = self.MAX_STEPS - self.steps
        timer_text = f"STEPS: {time_left}"
        text_width = self.font_small.size(timer_text)[0]
        draw_text(timer_text, self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - text_width - 20, 20))

        if self.game_over:
            board_empty = not np.any(self.grid)
            msg = "BOARD CLEARED!" if board_empty else "TIME'S UP!"
            msg_width, msg_height = self.font_large.size(msg)
            draw_text(msg, self.font_large, self.COLOR_FLASH, 
                      ((self.SCREEN_WIDTH - msg_width) // 2, (self.SCREEN_HEIGHT - msg_height) // 2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

# This block allows for human play and testing
if __name__ == '__main__':
    import sys
    
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    running = True
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Slider")
    
    print("--- Human Controls ---")
    print(GameEnv.user_guide)
    print("Press 'R' to Reset, ESC to Quit.")
    
    while running:
        move_action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    move_action = 1
                elif event.key == pygame.K_DOWN:
                    move_action = 2
                elif event.key == pygame.K_LEFT:
                    move_action = 3
                elif event.key == pygame.K_RIGHT:
                    move_action = 4
                elif event.key == pygame.K_r:
                    env.reset()

        if move_action != 0:
            action = [move_action, 0, 0]
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")
            if terminated:
                print("Game Over! Press 'R' to reset.")

        obs = env._get_observation()
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)
        
    env.close()
    sys.exit()