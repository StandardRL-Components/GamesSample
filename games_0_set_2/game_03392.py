
# Generated: 2025-08-27T23:12:40.252504
# Source Brief: brief_03392.md
# Brief Index: 3392

        
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
        "Controls: Arrow keys to move the cursor. Space to reveal a tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist minesweeper. Reveal all safe tiles to win, but avoid the mines!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array", grid_size=(9, 9), num_mines=10):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_mines = num_mines
        self.width, self.height = 640, 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Fonts and Colors
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.COLOR_BG = (40, 42, 54)
        self.COLOR_TILE_HIDDEN = (68, 71, 90)
        self.COLOR_TILE_REVEALED = (98, 114, 164)
        self.COLOR_GRID_LINE = (50, 52, 64)
        self.COLOR_CURSOR = (80, 250, 123)
        self.COLOR_MINE = (255, 85, 85)
        self.COLOR_FLAG = (241, 250, 140)
        self.COLOR_TEXT_UI = (248, 248, 242)
        self.COLOR_TEXT_WIN = (80, 250, 123)
        self.COLOR_TEXT_LOSE = (255, 85, 85)
        
        self.NUMBER_COLORS = {
            1: (139, 233, 253), # Cyan
            2: (80, 250, 123),  # Green
            3: (255, 121, 198), # Pink
            4: (189, 147, 249), # Purple
            5: (255, 184, 108), # Orange
            6: (255, 85, 85),   # Red
            7: (241, 250, 140), # Yellow
            8: (200, 200, 200)  # Light Grey
        }

        # Game dimensions
        self.tile_size = 36
        self.grid_w = self.grid_size[0] * self.tile_size
        self.grid_h = self.grid_size[1] * self.tile_size
        self.grid_offset_x = (self.width - self.grid_w) // 2
        self.grid_offset_y = (self.height - self.grid_h) // 2
        
        # Initialize state variables
        self.mine_grid = None
        self.number_grid = None
        self.revealed_grid = None
        self.cursor_pos = None
        self.total_safe_tiles = 0
        self.revealed_safe_count = 0
        self.win = False
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_action_was_reveal = False

        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.last_action_was_reveal = False

        # Initialize grids
        rows, cols = self.grid_size
        self.revealed_grid = np.zeros(self.grid_size, dtype=bool)
        self.mine_grid = np.zeros(self.grid_size, dtype=int)
        
        # Place mines
        mine_indices = self.np_random.choice(rows * cols, self.num_mines, replace=False)
        self.mine_grid.flat[mine_indices] = 1
        
        # Calculate numbers
        self.number_grid = np.zeros(self.grid_size, dtype=int)
        for r in range(rows):
            for c in range(cols):
                if self.mine_grid[r, c] == 1:
                    continue
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and self.mine_grid[nr, nc] == 1:
                            count += 1
                self.number_grid[r, c] = count
        
        self.total_safe_tiles = (rows * cols) - self.num_mines
        self.revealed_safe_count = 0
        
        self.cursor_pos = [rows // 2, cols // 2]
        
        return self._get_observation(), self._get_info()

    def _flood_fill(self, r, c):
        """Reveals a tile and recursively reveals its neighbors if it's a '0'."""
        rows, cols = self.grid_size
        stack = [(r, c)]
        
        while stack:
            curr_r, curr_c = stack.pop()
            
            if not (0 <= curr_r < rows and 0 <= curr_c < cols):
                continue
            if self.revealed_grid[curr_r, curr_c]:
                continue
            
            self.revealed_grid[curr_r, curr_c] = True
            self.revealed_safe_count += 1
            
            if self.number_grid[curr_r, curr_c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        stack.append((curr_r + dr, curr_c + dc))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = 0.0
        terminated = False
        
        # Reset reveal flag
        self.last_action_was_reveal = False

        # --- Action Handling ---
        # Movement
        r, c = self.cursor_pos
        if movement == 1: # Up
            self.cursor_pos[0] = (r - 1 + self.grid_size[0]) % self.grid_size[0]
        elif movement == 2: # Down
            self.cursor_pos[0] = (r + 1) % self.grid_size[0]
        elif movement == 3: # Left
            self.cursor_pos[1] = (c - 1 + self.grid_size[1]) % self.grid_size[1]
        elif movement == 4: # Right
            self.cursor_pos[1] = (c + 1) % self.grid_size[1]

        # Reveal tile
        if space_held:
            r, c = self.cursor_pos
            self.last_action_was_reveal = True
            
            if not self.revealed_grid[r, c]:
                # Hit a mine
                if self.mine_grid[r, c] == 1:
                    # sfx: explosion
                    reward = -100.0
                    self.score += reward
                    terminated = True
                    self.game_over = True
                    self.win = False
                # Hit a safe tile
                else:
                    # sfx: click_reveal
                    # Base reward for revealing a safe tile
                    reward = 1.0

                    # Check for the adjacent-to-zero penalty
                    has_zero_neighbor = False
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.grid_size[0] and 0 <= nc < self.grid_size[1]:
                                if self.number_grid[nr, nc] == 0 and self.mine_grid[nr, nc] == 0:
                                    has_zero_neighbor = True
                                    break
                        if has_zero_neighbor: break
                    
                    if has_zero_neighbor:
                        reward -= 0.2

                    # Apply flood fill if it's a zero tile
                    if self.number_grid[r, c] == 0:
                        self._flood_fill(r, c)
                    else:
                        self.revealed_grid[r, c] = True
                        self.revealed_safe_count += 1
                    
                    self.score += reward

                    # Check for win condition
                    if self.revealed_safe_count == self.total_safe_tiles:
                        # sfx: win_jingle
                        win_bonus = 100.0
                        reward += win_bonus
                        self.score += win_bonus
                        terminated = True
                        self.game_over = True
                        self.win = True
            else:
                # Penalty for clicking an already revealed tile
                reward = -0.1
                self.score += reward

        self.steps += 1
        if self.steps >= 1000 and not terminated:
            terminated = True
            self.game_over = True
            self.score -= 50 # Penalty for running out of time
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _render_game(self):
        rows, cols = self.grid_size
        
        # Draw tiles
        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(
                    self.grid_offset_x + c * self.tile_size,
                    self.grid_offset_y + r * self.tile_size,
                    self.tile_size, self.tile_size
                )
                
                if self.revealed_grid[r, c] or (self.game_over and self.mine_grid[r,c] == 1):
                    # Revealed tiles
                    if self.mine_grid[r, c] == 1:
                        pygame.draw.rect(self.screen, self.COLOR_MINE, rect)
                        # Draw mine icon
                        cx, cy = rect.center
                        pygame.gfxdraw.filled_circle(self.screen, cx, cy, self.tile_size // 4, (30,30,30))
                        pygame.gfxdraw.aacircle(self.screen, cx, cy, self.tile_size // 4, (30,30,30))
                    else:
                        pygame.draw.rect(self.screen, self.COLOR_TILE_REVEALED, rect)
                        num = self.number_grid[r, c]
                        if num > 0:
                            num_color = self.NUMBER_COLORS.get(num, self.COLOR_TEXT_UI)
                            text_surf = self.font_main.render(str(num), True, num_color)
                            text_rect = text_surf.get_rect(center=rect.center)
                            self.screen.blit(text_surf, text_rect)
                else:
                    # Hidden tiles
                    pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, rect)

        # Draw grid lines
        for i in range(rows + 1):
            y = self.grid_offset_y + i * self.tile_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.grid_offset_x, y), (self.grid_offset_x + self.grid_w, y), 2)
        for i in range(cols + 1):
            x = self.grid_offset_x + i * self.tile_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x, self.grid_offset_y), (x, self.grid_offset_y + self.grid_h), 2)

        # Draw cursor
        cur_r, cur_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_offset_x + cur_c * self.tile_size,
            self.grid_offset_y + cur_r * self.tile_size,
            self.tile_size, self.tile_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

        # Explosion effect on last reveal if it was a mine
        if self.game_over and not self.win and self.last_action_was_reveal:
            r, c = self.cursor_pos
            if self.mine_grid[r, c] == 1:
                cx = self.grid_offset_x + c * self.tile_size + self.tile_size // 2
                cy = self.grid_offset_y + r * self.tile_size + self.tile_size // 2
                pygame.gfxdraw.filled_circle(self.screen, cx, cy, self.tile_size, self.COLOR_MINE)


    def _render_ui(self):
        # Tiles left
        safe_left = self.total_safe_tiles - self.revealed_safe_count
        ui_text = f"Safe Tiles: {safe_left}"
        text_surf = self.font_main.render(ui_text, True, self.COLOR_TEXT_UI)
        self.screen.blit(text_surf, (10, 10))
        
        # Score
        score_text = f"Score: {self.score:.1f}"
        text_surf = self.font_main.render(score_text, True, self.COLOR_TEXT_UI)
        text_rect = text_surf.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(text_surf, text_rect)

        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_TEXT_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_TEXT_LOSE
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.width // 2, self.height // 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((40, 42, 54, 200))
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(text_surf, text_rect)

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
            "safe_tiles_remaining": self.total_safe_tiles - self.revealed_safe_count,
        }

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
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        assert self.num_mines == np.sum(self.mine_grid)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv()
    
    # --- Manual Play ---
    # To play manually, you need a window.
    # The environment is headless by default.
    # The following code sets up a window for human play.
    
    try:
        screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("Minesweeper Gym Env")
        
        obs, info = env.reset()
        done = False
        
        print(env.user_guide)
        
        while not done:
            # Map pygame keys to the MultiDiscrete action space
            movement = 0 # No-op
            space = 0
            shift = 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift = 1

            action = [movement, space, shift]
            
            # Since auto_advance is False, we only step on an action
            # For human play, we want to step every frame to see cursor movement
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # For human play, we need a delay to make it playable
            pygame.time.wait(50) # ~20 FPS for responsive controls

    finally:
        env.close()