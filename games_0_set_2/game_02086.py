
# Generated: 2025-08-27T19:13:20.576778
# Source Brief: brief_02086.md
# Brief Index: 2086

        
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
        "Controls: Arrow keys to move cursor. Space to reveal a tile. Shift to flag a tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic puzzle game. Reveal all safe tiles on the grid while avoiding the hidden mines."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_SIZE = 10
    NUM_MINES = 10
    MAX_STEPS = 1000
    
    # Colors
    COLOR_BG = (30, 30, 40)
    COLOR_GRID = (50, 50, 60)
    COLOR_TILE_COVERED = (140, 140, 160)
    COLOR_TILE_REVEALED = (100, 100, 110)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_FLAG = (255, 120, 0)
    COLOR_MINE = (220, 40, 40)
    COLOR_TEXT_UI = (220, 220, 220)
    COLOR_WIN = (80, 220, 80)
    COLOR_LOSE = (220, 80, 80)
    
    NUMBER_COLORS = {
        1: (80, 80, 255),
        2: (80, 180, 80),
        3: (255, 80, 80),
        4: (80, 80, 180),
        5: (180, 80, 80),
        6: (80, 180, 180),
        7: (40, 40, 40),
        8: (120, 120, 120)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_tile = pygame.font.SysFont("dejavusansmono", 22, bold=True)
        self.font_ui = pygame.font.SysFont("dejavusansmono", 18)
        self.font_msg = pygame.font.SysFont("dejavusansmono", 50, bold=True)
        
        # Initialize state variables
        self.mine_grid = None
        self.revealed_grid = None
        self.flagged_grid = None
        self.number_grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.total_safe_tiles = 0
        self.revealed_safe_tiles = 0
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.cursor_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        
        self.mine_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.revealed_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.flagged_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.number_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)

        self._place_mines()
        self._calculate_numbers()
        
        self.total_safe_tiles = self.GRID_SIZE * self.GRID_SIZE - self.NUM_MINES
        self.revealed_safe_tiles = 0
        
        return self._get_observation(), self._get_info()

    def _place_mines(self):
        coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(coords)
        for i in range(self.NUM_MINES):
            x, y = coords[i]
            self.mine_grid[y, x] = True

    def _calculate_numbers(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.mine_grid[y, x]:
                    self.number_grid[y, x] = -1  # -1 for mine
                    continue
                
                count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                            if self.mine_grid[ny, nx]:
                                count += 1
                self.number_grid[y, x] = count

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False

        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)
        
        cx, cy = self.cursor_pos[0], self.cursor_pos[1]

        # 2. Handle flagging (Shift)
        if shift_held and not self.revealed_grid[cy, cx]:
            self.flagged_grid[cy, cx] = not self.flagged_grid[cy, cx]
            # sfx: flag_toggle.wav

        # 3. Handle revealing (Space)
        elif space_held and not self.revealed_grid[cy, cx] and not self.flagged_grid[cy, cx]:
            self.revealed_grid[cy, cx] = True
            
            if self.mine_grid[cy, cx]:
                # sfx: explosion.wav
                reward = -100
                self.game_over = True
                self.win = False
                terminated = True
            else:
                # sfx: reveal.wav
                tiles_revealed = 1
                if self.number_grid[cy, cx] == 0:
                    tiles_revealed += self._flood_fill(cx, cy)
                
                reward = tiles_revealed
                self.revealed_safe_tiles = np.sum(self.revealed_grid & ~self.mine_grid)
                
                if self.revealed_safe_tiles == self.total_safe_tiles:
                    # sfx: win.wav
                    reward += 100
                    self.game_over = True
                    self.win = True
                    terminated = True

        self.steps += 1
        self.score += reward
        
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _flood_fill(self, x, y):
        stack = []
        # Initial reveal was a zero, so we check its neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and not self.revealed_grid[ny, nx] and not self.flagged_grid[ny, nx]:
                    stack.append((nx, ny))
        
        extra_revealed = 0
        while stack:
            cx, cy = stack.pop()
            if self.revealed_grid[cy, cx]: continue

            self.revealed_grid[cy, cx] = True
            extra_revealed += 1
            
            if self.number_grid[cy, cx] == 0:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and not self.revealed_grid[ny, nx] and not self.flagged_grid[ny, nx]:
                            stack.append((nx, ny))
        return extra_revealed

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        margin = 20
        grid_area_size = min(self.screen.get_width() - 2 * margin, self.screen.get_height() - 2 * margin)
        cell_size = grid_area_size // self.GRID_SIZE
        
        start_x = (self.screen.get_width() - cell_size * self.GRID_SIZE) // 2
        start_y = (self.screen.get_height() - cell_size * self.GRID_SIZE) // 2

        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(start_x + x * cell_size, start_y + y * cell_size, cell_size, cell_size)
                
                if self.revealed_grid[y, x] or (self.game_over and self.mine_grid[y, x]):
                    pygame.draw.rect(self.screen, self.COLOR_TILE_REVEALED, rect)
                    
                    if self.mine_grid[y, x]:
                        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, int(cell_size * 0.3), self.COLOR_MINE)
                        pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, int(cell_size * 0.3), self.COLOR_MINE)
                    else:
                        num = self.number_grid[y, x]
                        if num > 0:
                            color = self.NUMBER_COLORS.get(num, self.COLOR_TEXT_UI)
                            text_surf = self.font_tile.render(str(num), True, color)
                            text_rect = text_surf.get_rect(center=rect.center)
                            self.screen.blit(text_surf, text_rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_TILE_COVERED, rect)
                    if self.flagged_grid[y, x]:
                        # Draw a flag
                        p1 = (rect.centerx, rect.top + int(cell_size * 0.2))
                        p2 = (rect.left + int(cell_size * 0.2), rect.top + int(cell_size * 0.4))
                        p3 = (rect.centerx, rect.top + int(cell_size * 0.6))
                        pygame.draw.polygon(self.screen, self.COLOR_FLAG, [p1, p2, p3])
                        pygame.draw.line(self.screen, self.COLOR_FLAG, p1, (rect.centerx, rect.bottom - int(cell_size*0.2)), 2)

                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw cursor
        cursor_rect = pygame.Rect(
            start_x + self.cursor_pos[0] * cell_size,
            start_y + self.cursor_pos[1] * cell_size,
            cell_size,
            cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        score_text = f"Score: {self.score}"
        steps_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        mines_left = self.NUM_MINES - np.sum(self.flagged_grid)
        mines_text = f"Mines: {mines_left}"

        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT_UI)
        steps_surf = self.font_ui.render(steps_text, True, self.COLOR_TEXT_UI)
        mines_surf = self.font_ui.render(mines_text, True, self.COLOR_TEXT_UI)
        
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(steps_surf, (10, 30))
        self.screen.blit(mines_surf, (self.screen.get_width() - mines_surf.get_width() - 10, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_WIN if self.win else self.COLOR_LOSE
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
            
            # Add a semi-transparent background for the message
            bg_rect = msg_rect.inflate(40, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos.tolist(),
            "mines_left": self.NUM_MINES - np.sum(self.flagged_grid),
            "safe_tiles_revealed": self.revealed_safe_tiles,
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
        
        # Game-specific assertions
        self.reset()
        assert np.sum(self.mine_grid) == self.NUM_MINES
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up the display window
    pygame.display.set_caption("Minesweeper Gym Env")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()

    action = np.array([0, 0, 0]) # No-op, release, release
    
    while not done:
        # --- Human Controls ---
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                
        action = np.array([movement, space, shift])

        # --- Step the environment ---
        # Only step if an action was taken, since auto_advance is False
        if np.any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward}, Score: {info['score']}, Done: {done}")
        
        # --- Render the game ---
        # The observation is the rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit frame rate

    env.close()
    print("Game Over!")