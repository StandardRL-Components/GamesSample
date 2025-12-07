
# Generated: 2025-08-28T07:14:35.727653
# Source Brief: brief_03177.md
# Brief Index: 3177

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Tile:
    """Helper class to store the state of a single grid tile."""
    def __init__(self):
        self.is_mine = False
        self.is_revealed = False
        self.adjacent_mines = 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Press Space to reveal a tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist version of Minesweeper. Navigate the grid and reveal all safe tiles "
        "while avoiding the hidden mines. Numbers on revealed tiles indicate how many "
        "mines are adjacent."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 8, 6
        self.NUM_MINES = 10
        self.MAX_STEPS = 1000
        self.NUM_SAFE_TILES = self.GRID_W * self.GRID_H - self.NUM_MINES

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_tile = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_game_over = pygame.font.SysFont("Verdana", 60, bold=True)

        # Colors
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_GRID = (80, 80, 90)
        self.COLOR_UNREVEALED = (120, 120, 140)
        self.COLOR_REVEALED = (90, 90, 105)
        self.COLOR_CURSOR = (255, 220, 0)
        self.COLOR_MINE_BG = (200, 60, 60)
        self.COLOR_MINE_CIRCLE = (20, 20, 20)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_WIN = (80, 220, 120)
        self.COLOR_LOSS = (220, 80, 80)
        
        self.TILE_COLORS = {
            1: (50, 120, 220),
            2: (50, 180, 100),
            3: (220, 80, 80),
            4: (50, 50, 180),
            5: (180, 50, 50),
            6: (50, 180, 180),
            7: (150, 50, 150),
            8: (100, 100, 100),
        }

        # Game state variables (initialized in reset)
        self.grid = []
        self.cursor_pos = [0, 0]
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.revealed_count = 0
        self.last_clicked_mine_pos = None

        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.revealed_count = 0
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.last_clicked_mine_pos = None

        self._generate_grid()
        
        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        """Creates a new grid, places mines, and calculates adjacent numbers."""
        self.grid = [[Tile() for _ in range(self.GRID_W)] for _ in range(self.GRID_H)]
        
        mine_positions = []
        while len(mine_positions) < self.NUM_MINES:
            pos = (self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H))
            if pos not in mine_positions:
                mine_positions.append(pos)
                self.grid[pos[1]][pos[0]].is_mine = True
        
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if not self.grid[y][x].is_mine:
                    count = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0: continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and self.grid[ny][nx].is_mine:
                                count += 1
                    self.grid[y][x].adjacent_mines = count

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]
        space_pressed = action[1] == 1
        
        reward = 0.0
        
        if movement == 1:  # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_H) % self.GRID_H
        elif movement == 2:  # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_H
        elif movement == 3:  # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_W) % self.GRID_W
        elif movement == 4:  # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_W
            
        if space_pressed:
            x, y = self.cursor_pos
            tile = self.grid[y][x]
            
            if tile.is_revealed:
                reward = -0.1
            else:
                if tile.is_mine:
                    reward = -100.0
                    self.game_over = True
                    self.win = False
                    tile.is_revealed = True
                    self.last_clicked_mine_pos = (x, y)
                    # sound: explosion
                else:
                    reward = self._cascade_reveal(x, y)
                    if self.revealed_count == self.NUM_SAFE_TILES:
                        reward += 100.0
                        self.game_over = True
                        self.win = True
                        # sound: victory_jingle

        self.steps += 1
        self.score += reward
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _cascade_reveal(self, start_x, start_y):
        """Recursively reveals tiles starting from a point, returns total reward."""
        q = deque([(start_x, start_y)])
        visited = set()
        total_reward = 0.0

        while q:
            x, y = q.popleft()

            if not (0 <= x < self.GRID_W and 0 <= y < self.GRID_H): continue
            if (x, y) in visited: continue
            
            visited.add((x, y))
            tile = self.grid[y][x]

            if tile.is_revealed: continue

            tile.is_revealed = True
            self.revealed_count += 1
            # sound: tile_reveal_pop

            if tile.adjacent_mines == 0:
                total_reward += -0.2
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        q.append((x + dx, y + dy))
            else:
                total_reward += 1.0
        
        return total_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        tile_w = (self.WIDTH - 100) / self.GRID_W
        tile_h = (self.HEIGHT - 100) / self.GRID_H
        grid_px_w = tile_w * self.GRID_W
        grid_px_h = tile_h * self.GRID_H
        offset_x = (self.WIDTH - grid_px_w) / 2
        offset_y = (self.HEIGHT - grid_px_h) / 2

        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                tile = self.grid[y][x]
                rect = pygame.Rect(offset_x + x * tile_w, offset_y + y * tile_h, tile_w, tile_h)
                
                if tile.is_revealed:
                    if tile.is_mine:
                        pygame.draw.rect(self.screen, self.COLOR_MINE_BG, rect)
                        if self.last_clicked_mine_pos == (x,y):
                            cx, cy = int(rect.centerx), int(rect.centery)
                            pygame.gfxdraw.filled_circle(self.screen, cx, cy, int(tile_w*0.45), (255, 180, 0))
                            pygame.gfxdraw.filled_circle(self.screen, cx, cy, int(tile_w*0.3), (255, 100, 0))
                        pygame.gfxdraw.filled_circle(self.screen, int(rect.centerx), int(rect.centery), int(tile_w*0.2), self.COLOR_MINE_CIRCLE)
                    else:
                        pygame.draw.rect(self.screen, self.COLOR_REVEALED, rect)
                        if tile.adjacent_mines > 0:
                            num_color = self.TILE_COLORS.get(tile.adjacent_mines, self.COLOR_UI_TEXT)
                            self._draw_text(str(tile.adjacent_mines), self.font_tile, num_color, rect.center)
                else:
                    if self.game_over and tile.is_mine:
                        pygame.draw.rect(self.screen, self.COLOR_REVEALED, rect)
                        pygame.gfxdraw.filled_circle(self.screen, int(rect.centerx), int(rect.centery), int(tile_w*0.2), self.COLOR_MINE_CIRCLE)
                    else:
                        pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, rect)

                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 2)
        
        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(offset_x + cursor_x * tile_w, offset_y + cursor_y * tile_h, tile_w, tile_h)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4)

    def _render_ui(self):
        self._draw_text(f"Score: {self.score:.1f}", self.font_main, self.COLOR_UI_TEXT, (10, 10), align="topleft")
        self._draw_text(f"Steps: {self.steps}", self.font_main, self.COLOR_UI_TEXT, (self.WIDTH - 10, 10), align="topright")
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                self._draw_text("VICTORY!", self.font_game_over, self.COLOR_WIN, (self.WIDTH // 2, self.HEIGHT // 2))
            else:
                self._draw_text("GAME OVER", self.font_game_over, self.COLOR_LOSS, (self.WIDTH // 2, self.HEIGHT // 2))

    def _draw_text(self, text, font, color, pos, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = (int(pos[0]), int(pos[1]))
        elif align == "topleft":
            text_rect.topleft = (int(pos[0]), int(pos[1]))
        elif align == "topright":
            text_rect.topright = (int(pos[0]), int(pos[1]))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "revealed_tiles": self.revealed_count,
            "is_win": self.win if self.game_over else None
        }

    def close(self):
        pygame.font.quit()
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

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Minesweeper Gym Env")
    clock = pygame.time.Clock()

    print(env.user_guide)
    print(env.game_description)
    
    while not done:
        movement = 0
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_r: # Add a reset key for convenience
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                    continue
        
        if movement > 0 or space > 0:
            action = [movement, space, 0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if reward != 0:
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")
        
        # Render the current observation to the display window
        current_obs = env._get_observation()
        surf = pygame.surfarray.make_surface(np.transpose(current_obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print("Game Over!")
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False
        
        clock.tick(30)

    env.close()