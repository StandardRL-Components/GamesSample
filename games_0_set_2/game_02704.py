
# Generated: 2025-08-27T21:10:48.702157
# Source Brief: brief_02704.md
# Brief Index: 2704

        
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
        "Controls: Arrow keys to move cursor. Space to reveal a tile. Shift to place/remove a flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic puzzle game. Navigate a grid, revealing safe tiles while avoiding hidden mines to clear the board."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 9, 9
        self.NUM_MINES = 10
        self.MAX_STEPS = 1000

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
        self.font_small = pygame.font.SysFont("sans-serif", 20)
        self.font_large = pygame.font.SysFont("sans-serif", 48)
        self.font_tile = pygame.font.SysFont("monospace", 28, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_TILE_HIDDEN = (70, 80, 90)
        self.COLOR_TILE_REVEALED = (40, 50, 60)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_FLAG = (255, 255, 255)
        self.COLOR_MINE = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.NUMBER_COLORS = {
            1: (50, 150, 255), 2: (50, 200, 50), 3: (255, 50, 50),
            4: (150, 50, 255), 5: (255, 120, 0), 6: (50, 200, 200),
            7: (0, 0, 0), 8: (128, 128, 128)
        }
        
        # Grid layout
        self.tile_size = 36
        self.grid_offset_x = (self.WIDTH - self.GRID_WIDTH * self.tile_size) // 2
        self.grid_offset_y = (self.HEIGHT - self.GRID_HEIGHT * self.tile_size) // 2
        
        # State variables are initialized in reset()
        self.grid = None
        self.revealed = None
        self.flags = None
        self.cursor_pos = None
        self.game_over = None
        self.win = None
        self.score = None
        self.steps = None
        self.first_move = None
        self.np_random = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.first_move = True
        
        self.cursor_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.revealed = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=bool)
        self.flags = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=bool)
        
        return self._get_observation(), self._get_info()

    def _place_mines(self, safe_pos):
        safe_y, safe_x = safe_pos
        possible_coords = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if r != safe_y or c != safe_x:
                    possible_coords.append((r, c))
        
        mine_indices = self.np_random.choice(len(possible_coords), self.NUM_MINES, replace=False)
        mine_coords = [possible_coords[i] for i in mine_indices]

        for r, c in mine_coords:
            self.grid[r, c] = -1 # -1 represents a mine
        
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != -1:
                    count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH and self.grid[nr, nc] == -1:
                                count += 1
                    self.grid[r, c] = count
    
    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_HEIGHT
        elif movement == 2: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_HEIGHT
        elif movement == 3: self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_WIDTH
        elif movement == 4: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_WIDTH

        cy, cx = self.cursor_pos[1], self.cursor_pos[0]

        # 2. Handle actions (reveal or flag)
        if space_pressed: # Reveal tile
            if not self.flags[cy, cx] and not self.revealed[cy, cx]:
                if self.first_move:
                    self._place_mines(self.cursor_pos)
                    self.first_move = False
                
                reward += self._reveal_tile(cy, cx)

        elif shift_pressed: # Toggle flag
            if not self.revealed[cy, cx]:
                self.flags[cy, cx] = not self.flags[cy, cx]
                reward -= 0.1 # Small penalty for flagging
        
        self.score += reward
        self.steps += 1
        
        # 3. Check for termination
        if self.win:
            self.game_over = True
            self.score += 100
            reward += 100
            terminated = True
        elif self.game_over: # Loss condition is set inside _reveal_tile
            self.score -= 100
            reward -= 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _reveal_tile(self, y, x):
        if y < 0 or y >= self.GRID_HEIGHT or x < 0 or x >= self.GRID_WIDTH:
            return 0
        if self.revealed[y, x] or self.flags[y, x]:
            return 0

        self.revealed[y, x] = True
        
        if self.grid[y, x] == -1: # Hit a mine
            self.game_over = True
            return 0 # The -100 penalty is applied in step()

        reward = 1.0
        
        if self.grid[y, x] == 0: # Flood fill for empty tiles
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0: continue
                    reward += self._reveal_tile(y + dy, x + dx)
        
        # Check for win condition
        if np.sum(self.revealed) == (self.GRID_WIDTH * self.GRID_HEIGHT) - self.NUM_MINES:
            self.win = True
            
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    self.grid_offset_x + c * self.tile_size,
                    self.grid_offset_y + r * self.tile_size,
                    self.tile_size, self.tile_size
                )
                
                # Draw revealed tiles
                if self.revealed[r, c] or (self.game_over and self.grid[r, c] == -1):
                    pygame.draw.rect(self.screen, self.COLOR_TILE_REVEALED, rect)
                    
                    if self.grid[r, c] == -1: # Mine
                        pygame.draw.circle(self.screen, self.COLOR_MINE, rect.center, self.tile_size // 3)
                        pygame.draw.line(self.screen, self.COLOR_MINE, rect.topleft, rect.bottomright, 3)
                        pygame.draw.line(self.screen, self.COLOR_MINE, rect.topright, rect.bottomleft, 3)
                    elif self.grid[r, c] > 0: # Number
                        num_text = self.font_tile.render(str(self.grid[r, c]), True, self.NUMBER_COLORS[self.grid[r, c]])
                        text_rect = num_text.get_rect(center=rect.center)
                        self.screen.blit(num_text, text_rect)
                
                # Draw hidden tiles and flags
                else:
                    pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, rect)
                    if self.flags[r, c]:
                        # Draw a simple flag
                        flag_points = [
                            (rect.centerx - 5, rect.top + 5),
                            (rect.centerx - 5, rect.bottom - 5),
                            (rect.centerx + 8, rect.centery - 5),
                        ]
                        pygame.draw.line(self.screen, self.COLOR_FLAG, (rect.centerx - 5, rect.top + 5), (rect.centerx - 5, rect.bottom - 5), 2)
                        pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw cursor
        cursor_y, cursor_x = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_offset_x + cursor_x * self.tile_size,
            self.grid_offset_y + cursor_y * self.tile_size,
            self.tile_size, self.tile_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        # Score display
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Mines remaining display
        mines_left = self.NUM_MINES - np.sum(self.flags)
        mines_text = self.font_small.render(f"Mines: {mines_left}", True, self.COLOR_TEXT)
        self.screen.blit(mines_text, (self.WIDTH - mines_text.get_width() - 10, 10))
        
        # Game over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                message = "YOU WIN!"
                color = (100, 255, 100)
            else:
                message = "GAME OVER"
                color = self.COLOR_MINE
            
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos.tolist(),
            "win": self.win,
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


# Example usage for human play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for display
    pygame.display.set_caption("Minesweeper Gym Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    print(env.game_description)
    print(env.user_guide)

    running = True
    while running:
        action = [0, 0, 0] # no-op, released, released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
            if terminated:
                print("Game Over! Press 'R' to reset.")

        # Render the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()