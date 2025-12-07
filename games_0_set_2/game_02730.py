
# Generated: 2025-08-27T21:15:09.130891
# Source Brief: brief_02730.md
# Brief Index: 2730

        
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
        "A classic puzzle game. Navigate a grid, revealing safe tiles while avoiding hidden mines. "
        "Numbers on revealed tiles indicate how many mines are adjacent."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.GRID_DIM = 5
        self.NUM_MINES = 10
        self.MAX_STEPS = 1000
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        self.font_large = pygame.font.SysFont("Consolas", 40, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # Colors
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_GRID = (60, 65, 75)
        self.COLOR_UNREVEALED = (80, 88, 102)
        self.COLOR_REVEALED = (45, 50, 58)
        self.COLOR_CURSOR = (100, 180, 255)
        self.COLOR_MINE = (255, 80, 80)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_NUM = {
            1: (80, 160, 255),
            2: (80, 200, 120),
            3: (255, 120, 120),
            4: (180, 100, 255),
            5: (255, 180, 80),
            6: (80, 220, 220),
            7: (220, 220, 80),
            8: (200, 200, 200)
        }

        # Grid rendering properties
        self.tile_size = 60
        self.grid_width = self.GRID_DIM * self.tile_size
        self.grid_height = self.GRID_DIM * self.tile_size
        self.grid_offset_x = (self.WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.HEIGHT - self.grid_height) // 2 + 20

        # Initialize state variables
        self.grid = None
        self.revealed_grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.safe_tiles_to_reveal = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.cursor_pos = [self.GRID_DIM // 2, self.GRID_DIM // 2]
        
        self.grid = np.zeros((self.GRID_DIM, self.GRID_DIM), dtype=int)
        self.revealed_grid = np.zeros((self.GRID_DIM, self.GRID_DIM), dtype=bool)
        
        self._place_mines()
        self._calculate_numbers()
        self.safe_tiles_to_reveal = self.GRID_DIM**2 - self.NUM_MINES
        
        return self._get_observation(), self._get_info()

    def _place_mines(self):
        mine_positions = random.sample(range(self.GRID_DIM**2), self.NUM_MINES)
        for pos in mine_positions:
            x, y = pos % self.GRID_DIM, pos // self.GRID_DIM
            self.grid[y, x] = -1 # -1 represents a mine

    def _calculate_numbers(self):
        for y in range(self.GRID_DIM):
            for x in range(self.GRID_DIM):
                if self.grid[y, x] == -1:
                    continue
                mine_count = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.GRID_DIM and 0 <= nx < self.GRID_DIM and self.grid[ny, nx] == -1:
                            mine_count += 1
                self.grid[y, x] = mine_count

    def step(self, action):
        movement = action[0]
        space_pressed = action[1] == 1
        
        reward = 0
        terminated = False

        if not self.game_over:
            # Handle cursor movement
            prev_cursor_pos = list(self.cursor_pos)
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            
            # Wrap cursor around edges
            self.cursor_pos[0] %= self.GRID_DIM
            self.cursor_pos[1] %= self.GRID_DIM

            # Handle reveal action
            if space_pressed:
                # sfx: tile_click.wav
                x, y = self.cursor_pos
                if self.revealed_grid[y, x]:
                    reward = -0.1 # Penalty for clicking revealed tile
                else:
                    if self.grid[y, x] == -1: # Clicked a mine
                        # sfx: explosion.wav
                        reward = -100
                        self.score += reward
                        self.game_over = True
                        terminated = True
                        self.win = False
                        # Reveal all mines on loss
                        self.revealed_grid[self.grid == -1] = True
                    else: # Clicked a safe tile
                        # sfx: reveal.wav
                        revealed_count = self._reveal_tile(x, y)
                        reward = revealed_count # Reward is number of newly revealed tiles
                        self.score += reward
                        self.safe_tiles_to_reveal -= revealed_count
                        
                        if self.safe_tiles_to_reveal <= 0:
                            # sfx: win_jingle.wav
                            reward += 100
                            self.score += 100
                            self.game_over = True
                            terminated = True
                            self.win = True

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _reveal_tile(self, x, y):
        if not (0 <= y < self.GRID_DIM and 0 <= x < self.GRID_DIM):
            return 0
        if self.revealed_grid[y, x]:
            return 0

        self.revealed_grid[y, x] = True
        count = 1

        if self.grid[y, x] == 0:
            # Cascade reveal for empty tiles
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    count += self._reveal_tile(x + dx, y + dy)
        return count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for y in range(self.GRID_DIM):
            for x in range(self.GRID_DIM):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.tile_size,
                    self.grid_offset_y + y * self.tile_size,
                    self.tile_size, self.tile_size
                )

                if self.revealed_grid[y, x]:
                    pygame.draw.rect(self.screen, self.COLOR_REVEALED, rect)
                    value = self.grid[y, x]
                    if value == -1: # Mine
                        self._draw_mine(rect.center)
                    elif value > 0:
                        self._draw_text(
                            str(value), rect.center, self.font_large, self.COLOR_NUM.get(value, self.COLOR_TEXT)
                        )
                else:
                    pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, rect)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 2)
        
        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_offset_x + cursor_x * self.tile_size,
            self.grid_offset_y + cursor_y * self.tile_size,
            self.tile_size, self.tile_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=4)
    
    def _draw_mine(self, center):
        cx, cy = center
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, self.tile_size // 4, self.COLOR_MINE)
        pygame.gfxdraw.aacircle(self.screen, cx, cy, self.tile_size // 4, self.COLOR_MINE)
        # Spikes
        for i in range(8):
            angle = i * math.pi / 4
            x1 = cx + math.cos(angle) * self.tile_size * 0.2
            y1 = cy + math.sin(angle) * self.tile_size * 0.2
            x2 = cx + math.cos(angle) * self.tile_size * 0.35
            y2 = cy + math.sin(angle) * self.tile_size * 0.35
            pygame.draw.line(self.screen, self.COLOR_MINE, (x1, y1), (x2, y2), 3)

    def _render_ui(self):
        # Draw Score
        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Draw Steps
        steps_text = self.font_medium.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 20, 10))

        # Draw Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg = "BOARD CLEARED!"
                color = self.COLOR_NUM[2]
            else:
                msg = "GAME OVER"
                color = self.COLOR_MINE
                
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _draw_text(self, text, pos, font, color):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=pos)
        self.screen.blit(text_surface, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": list(self.cursor_pos),
            "game_over": self.game_over,
            "win": self.win
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Minesweeper Grid")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0  # No-op
        space_pressed = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space_pressed = 1
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if movement != 0 or space_pressed != 0:
            action = [movement, space_pressed, 0] # Shift is unused
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Win: {info['win']}")
                # Wait for a moment before allowing reset
                pygame.time.wait(2000)
                obs, info = env.reset()

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate for human play
        
    env.close()