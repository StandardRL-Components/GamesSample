
# Generated: 2025-08-27T20:59:12.961977
# Source Brief: brief_02642.md
# Brief Index: 2642

        
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

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to reveal a square. Shift to toggle a flag."
    )

    game_description = (
        "A classic puzzle game. Navigate a grid, revealing safe squares while avoiding hidden mines. "
        "Numbers on revealed squares indicate the count of adjacent mines. "
        "Use logic to clear the board without detonating a mine."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_SIZE = (25, 25)
        self.NUM_MINES = 10
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAX_STEPS = self.GRID_SIZE[0] * self.GRID_SIZE[1]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        try:
            self.GRID_FONT = pygame.font.SysFont("consolas", 14, bold=True)
            self.UI_FONT = pygame.font.SysFont("impact", 30)
            self.MSG_FONT = pygame.font.SysFont("impact", 50)
        except pygame.error:
            self.GRID_FONT = pygame.font.SysFont(None, 16)
            self.UI_FONT = pygame.font.SysFont(None, 36)
            self.MSG_FONT = pygame.font.SysFont(None, 60)


        # Colors
        self.COLOR_BG = (40, 42, 54)
        self.COLOR_GRID_LINE = (68, 71, 90)
        self.COLOR_UNREVEALED = (98, 114, 164)
        self.COLOR_REVEALED = (50, 52, 66)
        self.COLOR_CURSOR = (241, 250, 140)
        self.COLOR_MINE = (255, 85, 85)
        self.COLOR_FLAG = (80, 250, 123)
        self.COLOR_TEXT = (248, 248, 242)
        self.COLOR_NUMBER = {
            1: (139, 233, 253), 2: (80, 250, 123), 3: (255, 184, 108),
            4: (255, 121, 198), 5: (189, 147, 249), 6: (66, 193, 210),
            7: (255, 255, 255), 8: (150, 150, 150)
        }
        
        # Grid rendering properties
        self.cell_size = self.SCREEN_HEIGHT // self.GRID_SIZE[1]
        self.grid_width = self.cell_size * self.GRID_SIZE[0]
        self.grid_height = self.cell_size * self.GRID_SIZE[1]
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2

        # Initialize state variables
        self.mine_grid = None
        self.number_grid = None
        self.revealed_grid = None
        self.flagged_grid = None
        self.cursor_pos = None
        self.revealed_safe_count = None
        self.total_safe_squares = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.cursor_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        
        # Initialize grids
        self.mine_grid = np.zeros(self.GRID_SIZE, dtype=bool)
        self.number_grid = np.zeros(self.GRID_SIZE, dtype=int)
        self.revealed_grid = np.zeros(self.GRID_SIZE, dtype=bool)
        self.flagged_grid = np.zeros(self.GRID_SIZE, dtype=bool)

        # Place mines
        total_cells = self.GRID_SIZE[0] * self.GRID_SIZE[1]
        flat_indices = self.np_random.choice(total_cells, self.NUM_MINES, replace=False)
        mine_coords = np.unravel_index(flat_indices, self.GRID_SIZE)
        self.mine_grid[mine_coords] = True

        # Pre-calculate adjacent mine counts
        for x in range(self.GRID_SIZE[0]):
            for y in range(self.GRID_SIZE[1]):
                if self.mine_grid[x, y]:
                    continue
                count = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_SIZE[0] and 0 <= ny < self.GRID_SIZE[1] and self.mine_grid[nx, ny]:
                            count += 1
                self.number_grid[x, y] = count

        self.revealed_safe_count = 0
        self.total_safe_squares = total_cells - self.NUM_MINES

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False
        
        if not self.game_over:
            # Handle cursor movement
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE[1] - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE[0] - 1, self.cursor_pos[0] + 1)

            x, y = self.cursor_pos
            
            # Prioritize reveal over flag
            if space_pressed:
                if self.flagged_grid[x, y]:
                    pass # Cannot reveal a flagged square
                elif self.revealed_grid[x, y]:
                    reward = -0.2 # Penalty for revealing an already-revealed square
                elif self.mine_grid[x, y]:
                    # Oops, a mine!
                    self.revealed_grid[x, y] = True
                    self.game_over = True
                    reward = -100.0
                    # sfx: explosion
                else:
                    # Revealed a safe square
                    initial_revealed_count = self.revealed_safe_count
                    if self.number_grid[x, y] == 0:
                        self._flood_fill_reveal(x, y)
                    else:
                        if not self.revealed_grid[x, y]:
                            self.revealed_grid[x, y] = True
                            self.revealed_safe_count += 1
                    
                    newly_revealed = self.revealed_safe_count - initial_revealed_count
                    reward = 1.0 * newly_revealed # +1 for each new safe square
                    # sfx: click_reveal

            elif shift_pressed:
                if not self.revealed_grid[x, y]:
                    is_mine = self.mine_grid[x, y]
                    is_flagged = self.flagged_grid[x, y]
                    
                    if not is_flagged: # Placing a flag
                        self.flagged_grid[x, y] = True
                        reward = 5.0 if is_mine else -5.0
                        # sfx: flag_place
                    else: # Removing a flag
                        self.flagged_grid[x, y] = False
                        reward = -5.0 if is_mine else 5.0
                        # sfx: flag_remove

            # Check for win condition
            if self.revealed_safe_count == self.total_safe_squares:
                self.game_over = True
                self.game_won = True
                reward += 100.0
                # sfx: win_fanfare

        self.steps += 1
        self.score += reward
        
        if self.game_over or self.steps >= self.MAX_STEPS:
            terminated = True
            if self.game_over and not self.game_won:
                # Reveal all mines on loss
                self.revealed_grid = np.logical_or(self.revealed_grid, self.mine_grid)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _flood_fill_reveal(self, start_x, start_y):
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])

        while q:
            x, y = q.popleft()

            if not self.revealed_grid[x, y]:
                self.revealed_grid[x, y] = True
                self.revealed_safe_count += 1
            
            # If this is an empty square, expand to neighbors
            if self.number_grid[x, y] == 0:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        nx, ny = x + dx, y + dy
                        
                        if 0 <= nx < self.GRID_SIZE[0] and 0 <= ny < self.GRID_SIZE[1] and \
                           not self.revealed_grid[nx, ny] and not self.flagged_grid[nx, ny] and \
                           (nx, ny) not in visited:
                            
                            visited.add((nx, ny))
                            q.append((nx, ny))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(self.GRID_SIZE[0]):
            for y in range(self.GRID_SIZE[1]):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                
                # Draw cell background
                bg_color = self.COLOR_REVEALED if self.revealed_grid[x, y] else self.COLOR_UNREVEALED
                pygame.draw.rect(self.screen, bg_color, rect)

                # Draw content if revealed
                if self.revealed_grid[x, y]:
                    if self.mine_grid[x, y]:
                        pygame.gfxdraw.filled_circle(
                            self.screen, rect.centerx, rect.centery,
                            self.cell_size // 3, self.COLOR_MINE
                        )
                    elif self.number_grid[x, y] > 0:
                        num_text = str(self.number_grid[x, y])
                        color = self.COLOR_NUMBER.get(self.number_grid[x, y], self.COLOR_TEXT)
                        text_surf = self.GRID_FONT.render(num_text, True, color)
                        text_rect = text_surf.get_rect(center=rect.center)
                        self.screen.blit(text_surf, text_rect)
                
                # Draw flag if flagged
                elif self.flagged_grid[x, y]:
                    p1 = (rect.centerx, rect.top + 3)
                    p2 = (rect.centerx, rect.bottom - 3)
                    p3 = (rect.left + 3, rect.centery - 2)
                    pygame.draw.polygon(self.screen, self.COLOR_FLAG, [p1, p3, p2])

                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.cell_size,
            self.grid_offset_y + self.cursor_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)

    def _render_ui(self):
        # Render score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.UI_FONT.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 10))

        # Render game over/win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg_text = "YOU WIN!" if self.game_won else "GAME OVER"
            msg_color = self.COLOR_FLAG if self.game_won else self.COLOR_MINE
            msg_surf = self.MSG_FONT.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "revealed_safe": self.revealed_safe_count,
            "total_safe": self.total_safe_squares,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert np.sum(self.mine_grid) == self.NUM_MINES, f"Expected {self.NUM_MINES} mines, found {np.sum(self.mine_grid)}"
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

        # Test cursor boundary
        self.reset()
        self.cursor_pos = [0, 0]
        self.step([3, 0, 0]) # Move left
        assert self.cursor_pos[0] == 0
        self.step([1, 0, 0]) # Move up
        assert self.cursor_pos[1] == 0

        self.cursor_pos = [self.GRID_SIZE[0]-1, self.GRID_SIZE[1]-1]
        self.step([4, 0, 0]) # Move right
        assert self.cursor_pos[0] == self.GRID_SIZE[0]-1
        self.step([2, 0, 0]) # Move down
        assert self.cursor_pos[1] == self.GRID_SIZE[1]-1

        # Test mine reveal
        self.reset()
        try:
            mine_pos = np.argwhere(self.mine_grid)[0]
            self.cursor_pos = list(mine_pos)
            _, reward, terminated, _, _ = self.step([0, 1, 0]) # Reveal
            assert reward == -100.0, f"Expected -100 reward for hitting a mine, got {reward}"
            assert terminated is True, "Game should terminate after hitting a mine"
        except IndexError:
            print("Warning: No mines found for test, skipping mine reveal test.")

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Minesweeper Gym Env")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    action = [0, 0, 0] # No-op
    
    print(GameEnv.user_guide)

    while not done:
        # Get user input
        space_pressed = 0
        shift_pressed = 0
        movement = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_pressed = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_pressed = 1
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    print("--- Game Reset ---")

        # Only perform an action if a key was pressed
        if any([movement, space_pressed, shift_pressed]):
            action = [movement, space_pressed, shift_pressed]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            print(f"Action: {action}, Reward: {reward:.1f}, Score: {info['score']:.1f}, Done: {terminated}")
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit frame rate
        
    print("Game Over!")
    pygame.time.wait(2000)
    env.close()