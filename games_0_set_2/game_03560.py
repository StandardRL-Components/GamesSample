
# Generated: 2025-08-27T23:43:25.903546
# Source Brief: brief_03560.md
# Brief Index: 3560

        
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
        "Controls: Arrow keys to move cursor. Space to reveal a square. Shift to place/remove a flag."
    )

    game_description = (
        "A classic mine-sweeping puzzle. Reveal all safe squares without hitting a mine. Numbers indicate adjacent mines."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.GRID_SIZE = 20
        self.NUM_MINES = 40 # Increased from 10 to make it more challenging
        self.MAX_STEPS = self.GRID_SIZE * self.GRID_SIZE

        # Visual constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 16
        self.GRID_LINE_WIDTH = 1
        self.GRID_WIDTH = self.GRID_SIZE * (self.CELL_SIZE + self.GRID_LINE_WIDTH) + self.GRID_LINE_WIDTH
        self.GRID_HEIGHT = self.GRID_SIZE * (self.CELL_SIZE + self.GRID_LINE_WIDTH) + self.GRID_LINE_WIDTH
        self.X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20 # Make space for UI at top

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (60, 60, 70)
        self.COLOR_UNREVEALED = (120, 130, 140)
        self.COLOR_REVEALED = (45, 45, 55)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_FLAG = (255, 190, 0)
        self.COLOR_MINE = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_WIN = (0, 255, 127)
        self.COLOR_LOSE = (255, 80, 80)
        self.NUM_COLORS = [
            None,
            (0, 127, 255),  # 1
            (0, 190, 0),    # 2
            (255, 0, 0),    # 3
            (0, 0, 127),    # 4
            (127, 0, 0),    # 5
            (0, 190, 190),  # 6
            (0, 0, 0),      # 7
            (100, 100, 100) # 8
        ]

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_grid = pygame.font.Font(None, self.CELL_SIZE + 2)
        self.font_game_over = pygame.font.Font(None, 60)
        
        # Game state variables
        self.cursor_pos = None
        self.mine_grid = None
        self.revealed_grid = None
        self.flagged_grid = None
        self.number_grid = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.first_reveal_done = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.cursor_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        self.revealed_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.flagged_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        
        # Defer mine generation until the first reveal
        self.mine_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.number_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.first_reveal_done = False
        
        return self._get_observation(), self._get_info()

    def _generate_minefield(self, first_click_pos):
        # Generate mines away from the first click
        safe_zone = []
        for r_offset in range(-1, 2):
            for c_offset in range(-1, 2):
                safe_zone.append((first_click_pos[1] + r_offset, first_click_pos[0] + c_offset))
        
        possible_mine_indices = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (r, c) not in safe_zone:
                    possible_mine_indices.append(r * self.GRID_SIZE + c)
        
        mine_indices = self.np_random.choice(possible_mine_indices, self.NUM_MINES, replace=False)
        
        flat_grid = np.zeros(self.GRID_SIZE * self.GRID_SIZE, dtype=np.int8)
        flat_grid[mine_indices] = 1
        self.mine_grid = flat_grid.reshape((self.GRID_SIZE, self.GRID_SIZE))

        # Calculate numbers for all squares
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.mine_grid[r, c] == 0:
                    mine_count = np.sum(self.mine_grid[max(0, r-1):r+2, max(0, c-1):c+2])
                    self.number_grid[r, c] = mine_count
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)
        
        cx, cy = self.cursor_pos

        # 2. Handle actions (Reveal > Flag)
        if space_held:
            if not self.flagged_grid[cy, cx] and not self.revealed_grid[cy, cx]:
                # First reveal of the game: generate field
                if not self.first_reveal_done:
                    self._generate_minefield(self.cursor_pos)
                    self.first_reveal_done = True

                # Hit a mine
                if self.mine_grid[cy, cx] == 1:
                    # sfx: explosion
                    self.game_over = True
                    self.win = False
                    reward = -100.0
                    self.revealed_grid[cy, cx] = 1
                # Hit a safe square
                else:
                    # sfx: click_reveal
                    revealed_count = self._flood_fill_reveal(cx, cy)
                    reward = float(revealed_count) # +1 per revealed square

        elif shift_held:
            if not self.revealed_grid[cy, cx]:
                is_currently_flagged = self.flagged_grid[cy, cx] == 1
                self.flagged_grid[cy, cx] = 1 - self.flagged_grid[cy, cx] # Toggle
                is_now_flagged = self.flagged_grid[cy, cx] == 1
                
                if is_now_flagged and not is_currently_flagged:
                    # sfx: place_flag
                    if self.mine_grid[cy, cx] == 1 and self.first_reveal_done:
                        reward = 0.2
                    else:
                        reward = -0.2
                # else: sfx: remove_flag
        
        self.score += reward
        self.steps += 1

        # 3. Check for termination
        num_safe_squares = self.GRID_SIZE * self.GRID_SIZE - self.NUM_MINES
        if self.first_reveal_done and np.sum(self.revealed_grid) == num_safe_squares and not self.game_over:
            self.game_over = True
            self.win = True
            reward += 100.0
            self.score += 100.0
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _flood_fill_reveal(self, x, y):
        stack = [(x, y)]
        revealed_count = 0
        while stack:
            cx, cy = stack.pop()
            if not (0 <= cx < self.GRID_SIZE and 0 <= cy < self.GRID_SIZE):
                continue
            if self.revealed_grid[cy, cx] or self.flagged_grid[cy, cx]:
                continue
            
            self.revealed_grid[cy, cx] = 1
            revealed_count += 1
            
            if self.number_grid[cy, cx] == 0:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        stack.append((cx + dx, cy + dy))
        return revealed_count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(
                    self.X_OFFSET + c * (self.CELL_SIZE + self.GRID_LINE_WIDTH),
                    self.Y_OFFSET + r * (self.CELL_SIZE + self.GRID_LINE_WIDTH),
                    self.CELL_SIZE + self.GRID_LINE_WIDTH,
                    self.CELL_SIZE + self.GRID_LINE_WIDTH
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, cell_rect)

                inner_rect = pygame.Rect(
                    cell_rect.x + self.GRID_LINE_WIDTH,
                    cell_rect.y + self.GRID_LINE_WIDTH,
                    self.CELL_SIZE, self.CELL_SIZE
                )

                if self.revealed_grid[r, c]:
                    pygame.draw.rect(self.screen, self.COLOR_REVEALED, inner_rect)
                    if self.mine_grid[r, c]:
                        pygame.gfxdraw.filled_circle(self.screen, inner_rect.centerx, inner_rect.centery, self.CELL_SIZE // 3, self.COLOR_MINE)
                        pygame.gfxdraw.aacircle(self.screen, inner_rect.centerx, inner_rect.centery, self.CELL_SIZE // 3, self.COLOR_MINE)
                    elif self.number_grid[r, c] > 0:
                        num_text = self.font_grid.render(str(self.number_grid[r, c]), True, self.NUM_COLORS[self.number_grid[r, c]])
                        text_rect = num_text.get_rect(center=inner_rect.center)
                        self.screen.blit(num_text, text_rect)
                else: # Unrevealed
                    pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, inner_rect)
                    if self.flagged_grid[r, c]:
                        # Draw flag
                        flag_points = [
                            (inner_rect.centerx, inner_rect.top + 3),
                            (inner_rect.right - 4, inner_rect.top + 6),
                            (inner_rect.centerx, inner_rect.top + 9)
                        ]
                        pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)
                        pygame.draw.line(self.screen, self.COLOR_FLAG, (inner_rect.centerx, inner_rect.top + 3), (inner_rect.centerx, inner_rect.bottom - 3), 1)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.X_OFFSET + self.cursor_pos[0] * (self.CELL_SIZE + self.GRID_LINE_WIDTH),
            self.Y_OFFSET + self.cursor_pos[1] * (self.CELL_SIZE + self.GRID_LINE_WIDTH),
            self.CELL_SIZE + self.GRID_LINE_WIDTH,
            self.CELL_SIZE + self.GRID_LINE_WIDTH
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)
        
        # On game over, reveal all mines
        if self.game_over and not self.win:
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    if self.mine_grid[r,c] and not self.revealed_grid[r,c]:
                        inner_rect = pygame.Rect(
                            self.X_OFFSET + c * (self.CELL_SIZE + self.GRID_LINE_WIDTH) + self.GRID_LINE_WIDTH,
                            self.Y_OFFSET + r * (self.CELL_SIZE + self.GRID_LINE_WIDTH) + self.GRID_LINE_WIDTH,
                            self.CELL_SIZE, self.CELL_SIZE
                        )
                        pygame.gfxdraw.filled_circle(self.screen, inner_rect.centerx, inner_rect.centery, self.CELL_SIZE // 4, self.COLOR_GRID)


    def _render_ui(self):
        # Top UI bar
        flags_placed = np.sum(self.flagged_grid)
        mines_left = self.NUM_MINES - flags_placed if self.first_reveal_done else self.NUM_MINES
        
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        mines_text = self.font_main.render(f"Mines: {mines_left}", True, self.COLOR_TEXT)
        mines_rect = mines_text.get_rect(centerx=self.WIDTH // 2, y=10)
        self.screen.blit(mines_text, mines_rect)

        steps_text = self.font_main.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        steps_rect = steps_text.get_rect(right=self.WIDTH - 10, y=10)
        self.screen.blit(steps_text, steps_rect)

        # Game over message
        if self.game_over:
            if self.win:
                end_text = self.font_game_over.render("YOU WIN!", True, self.COLOR_WIN)
            else:
                end_text = self.font_game_over.render("GAME OVER", True, self.COLOR_LOSE)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos.tolist(),
            "mines_remaining": self.NUM_MINES - np.sum(self.flagged_grid) if self.first_reveal_done else self.NUM_MINES,
            "win": self.win
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

if __name__ == "__main__":
    # Example of how to run the environment
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    pygame.display.set_caption("Minesweeper Gym Env")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    
    while not done:
        # Default action is no-op
        action = [0, 0, 0] # move, space, shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
        
        # Only step if a key was pressed
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {terminated}")

        # Render the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)
        
    print("Game Over!")
    pygame.time.wait(2000)
    env.close()