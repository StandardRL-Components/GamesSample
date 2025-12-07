
# Generated: 2025-08-27T16:29:00.371371
# Source Brief: brief_01240.md
# Brief Index: 1240

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to reveal a square and shift to place/remove a flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Clear the grid by revealing all safe squares while avoiding the hidden mines."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 9
    NUM_MINES = 10
    MAX_STEPS = 1000

    # Cell states
    STATE_UNREVEALED = 0
    STATE_REVEALED = 1
    STATE_FLAGGED = 2

    # Colors
    COLOR_BG = (44, 62, 80) # Dark Blue
    COLOR_GRID = (52, 73, 94) # Slightly lighter blue
    COLOR_UNREVEALED = (149, 165, 166) # Grey
    COLOR_UNREVEALED_LIGHT = (189, 195, 199)
    COLOR_UNREVEALED_DARK = (127, 140, 141)
    COLOR_REVEALED = (236, 240, 241) # Light Grey
    COLOR_CURSOR = (241, 196, 15) # Yellow
    COLOR_FLAG = (46, 204, 113) # Green
    COLOR_MINE = (231, 76, 60) # Red
    COLOR_TEXT = (236, 240, 241) # White
    COLOR_NUMBER = [
        (0, 0, 0), # Not used for 0
        (52, 152, 219),  # 1: Blue
        (46, 204, 113),  # 2: Green
        (231, 76, 60),   # 3: Red
        (142, 68, 173),  # 4: Purple
        (241, 196, 15),  # 5: Yellow
        (230, 126, 34),  # 6: Orange
        (52, 73, 94),    # 7: Dark Grey
        (255, 255, 255), # 8: White
    ]

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
        self.font_large = pygame.font.SysFont("sans-serif", 48, bold=True)
        self.font_medium = pygame.font.SysFont("sans-serif", 24, bold=True)
        self.font_small = pygame.font.SysFont("sans-serif", 18, bold=True)

        # Game state variables
        self.mine_grid = None
        self.number_grid = None
        self.state_grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.revealed_safe_squares = 0
        self.flags_placed = 0
        
        # Grid rendering properties
        self.cell_size = 36
        self.grid_width = self.GRID_SIZE * self.cell_size
        self.grid_height = self.GRID_SIZE * self.cell_size
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2 + 20

        self.reset()
        
        # Validate implementation after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.revealed_safe_squares = 0
        self.flags_placed = 0

        self.cursor_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        self._generate_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False
        
        # 1. Handle cursor movement
        if movement == 1: # Up
            self.cursor_pos[1] -= 1
        elif movement == 2: # Down
            self.cursor_pos[1] += 1
        elif movement == 3: # Left
            self.cursor_pos[0] -= 1
        elif movement == 4: # Right
            self.cursor_pos[0] += 1
        self.cursor_pos = np.clip(self.cursor_pos, 0, self.GRID_SIZE - 1)

        cx, cy = self.cursor_pos

        # 2. Handle actions (Shift takes precedence over Space)
        if shift_pressed:
            reward += self._toggle_flag(cx, cy)
        elif space_pressed:
            reward += self._reveal_square(cx, cy)
        
        # 3. Check termination conditions
        if self.game_over:
            terminated = True
            if self.win:
                reward += 100.0  # Win bonus
            else:
                reward = -100.0 # Lose penalty (override any other rewards this step)
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _generate_grid(self):
        self.mine_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.number_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.state_grid = np.full((self.GRID_SIZE, self.GRID_SIZE), self.STATE_UNREVEALED, dtype=int)

        # Place mines
        flat_indices = np.arange(self.GRID_SIZE * self.GRID_SIZE)
        self.np_random.shuffle(flat_indices)
        mine_indices = flat_indices[:self.NUM_MINES]
        rows, cols = np.unravel_index(mine_indices, (self.GRID_SIZE, self.GRID_SIZE))
        self.mine_grid[rows, cols] = True

        # Calculate numbers
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if not self.mine_grid[r, c]:
                    count = np.sum(self.mine_grid[max(0, r-1):r+2, max(0, c-1):c+2])
                    self.number_grid[r, c] = count
    
    def _toggle_flag(self, x, y):
        if self.state_grid[y, x] == self.STATE_UNREVEALED:
            self.state_grid[y, x] = self.STATE_FLAGGED
            self.flags_placed += 1
            # Correctly flagging a mine is rewarded
            return 5.0 if self.mine_grid[y, x] else -1.0 # Penalize wrong flags
        elif self.state_grid[y, x] == self.STATE_FLAGGED:
            self.state_grid[y, x] = self.STATE_UNREVEALED
            self.flags_placed -= 1
            # Removing a correct flag is penalized
            return -5.0 if self.mine_grid[y, x] else 1.0
        return 0.0

    def _reveal_square(self, x, y):
        if self.state_grid[y, x] != self.STATE_UNREVEALED:
            return -0.1 # Penalty for revealing an already revealed/flagged square

        if self.mine_grid[y, x]:
            self.game_over = True
            self.win = False
            # The final -100 is applied in step()
            return 0.0
        else:
            revealed_count, is_zero_area = self._flood_fill(x, y)
            self.revealed_safe_squares += revealed_count
            
            # Check for win
            total_safe_squares = self.GRID_SIZE * self.GRID_SIZE - self.NUM_MINES
            if self.revealed_safe_squares >= total_safe_squares:
                self.game_over = True
                self.win = True

            # Reward calculation
            reward = revealed_count
            if is_zero_area:
                reward -= 0.2 # Discourage revealing empty areas
            return reward

    def _flood_fill(self, x, y):
        """Reveals a square and any adjacent empty squares."""
        if self.state_grid[y, x] != self.STATE_UNREVEALED:
            return 0, False

        q = deque([(x, y)])
        self.state_grid[y, x] = self.STATE_REVEALED
        count = 1
        is_zero_area = self.number_grid[y, x] == 0

        while q:
            cx, cy = q.popleft()
            if self.number_grid[cy, cx] == 0:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.state_grid[ny, nx] == self.STATE_UNREVEALED:
                            self.state_grid[ny, nx] = self.STATE_REVEALED
                            count += 1
                            q.append((nx, ny))
        return count, is_zero_area

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
            "cursor_pos": self.cursor_pos.tolist(),
            "flags_remaining": self.NUM_MINES - self.flags_placed,
            "revealed_safe": self.revealed_safe_squares,
        }
        
    def _render_text(self, text, font, color, center):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=center)
        self.screen.blit(text_surface, text_rect)

    def _render_game(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.grid_offset_x + c * self.cell_size,
                    self.grid_offset_y + r * self.cell_size,
                    self.cell_size, self.cell_size
                )
                
                state = self.state_grid[r, c]
                
                # Draw cell based on state
                if state == self.STATE_UNREVEALED or (self.game_over and self.mine_grid[r, c] and state == self.STATE_FLAGGED):
                    # Beveled unrevealed square
                    pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, rect)
                    pygame.draw.line(self.screen, self.COLOR_UNREVEALED_LIGHT, rect.topleft, rect.topright, 2)
                    pygame.draw.line(self.screen, self.COLOR_UNREVEALED_LIGHT, rect.topleft, rect.bottomleft, 2)
                    pygame.draw.line(self.screen, self.COLOR_UNREVEALED_DARK, rect.bottomleft, rect.bottomright, 2)
                    pygame.draw.line(self.screen, self.COLOR_UNREVEALED_DARK, rect.topright, rect.bottomright, 2)
                elif state == self.STATE_REVEALED:
                    pygame.draw.rect(self.screen, self.COLOR_REVEALED, rect)
                    num = self.number_grid[r, c]
                    if num > 0:
                        self._render_text(str(num), self.font_medium, self.COLOR_NUMBER[num-1], rect.center)
                elif state == self.STATE_FLAGGED:
                    # Draw unrevealed square underneath flag
                    pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, rect)
                    # Draw flag
                    flag_points = [
                        (rect.centerx - 5, rect.top + 5),
                        (rect.centerx + 8, rect.centery - 5),
                        (rect.centerx - 5, rect.centery)
                    ]
                    pygame.gfxdraw.aapolygon(self.screen, flag_points, self.COLOR_FLAG)
                    pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_FLAG)
                    pygame.draw.line(self.screen, self.COLOR_FLAG, (rect.centerx - 5, rect.top + 5), (rect.centerx - 5, rect.bottom - 5), 2)
                
                # On game over, reveal mines
                if self.game_over and not self.win and self.mine_grid[r, c] and state != self.STATE_FLAGGED:
                    pygame.draw.rect(self.screen, self.COLOR_MINE, rect)
                    pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, self.cell_size // 4, (0,0,0))
                    pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, self.cell_size // 4, (0,0,0))

                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.cell_size,
            self.grid_offset_y + self.cursor_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        # Transparent fill
        s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        s.fill((*self.COLOR_CURSOR, 60))
        self.screen.blit(s, cursor_rect.topleft)
        # Border
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        # Render score
        score_text = f"Score: {int(self.score)}"
        self._render_text(score_text, self.font_medium, self.COLOR_TEXT, (100, 25))

        # Render flags remaining
        flags_text = f"Mines: {self.NUM_MINES - self.flags_placed}"
        self._render_text(flags_text, self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH - 100, 25))

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            if self.win:
                self._render_text("YOU WIN!", self.font_large, (46, 204, 113), self.screen.get_rect().center)
            else:
                self._render_text("GAME OVER", self.font_large, self.COLOR_MINE, self.screen.get_rect().center)
                
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    # --- Example of how to run the environment ---
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Example ---
    # To run this, you need to install pygame and have a display.
    # On a headless server, this part will fail, but the environment itself is headless.
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Minesweeper RL")
        clock = pygame.time.Clock()

        running = True
        while running:
            movement, space, shift = 0, 0, 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_UP]: movement = 1
                    elif keys[pygame.K_DOWN]: movement = 2
                    elif keys[pygame.K_LEFT]: movement = 3
                    elif keys[pygame.K_RIGHT]: movement = 4

                    if keys[pygame.K_SPACE]: space = 1
                    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
                    
                    action = np.array([movement, space, shift])
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

                    if terminated:
                        print("Game Over! Resetting in 3 seconds...")
                        screen.blit(pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2))), (0, 0))
                        pygame.display.flip()
                        pygame.time.wait(3000)
                        obs, info = env.reset()

            # Draw the current state
            screen.blit(pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2))), (0, 0))
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()
        
    except pygame.error as e:
        print(f"Pygame display error: {e}")
        print("This is expected on a headless server. The environment itself is functional.")
        print("Running a simple step test without rendering to display.")
        
        env = GameEnv()
        env.reset()
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Episode terminated at step {i}.")
                env.reset()
        print("Headless step test completed.")