
# Generated: 2025-08-27T15:43:14.937631
# Source Brief: brief_01050.md
# Brief Index: 1050

        
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
    """
    A Gymnasium environment for a Minesweeper-style puzzle game.
    The player navigates a grid, reveals safe squares, and flags suspected mines.
    The goal is to reveal all safe squares without detonating a mine.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to reveal a square and shift to place/remove a flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a grid, revealing safe squares while avoiding hidden mines to clear the field."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 9
        self.NUM_MINES = 10
        self.MAX_STEPS = 1000

        # Visuals
        self.COLOR_BG = (22, 28, 38)
        self.COLOR_GRID = (50, 60, 75)
        self.COLOR_HIDDEN = (41, 51, 66)
        self.COLOR_REVEALED = (190, 200, 215)
        self.COLOR_CURSOR = (255, 204, 0)
        self.COLOR_MINE_BG = (236, 85, 93)
        self.COLOR_FLAG = (82, 204, 152)
        self.COLOR_TEXT_UI = (220, 220, 220)
        self.COLOR_TEXT_WIN = (100, 255, 150)
        self.COLOR_TEXT_LOSE = (255, 100, 100)
        self.NUMBER_COLORS = {
            1: (56, 153, 224),
            2: (82, 204, 152),
            3: (236, 85, 93),
            4: (119, 91, 189),
            5: (199, 106, 38),
            6: (42, 181, 186),
            7: (200, 200, 200),
            8: (120, 120, 120),
        }

        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.START_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.START_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_game = pygame.font.Font(None, int(self.CELL_SIZE * 0.8))
        self.font_ui = pygame.font.Font(None, 32)
        self.font_msg = pygame.font.Font(None, 64)
        
        # --- Game State ---
        self.grid = None
        self.revealed_grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_state = False
        self.rng = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_state = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        
        self._place_mines_and_calculate_numbers()
        
        return self._get_observation(), self._get_info()

    def _place_mines_and_calculate_numbers(self):
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.revealed_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int) # 0:hidden, 1:revealed, 2:flagged

        # Place mines
        mine_indices = self.rng.choice(self.GRID_SIZE * self.GRID_SIZE, self.NUM_MINES, replace=False)
        mine_coords = np.unravel_index(mine_indices, (self.GRID_SIZE, self.GRID_SIZE))
        self.grid[mine_coords] = -1 # -1 represents a mine

        assert np.sum(self.grid == -1) == self.NUM_MINES, "Incorrect number of mines placed"

        # Calculate numbers
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y, x] == -1:
                    continue
                count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.grid[ny, nx] == -1:
                            count += 1
                self.grid[y, x] = count

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        self._move_cursor(movement)
        
        x, y = self.cursor_pos
        is_revealed = self.revealed_grid[y, x] == 1
        is_flagged = self.revealed_grid[y, x] == 2
        is_mine = self.grid[y, x] == -1

        if space_pressed and not is_revealed and not is_flagged:
            # Reveal action
            if is_mine:
                # sound: explosion
                reward = -100.0
                self.game_over = True
                self.win_state = False
                self.revealed_grid[self.grid == -1] = 1 # Reveal all mines
            else:
                # sound: click_safe
                reward = 1.0
                if self.grid[y, x] == 0:
                    reward -= 0.2
                    self._flood_fill(x, y)
                else:
                    self.revealed_grid[y, x] = 1

        elif shift_pressed and not is_revealed:
            if is_flagged:
                # Unflagging
                # sound: unflag
                self.revealed_grid[y, x] = 0
                reward = -5.0 if is_mine else 5.0
            else:
                # Flagging
                # sound: flag
                self.revealed_grid[y, x] = 2
                reward = 5.0 if is_mine else -5.0

        self.steps += 1
        self.score += reward
        
        win_condition_met = not is_mine and self._check_win_condition()
        
        if win_condition_met:
            # sound: win_fanfare
            win_bonus = 100.0
            reward += win_bonus
            self.score += win_bonus
            self.game_over = True
            self.win_state = True
            self.revealed_grid[self.grid == -1] = 2 # Auto-flag remaining mines
            
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _move_cursor(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        y, x = self.cursor_pos
        if movement == 1: y -= 1
        elif movement == 2: y += 1
        elif movement == 3: x -= 1
        elif movement == 4: x += 1
        
        # Wrap around
        self.cursor_pos = [y % self.GRID_SIZE, x % self.GRID_SIZE]

    def _flood_fill(self, x, y):
        stack = [(x, y)]
        visited = set()
        while stack:
            cx, cy = stack.pop()
            if not (0 <= cx < self.GRID_SIZE and 0 <= cy < self.GRID_SIZE): continue
            if (cx, cy) in visited: continue
            if self.revealed_grid[cy, cx] != 0: continue # Don't flood fill into flagged squares

            visited.add((cx, cy))
            self.revealed_grid[cy, cx] = 1

            if self.grid[cy, cx] == 0:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        stack.append((cx + dx, cy + dy))
    
    def _check_win_condition(self):
        num_revealed = np.sum(self.revealed_grid == 1)
        num_safe_squares = self.GRID_SIZE * self.GRID_SIZE - self.NUM_MINES
        return num_revealed == num_safe_squares

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw cells
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.START_X + x * self.CELL_SIZE,
                    self.START_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                
                state = self.revealed_grid[y, x]
                if state == 1: # Revealed
                    is_mine = self.grid[y, x] == -1
                    cell_color = self.COLOR_MINE_BG if is_mine and self.game_over and not self.win_state else self.COLOR_REVEALED
                    pygame.draw.rect(self.screen, cell_color, rect)
                    
                    if self.grid[y, x] > 0:
                        num_text = str(self.grid[y, x])
                        color = self.NUMBER_COLORS.get(self.grid[y, x], self.COLOR_TEXT_UI)
                        text_surf = self.font_game.render(num_text, True, color)
                        text_rect = text_surf.get_rect(center=rect.center)
                        self.screen.blit(text_surf, text_rect)
                    elif is_mine and self.game_over and not self.win_state:
                        # Draw mine icon
                        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, int(self.CELL_SIZE * 0.3), (30,30,30))
                        pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, int(self.CELL_SIZE * 0.3), (30,30,30))

                elif state == 2: # Flagged
                    pygame.draw.rect(self.screen, self.COLOR_HIDDEN, rect)
                    flag_points = [
                        (rect.centerx, rect.top + 5),
                        (rect.right - 10, rect.centery - 5),
                        (rect.centerx, rect.centery + 5)
                    ]
                    pygame.gfxdraw.aapolygon(self.screen, flag_points, self.COLOR_FLAG)
                    pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_FLAG)
                    pygame.draw.line(self.screen, self.COLOR_FLAG, (rect.centerx, rect.top + 5), (rect.centerx, rect.bottom - 5), 2)

                else: # Hidden
                    pygame.draw.rect(self.screen, self.COLOR_HIDDEN, rect)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.START_X + i * self.CELL_SIZE, self.START_Y), (self.START_X + i * self.CELL_SIZE, self.START_Y + self.GRID_HEIGHT), 1)
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.START_X, self.START_Y + i * self.CELL_SIZE), (self.START_X + self.GRID_WIDTH, self.START_Y + i * self.CELL_SIZE), 1)

        # Draw cursor
        cur_y, cur_x = self.cursor_pos
        cursor_rect = pygame.Rect(self.START_X + cur_x * self.CELL_SIZE, self.START_Y + cur_y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        glow_rect = cursor_rect.inflate(6, 6)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_CURSOR, 50), glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=3)

    def _render_ui(self):
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT_UI)
        self.screen.blit(score_surf, (15, 10))

        steps_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        steps_surf = self.font_ui.render(steps_text, True, self.COLOR_TEXT_UI)
        steps_rect = steps_surf.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(steps_surf, steps_rect)
        
        if self.game_over:
            msg_text = "YOU WIN!" if self.win_state else "GAME OVER"
            msg_color = self.COLOR_TEXT_WIN if self.win_state else self.COLOR_TEXT_LOSE
            msg_surf = self.font_msg.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            bg_rect = msg_rect.inflate(40, 20)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((*self.COLOR_BG, 200))
            self.screen.blit(bg_surf, bg_rect)
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "win": self.win_state,
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

# Example usage for testing
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a display for visualization
    pygame.display.set_caption("Minesweeper Gym Env")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    print(env.user_guide)
    print(env.game_description)

    running = True
    terminated = False
    
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if terminated: # If game is over, any key resets
                    terminated = False
                    obs, info = env.reset()
                    break

                # Translate pygame keys to our action space
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                
                if event.key == pygame.K_SPACE: action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1

                # Since auto_advance is False, we only step on a key press
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.1f}, Score: {info['score']:.1f}, Terminated: {terminated}")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    env.close()