
# Generated: 2025-08-28T04:11:09.655376
# Source Brief: brief_05170.md
# Brief Index: 5170

        
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
        "Controls: Arrow keys to move cursor. Space to reveal a tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic mine-sweeping puzzle game. Reveal all safe tiles without hitting a mine."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 8
    GRID_HEIGHT = 6
    NUM_MINES = 10
    MAX_STEPS = 1000

    TILE_SIZE = 50
    GRID_PIXEL_WIDTH = GRID_WIDTH * TILE_SIZE
    GRID_PIXEL_HEIGHT = GRID_HEIGHT * TILE_SIZE
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_PIXEL_WIDTH) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_PIXEL_HEIGHT) // 2 + 20

    COLOR_BG = (20, 20, 30)
    COLOR_GRID_LINE = (50, 50, 60)
    COLOR_HIDDEN = (70, 80, 90)
    COLOR_REVEALED = (110, 120, 130)
    COLOR_CURSOR = (255, 255, 0, 100)
    COLOR_MINE_EXPLOSION = (255, 80, 80)
    COLOR_MINE_ICON = (40, 40, 40)
    COLOR_WIN_OVERLAY = (0, 255, 0, 50)
    COLOR_LOSE_OVERLAY = (255, 0, 0, 50)

    NUMBER_COLORS = {
        1: (100, 180, 255),
        2: (100, 255, 100),
        3: (255, 100, 100),
        4: (150, 100, 255),
        5: (255, 150, 50),
        6: (50, 200, 200),
        7: (220, 220, 220),
        8: (180, 180, 180),
    }

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
        self.clock = pygame.time.Clock()
        self.tile_font = pygame.font.SysFont("Consolas", 32, bold=True)
        self.ui_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.end_font = pygame.font.SysFont("Arial", 64, bold=True)
        
        # Initialize state variables
        self.mine_grid = None
        self.numbers_grid = None
        self.revealed_grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.total_safe_squares = 0
        self.revealed_safe_squares = 0
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.cursor_pos = [0, 0]
        self.revealed_safe_squares = 0
        self._generate_grid()
        
        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        self.mine_grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=bool)
        self.numbers_grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.revealed_grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=bool)
        self.total_safe_squares = self.GRID_WIDTH * self.GRID_HEIGHT - self.NUM_MINES

        # Place mines, ensuring the initial cursor position (0,0) is safe
        mines_placed = 0
        while mines_placed < self.NUM_MINES:
            x = self.np_random.integers(0, self.GRID_WIDTH)
            y = self.np_random.integers(0, self.GRID_HEIGHT)
            if not self.mine_grid[y, x] and not (x == 0 and y == 0):
                self.mine_grid[y, x] = True
                mines_placed += 1

        # Calculate adjacent mine counts
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.mine_grid[y, x]:
                    continue
                count = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.mine_grid[ny, nx]:
                            count += 1
                self.numbers_grid[y, x] = count
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Handle Movement ---
        if movement == 1: # Up
            self.cursor_pos[1] -= 1
        elif movement == 2: # Down
            self.cursor_pos[1] += 1
        elif movement == 3: # Left
            self.cursor_pos[0] -= 1
        elif movement == 4: # Right
            self.cursor_pos[0] += 1
        
        # Wrap cursor around grid
        self.cursor_pos[0] %= self.GRID_WIDTH
        self.cursor_pos[1] %= self.GRID_HEIGHT

        # --- Handle Reveal Action ---
        if space_held:
            cx, cy = self.cursor_pos
            if self.revealed_grid[cy, cx]:
                reward = -0.1 # Penalty for revealing an already-revealed tile
            else:
                # Check for mine
                if self.mine_grid[cy, cx]:
                    # SFX: Explosion
                    self.game_over = True
                    self.win = False
                    reward = -100.0
                    self.revealed_grid[cy, cx] = True # Reveal the mine
                else:
                    # SFX: Click
                    # Safe square
                    num = self.numbers_grid[cy, cx]
                    if num == 0:
                        # Flood fill for empty squares
                        revealed_count = self._flood_fill(cx, cy)
                        reward = float(revealed_count)
                    else:
                        self.revealed_grid[cy, cx] = True
                        self.revealed_safe_squares += 1
                        reward = 1.0
        
        self.score += reward

        # --- Check for Win Condition ---
        if not self.game_over and self.revealed_safe_squares == self.total_safe_squares:
            # SFX: Win Jingle
            self.game_over = True
            self.win = True
            self.score += 100.0
            reward += 100.0

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _flood_fill(self, x, y):
        stack = [(x, y)]
        visited = set()
        count = 0
        while stack:
            cx, cy = stack.pop()
            if not (0 <= cx < self.GRID_WIDTH and 0 <= cy < self.GRID_HEIGHT):
                continue
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            if not self.revealed_grid[cy, cx]:
                self.revealed_grid[cy, cx] = True
                self.revealed_safe_squares += 1
                count += 1
                if self.numbers_grid[cy, cx] == 0:
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dx == 0 and dy == 0:
                                continue
                            stack.append((cx + dx, cy + dy))
        return count
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    self.GRID_X_OFFSET + x * self.TILE_SIZE,
                    self.GRID_Y_OFFSET + y * self.TILE_SIZE,
                    self.TILE_SIZE,
                    self.TILE_SIZE
                )
                
                is_revealed = self.revealed_grid[y, x]
                is_mine = self.mine_grid[y, x]

                if is_revealed:
                    if is_mine:
                        pygame.draw.rect(self.screen, self.COLOR_MINE_EXPLOSION, rect)
                        # Draw mine icon
                        cx, cy = rect.center
                        pygame.draw.circle(self.screen, self.COLOR_MINE_ICON, (cx, cy), self.TILE_SIZE // 4)
                    else:
                        pygame.draw.rect(self.screen, self.COLOR_REVEALED, rect)
                        num = self.numbers_grid[y, x]
                        if num > 0:
                            num_text = self.tile_font.render(str(num), True, self.NUMBER_COLORS[num])
                            text_rect = num_text.get_rect(center=rect.center)
                            self.screen.blit(num_text, text_rect)
                else:
                    # If game is over and won, show all tiles as revealed but safe
                    if self.game_over and self.win:
                         pygame.draw.rect(self.screen, self.COLOR_REVEALED, rect)
                    else:
                        pygame.draw.rect(self.screen, self.COLOR_HIDDEN, rect)

        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_X_OFFSET + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_PIXEL_HEIGHT), 1)
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_Y_OFFSET + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_PIXEL_WIDTH, y), 1)
        
        # Draw cursor
        cursor_rect = pygame.Rect(
            self.GRID_X_OFFSET + self.cursor_pos[0] * self.TILE_SIZE,
            self.GRID_Y_OFFSET + self.cursor_pos[1] * self.TILE_SIZE,
            self.TILE_SIZE,
            self.TILE_SIZE
        )
        cursor_surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        cursor_surface.fill(self.COLOR_CURSOR)
        self.screen.blit(cursor_surface, cursor_rect.topleft)
        pygame.gfxdraw.rectangle(self.screen, cursor_rect, (255, 255, 0))


    def _render_ui(self):
        # Draw score
        score_text = self.ui_font.render(f"Score: {self.score:.1f}", True, (200, 200, 220))
        self.screen.blit(score_text, (10, 10))

        # Draw steps
        steps_text = self.ui_font.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, (200, 200, 220))
        steps_rect = steps_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(steps_text, steps_rect)

        # Draw game over/win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            if self.win:
                overlay.fill(self.COLOR_WIN_OVERLAY)
                end_text_str = "YOU WIN!"
                end_text_color = (150, 255, 150)
            else:
                overlay.fill(self.COLOR_LOSE_OVERLAY)
                end_text_str = "GAME OVER"
                end_text_color = (255, 150, 150)
            
            self.screen.blit(overlay, (0, 0))
            end_text = self.end_font.render(end_text_str, True, end_text_color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 100))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": tuple(self.cursor_pos),
            "win": self.win,
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Example ---
    # To play manually, you need a different setup that captures keyboard events.
    # The following is a simple loop for an agent.
    
    print(f"Description: {env.game_description}")
    print(f"Controls: {env.user_guide}")
    
    done = False
    total_reward = 0
    step_count = 0
    
    # For visualization
    pygame.display.set_caption("MineSweeper Gym Environment")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    # Game loop for visualization
    running = True
    while running:
        action = env.action_space.sample() # Replace with your agent's action
        
        # --- Pygame event handling for manual control (optional) ---
        # This part is for human play, not for agent training.
        # To use, comment out `action = env.action_space.sample()`
        # and uncomment the block below.
        
        # move_action = 0
        # space_action = 0
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         running = False
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_UP: move_action = 1
        #         elif event.key == pygame.K_DOWN: move_action = 2
        #         elif event.key == pygame.K_LEFT: move_action = 3
        #         elif event.key == pygame.K_RIGHT: move_action = 4
        #         elif event.key == pygame.K_SPACE: space_action = 1
        #         elif event.key == pygame.K_r: # Reset on 'r'
        #             obs, info = env.reset()
        #             total_reward = 0
        #             step_count = 0
        #             done = False
        
        # action = [move_action, space_action, 0]

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print(f"Episode finished in {step_count} steps. Total reward: {total_reward:.2f}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            step_count = 0
            done = False
        
        # Since auto_advance is False, we only need a small delay for visualization
        pygame.time.wait(100) # Control speed of random agent

    env.close()