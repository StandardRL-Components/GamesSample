
# Generated: 2025-08-28T04:15:24.727495
# Source Brief: brief_02261.md
# Brief Index: 2261

        
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
    A Minesweeper-style game where the agent must clear a grid of safe squares
    while avoiding hidden mines. The game is presented in a clean, minimalist style.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: ↑↓←→ to move the cursor. Press space to reveal a square."
    )

    # User-facing description of the game
    game_description = (
        "Navigate a grid, revealing safe squares while avoiding hidden mines to clear the board as quickly as possible."
    )

    # Game is turn-based, so it only advances on action.
    auto_advance = False
    
    # --- Constants ---
    GRID_SIZE = 9
    NUM_MINES = 10
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID_LINES = (60, 60, 70)
    COLOR_UNREVEALED = (45, 45, 55)
    COLOR_REVEALED_SAFE = (70, 80, 100)
    COLOR_REVEALED_MINE = (200, 50, 50)
    COLOR_CURSOR = (100, 255, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_TEXT = (200, 200, 200)
    NUMBER_COLORS = [
        COLOR_REVEALED_SAFE,  # 0 is not shown
        (50, 150, 255),  # 1
        (80, 200, 80),   # 2
        (255, 80, 80),   # 3
        (150, 80, 200),  # 4
        (200, 150, 50),  # 5
        (50, 200, 200),  # 6
        (200, 50, 200),  # 7
        (100, 100, 100)  # 8
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_grid = pygame.font.Font(None, 36)
        self.font_ui = pygame.font.Font(None, 28)
        
        # Game state variables
        self.cursor_pos = [0, 0]
        self.mine_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.revealed_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.number_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None

        # Grid rendering properties
        self.grid_area_size = min(self.SCREEN_WIDTH, self.SCREEN_HEIGHT) * 0.9
        self.cell_size = self.grid_area_size / self.GRID_SIZE
        self.grid_start_x = (self.SCREEN_WIDTH - self.grid_area_size) / 2
        self.grid_start_y = (self.SCREEN_HEIGHT - self.grid_area_size) / 2

        # Initialize state
        self.reset()

        # Run validation check
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Center the cursor
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        
        # Generate mine locations
        self.revealed_grid.fill(False)
        self.mine_grid.fill(False)
        
        flat_indices = np.arange(self.GRID_SIZE * self.GRID_SIZE)
        self.np_random.shuffle(flat_indices)
        mine_indices = flat_indices[:self.NUM_MINES]
        
        rows, cols = np.unravel_index(mine_indices, (self.GRID_SIZE, self.GRID_SIZE))
        self.mine_grid[rows, cols] = True
        
        # Pre-calculate numbers for all cells
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                self.number_grid[r, c] = self._calculate_adjacent_mines(r, c)
                
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_pressed = action[1] == 1

        reward = 0
        terminated = False

        # 1. Handle movement
        if movement == 1:  # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 2:  # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE
        elif movement == 3:  # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 4:  # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE

        # 2. Handle reveal action
        if space_pressed:
            r, c = self.cursor_pos
            if self.revealed_grid[r, c]:
                reward = -0.1  # Penalty for revealing an already revealed square
            else:
                if self.mine_grid[r, c]:
                    # Revealed a mine - Game Over
                    reward = -100
                    self.game_over = True
                    terminated = True
                    self.revealed_grid[r, c] = True # Show the mine
                    # SFX: Explosion
                else:
                    # Revealed a safe square
                    if self.number_grid[r, c] == 0:
                        # Flood fill for 0s
                        revealed_count = self._flood_fill(r, c)
                        reward = float(revealed_count) # +1 for each new square
                        # SFX: Multiple soft clicks
                    else:
                        self.revealed_grid[r, c] = True
                        reward = 1.0 # +1 for a single square
                        # SFX: Single click
        
        # 3. Check for win condition
        if not self.game_over:
            safe_squares = self.GRID_SIZE * self.GRID_SIZE - self.NUM_MINES
            if np.sum(self.revealed_grid) == safe_squares:
                reward += 100 # Win bonus
                self.game_over = True
                terminated = True
                # SFX: Victory fanfare
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True # End due to step limit
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _flood_fill(self, r, c):
        """Recursively reveals empty areas."""
        q = [(r, c)]
        visited = set(q)
        revealed_count = 0
        
        while q:
            curr_r, curr_c = q.pop(0)
            
            if not self.revealed_grid[curr_r, curr_c]:
                self.revealed_grid[curr_r, curr_c] = True
                revealed_count += 1

            if self.number_grid[curr_r, curr_c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        
                        nr, nc = curr_r + dr, curr_c + dc
                        
                        if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and (nr, nc) not in visited:
                            q.append((nr, nc))
                            visited.add((nr, nc))
        return revealed_count

    def _calculate_adjacent_mines(self, r, c):
        if self.mine_grid[r, c]:
            return -1 # Represents a mine
        
        mine_count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and self.mine_grid[nr, nc]:
                    mine_count += 1
        return mine_count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid cells
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.grid_start_x + c * self.cell_size,
                    self.grid_start_y + r * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                if self.revealed_grid[r, c]:
                    if self.mine_grid[r, c]:
                        pygame.draw.rect(self.screen, self.COLOR_REVEALED_MINE, rect)
                        # Draw mine symbol
                        cx, cy = rect.center
                        pygame.draw.circle(self.screen, self.COLOR_TEXT, (cx, cy), self.cell_size * 0.25)
                    else:
                        pygame.draw.rect(self.screen, self.COLOR_REVEALED_SAFE, rect)
                        num = self.number_grid[r, c]
                        if num > 0:
                            num_color = self.NUMBER_COLORS[num]
                            text_surf = self.font_grid.render(str(num), True, num_color)
                            text_rect = text_surf.get_rect(center=rect.center)
                            self.screen.blit(text_surf, text_rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, rect)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.grid_start_x + i * self.cell_size, self.grid_start_y)
            end_pos = (self.grid_start_x + i * self.cell_size, self.grid_start_y + self.grid_area_size)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos, 2)
            # Horizontal lines
            start_pos = (self.grid_start_x, self.grid_start_y + i * self.cell_size)
            end_pos = (self.grid_start_x + self.grid_area_size, self.grid_start_y + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos, 2)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_start_x + cursor_c * self.cell_size,
            self.grid_start_y + cursor_r * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        score_text = f"Score: {self.score:.1f}"
        text_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))

        steps_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        text_surf = self.font_ui.render(steps_text, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "game_over": self.game_over,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Use a different screen for display to not interfere with the headless one
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Minesweeper Gym Environment")
    
    done = False
    clock = pygame.time.Clock()
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # Default action is a no-op
        action = [0, 0, 0] # [movement, space, shift]

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
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    print("Game Reset.")
        
        # Only step if an action was taken
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.1f}, Terminated: {terminated}, Info: {info}")
            if terminated:
                print("--- GAME OVER ---")
                print("Press 'R' to reset.")

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()