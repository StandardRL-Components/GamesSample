
# Generated: 2025-08-27T12:49:21.189019
# Source Brief: brief_00172.md
# Brief Index: 172

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a Minesweeper-style puzzle game.

    The player navigates a 9x9 grid with a cursor and reveals tiles. The goal is to
    reveal all tiles that do not contain mines. Revealing a mine ends the game.
    Revealing a safe tile shows a number indicating how many of its 8 neighbors
    are mines. Revealing a tile with 0 adjacent mines will automatically reveal
    all of its neighbors as well (a flood fill).

    The environment prioritizes visual clarity and a clean, responsive feel,
    making it suitable for both human players and reinforcement learning agents.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to reveal the selected tile."
    )

    # Short, user-facing description of the game
    game_description = (
        "A classic puzzle game of logic and risk. Navigate a grid and reveal tiles while avoiding "
        "hidden mines. Uncover all the safe tiles to win."
    )

    # The game is turn-based, so it should only advance on an action.
    auto_advance = False

    # --- Constants ---
    GRID_SIZE = 9
    NUM_MINES = 10
    MAX_STEPS = 1000

    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_LINES = (40, 60, 80)
    COLOR_TILE_HIDDEN = (60, 80, 100)
    COLOR_TILE_REVEALED = (90, 110, 130)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_TEXT = (230, 240, 255)
    COLOR_MINE = (255, 80, 80)
    COLOR_GAMEOVER_TINT = (0, 0, 0, 180) # Semi-transparent black

    # Map mine counts to colors for visual distinction
    NUMBER_COLORS = {
        1: (100, 150, 255),  # Blue
        2: (100, 220, 100),  # Green
        3: (255, 100, 100),  # Red
        4: (150, 100, 255),  # Purple
        5: (255, 150, 50),   # Orange
        6: (50, 200, 200),   # Cyan
        7: (220, 220, 50),   # Yellow
        8: (200, 200, 200),  # Light Grey
    }


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Fonts and Pre-rendered Surfaces ---
        self.ui_font = pygame.font.SysFont("Consolas", 24, bold=True)
        self.tile_font = pygame.font.SysFont("Consolas", 20, bold=True)
        self.game_over_font = pygame.font.SysFont("Verdana", 48, bold=True)
        self._pre_render_numbers()

        # --- Game State Initialization ---
        self.grid_solution = None
        self.grid_visible = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.safe_tiles_remaining = 0

        # --- Grid Layout Calculation ---
        self.tile_size = 36
        self.grid_width = self.GRID_SIZE * self.tile_size
        self.grid_height = self.GRID_SIZE * self.tile_size
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2

        self.reset()
        
        # self.validate_implementation() # Uncomment for debugging

    def _pre_render_numbers(self):
        """Pre-renders number surfaces for faster drawing."""
        self.number_surfaces = {}
        for num, color in self.NUMBER_COLORS.items():
            self.number_surfaces[num] = self.tile_font.render(str(num), True, color)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._initialize_grid()
        self.cursor_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.safe_tiles_remaining = (self.GRID_SIZE * self.GRID_SIZE) - self.NUM_MINES

        return self._get_observation(), self._get_info()

    def _initialize_grid(self):
        """Creates the minefield solution and visibility grids."""
        # -1 for mine, 0-8 for safe tiles
        self.grid_solution = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        # 0 for hidden, 1 for revealed
        self.grid_visible = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Place mines
        mine_indices = self.np_random.choice(self.GRID_SIZE * self.GRID_SIZE, self.NUM_MINES, replace=False)
        mine_coords = np.unravel_index(mine_indices, (self.GRID_SIZE, self.GRID_SIZE))
        self.grid_solution[mine_coords] = -1

        # Calculate neighbor counts
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid_solution[r, c] == -1:
                    continue
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and self.grid_solution[nr, nc] == -1:
                            count += 1
                self.grid_solution[r, c] = count

    def step(self, action):
        reward = 0.0
        self.game_over = False # Reset on each step, will be set by logic if needed

        # --- Unpack and Process Actions ---
        movement, space_pressed, _ = action
        space_pressed = space_pressed == 1

        # 1. Handle cursor movement
        if movement == 1:  # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_SIZE
        elif movement == 2:  # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE
        elif movement == 3:  # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_SIZE
        elif movement == 4:  # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE

        # 2. Handle tile reveal action
        if space_pressed:
            r, c = self.cursor_pos
            # Only act if the tile is hidden
            if self.grid_visible[r, c] == 0:
                reward += self._reveal_tile(r, c)

        # 3. Check for win condition
        if self.safe_tiles_remaining == 0 and not self.game_over:
            self.win = True
            self.game_over = True
            reward += 100.0
            self.score += 100

        self.steps += 1
        terminated = self._check_termination()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _reveal_tile(self, r, c):
        """Reveals a single tile and handles consequences. Returns reward."""
        # Base case: already revealed
        if self.grid_visible[r, c] == 1:
            return 0.0

        self.grid_visible[r, c] = 1
        tile_value = self.grid_solution[r, c]

        if tile_value == -1:  # Hit a mine
            # sfx: explosion
            self.game_over = True
            self.win = False
            return -100.0
        else:
            # sfx: click_safe
            self.safe_tiles_remaining -= 1
            self.score += 1
            
            reward = 1.0
            # Per brief: -0.2 for revealing a tile with 0 adjacent mines
            if tile_value == 0:
                reward -= 0.2
                # Flood fill for zero-value tiles
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                            reward += self._reveal_tile(nr, nc)
            return reward


    def _check_termination(self):
        """Check if the episode should end."""
        return self.game_over or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": tuple(self.cursor_pos),
            "safe_tiles_remaining": self.safe_tiles_remaining,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the game grid, tiles, and cursor."""
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.grid_offset_x + c * self.tile_size,
                    self.grid_offset_y + r * self.tile_size,
                    self.tile_size,
                    self.tile_size,
                )

                # Draw tile background
                is_revealed = self.grid_visible[r, c] == 1 or (self.game_over and self.grid_solution[r, c] == -1)
                tile_color = self.COLOR_TILE_REVEALED if is_revealed else self.COLOR_TILE_HIDDEN
                pygame.draw.rect(self.screen, tile_color, rect)

                # Draw tile content if revealed
                if is_revealed:
                    value = self.grid_solution[r, c]
                    if value == -1: # Mine
                        # Draw a stylized mine
                        center = rect.center
                        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.tile_size // 3, self.COLOR_MINE)
                        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.tile_size // 3, self.COLOR_MINE)
                    elif value > 0: # Number
                        num_surf = self.number_surfaces[value]
                        num_rect = num_surf.get_rect(center=rect.center)
                        self.screen.blit(num_surf, num_rect)

        # Draw grid lines over tiles
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.grid_offset_x + i * self.tile_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + i * self.tile_size, self.grid_offset_y + self.grid_height)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.grid_offset_x, self.grid_offset_y + i * self.tile_size)
            end_pos = (self.grid_offset_x + self.grid_width, self.grid_offset_y + i * self.tile_size)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos, 1)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_offset_x + cursor_c * self.tile_size,
            self.grid_offset_y + cursor_r * self.tile_size,
            self.tile_size,
            self.tile_size,
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        """Renders score and step count."""
        score_text = f"SCORE: {self.score}"
        score_surf = self.ui_font.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 10))

        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        steps_surf = self.ui_font.render(steps_text, True, self.COLOR_TEXT)
        steps_rect = steps_surf.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(steps_surf, steps_rect)

    def _render_game_over_screen(self):
        """Renders the semi-transparent overlay and game over text."""
        tint_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        tint_surface.fill(self.COLOR_GAMEOVER_TINT)
        self.screen.blit(tint_surface, (0, 0))

        if self.win:
            message = "YOU WIN!"
            color = (100, 255, 100)
        else:
            message = "GAME OVER"
            color = self.COLOR_MINE

        text_surf = self.game_over_font.render(message, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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
        assert self.grid_solution is not None
        assert np.sum(self.grid_solution == -1) == self.NUM_MINES
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set up Pygame for human play
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use 'dummy' for headless, 'x11' or 'windows' for display

    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Play Loop ---
    # env.validate_implementation() # Run validation
    
    obs, info = env.reset()
    done = False
    
    # Set up display window
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Minesweeper Gym Environment")
    
    running = True
    while running:
        # Convert observation back to Pygame surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if done:
            print(f"Game Over! Final Info: {info}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False
            continue

        # Map keyboard inputs to actions
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
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
                
                # If any key was pressed, step the environment
                if any(a != 0 for a in action):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

    env.close()