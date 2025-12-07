# Generated: 2025-08-27T14:23:33.286147
# Source Brief: brief_00670.md
# Brief Index: 670


import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Minesweeper-style game environment for Gymnasium.

    The player controls a cursor on a grid of hidden tiles. The goal is to
    reveal all tiles that do not contain mines. Revealing a mine ends the game.
    Revealed tiles show the number of adjacent mines, providing clues for the
    player's next move.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to reveal a tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic puzzle game. Navigate a grid and reveal tiles to find all the "
        "safe squares without hitting a mine."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 8
    NUM_MINES = 10
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (30, 30, 40)
    COLOR_TILE_HIDDEN = (100, 100, 110)
    COLOR_TILE_REVEALED = (60, 70, 60)
    COLOR_GRID = (120, 120, 130)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_MINE = (255, 50, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    COLOR_NUMBERS = [
        (0, 0, 0, 0),  # 0 is transparent
        (0, 150, 255),  # 1
        (0, 200, 0),  # 2
        (255, 100, 0),  # 3
        (0, 0, 200),  # 4
        (150, 0, 0),  # 5
        (0, 150, 150),  # 6
        (100, 0, 100),  # 7
        (50, 50, 50),  # 8
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_tile = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)

        # Grid layout calculation
        self.tile_size = 40
        self.grid_width = self.GRID_SIZE * self.tile_size
        self.grid_height = self.GRID_SIZE * self.tile_size
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2 + 20

        # Initialize state variables
        self.grid_mines = None
        self.grid_revealed = None
        self.grid_numbers = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.revealed_safe_tiles = 0

        # The environment state must be initialized before the validation check.
        # reset() sets up the random number generator and generates the initial grid.
        self.reset()

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.revealed_safe_tiles = 0
        self.cursor_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])

        self._generate_grid()

        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        """Creates a new minefield."""
        self.grid_mines = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.grid_revealed = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self.grid_numbers = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)

        # Place mines
        mine_indices = self.np_random.choice(self.GRID_SIZE * self.GRID_SIZE, self.NUM_MINES, replace=False)
        mine_coords = np.unravel_index(mine_indices, (self.GRID_SIZE, self.GRID_SIZE))
        self.grid_mines[mine_coords] = True
        assert np.sum(self.grid_mines) == self.NUM_MINES, "Incorrect number of mines placed"

        # Calculate adjacent mine counts
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if self.grid_mines[x, y]:
                    continue
                count = 0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        nx, ny = x + i, y + j
                        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.grid_mines[nx, ny]:
                            count += 1
                self.grid_numbers[x, y] = count

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_press = action[1] == 1
        # shift_held is unused

        reward = 0
        terminated = False

        # 1. Handle cursor movement
        prev_cursor_pos = self.cursor_pos.copy()
        if movement == 1:  # Up
            self.cursor_pos[1] -= 1
        elif movement == 2:  # Down
            self.cursor_pos[1] += 1
        elif movement == 3:  # Left
            self.cursor_pos[0] -= 1
        elif movement == 4:  # Right
            self.cursor_pos[0] += 1

        # Wrap cursor around grid edges
        self.cursor_pos[0] %= self.GRID_SIZE
        self.cursor_pos[1] %= self.GRID_SIZE

        # 2. Handle reveal action
        if space_press:
            reward = self._reveal_tile(self.cursor_pos[0], self.cursor_pos[1])

        self.score += reward

        # 3. Check for win condition
        total_safe_tiles = self.GRID_SIZE ** 2 - self.NUM_MINES
        if not self.game_over and self.revealed_safe_tiles == total_safe_tiles:
            self.win = True
            self.game_over = True
            win_reward = 100
            reward += win_reward
            self.score += win_reward

        # 4. Check for termination
        self.steps += 1
        if self.game_over or self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _reveal_tile(self, x, y):
        """Logic for revealing a single tile, returns reward."""
        if self.grid_revealed[x, y]:
            return -0.1  # Penalty for re-clicking

        self.grid_revealed[x, y] = True

        if self.grid_mines[x, y]:
            # Game over - Hit a mine
            # sound: explosion
            self.game_over = True
            return -100  # Loss penalty

        # Revealed a safe tile
        # sound: click
        self.revealed_safe_tiles += 1

        # If the tile is empty (0 adjacent mines), flood fill
        if self.grid_numbers[x, y] == 0:
            self._flood_fill(x, y)

        return 1.0  # Reward for revealing a safe tile

    def _flood_fill(self, start_x, start_y):
        """Recursively reveal empty areas connected to the start tile."""
        q = deque([(start_x, start_y)])
        visited = {(start_x, start_y)}

        while q:
            x, y = q.popleft()

            # Reveal neighbors
            for i in range(-1, 2):
                for j in range(-1, 2):
                    nx, ny = x + i, y + j

                    if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and not self.grid_revealed[nx, ny] and (
                    nx, ny) not in visited:
                        if not self.grid_mines[nx,ny]: # Make sure not to auto-reveal mines
                            self.revealed_safe_tiles += 1
                        self.grid_revealed[nx, ny] = True

                        if self.grid_numbers[nx, ny] == 0:
                            q.append((nx, ny))
                            visited.add((nx, ny))

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the grid, tiles, and cursor."""
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.tile_size,
                    self.grid_offset_y + y * self.tile_size,
                    self.tile_size,
                    self.tile_size
                )

                if self.grid_revealed[x, y]:
                    pygame.draw.rect(self.screen, self.COLOR_TILE_REVEALED, rect)
                    if self.grid_mines[x, y]:
                        # Draw a mine
                        pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, self.tile_size // 4,
                                                self.COLOR_MINE)
                        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, self.tile_size // 4,
                                                     self.COLOR_MINE)
                    elif self.grid_numbers[x, y] > 0:
                        # Draw number
                        num = self.grid_numbers[x, y]
                        color = self.COLOR_NUMBERS[num]
                        text_surf = self.font_tile.render(str(num), True, color)
                        text_rect = text_surf.get_rect(center=rect.center)
                        self.screen.blit(text_surf, text_rect)
                else:
                    # Draw hidden tile
                    pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, rect)

                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.tile_size,
            self.grid_offset_y + self.cursor_pos[1] * self.tile_size,
            self.tile_size,
            self.tile_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)  # Thicker border for cursor

    def _render_ui(self):
        """Renders the score, steps, and game over messages."""

        # Helper to draw text with a shadow
        def draw_text(text, font, color, pos, shadow_color, shadow_offset=(2, 2)):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Draw score and steps
        safe_tiles_total = self.GRID_SIZE ** 2 - self.NUM_MINES
        score_text = f"Revealed: {self.revealed_safe_tiles} / {safe_tiles_total}"
        steps_text = f"Steps: {self.steps} / {self.MAX_STEPS}"
        draw_text(score_text, self.font_ui, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)
        draw_text(steps_text, self.font_ui, self.COLOR_TEXT, (self.SCREEN_WIDTH - 180, 10), self.COLOR_TEXT_SHADOW)

        # Draw Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            if self.win:
                message = "YOU WIN!"
                color = (100, 255, 100)
            else:
                message = "GAME OVER"
                color = self.COLOR_MINE

            draw_text(message, self.font_game_over, color,
                      (self.SCREEN_WIDTH // 2 - self.font_game_over.size(message)[0] // 2,
                       self.SCREEN_HEIGHT // 2 - 50), self.COLOR_TEXT_SHADOW)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos.tolist(),
            "revealed_safe_tiles": self.revealed_safe_tiles,
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
    # This block allows you to play the game directly
    # The environment must be created with render_mode="rgb_array" for the main block to work
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Create a window to display the game
    pygame.display.set_caption("Minesweeper Gym Environment")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    running = True
    while running:
        # Human input mapping
        movement = 0  # no-op
        space_press = 0

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
                    space_press = 1
                elif event.key == pygame.K_r:  # Reset on 'r'
                    obs, info = env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # Only step if an action was taken
        if movement != 0 or space_press != 0:
            action = [movement, space_press, 0]  # shift is unused
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.1f}, Terminated: {terminated}, Info: {info}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)  # Limit FPS

    env.close()