import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to draw a path over matching blocks. "
        "Press Space to clear the path. Press Shift to select a new starting block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Connect lines of matching colored blocks to clear them from the board. "
        "Clear 15 lines to win, but don't let the board fill up!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_COLS = 12
        self.GRID_ROWS = 8
        self.BLOCK_SIZE = 40
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_COLS * self.BLOCK_SIZE) // 2
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_ROWS * self.BLOCK_SIZE) // 2
        self.BORDER_RADIUS = 8
        self.LINE_WIDTH = 8
        self.WIN_CONDITION = 15
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_BG = (30, 45, 60)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_TEXT_SHADOW = (10, 15, 20)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLORS = [
            (239, 83, 80),  # Red
            (3, 169, 244),  # Blue
            (139, 195, 74),  # Green
            (255, 235, 59),  # Yellow
            (156, 39, 176),  # Purple
            (0, 188, 212),  # Cyan
            (255, 152, 0),  # Orange
            (233, 30, 99),  # Pink
        ]

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
        self.font_main = pygame.font.SysFont("sans-serif", 24, bold=True)
        self.font_large = pygame.font.SysFont("sans-serif", 48, bold=True)

        # State variables (initialized in reset)
        self.grid = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.drag_path = []
        self.cursor_pos = None
        self.np_random = None
        self.blocks_to_clear_anim = []

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.win = False
        self.drag_path = []
        self.cursor_pos = None
        self.blocks_to_clear_anim = []

        while True:
            self._create_grid()
            if self._check_for_valid_moves():
                break

        self._select_new_start_block()
        if not self.drag_path:
            self.game_over = True

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # After the game is over, reset to a new game on the next step
            obs, info = self.reset()
            return obs, 0, True, False, info

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1
        self.blocks_to_clear_anim = []

        if shift_held:
            # sound effect: reset_selection.wav
            self._select_new_start_block()

        elif space_held:
            if len(self.drag_path) > 1:
                # sound effect: clear_success.wav
                num_cleared = len(self.drag_path)
                reward += num_cleared
                reward += 10
                self.score += num_cleared
                self.lines_cleared += 1

                self.blocks_to_clear_anim = list(self.drag_path)
                for x, y in self.drag_path:
                    self.grid[y][x] = -1

                self._apply_gravity_and_refill()
                while not self._check_for_valid_moves():
                    self._create_grid() # Ensure new grid has moves
                self._select_new_start_block()
            else:
                # sound effect: clear_fail.wav
                reward -= 0.1
                self._select_new_start_block()

        elif movement > 0 and self.drag_path:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            curr_x, curr_y = self.cursor_pos
            target_x, target_y = curr_x + dx, curr_y + dy

            if 0 <= target_x < self.GRID_COLS and 0 <= target_y < self.GRID_ROWS:
                start_color_idx = self.grid[self.drag_path[0][1]][self.drag_path[0][0]]
                if start_color_idx != -1:
                    target_color_idx = self.grid[target_y][target_x]
                    if target_color_idx == start_color_idx and (target_x, target_y) not in self.drag_path:
                        # sound effect: select_block.wav
                        self.drag_path.append((target_x, target_y))
                        self.cursor_pos = (target_x, target_y)
                    # else: sound effect: bump.wav

        terminated = False
        if self.lines_cleared >= self.WIN_CONDITION:
            reward += 100
            self.game_over = True
            self.win = True
            terminated = True
        elif not self._check_for_valid_moves():
            reward -= 50
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_grid(self):
        self.grid = self.np_random.integers(len(self.COLORS), size=(self.GRID_ROWS, self.GRID_COLS)).tolist()

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_COLS):
            col = [self.grid[y][x] for y in range(self.GRID_ROWS) if self.grid[y][x] != -1]
            num_new = self.GRID_ROWS - len(col)
            new_blocks = self.np_random.integers(len(self.COLORS), size=num_new).tolist()
            new_col = new_blocks + col
            for y in range(self.GRID_ROWS):
                self.grid[y][x] = new_col[y]

    def _find_valid_starts(self):
        starts = []
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                color = self.grid[y][x]
                if color == -1: continue
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                        if self.grid[ny][nx] == color:
                            starts.append((x, y))
                            break
        return starts

    def _select_new_start_block(self):
        valid_starts = self._find_valid_starts()
        if not valid_starts:
            self.drag_path = []
            self.cursor_pos = None
            return

        start_pos = valid_starts[self.np_random.integers(len(valid_starts))]
        self.cursor_pos = start_pos
        self.drag_path = [start_pos]

    def _check_for_valid_moves(self):
        return len(self._find_valid_starts()) > 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET,
                                self.GRID_COLS * self.BLOCK_SIZE, self.GRID_ROWS * self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=self.BORDER_RADIUS)

        # Draw blocks
        if self.grid is not None:
            for y in range(self.GRID_ROWS):
                for x in range(self.GRID_COLS):
                    color_idx = self.grid[y][x]
                    if color_idx == -1: continue

                    rect = pygame.Rect(
                        self.GRID_X_OFFSET + x * self.BLOCK_SIZE + 2,
                        self.GRID_Y_OFFSET + y * self.BLOCK_SIZE + 2,
                        self.BLOCK_SIZE - 4,
                        self.BLOCK_SIZE - 4,
                    )

                    if (x, y) in self.blocks_to_clear_anim:
                        # Clear animation
                        pygame.draw.rect(self.screen, (255, 255, 255), rect, border_radius=self.BORDER_RADIUS)
                    else:
                        color = self.COLORS[color_idx]
                        pygame.draw.rect(self.screen, color, rect, border_radius=self.BORDER_RADIUS)

        # Draw drag path highlight and line
        if len(self.drag_path) > 0 and self.grid is not None:
            start_block_y, start_block_x = self.drag_path[0][1], self.drag_path[0][0]
            if self.grid[start_block_y][start_block_x] != -1:
                # Highlight selected blocks
                for x, y in self.drag_path:
                    highlight_rect = pygame.Rect(
                        self.GRID_X_OFFSET + x * self.BLOCK_SIZE,
                        self.GRID_Y_OFFSET + y * self.BLOCK_SIZE,
                        self.BLOCK_SIZE,
                        self.BLOCK_SIZE,
                    )
                    pygame.draw.rect(self.screen, (255, 255, 255), highlight_rect, width=3,
                                     border_radius=self.BORDER_RADIUS + 2)

                # Draw connecting lines
                if len(self.drag_path) > 1:
                    points = []
                    for x, y in self.drag_path:
                        points.append((
                            self.GRID_X_OFFSET + x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2,
                            self.GRID_Y_OFFSET + y * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
                        ))
                    pygame.draw.lines(self.screen, (255, 255, 255, 180), False, points, self.LINE_WIDTH)

        # Draw cursor
        if self.cursor_pos:
            x, y = self.cursor_pos
            cursor_rect = pygame.Rect(
                self.GRID_X_OFFSET + x * self.BLOCK_SIZE,
                self.GRID_Y_OFFSET + y * self.BLOCK_SIZE,
                self.BLOCK_SIZE,
                self.BLOCK_SIZE,
            )
            # Pulsating alpha effect for the cursor
            alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.01)
            cursor_surface = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, alpha), cursor_surface.get_rect(), width=4,
                             border_radius=self.BORDER_RADIUS)
            self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _render_ui(self):
        # Render score
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, (20, 10), self.font_main)

        # Render lines cleared
        lines_text = f"LINES: {self.lines_cleared} / {self.WIN_CONDITION}"
        text_width = self.font_main.size(lines_text)[0]
        self._draw_text(lines_text, (self.SCREEN_WIDTH - text_width - 20, 10), self.font_main)

        # Render game over/win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            message = "YOU WIN!" if self.win else "GAME OVER"
            text_width, text_height = self.font_large.size(message)
            pos = ((self.SCREEN_WIDTH - text_width) // 2, (self.SCREEN_HEIGHT - text_height) // 2)
            self._draw_text(message, pos, self.font_large)

    def _draw_text(self, text, pos, font):
        x, y = pos
        shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surface = font.render(text, True, self.COLOR_TEXT)
        self.screen.blit(shadow_surface, (x + 2, y + 2))
        self.screen.blit(text_surface, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test reset - MUST be called before tests that rely on game state
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)

        # Test observation space (now that grid is initialized)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game with keyboard controls
    env = GameEnv()
    obs, info = env.reset()

    running = True
    terminated = False

    # Create a window to display the game
    pygame.display.set_caption("Block Clear")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    while running:
        # Action defaults
        movement = 0  # none
        space = 0  # released
        shift = 0  # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4

            if keys[pygame.K_SPACE]:
                space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Lines: {info['lines_cleared']}")
                # A small delay to show the final screen before the game might auto-reset
                pygame.time.wait(2000)


        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we need a small delay for human playability
        pygame.time.wait(50)

    pygame.quit()