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
        "Controls: Arrow keys to move cursor. Space to place a color. Shift to cycle colors."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate a hidden pixel art image by filling a grid. You have a limited number of moves. Plan carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_MOVES = 20
        self.GRID_AREA_WIDTH = 360
        self.CELL_SIZE = self.GRID_AREA_WIDTH // self.GRID_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_AREA_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_AREA_WIDTH) // 2

        # --- Colors ---
        self.COLOR_BG = (44, 62, 80)  # Dark Blue-Gray
        self.COLOR_GRID = (52, 73, 94)  # Slightly Lighter Blue-Gray
        self.COLOR_CURSOR = (241, 196, 15)  # Yellow
        self.COLOR_TEXT = (236, 240, 241)  # Light Gray/White
        self.COLOR_TEXT_SHADOW = (40, 40, 40)
        self.COLOR_OUTLINE = (127, 140, 141, 100)  # Semi-transparent Gray for hint

        # Palette: 0=Empty, 1-4 are placeable colors
        self.PALETTE = [
            (0, 0, 0),  # Index 0 is not used for placement
            (231, 76, 60),  # Red
            (46, 204, 113),  # Green
            (52, 152, 219),  # Blue
            (155, 89, 182),  # Purple
        ]
        self.NUM_COLORS = len(self.PALETTE) - 1

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
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Target images (0=empty, 1-4=color index)
        self.target_patterns = self._create_patterns()

        # Initialize state variables
        self.player_grid = None
        self.target_image = None
        self.cursor_pos = None
        self.selected_color_idx = None
        self.remaining_moves = None
        self.score = None
        self.game_over = None
        self.steps = None
        self.last_space_held = None
        self.last_shift_held = None
        self.particles = None
        self.completed_rows = None
        self.completed_cols = None
        self.win_state = None

        # Initialize state
        # self.reset() is called by the test harness, no need to call it here.

    def _create_patterns(self):
        # Smiley Face
        smiley = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        smiley[2, 2] = smiley[2, 7] = 1  # Eyes
        smiley[3, 2] = smiley[3, 7] = 1
        smiley[5, 2:8] = 1
        smiley[6, 3:7] = 1
        smiley[7, 4:6] = 1

        # Heart
        heart = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        heart[2, [2, 3, 6, 7]] = 2
        heart[3, [1, 2, 3, 4, 5, 6, 7, 8]] = 2
        heart[4, 1:9] = 2
        heart[5, 2:8] = 2
        heart[6, 3:7] = 2
        heart[7, 4:6] = 2

        # Spaceship
        ship = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        ship[2, 4] = 3
        ship[3, 3:6] = 3
        ship[4, 2:7] = 3
        ship[5, 1:8] = 4
        ship[6, [2, 6]] = 4
        ship[7, [3, 5]] = 1

        return [smiley, heart, ship]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.remaining_moves = self.MAX_MOVES
        self.player_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.target_image = self.target_patterns[self.np_random.integers(0, len(self.target_patterns))].copy()
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_color_idx = 0  # Index for self.PALETTE[1:]
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.completed_rows = set()
        self.completed_cols = set()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Handle Actions ---
        if not self.game_over:
            # 1. Cursor Movement
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            self.cursor_pos[0] %= self.GRID_SIZE
            self.cursor_pos[1] %= self.GRID_SIZE

            # 2. Cycle Color (on press)
            if shift_held and not self.last_shift_held:
                self.selected_color_idx = (self.selected_color_idx + 1) % self.NUM_COLORS
                # Sound: UI_switch.wav

            # 3. Place Color (on press)
            if space_held and not self.last_space_held and self.remaining_moves > 0:
                cx, cy = self.cursor_pos
                if self.player_grid[cy, cx] == 0:  # Can only place on empty squares
                    self.remaining_moves -= 1
                    placed_color = self.selected_color_idx + 1
                    target_color = self.target_image[cy, cx]

                    self.player_grid[cy, cx] = placed_color

                    # Reward for correct/incorrect placement
                    if placed_color == target_color:
                        reward += 1
                        self.score += 10
                        # Sound: place_correct.wav
                    else:
                        reward -= 1
                        self.score -= 5
                        # Sound: place_incorrect.wav

                    self._create_particles(cx, cy, self.PALETTE[placed_color])

                    # Check for row/column completion reward
                    reward += self._check_line_completion(cx, cy)

        # --- Update State & Check Termination ---
        self.steps += 1

        is_win = np.array_equal(self.player_grid, self.target_image)
        is_loss = self.remaining_moves <= 0 and not is_win
        terminated = is_win or is_loss

        if terminated and not self.game_over:
            self.game_over = True
            if is_win:
                reward += 100
                self.score += 1000
                self.win_state = True
                # Sound: win.wav
            if is_loss:
                reward -= 50
                self.score -= 500
                # Sound: lose.wav

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _check_line_completion(self, x, y):
        reward = 0
        # Check column
        if y not in self.completed_rows:
            if np.array_equal(self.player_grid[y, :], self.target_image[y, :]):
                reward += 10
                self.score += 100
                self.completed_rows.add(y)
        # Check row
        if x not in self.completed_cols:
            if np.array_equal(self.player_grid[:, x], self.target_image[:, x]):
                reward += 10
                self.score += 100
                self.completed_cols.add(x)
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_and_draw_particles()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (
                self.GRID_OFFSET_X + i * self.CELL_SIZE,
                self.GRID_OFFSET_Y + self.GRID_AREA_WIDTH,
            )
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (
                self.GRID_OFFSET_X + self.GRID_AREA_WIDTH,
                self.GRID_OFFSET_Y + i * self.CELL_SIZE,
            )
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )

                # Draw target outline hint
                target_color_idx = self.target_image[y, x]
                if target_color_idx > 0:
                    pygame.gfxdraw.rectangle(self.screen, cell_rect, self.COLOR_OUTLINE)

                # Draw player's placed colors
                player_color_idx = self.player_grid[y, x]
                if player_color_idx > 0:
                    color = self.PALETTE[player_color_idx]
                    pygame.draw.rect(self.screen, color, cell_rect.inflate(-2, -2))

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE,
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            surface = font.render(text, True, color)
            self.screen.blit(surface, pos)

        # Top UI: Moves and Score
        draw_text(
            f"Moves: {self.remaining_moves}", self.font_medium, self.COLOR_TEXT, (20, 10)
        )
        score_text = f"Score: {self.score}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_TEXT)
        draw_text(
            score_text,
            self.font_medium,
            self.COLOR_TEXT,
            (self.WIDTH - score_surf.get_width() - 20, 10),
        )

        # Bottom UI: Color Palette
        palette_y = self.HEIGHT - 45
        palette_start_x = self.GRID_OFFSET_X
        for i in range(self.NUM_COLORS):
            color = self.PALETTE[i + 1]
            rect = pygame.Rect(palette_start_x + i * 40, palette_y, 35, 35)
            pygame.draw.rect(self.screen, color, rect)
            if i == self.selected_color_idx:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            message = "COMPLETE!" if self.win_state else "GAME OVER"
            color = (46, 204, 113) if self.win_state else (231, 76, 60)

            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _create_particles(self, grid_x, grid_y, color):
        center_x = self.GRID_OFFSET_X + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        center_y = self.GRID_OFFSET_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.randint(15, 30)  # frames
            size = random.randint(4, 8)
            # Particle: [x, y, vx, vy, current_lifetime, initial_lifetime, initial_size, color]
            self.particles.append([center_x, center_y, vx, vy, lifetime, lifetime, size, color])

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            # p = [x, y, vx, vy, current_lifetime, initial_lifetime, initial_size, color]
            p[0] += p[2]  # x += vx
            p[1] += p[3]  # y += vy
            p[4] -= 1     # current_lifetime -= 1
            if p[4] > 0:
                active_particles.append(p)
                pos = (int(p[0]), int(p[1]))
                
                initial_lifetime = p[5]
                if initial_lifetime > 0:
                    # size shrinks over time
                    size = int(p[6] * (p[4] / initial_lifetime))
                else:
                    size = 0
                
                if size > 0:
                    color = p[7]
                    pygame.draw.rect(self.screen, color, (pos[0], pos[1], size, size))
        self.particles = active_particles

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_moves": self.remaining_moves,
            "cursor_pos": self.cursor_pos,
            "win": self.win_state,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc is False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == "__main__":
    import sys

    # For human play, we need to set up the window
    class HumanGameEnv(GameEnv):
        def __init__(self):
            super().__init__(render_mode="human")
            # The next line is the only change from the base class for human mode.
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Pixel Painter")

        def _get_observation(self):
            # In human mode, we render to the display instead of a surface array
            # The parent method draws everything to self.screen
            super()._get_observation()
            # Update the full display Surface to the screen
            pygame.display.flip()
            # The observation is not used in human mode, so return a dummy one.
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

    env = HumanGameEnv()
    obs, info = env.reset()
    done = False

    print("\n" + "=" * 30)
    print("Pixel Painter - Human Player")
    print(env.game_description)
    print(env.user_guide)
    print("=" * 30 + "\n")

    while not done:
        # Map pygame keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()

        movement = 0  # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Event handling for closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()
                done = False

        env.clock.tick(30)  # Limit to 30 FPS for human play

    print("Game Over!")
    print(f"Final Info: {info}")

    # Keep window open for a bit to see the final state
    end_time = pygame.time.get_ticks() + 3000
    while pygame.time.get_ticks() < end_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        env.clock.tick(30)

    env.close()