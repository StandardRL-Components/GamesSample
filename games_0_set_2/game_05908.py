import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to paint adjacent empty squares."
    )

    # User-facing description of the game
    game_description = (
        "A fast-paced puzzle game. Fill the 8x8 grid with a single color before time runs out."
    )

    # Frames auto-advance for real-time timer
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 8
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2

        # Time and step limits
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.FPS * self.TIME_LIMIT_SECONDS

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
        self.UI_FONT = pygame.font.Font(None, 36)
        self.UI_FONT_SMALL = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID_LINE = (50, 55, 70)
        self.COLOR_EMPTY = (35, 40, 52)
        self.COLORS = [
            self.COLOR_EMPTY,
            (231, 76, 60),  # Red
            (46, 204, 113),  # Green
            (52, 152, 219),  # Blue
            (241, 196, 15),  # Yellow
        ]
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_CURSOR = (255, 255, 255)

        # Particle effect settings
        self.PARTICLE_LIFETIME = 15  # frames

        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.score = 0
        self.steps = 0
        self.timer = 0
        self.game_over = False
        self.particles = []
        self.completed_4x4_origins = set()
        self.rng = np.random.default_rng()

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.score = 0
        self.steps = 0
        self.timer = self.MAX_STEPS
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.particles = []
        self.completed_4x4_origins = set()

        self._initialize_grid()

        return self._get_observation(), self._get_info()

    def _initialize_grid(self):
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        num_starters = self.rng.integers(4, 7)
        for _ in range(num_starters):
            r, c = self.rng.integers(0, self.GRID_SIZE, size=2)
            color_idx = self.rng.integers(1, len(self.COLORS))
            self.grid[r, c] = color_idx

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1

        self.steps += 1
        self.timer -= 1
        reward = 0.0

        self._move_cursor(movement)

        prev_filled_count = np.count_nonzero(self.grid)
        grid_changed = False

        if space_pressed:
            grid_changed = self._perform_paint_action()

        if grid_changed:
            current_filled_count = np.count_nonzero(self.grid)
            newly_filled_count = current_filled_count - prev_filled_count
            reward += float(newly_filled_count)  # +1 per new square

            bonus_reward = self._check_4x4_bonus()
            reward += bonus_reward
        elif space_pressed:
            reward -= 0.2  # Penalty for ineffective click

        self.score += reward

        terminated = False
        is_victory = self._check_victory()

        if self.timer <= 0:
            terminated = True
            if not is_victory:
                reward -= 50  # Time out penalty

        if is_victory:
            terminated = True
            reward += 100  # Victory bonus
            # Add to score as well to reflect in UI
            self.score += 100

        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_cursor(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)

    def _perform_paint_action(self):
        r, c = self.cursor_pos
        color_to_paint = self.grid[r, c]

        if color_to_paint == 0:
            return False

        grid_changed = False
        neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

        for nr, nc in neighbors:
            if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                if self.grid[nr, nc] == 0:
                    self.grid[nr, nc] = color_to_paint
                    self.particles.append({
                        "pos": (nr, nc),
                        "lifetime": self.PARTICLE_LIFETIME,
                    })
                    grid_changed = True

        return grid_changed

    def _check_4x4_bonus(self):
        bonus = 0
        for r_start in range(self.GRID_SIZE - 3):
            for c_start in range(self.GRID_SIZE - 3):
                if (r_start, c_start) not in self.completed_4x4_origins:
                    sub_grid = self.grid[r_start:r_start + 4, c_start:c_start + 4]
                    first_color = sub_grid[0, 0]
                    if first_color != 0 and np.all(sub_grid == first_color):
                        bonus += 5
                        self.completed_4x4_origins.add((r_start, c_start))
        return bonus

    def _check_victory(self):
        if self.grid is None or self.grid.size == 0:
            return False
        first_color = self.grid[0, 0]
        if first_color == 0:
            return False
        return np.all(self.grid == first_color)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_and_render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # This check prevents rendering before reset() is called
        if self.grid is None:
            return

        # Draw grid cells and lines
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.grid[r, c]
                color = self.COLORS[color_idx]
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + cursor_c * self.CELL_SIZE,
            self.GRID_OFFSET_Y + cursor_r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )

        # Pulsing effect for cursor border
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # 0 to 1
        border_width = 2 + int(pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, border_width)

    def _update_and_render_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.pop(i)
            else:
                r, c = p["pos"]
                progress = p["lifetime"] / self.PARTICLE_LIFETIME
                size = int(self.CELL_SIZE * (1 - progress))
                alpha = int(255 * progress)

                center_x = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                center_y = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2

                rect = pygame.Rect(center_x - size // 2, center_y - size // 2, size, size)

                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                pygame.draw.rect(temp_surf, (255, 255, 255, alpha), temp_surf.get_rect(), border_radius=size // 4)
                self.screen.blit(temp_surf, rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.UI_FONT.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_color = self.COLOR_TEXT if time_left > 10 else (231, 76, 60)
        timer_text = self.UI_FONT.render(f"Time: {time_left:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(timer_text, timer_rect)

        # Game Over / Victory Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            is_victory = self._check_victory()
            msg = "VICTORY!" if is_victory else "TIME'S UP!"
            color = self.COLORS[2] if is_victory else self.COLORS[1]

            msg_text = pygame.font.Font(None, 80).render(msg, True, color)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left_seconds": max(0, self.timer / self.FPS),
            "grid_coverage": np.count_nonzero(self.grid) / (self.GRID_SIZE ** 2) if self.grid is not None else 0
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test reset first to initialize the state
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)

        # Now test observation space on a valid state
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8

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
    # This block allows you to play the game manually
    # It will create a graphical window and may not work in a headless environment
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Setup Pygame window for human play
    pygame.display.set_caption("Grid Painter")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement = 0  # No-op
        space_pressed = 0
        shift_pressed = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space_pressed = 1
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_pressed = 1

        action = [movement, space_pressed, shift_pressed]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000)  # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)

    env.close()