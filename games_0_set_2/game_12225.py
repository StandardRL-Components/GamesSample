import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import os
import pygame


# Set SDL to dummy mode for headless operation, required for Gymnasium
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the goal is to balance water levels across three
    reservoirs. The player controls the tilt of the system, causing water to flow
    between adjacent reservoirs. Random rainfall adds water, and the player must
    maintain all reservoirs above 60% capacity for 60 consecutive seconds to win.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing attributes ---
    game_description = (
        "Balance water levels across three reservoirs by tilting the system. "
        "Counteract random rainfall and keep all reservoirs above 60% capacity to win."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to tilt the system and transfer water between the reservoirs."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60

    COLOR_BG = (25, 35, 45)
    COLOR_RESERVOIR = (100, 110, 120)
    COLOR_WATER = (50, 150, 255)
    COLOR_WATER_SURFACE = (150, 200, 255)
    COLOR_TARGET_LINE = (100, 255, 150)
    COLOR_WARN_GLOW = (255, 50, 50, 50) # RGBA for transparency
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_RAIN = (180, 200, 220)

    RESERVOIR_COUNT = 3
    RESERVOIR_WIDTH = 120
    RESERVOIR_HEIGHT = 250
    RESERVOIR_Y = 80
    RESERVOIR_SPACING = (SCREEN_WIDTH - RESERVOIR_COUNT * RESERVOIR_WIDTH) / (RESERVOIR_COUNT + 1)

    MAX_STEPS = 6000  # 100 seconds at 60 FPS
    WIN_DURATION_SECONDS = 60
    WIN_DURATION_STEPS = WIN_DURATION_SECONDS * FPS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_m = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Game State Initialization ---
        self.reservoir_levels = [0.0] * self.RESERVOIR_COUNT
        self.tilt_momentum = 0.0
        self.win_timer = 0
        self.rain_particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_sync_bonus_step = -1
        self.last_win_duration_reward_time = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reservoir_levels = [50.0, 50.0, 50.0]
        self.tilt_momentum = 0.0
        self.win_timer = 0
        self.rain_particles = []
        self.last_sync_bonus_step = -self.FPS # Allow sync bonus from step 0
        self.last_win_duration_reward_time = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self._update_game_state(movement)

        terminated, won = self._check_termination()
        reward = self._calculate_reward(terminated, won)
        self.score += reward
        self.steps += 1

        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _update_game_state(self, movement):
        # 1. Update Tilt Momentum
        if movement == 3:  # Tilt Left
            self.tilt_momentum = max(-1.0, self.tilt_momentum - 0.08)
        elif movement == 4:  # Tilt Right
            self.tilt_momentum = min(1.0, self.tilt_momentum + 0.08)
        else:  # No tilt or invalid action
            self.tilt_momentum *= 0.95  # Decay momentum

        # 2. Water Transfer based on Momentum
        max_transfer_rate = 0.03 # 3% of source water per frame at full tilt
        transfer_rate = max_transfer_rate * abs(self.tilt_momentum)

        deltas = [0.0] * self.RESERVOIR_COUNT

        if self.tilt_momentum > 0: # Tilting Right (flow ->)
            # R1 to R2
            transfer_1_2 = self.reservoir_levels[0] * transfer_rate
            deltas[0] -= transfer_1_2
            deltas[1] += transfer_1_2
            # R2 to R3
            transfer_2_3 = self.reservoir_levels[1] * transfer_rate
            deltas[1] -= transfer_2_3
            deltas[2] += transfer_2_3
        elif self.tilt_momentum < 0: # Tilting Left (flow <-)
            # R2 to R1
            transfer_2_1 = self.reservoir_levels[1] * transfer_rate
            deltas[1] -= transfer_2_1
            deltas[0] += transfer_2_1
            # R3 to R2
            transfer_3_2 = self.reservoir_levels[2] * transfer_rate
            deltas[2] -= transfer_3_2
            deltas[1] += transfer_3_2

        for i in range(self.RESERVOIR_COUNT):
            self.reservoir_levels[i] += deltas[i]

        # 3. Rain
        if self.np_random.random() < 0.20: # 20% chance per frame
            reservoir_idx = self.np_random.integers(0, self.RESERVOIR_COUNT)
            rain_amount = self.np_random.uniform(1.0, 3.0)
            self.reservoir_levels[reservoir_idx] += rain_amount

            # Add rain particle for visual effect
            res_x = self.RESERVOIR_SPACING * (reservoir_idx + 1) + self.RESERVOIR_WIDTH * reservoir_idx
            particle_x = res_x + self.np_random.uniform(10, self.RESERVOIR_WIDTH - 10)
            self.rain_particles.append([particle_x, self.RESERVOIR_Y - 10, self.np_random.uniform(4, 8)])

        # 4. Sync Bonus
        min_level, max_level = min(self.reservoir_levels), max(self.reservoir_levels)
        if max_level - min_level <= 5.0 and self.steps > self.last_sync_bonus_step + self.FPS:
            bonus_water = 2.0 / self.RESERVOIR_COUNT
            for i in range(self.RESERVOIR_COUNT):
                self.reservoir_levels[i] += bonus_water
            self.last_sync_bonus_step = self.steps

        # 5. Clamp water levels
        self.reservoir_levels = [max(0.0, min(100.0, level)) for level in self.reservoir_levels]

    def _check_termination(self):
        # Loss condition: any reservoir is empty
        if any(level <= 0 for level in self.reservoir_levels):
            return True, False # terminated, won

        # Win condition logic
        if all(level >= 60.0 for level in self.reservoir_levels):
            self.win_timer += 1
        else:
            self.win_timer = 0

        if self.win_timer >= self.WIN_DURATION_STEPS:
            return True, True # terminated, won

        # Max steps reached
        if self.steps >= self.MAX_STEPS:
            return True, False # terminated, won

        return False, False

    def _calculate_reward(self, terminated, won):
        if terminated:
            if won:
                return 100.0
            else:
                return -100.0

        reward = 0.0

        # Continuous reward for keeping levels high
        for level in self.reservoir_levels:
            if level > 50.0:
                reward += 0.01

        # Event-based reward for being in sync
        min_level, max_level = min(self.reservoir_levels), max(self.reservoir_levels)
        if max_level - min_level <= 5.0 and self.steps == self.last_sync_bonus_step:
            reward += 1.0

        # Event-based reward for duration above 60%
        win_duration_seconds = self.win_timer // self.FPS
        if win_duration_seconds > self.last_win_duration_reward_time and win_duration_seconds > 0 and win_duration_seconds % 10 == 0:
            reward += 5.0
            self.last_win_duration_reward_time = win_duration_seconds

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game_elements()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win_timer_seconds": self.win_timer / self.FPS,
            "reservoir_levels": [round(l, 2) for l in self.reservoir_levels],
            "tilt_momentum": round(self.tilt_momentum, 2),
        }

    def _render_game_elements(self):
        # Draw reservoirs and water
        for i in range(self.RESERVOIR_COUNT):
            self._draw_reservoir(i, self.reservoir_levels[i])

        # Update and draw rain particles
        self._update_and_draw_particles()

    def _draw_reservoir(self, index, level):
        res_x = self.RESERVOIR_SPACING * (index + 1) + self.RESERVOIR_WIDTH * index
        res_rect = pygame.Rect(res_x, self.RESERVOIR_Y, self.RESERVOIR_WIDTH, self.RESERVOIR_HEIGHT)

        # Low water warning glow
        if level < 20.0:
            glow_surface = pygame.Surface((self.RESERVOIR_WIDTH + 20, self.RESERVOIR_HEIGHT + 20), pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, self.COLOR_WARN_GLOW, glow_surface.get_rect(), border_radius=15)
            self.screen.blit(glow_surface, (res_x - 10, self.RESERVOIR_Y - 10))

        # Water
        water_height = (level / 100.0) * self.RESERVOIR_HEIGHT
        water_rect = pygame.Rect(res_x, self.RESERVOIR_Y + self.RESERVOIR_HEIGHT - water_height, self.RESERVOIR_WIDTH, water_height)
        pygame.draw.rect(self.screen, self.COLOR_WATER, water_rect, border_bottom_left_radius=8, border_bottom_right_radius=8)

        # Water surface line
        if water_height > 0:
            surface_y = self.RESERVOIR_Y + self.RESERVOIR_HEIGHT - water_height
            pygame.draw.line(self.screen, self.COLOR_WATER_SURFACE, (res_x, surface_y), (res_x + self.RESERVOIR_WIDTH, surface_y), 2)

        # Reservoir outline
        pygame.draw.rect(self.screen, self.COLOR_RESERVOIR, res_rect, 4, border_radius=10)

        # 60% Target line
        target_y = self.RESERVOIR_Y + self.RESERVOIR_HEIGHT * (1.0 - 0.6)
        pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (res_x, target_y), (res_x + self.RESERVOIR_WIDTH, target_y), 2)

    def _update_and_draw_particles(self):
        for p in self.rain_particles[:]:
            p[1] += p[2] # Fall
            pygame.draw.line(self.screen, self.COLOR_RAIN, (p[0], p[1]), (p[0], p[1] + 10), 2)

            # Check for collision with water surface or removal
            removed = False
            for i in range(self.RESERVOIR_COUNT):
                water_surface_y = self.RESERVOIR_Y + self.RESERVOIR_HEIGHT * (1.0 - self.reservoir_levels[i] / 100.0)
                if p[1] > water_surface_y:
                    self.rain_particles.remove(p)
                    removed = True
                    break
            if not removed and p[1] > self.RESERVOIR_Y + self.RESERVOIR_HEIGHT:
                 self.rain_particles.remove(p)

    def _render_ui(self):
        # Draw text inside reservoirs
        for i in range(self.RESERVOIR_COUNT):
            res_x = self.RESERVOIR_SPACING * (i + 1) + self.RESERVOIR_WIDTH * i
            text = f"{self.reservoir_levels[i]:.1f}%"
            self._draw_text(text, self.font_m, (res_x + self.RESERVOIR_WIDTH / 2, self.RESERVOIR_Y + self.RESERVOIR_HEIGHT - 30))

        # Draw score and timers
        self._draw_text(f"Score: {self.score:.2f}", self.font_m, (10, 10), align="topleft")
        self._draw_text(f"Step: {self.steps}/{self.MAX_STEPS}", self.font_s, (10, 40), align="topleft")

        self._draw_text(f"Balance Timer: {self.win_timer / self.FPS:.1f}s / {self.WIN_DURATION_SECONDS}s", self.font_m, (self.SCREEN_WIDTH - 10, 10), align="topright")

        # Win progress bar
        bar_width = 300
        bar_height = 10
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 40
        pygame.draw.rect(self.screen, self.COLOR_RESERVOIR, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        win_progress = self.win_timer / self.WIN_DURATION_STEPS
        if win_progress > 0:
            pygame.draw.rect(self.screen, self.COLOR_TARGET_LINE, (bar_x, bar_y, bar_width * win_progress, bar_height), border_radius=5)

        # Draw tilt indicator
        indicator_y = self.SCREEN_HEIGHT - 20
        indicator_center_x = self.SCREEN_WIDTH / 2
        indicator_width = 100

        p1 = (indicator_center_x - indicator_width / 2, indicator_y)
        p2 = (indicator_center_x + indicator_width / 2, indicator_y)
        pygame.draw.line(self.screen, self.COLOR_RESERVOIR, p1, p2, 4)

        ball_x = indicator_center_x + (indicator_width / 2 - 5) * self.tilt_momentum
        pygame.gfxdraw.aacircle(self.screen, int(ball_x), int(indicator_y), 8, self.COLOR_WATER_SURFACE)
        pygame.gfxdraw.filled_circle(self.screen, int(ball_x), int(indicator_y), 8, self.COLOR_WATER_SURFACE)

    def _draw_text(self, text, font, pos, align="center"):
        text_surf = font.render(text, True, self.COLOR_TEXT)
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos

        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually
    # It will create a window and render the game
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    env = GameEnv()
    obs, info = env.reset()
    done = False

    # Game loop
    running = True
    while running:
        # --- Player Input ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4

        # We only care about the movement action for manual play
        action = [movement, 0, 0]

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    done = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Game Step ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Rendering ---
        # The observation is the rendered screen, so we need to display it
        # Pygame uses (width, height), numpy uses (height, width)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))

        # Create a display if one doesn't exist
        try:
            display_surf = pygame.display.get_surface()
            if display_surf is None:
                raise AttributeError
        except (pygame.error, AttributeError):
            display_surf = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
            pygame.display.set_caption("Reservoir Balance")

        display_surf.blit(surf, (0, 0))

        if done:
            # Display game over message
            overlay = pygame.Surface((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))

            won = info["win_timer_seconds"] >= GameEnv.WIN_DURATION_SECONDS
            msg = "YOU WIN!" if won else "GAME OVER"
            color = GameEnv.COLOR_TARGET_LINE if won else GameEnv.COLOR_WARN_GLOW

            font_l = pygame.font.SysFont("Consolas", 64, bold=True)
            text_surf = font_l.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 - 20))
            display_surf.blit(overlay, (0, 0))
            display_surf.blit(text_surf, text_rect)

            font_s = pygame.font.SysFont("Consolas", 24, bold=True)
            sub_text_surf = font_s.render("Press 'R' to Restart", True, GameEnv.COLOR_TEXT)
            sub_text_rect = sub_text_surf.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 + 40))
            display_surf.blit(sub_text_surf, sub_text_rect)

        pygame.display.flip()
        env.clock.tick(env.FPS)

    env.close()