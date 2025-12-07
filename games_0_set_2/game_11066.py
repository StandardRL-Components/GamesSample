import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Balance the output of two fluctuating power generators to meet energy demands and stabilize the system."
    )
    user_guide = (
        "Controls: Use ↑/↓ to adjust Generator 1 output and ←/→ to adjust Generator 2 output. "
        "Match the total output to the system's demand to achieve 100% efficiency."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 10000
        self.VICTORY_DURATION_STEPS = 30 * self.FPS  # 30 seconds

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 50)

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_DIM = (150, 150, 170)
        self.COLOR_GEN_BLUE = (0, 150, 255)
        self.COLOR_DEVICE_RED = (255, 80, 80)
        self.COLOR_DEVICE_YELLOW = (255, 200, 0)
        self.COLOR_EFF_GREEN = (0, 255, 120)
        self.COLOR_BAR_BG = (40, 50, 75)

        # Game state variables are initialized in reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Core game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.gen1_output = 5.0
        self.gen2_output = 5.0
        self.gen1_max_output = 10.0
        self.gen2_max_output = 10.0
        self.gen_fluctuation_range = 1.0
        self.total_demand = 20.0
        self.device_demand = self.total_demand / 5.0

        self.efficiency = 0.0
        self.time_at_100_efficiency = 0
        self.victory_timer_active = False
        self.first_time_100_efficiency = True

        # Display state for smooth interpolation
        self.display_gen1_output = self.gen1_output
        self.display_gen2_output = self.gen2_output
        self.display_efficiency = self.efficiency
        self.display_device_power = [0.0] * 5

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. HANDLE PLAYER INPUT ---
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        if movement == 1:  # Up: Gen 1 increase
            self.gen1_output += 1
        elif movement == 2:  # Down: Gen 1 decrease
            self.gen1_output -= 1
        elif movement == 4:  # Right: Gen 2 increase
            self.gen2_output += 1
        elif movement == 3:  # Left: Gen 2 decrease
            self.gen2_output -= 1

        self.gen1_output = np.clip(self.gen1_output, 1, self.gen1_max_output)
        self.gen2_output = np.clip(self.gen2_output, 1, self.gen2_max_output)

        # --- 2. UPDATE GAME STATE ---
        self.steps += 1

        # Natural generator fluctuation every second
        if self.steps % self.FPS == 0:
            fluctuation1 = self.np_random.uniform(-self.gen_fluctuation_range, self.gen_fluctuation_range)
            fluctuation2 = self.np_random.uniform(-self.gen_fluctuation_range, self.gen_fluctuation_range)
            self.gen1_output = np.clip(self.gen1_output + fluctuation1, 1, self.gen1_max_output)
            self.gen2_output = np.clip(self.gen2_output + fluctuation2, 1, self.gen2_max_output)

        # Increase fluctuation range every 10 seconds
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.gen_fluctuation_range = min(3.0, self.gen_fluctuation_range + 0.1)

        total_output = self.gen1_output + self.gen2_output
        self.efficiency = min(1.0, total_output / self.total_demand)

        # --- 3. CALCULATE REWARD & CHECK TERMINATION ---
        reward = 0
        terminated = False

        if self.efficiency >= 0.9:
            reward += 1
        if self.efficiency == 1.0:
            reward += 4  # Total of 5 for 100%

        if self.efficiency == 1.0:
            if not self.victory_timer_active:
                self.victory_timer_active = True

            if self.first_time_100_efficiency:
                self.first_time_100_efficiency = False
                reward += 10
                # Boost max output by 20%
                self.gen1_max_output *= 1.2
                self.gen2_max_output *= 1.2

            self.time_at_100_efficiency += 1
        else:
            if self.victory_timer_active:
                self.victory_timer_active = False
            self.time_at_100_efficiency = 0

        # Check termination conditions
        if self.time_at_100_efficiency >= self.VICTORY_DURATION_STEPS:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward += -100
            terminated = True
            self.game_over = True

        self.score += reward

        # --- 4. UPDATE PARTICLES ---
        if self.efficiency == 1.0:
            self._spawn_particles()
        self._update_particles()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "efficiency": self.efficiency,
            "gen1_output": self.gen1_output,
            "gen2_output": self.gen2_output,
            "time_at_100%": self.time_at_100_efficiency,
        }

    def _get_observation(self):
        # Interpolate display values for smooth visuals
        lerp_factor = 0.1
        self.display_gen1_output += (self.gen1_output - self.display_gen1_output) * lerp_factor
        self.display_gen2_output += (self.gen2_output - self.display_gen2_output) * lerp_factor
        self.display_efficiency += (self.efficiency - self.display_efficiency) * lerp_factor

        total_output = self.gen1_output + self.gen2_output
        power_per_device = total_output / 5.0
        for i in range(5):
            target_power = min(1.0, power_per_device / self.device_demand)
            self.display_device_power[i] += (target_power - self.display_device_power[i]) * lerp_factor

        # --- RENDER ---
        self.screen.fill(self.COLOR_BG)
        self._render_background_grid()
        self._render_particles()
        self._render_generators()
        self._render_devices()
        self._render_efficiency_gauge()
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- PARTICLE SYSTEM ---
    def _spawn_particles(self):
        if self.np_random.random() < 0.8:  # Spawn particles frequently
            # Particles from generators to center
            start_pos1 = (self.WIDTH * 0.25, self.HEIGHT * 0.5)
            start_pos2 = (self.WIDTH * 0.75, self.HEIGHT * 0.5)
            end_pos = (self.WIDTH * 0.5, self.HEIGHT * 0.3)
            for start_pos in [start_pos1, start_pos2]:
                p_angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
                p_angle += self.np_random.uniform(-0.3, 0.3)
                p_speed = self.np_random.uniform(2, 4)
                p_vel = (math.cos(p_angle) * p_speed, math.sin(p_angle) * p_speed)
                p_life = self.np_random.integers(40, 70)
                self.particles.append({'pos': list(start_pos), 'vel': p_vel, 'life': p_life, 'max_life': p_life, 'color': self.COLOR_GEN_BLUE})

            # Particles from center to devices
            start_pos_center = (self.WIDTH * 0.5, self.HEIGHT * 0.3)
            for i in range(5):
                end_pos_device = (self.WIDTH * (0.2 + i * 0.15), self.HEIGHT * 0.8)
                p_angle = math.atan2(end_pos_device[1] - start_pos_center[1], end_pos_device[0] - start_pos_center[0])
                p_angle += self.np_random.uniform(-0.2, 0.2)
                p_speed = self.np_random.uniform(3, 5)
                p_vel = (math.cos(p_angle) * p_speed, math.sin(p_angle) * p_speed)
                p_life = self.np_random.integers(50, 80)
                self.particles.append({'pos': list(start_pos_center), 'vel': p_vel, 'life': p_life, 'max_life': p_life, 'color': self.COLOR_EFF_GREEN})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    # --- RENDERING HELPERS ---
    def _render_background_grid(self):
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 2, color)

    def _render_generators(self):
        # Generator 1
        g1_rect = pygame.Rect(self.WIDTH * 0.15, self.HEIGHT * 0.4, 100, 150)
        self._draw_panel("GENERATOR 1", g1_rect)
        bar_val = self.display_gen1_output / self.gen1_max_output
        self._draw_vertical_bar(g1_rect.move(0, 30), bar_val, self.COLOR_GEN_BLUE)
        self._draw_text(f"{self.gen1_output:.1f}", (g1_rect.centerx, g1_rect.bottom - 15), self.font_medium, self.COLOR_TEXT)

        # Generator 2
        g2_rect = pygame.Rect(self.WIDTH * 0.85 - 100, self.HEIGHT * 0.4, 100, 150)
        self._draw_panel("GENERATOR 2", g2_rect)
        bar_val = self.display_gen2_output / self.gen2_max_output
        self._draw_vertical_bar(g2_rect.move(0, 30), bar_val, self.COLOR_GEN_BLUE)
        self._draw_text(f"{self.gen2_output:.1f}", (g2_rect.centerx, g2_rect.bottom - 15), self.font_medium, self.COLOR_TEXT)

    def _render_devices(self):
        base_y = self.HEIGHT * 0.8
        for i in range(5):
            x = self.WIDTH * (0.2 + i * 0.15)
            # Device Icon (a simple square)
            icon_rect = pygame.Rect(x - 15, base_y - 15, 30, 30)
            pygame.draw.rect(self.screen, self.COLOR_TEXT_DIM, icon_rect, 2, border_radius=4)
            self._draw_text(f"D{i + 1}", icon_rect.center, self.font_small, self.COLOR_TEXT_DIM)

            # Power Bar
            power_val = self.display_device_power[i]
            bar_color = self.COLOR_DEVICE_RED if power_val < 0.9 else (self.COLOR_EFF_GREEN if power_val == 1.0 else self.COLOR_DEVICE_YELLOW)
            bar_rect = pygame.Rect(x - 30, base_y + 25, 60, 10)
            self._draw_horizontal_bar(bar_rect, power_val, bar_color)

    def _render_efficiency_gauge(self):
        center = (self.WIDTH // 2, int(self.HEIGHT * 0.3))
        radius = 60

        # Glow effect when at 100%
        if self.display_efficiency > 0.99:
            glow_radius = radius + 15 + 10 * math.sin(self.steps * 0.1)
            glow_alpha = int(100 * self.display_efficiency)
            self._draw_glow(center, int(glow_radius), self.COLOR_EFF_GREEN, glow_alpha)

        self._draw_circular_gauge(center, radius, self.display_efficiency, self.COLOR_EFF_GREEN, self.COLOR_BAR_BG)

        eff_text = f"{self.efficiency:.1%}"
        self._draw_text(eff_text, center, self.font_large, self.COLOR_TEXT)
        self._draw_text("EFFICIENCY", (center[0], center[1] + 45), self.font_small, self.COLOR_TEXT_DIM)

    def _render_ui(self):
        # Score
        self._draw_text(f"SCORE: {int(self.score)}", (self.WIDTH - 10, 10), self.font_medium, self.COLOR_TEXT, align="topright")
        # Steps
        self._draw_text(f"STEP: {self.steps}", (10, 10), self.font_medium, self.COLOR_TEXT, align="topleft")

        # Victory Timer
        if self.victory_timer_active:
            remaining_steps = self.VICTORY_DURATION_STEPS - self.time_at_100_efficiency
            remaining_seconds = remaining_steps / self.FPS
            timer_text = f"SYSTEM STABLE: {remaining_seconds:.1f}s"
            self._draw_text(timer_text, (self.WIDTH // 2, self.HEIGHT - 20), self.font_medium, self.COLOR_EFF_GREEN)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY" if self.time_at_100_efficiency >= self.VICTORY_DURATION_STEPS else "SYSTEM TIMEOUT"
            color = self.COLOR_EFF_GREEN if self.time_at_100_efficiency >= self.VICTORY_DURATION_STEPS else self.COLOR_DEVICE_RED
            self._draw_text(msg, (self.WIDTH // 2, self.HEIGHT // 2 - 20), self.font_large, color)
            self._draw_text("Press RESET to play again", (self.WIDTH // 2, self.HEIGHT // 2 + 30), self.font_medium, self.COLOR_TEXT)

    def _draw_text(self, text, pos, font, color, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos
        self.screen.blit(text_surface, text_rect)

    def _draw_panel(self, title, rect):
        pygame.gfxdraw.box(self.screen, rect, (0, 0, 0, 50))
        pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 2, border_radius=5)
        self._draw_text(title, (rect.centerx, rect.top + 15), self.font_small, self.COLOR_TEXT_DIM)

    def _draw_horizontal_bar(self, rect, progress, color):
        progress = np.clip(progress, 0, 1)
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, rect, border_radius=3)
        fill_rect = pygame.Rect(rect.left, rect.top, int(rect.width * progress), rect.height)
        pygame.draw.rect(self.screen, color, fill_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_TEXT_DIM, rect, 1, border_radius=3)

    def _draw_vertical_bar(self, rect, progress, color):
        progress = np.clip(progress, 0, 1)
        bar_area = rect.inflate(-20, -60)
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, bar_area, border_radius=5)
        fill_height = int(bar_area.height * progress)
        fill_rect = pygame.Rect(bar_area.left, bar_area.bottom - fill_height, bar_area.width, fill_height)
        pygame.draw.rect(self.screen, color, fill_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TEXT_DIM, bar_area, 1, border_radius=5)

    def _draw_circular_gauge(self, center, radius, progress, color, bg_color):
        progress = np.clip(progress, 0, 1)
        thickness = 10
        for i in range(thickness):
            r = radius - i
            # Background arc
            pygame.gfxdraw.arc(self.screen, center[0], center[1], r, -225, 45, bg_color)
            # Foreground arc
            end_angle = int(-225 + (270 * progress))
            if end_angle > -225:
                pygame.gfxdraw.arc(self.screen, center[0], center[1], r, -225, end_angle, color)

    def _draw_glow(self, center, radius, color, alpha):
        if radius <= 0: return
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, color + (alpha,), (radius, radius), radius)
        self.screen.blit(surf, (center[0] - radius, center[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Generator Sync")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        action = [0, 0, 0]  # Default action: no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        # The other actions are not used in this game
        # action[1] = 1 if keys[pygame.K_SPACE] else 0
        # action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for 'R' to be pressed to reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False

        clock.tick(env.FPS)

    env.close()