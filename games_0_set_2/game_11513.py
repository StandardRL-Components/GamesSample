from gymnasium.spaces import MultiDiscrete
import os
import pygame


import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
from collections import deque
import numpy as np
from gymnasium.spaces import Box, MultiDiscrete

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player must survive a barrage of projectiles.
    The player controls gravity to deflect projectiles away from a base at the
    bottom of the screen. Deflecting projectiles creates temporary gravity wells
    that slow down other projectiles. Surviving for the full duration results in a win.
    """

    game_description = "Deflect incoming projectiles by switching gravity. Survive for two minutes to win."
    user_guide = "Use the ↑ and ↓ arrow keys to switch the direction of gravity and deflect projectiles away from your base."
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- CRITICAL: Action and Observation Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.render_mode = render_mode
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 24)

        # --- Visual & Game Constants ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_GRID = (20, 30, 50)
        self.COLOR_BASE = (0, 255, 128)
        self.COLOR_SHIELD = (0, 192, 255)
        self.COLOR_PROJECTILE = (255, 50, 50)
        self.COLOR_WELL = (176, 38, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TIMER_WARN = (255, 200, 0)
        self.COLOR_TIMER_CRIT = (255, 80, 80)

        self.BASE_HEIGHT = 20
        self.BASE_RECT = pygame.Rect(0, self.height - self.BASE_HEIGHT, self.width, self.BASE_HEIGHT)
        self.SHIELD_Y = self.height - 80

        self.MAX_TIME = 120 * self.metadata["render_fps"]  # 120 seconds
        self.DEFLECTIONS_FOR_FLIP = 10
        self.DIFFICULTY_INTERVAL = 10 * self.metadata["render_fps"]  # 10 seconds

        # --- Physics Constants ---
        self.GRAVITY_ACCEL = 0.08
        self.PROJECTILE_BASE_SPEED_Y = 2.0
        self.PROJECTILE_RADIUS = 7
        self.WELL_DURATION = 1.5 * self.metadata["render_fps"]
        self.WELL_MAX_RADIUS = 100
        self.WELL_SLOW_FACTOR = 0.5

        # --- Initialize state variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.gravity_direction = 0
        self.projectiles = []
        self.gravity_wells = []
        self.particles = []
        self.deflection_counter = 0
        self.projectile_spawn_timer = 0
        self.projectile_spawn_interval = 0
        self.difficulty_timer = 0
        self.events = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME
        self.gravity_direction = 1  # Start with gravity down

        self.projectiles = []
        self.gravity_wells = []
        self.particles = []

        self.deflection_counter = 0
        self.projectile_spawn_interval = self.metadata["render_fps"] * 1.5  # 1.5 seconds
        self.projectile_spawn_timer = self.projectile_spawn_interval
        self.difficulty_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self.events = {'deflected': 0, 'won': False}
        reward = 0.0

        if not self.game_over:
            # --- Handle Action ---
            if movement == 1:  # Up
                self.gravity_direction = -1
            elif movement == 2:  # Down
                self.gravity_direction = 1
            # Other actions are no-ops

            # --- Update Game State ---
            self.steps += 1
            self.timer -= 1

            self._update_difficulty()
            self._update_spawner()
            self._update_projectiles()
            self._update_wells()
            self._update_particles()

        # --- Calculate Reward ---
        reward = self._calculate_reward()

        # --- Check Termination ---
        terminated = self.game_over or self.timer <= 0
        if terminated and not self.game_over and self.timer <= 0:
            self.events['won'] = True
            reward += 100.0
            self.score += 1000

        truncated = False  # This game ends on a time limit or failure, not truncation.

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _calculate_reward(self):
        # Continuous survival reward + event-based deflection reward
        return 0.1 + self.events['deflected'] * 1.0

    def _update_difficulty(self):
        self.difficulty_timer += 1
        if self.difficulty_timer >= self.DIFFICULTY_INTERVAL:
            self.difficulty_timer = 0
            self.projectile_spawn_interval = max(15, self.projectile_spawn_interval * 0.95)

    def _update_spawner(self):
        self.projectile_spawn_timer -= 1
        if self.projectile_spawn_timer <= 0:
            self.projectile_spawn_timer = self.projectile_spawn_interval

            x_pos = self.np_random.uniform(self.PROJECTILE_RADIUS * 2, self.width - self.PROJECTILE_RADIUS * 2)
            x_vel = self.np_random.uniform(-1.0, 1.0)

            self.projectiles.append({
                'pos': pygame.Vector2(x_pos, -self.PROJECTILE_RADIUS),
                'vel': pygame.Vector2(x_vel, self.PROJECTILE_BASE_SPEED_Y),
                'trail': deque(maxlen=15)
            })

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj['trail'].append(pygame.Vector2(proj['pos']))

            # Check for gravity well slowdown
            is_slowed = False
            for well in self.gravity_wells:
                if proj['pos'].distance_to(well['pos']) < well['radius']:
                    is_slowed = True
                    break

            # Apply gravity
            proj['vel'].y += self.GRAVITY_ACCEL * self.gravity_direction

            # Update position
            speed_multiplier = self.WELL_SLOW_FACTOR if is_slowed else 1.0
            proj['pos'] += proj['vel'] * speed_multiplier

            # Deflection check
            if len(proj['trail']) > 1:
                was_above = proj['trail'][-2].y < self.SHIELD_Y
                is_below = proj['pos'].y >= self.SHIELD_Y
                correct_gravity_dir = (self.gravity_direction > 0 and proj['vel'].y > 0) or \
                                      (self.gravity_direction < 0 and proj['vel'].y < 0)

                if was_above and is_below and correct_gravity_dir:
                    self._handle_deflection(proj)

            # Boundary and collision checks
            if self.BASE_RECT.collidepoint(proj['pos']):
                self.game_over = True
                self._create_particles(proj['pos'], 100, (255, 255, 100), 5, 120)
                self.projectiles.remove(proj)
                continue

            if not self.screen.get_rect().inflate(50, 50).collidepoint(proj['pos']):
                self.projectiles.remove(proj)

    def _handle_deflection(self, proj):
        self.events['deflected'] += 1
        self.score += 10
        self.deflection_counter += 1

        proj['vel'].y *= -1.1  # Bounce with extra force
        proj['pos'].y = self.SHIELD_Y - (proj['pos'].y - self.SHIELD_Y)  # Correct position

        self._create_particles(proj['pos'], 30, self.COLOR_SHIELD, 3, 40)
        self.gravity_wells.append({
            'pos': pygame.Vector2(proj['pos']),
            'lifetime': self.WELL_DURATION,
            'max_lifetime': self.WELL_DURATION,
        })

        if self.deflection_counter >= self.DEFLECTIONS_FOR_FLIP:
            self.deflection_counter = 0
            self.gravity_direction *= -1

    def _update_wells(self):
        for well in self.gravity_wells[:]:
            well['lifetime'] -= 1
            if well['lifetime'] <= 0:
                self.gravity_wells.remove(well)
            else:
                progress = well['lifetime'] / well['max_lifetime']
                well['radius'] = self.WELL_MAX_RADIUS * (1.0 - math.sqrt(progress))

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.98  # Drag
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color, speed_max, lifetime_max):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'lifetime': self.np_random.integers(20, lifetime_max),
                'max_lifetime': lifetime_max,
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "deflections": self.deflection_counter,
        }

    def _render_game(self):
        self._draw_grid()
        self._draw_wells()
        self._draw_projectiles()
        self._draw_particles()
        self._draw_base_and_shield()
        self._draw_gravity_indicator()

    def _draw_grid(self):
        for x in range(0, self.width, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.height))
        for y in range(0, self.height, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.width, y))

    def _draw_wells(self):
        well_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for well in self.gravity_wells:
            progress = well['lifetime'] / well['max_lifetime']
            alpha = int(100 * progress)
            self._draw_glowing_circle(
                well_surf, self.COLOR_WELL, (int(well['pos'].x), int(well['pos'].y)),
                int(well['radius']), alpha, 3
            )
        self.screen.blit(well_surf, (0, 0))

    def _draw_projectiles(self):
        proj_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for proj in self.projectiles:
            # Draw trail
            if len(proj['trail']) > 1:
                points = [(int(p.x), int(p.y)) for p in proj['trail']]
                for i in range(len(points) - 1):
                    alpha = int(200 * (i / len(points)))
                    color = (*self.COLOR_PROJECTILE, alpha)
                    pygame.draw.line(proj_surf, color, points[i], points[i + 1], 2)

            # Draw main projectile
            self._draw_glowing_circle(
                proj_surf, self.COLOR_PROJECTILE, (int(proj['pos'].x), int(proj['pos'].y)),
                self.PROJECTILE_RADIUS, 255, 3
            )
        self.screen.blit(proj_surf, (0, 0))

    def _draw_particles(self):
        particle_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for p in self.particles:
            progress = p['lifetime'] / p['max_lifetime']
            alpha = int(255 * progress)
            color = (*p['color'], alpha)
            pos = (int(p['pos'].x), int(p['pos'].y))
            radius = int(p['radius'] * progress)
            if radius > 0:
                pygame.draw.circle(particle_surf, color, pos, radius)
        self.screen.blit(particle_surf, (0, 0))

    def _draw_base_and_shield(self):
        # Base
        glow_color = (*self.COLOR_BASE, 50)
        pygame.draw.rect(self.screen, glow_color, self.BASE_RECT.inflate(10, 10), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.BASE_RECT, border_radius=5)

        # Shield
        pulse = abs(math.sin(self.steps * 0.1))
        glow_width = 2 + int(pulse * 3)
        glow_alpha = 50 + int(pulse * 50)

        glow_color = (*self.COLOR_SHIELD, glow_alpha)
        pygame.draw.line(self.screen, glow_color, (0, self.SHIELD_Y), (self.width, self.SHIELD_Y), glow_width * 2)
        pygame.draw.line(self.screen, self.COLOR_SHIELD, (0, self.SHIELD_Y), (self.width, self.SHIELD_Y), 3)

    def _draw_gravity_indicator(self):
        num_arrows = 10
        arrow_spacing = self.width / (num_arrows + 1)
        arrow_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for i in range(num_arrows):
            x = (i + 1) * arrow_spacing
            y_start = self.SHIELD_Y - 20
            y_end = self.SHIELD_Y - 40

            pulse = abs(math.sin(self.steps * 0.05 + i))
            alpha = int(50 + pulse * 100)
            color = (*self.COLOR_SHIELD, alpha)

            if self.gravity_direction > 0:  # Down
                y_start, y_end = y_end, y_start
                points = [(x, y_end), (x - 5, y_start), (x + 5, y_start)]
            else:  # Up
                points = [(x, y_end), (x - 5, y_start), (x + 5, y_start)]
            
            pygame.gfxdraw.aapolygon(arrow_surf, points, color)
            pygame.gfxdraw.filled_polygon(arrow_surf, points, color)
        self.screen.blit(arrow_surf, (0, 0))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        seconds_left = max(0, self.timer / self.metadata["render_fps"])
        timer_str = f"{int(seconds_left // 60):02}:{int(seconds_left % 60):02}"
        timer_color = self.COLOR_TEXT
        if seconds_left < 10: timer_color = self.COLOR_TIMER_CRIT
        elif seconds_left < 30: timer_color = self.COLOR_TIMER_WARN
        timer_text = self.font_small.render(timer_str, True, timer_color)
        self.screen.blit(timer_text, (self.width - timer_text.get_width() - 10, 10))

        # Deflection Counter
        bar_width = 200
        bar_height = 10
        bar_x = (self.width - bar_width) / 2
        bar_y = 15
        fill_ratio = self.deflection_counter / self.DEFLECTIONS_FOR_FLIP
        fill_width = bar_width * fill_ratio

        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_SHIELD, (bar_x, bar_y, fill_width, bar_height), border_radius=3)

        # Game Over / Win message
        if self.game_over:
            self._draw_centered_text("BASE DESTROYED", self.COLOR_TIMER_CRIT)
        elif self.timer <= 0:
            self._draw_centered_text("SURVIVAL COMPLETE", self.COLOR_BASE)

    def _draw_centered_text(self, text, color):
        text_surf = self.font_large.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.width / 2, self.height / 2))

        bg_rect = text_rect.inflate(20, 20)
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 180))
        self.screen.blit(bg_surf, bg_rect)

        self.screen.blit(text_surf, text_rect)

    def _draw_glowing_circle(self, surface, color, center, radius, alpha, layers):
        if radius <= 0: return
        center_x, center_y = center
        for i in range(layers, 0, -1):
            layer_alpha = int(alpha * (1 / (layers - i + 2)) ** 2)
            layer_color = (*color, layer_alpha)
            pygame.gfxdraw.filled_circle(surface, center_x, center_y, radius - i, layer_color)

        main_radius = int(radius * 0.8)
        if main_radius > 0:
            main_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(surface, center_x, center_y, main_radius, main_color)
            pygame.gfxdraw.aacircle(surface, center_x, center_y, main_radius, main_color)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    pygame.display.init()
    render_screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Gravity Deflector")
    running = True

    action = env.action_space.sample()
    action.fill(0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                if event.key == pygame.K_UP:
                    action[0] = 1  # Gravity Up
                if event.key == pygame.K_DOWN:
                    action[0] = 2  # Gravity Down

        obs, reward, terminated, truncated, info = env.step(action)
        action[0] = 0  # Reset movement action after one step

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(render_surface, (0, 0))
            pygame.display.flip()
            
            wait_for_input = True
            while wait_for_input:
                event = pygame.event.wait()
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    running = False
                    wait_for_input = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    wait_for_input = False
            if not running: break

        if running:
            render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(render_surface, (0, 0))
            pygame.display.flip()

        env.clock.tick(env.metadata["render_fps"])

    env.close()