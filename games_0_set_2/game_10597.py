import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Generated: 2025-08-26T10:40:31.901613
# Source Brief: brief_00597.md
# Brief Index: 597
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your cities from incoming missiles by manipulating wind patterns. "
        "Create high and low-pressure zones to steer projectiles off-course."
    )
    user_guide = (
        "Use arrow keys to move the cursor. Hold Shift and use ↑/↓ to increase/decrease pressure. "
        "Press Space to launch a missile. Hold Shift and press Space to reset pressure at the cursor."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_WIDTH = SCREEN_WIDTH // GRID_WIDTH
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_HEIGHT

    MAX_STEPS = 1500
    TOTAL_MISSILES = 20

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (25, 40, 60)
    COLOR_LAND = (20, 50, 45)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_ACTION = (255, 100, 0)

    COLOR_MISSILE = (255, 80, 80)
    COLOR_MISSILE_TRAIL = (255, 150, 150)

    COLOR_CITY_HEALTHY = (0, 255, 150)
    COLOR_CITY_DAMAGED = (255, 200, 0)
    COLOR_CITY_CRITICAL = (255, 50, 50)

    COLOR_LOW_PRESSURE = (50, 100, 255)
    COLOR_HIGH_PRESSURE = (255, 100, 50)

    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 20)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)

        # Uninitialized variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cities = []
        self.pressure_grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT))
        self.wind_field = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT, 2))
        self.cursor_pos = [0, 0]
        self.missile = None
        self.missiles_remaining = 0
        self.successful_deflections = 0
        self.base_missile_speed = 0.0
        self.game_phase = "PLANNING"
        self.prev_space_held = False
        self.particles = []
        self.land_poly = []

        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Seed python's random and numpy's random for reproducibility
            random.seed(seed)
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_landmass()
        self.cities = self._generate_cities(5)

        self.pressure_grid = np.full((self.GRID_WIDTH, self.GRID_HEIGHT), 0.5)
        self._calculate_wind_field()

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]

        self.missile = None
        self.missiles_remaining = self.TOTAL_MISSILES
        self.successful_deflections = 0
        self.base_missile_speed = 2.0

        self.game_phase = "PLANNING"
        self.prev_space_held = False
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for time passing

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        input_reward = self._handle_input(movement, space_held, shift_held)
        reward += input_reward

        # --- Update Game State ---
        if self.game_phase == "IN_FLIGHT":
            flight_reward = self._update_missile()
            reward += flight_reward

        self._update_particles()

        # --- Check Termination ---
        total_damage = sum(100 - city['health'] for city in self.cities)
        total_max_damage = len(self.cities) * 100 if self.cities else 0
        damage_percent = (total_damage / total_max_damage) * 100 if total_max_damage > 0 else 0

        terminated = False
        if self.missiles_remaining <= 0 and self.game_phase == "PLANNING":
            terminated = True
            if damage_percent < 10:
                reward += 50  # Victory bonus
        elif damage_percent >= 90 and self.cities:
            terminated = True
            reward -= 50  # Catastrophic failure penalty
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated

        self.prev_space_held = space_held

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0

        # --- Cursor Movement (not holding shift) ---
        if not shift_held:
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        # --- Pressure Manipulation (holding shift) ---
        if shift_held:
            cx, cy = self.cursor_pos
            if movement == 1:  # Increase pressure
                self.pressure_grid[cx, cy] = min(1.0, self.pressure_grid[cx, cy] + 0.1)
                self._calculate_wind_field()
            elif movement == 2:  # Decrease pressure
                self.pressure_grid[cx, cy] = max(0.0, self.pressure_grid[cx, cy] - 0.1)
                self._calculate_wind_field()

        # --- Reset Pressure (Shift + Space) ---
        if shift_held and space_held:
            cx, cy = self.cursor_pos
            if self.pressure_grid[cx, cy] != 0.5:
                self.pressure_grid[cx, cy] = 0.5
                self._calculate_wind_field()

        # --- Launch Missile (Space press) ---
        if self.game_phase == "PLANNING" and space_held and not self.prev_space_held and not shift_held:
            if self.missiles_remaining > 0 and self.cities:
                self._launch_missile()
                self.game_phase = "IN_FLIGHT"
                self.missiles_remaining -= 1

        return reward

    def _update_missile(self):
        if not self.missile:
            return 0

        reward = 0

        # Apply wind
        wind = self._get_wind_at(self.missile['pos'][0], self.missile['pos'][1])
        self.missile['vel'][0] += wind[0] * 0.05  # Wind influence factor
        self.missile['vel'][1] += wind[1] * 0.05

        # Clamp velocity to prevent extreme speeds
        speed = math.hypot(self.missile['vel'][0], self.missile['vel'][1])
        if speed > self.base_missile_speed * 2:
            self.missile['vel'][0] = (self.missile['vel'][0] / speed) * self.base_missile_speed * 2
            self.missile['vel'][1] = (self.missile['vel'][1] / speed) * self.base_missile_speed * 2

        # Update position
        self.missile['pos'][0] += self.missile['vel'][0]
        self.missile['pos'][1] += self.missile['vel'][1]

        self.missile['trail'].append(tuple(self.missile['pos']))
        if len(self.missile['trail']) > 20:
            self.missile['trail'].pop(0)

        # Deflection reward
        initial_path_vec = (self.missile['target'][0] - self.missile['start'][0], self.missile['target'][1] - self.missile['start'][1])
        current_pos_vec = (self.missile['pos'][0] - self.missile['start'][0], self.missile['pos'][1] - self.missile['start'][1])

        len_initial = math.hypot(*initial_path_vec)
        if len_initial > 0:
            proj = (current_pos_vec[0] * initial_path_vec[0] + current_pos_vec[1] * initial_path_vec[1]) / len_initial
            proj_point = (self.missile['start'][0] + (proj / len_initial) * initial_path_vec[0],
                          self.missile['start'][1] + (proj / len_initial) * initial_path_vec[1])
            deflection = math.hypot(self.missile['pos'][0] - proj_point[0], self.missile['pos'][1] - proj_point[1])
            reward += min(0.1 * (deflection / 10), 1.0)  # Reward for deflection, capped

        # Check for city collision
        for city in self.cities:
            dist = math.hypot(self.missile['pos'][0] - city['pos'][0], self.missile['pos'][1] - city['pos'][1])
            if dist < city['radius']:
                damage = random.randint(25, 40)
                city['health'] = max(0, city['health'] - damage)
                self.score -= 5
                reward -= 5
                self._create_explosion(city['pos'], self.COLOR_CITY_CRITICAL, 50)
                self.missile = None
                self.game_phase = "PLANNING"
                return reward

        # Check for off-screen
        x, y = self.missile['pos']
        if not (0 < x < self.SCREEN_WIDTH and 0 < y < self.SCREEN_HEIGHT):
            self.score += 10
            reward += 10
            self.successful_deflections += 1
            if self.successful_deflections > 0 and self.successful_deflections % 5 == 0:
                self.base_missile_speed += 0.2
            self._create_explosion(self.missile['pos'], self.COLOR_TEXT, 20)
            self.missile = None
            self.game_phase = "PLANNING"

        return reward

    def _launch_missile(self):
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            start_pos = [random.uniform(0, self.SCREEN_WIDTH), -10]
        elif edge == 'bottom':
            start_pos = [random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 10]
        elif edge == 'left':
            start_pos = [-10, random.uniform(0, self.SCREEN_HEIGHT)]
        else:  # right
            start_pos = [self.SCREEN_WIDTH + 10, random.uniform(0, self.SCREEN_HEIGHT)]

        target_city = random.choice(self.cities)
        target_pos = [
            target_city['pos'][0] + random.uniform(-10, 10),
            target_city['pos'][1] + random.uniform(-10, 10)
        ]

        angle = math.atan2(target_pos[1] - start_pos[1], target_pos[0] - start_pos[0])
        velocity = [math.cos(angle) * self.base_missile_speed, math.sin(angle) * self.base_missile_speed]

        self.missile = {
            'pos': start_pos,
            'vel': velocity,
            'start': tuple(start_pos),
            'target': tuple(target_pos),
            'trail': []
        }

    def _calculate_wind_field(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                p_center = self.pressure_grid[x, y]
                p_right = self.pressure_grid[min(x + 1, self.GRID_WIDTH - 1), y]
                p_left = self.pressure_grid[max(x - 1, 0), y]
                p_down = self.pressure_grid[x, min(y + 1, self.GRID_HEIGHT - 1)]
                p_up = self.pressure_grid[x, max(y - 1, 0)]

                # Pressure gradient -> Wind. Wind flows from high to low pressure.
                grad_x = (p_right - p_left) / 2.0
                grad_y = (p_down - p_up) / 2.0

                wind_x = -grad_x * 5.0  # Wind strength factor
                wind_y = -grad_y * 5.0
                self.wind_field[x, y] = [wind_x, wind_y]

    def _get_wind_at(self, x, y):
        grid_x = x / self.CELL_WIDTH
        grid_y = y / self.CELL_HEIGHT

        x1_raw, y1_raw = int(grid_x), int(grid_y)
        x2_raw, y2_raw = x1_raw + 1, y1_raw + 1

        # Clamp indices to grid bounds. This collapses interpolation at the edges.
        x1 = np.clip(x1_raw, 0, self.GRID_WIDTH - 1)
        x2 = np.clip(x2_raw, 0, self.GRID_WIDTH - 1)
        y1 = np.clip(y1_raw, 0, self.GRID_HEIGHT - 1)
        y2 = np.clip(y2_raw, 0, self.GRID_HEIGHT - 1)

        # Fractional part should be relative to the original integer part.
        # Clip to [0,1] to handle extrapolation for off-screen points.
        fx = np.clip(grid_x - x1_raw, 0.0, 1.0)
        fy = np.clip(grid_y - y1_raw, 0.0, 1.0)

        w11 = self.wind_field[x1, y1]
        w12 = self.wind_field[x1, y2]
        w21 = self.wind_field[x2, y1]
        w22 = self.wind_field[x2, y2]

        top = w11 * (1 - fx) + w21 * fx
        bottom = w12 * (1 - fx) + w22 * fx
        wind = top * (1 - fy) + bottom * fy

        return wind

    def _generate_cities(self, num_cities):
        cities = []
        for _ in range(num_cities):
            for _ in range(100): # Try 100 times to place a city
                pos = [
                    random.uniform(self.SCREEN_WIDTH * 0.1, self.SCREEN_WIDTH * 0.9),
                    random.uniform(self.SCREEN_HEIGHT * 0.1, self.SCREEN_HEIGHT * 0.9)
                ]
                # Ensure cities are not too close to each other
                if all(math.hypot(pos[0] - c['pos'][0], pos[1] - c['pos'][1]) > 80 for c in cities):
                    cities.append({'pos': pos, 'health': 100, 'radius': 15})
                    break
        return cities

    def _generate_landmass(self):
        self.land_poly = []
        cx, cy = self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2
        num_points = 12
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            r = random.uniform(min(cx, cy) * 0.6, min(cx, cy) * 1.2)
            px = cx + math.cos(angle) * r
            py = cy + math.sin(angle) * r
            self.land_poly.append((px, py))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(10, 30),
                'color': color,
                'radius': random.uniform(2, 5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_wind_field()
        self._render_pressure_heatmap()
        self._render_cities()
        self._render_particles()
        self._render_missile()
        self._render_cursor()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        if self.land_poly:
            pygame.gfxdraw.filled_polygon(self.screen, self.land_poly, self.COLOR_LAND)
            pygame.gfxdraw.aapolygon(self.screen, self.land_poly, self.COLOR_LAND)
        for x in range(self.GRID_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.CELL_WIDTH, 0), (x * self.CELL_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.CELL_HEIGHT), (self.SCREEN_WIDTH, y * self.CELL_HEIGHT))

    def _render_pressure_heatmap(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                pressure = self.pressure_grid[x, y]
                if abs(pressure - 0.5) > 0.01:
                    alpha = int(abs(pressure - 0.5) * 2 * 100)
                    color = self.COLOR_HIGH_PRESSURE if pressure > 0.5 else self.COLOR_LOW_PRESSURE

                    temp_surf = pygame.Surface((self.CELL_WIDTH, self.CELL_HEIGHT), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, (*color, alpha), (self.CELL_WIDTH // 2, self.CELL_HEIGHT // 2), self.CELL_WIDTH // 2)
                    self.screen.blit(temp_surf, (x * self.CELL_WIDTH, y * self.CELL_HEIGHT))

    def _render_wind_field(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                center_x = int(x * self.CELL_WIDTH + self.CELL_WIDTH / 2)
                center_y = int(y * self.CELL_HEIGHT + self.CELL_HEIGHT / 2)
                wind_vec = self.wind_field[x, y]

                strength = math.hypot(*wind_vec)
                if strength > 0.1:
                    end_x = center_x + wind_vec[0] * 3
                    end_y = center_y + wind_vec[1] * 3
                    color = (50, 70, 100, min(255, int(strength * 50)))
                    pygame.draw.line(self.screen, color, (center_x, center_y), (end_x, end_y), 1)

    def _render_cities(self):
        for city in self.cities:
            x, y = int(city['pos'][0]), int(city['pos'][1])
            radius = int(city['radius'])
            health_frac = city['health'] / 100.0

            if health_frac > 0.66:
                color = self.COLOR_CITY_HEALTHY
            elif health_frac > 0.33:
                color = self.COLOR_CITY_DAMAGED
            else:
                color = self.COLOR_CITY_CRITICAL

            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

            # Health bar
            bar_width = 30
            bar_height = 5
            bar_x = x - bar_width // 2
            bar_y = y + radius + 5
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, int(bar_width * health_frac), bar_height))

    def _render_missile(self):
        if self.missile:
            # Trail
            if len(self.missile['trail']) > 1:
                for i in range(len(self.missile['trail']) - 1):
                    p1 = self.missile['trail'][i]
                    p2 = self.missile['trail'][i + 1]
                    alpha = int((i / len(self.missile['trail'])) * 255)
                    pygame.draw.line(self.screen, (*self.COLOR_MISSILE_TRAIL, alpha), (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 2)

            # Missile head
            x, y = int(self.missile['pos'][0]), int(self.missile['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, 4, self.COLOR_MISSILE)
            pygame.gfxdraw.aacircle(self.screen, x, y, 4, self.COLOR_MISSILE)
            pygame.gfxdraw.aacircle(self.screen, x, y, 6, (*self.COLOR_MISSILE, 128))  # Glow

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'] * (p['life'] / 30.0))
            if radius > 0:
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        x, y = cx * self.CELL_WIDTH, cy * self.CELL_HEIGHT
        # This check is for human play, not strictly needed for headless mode
        shift_pressed = False
        try:
            shift_pressed = pygame.key.get_pressed()[pygame.K_LSHIFT]
        except pygame.error: # When no display is initialized, get_pressed fails
            pass
        color = self.COLOR_CURSOR_ACTION if shift_pressed else self.COLOR_CURSOR
        rect = (x, y, self.CELL_WIDTH, self.CELL_HEIGHT)
        pygame.draw.rect(self.screen, color, rect, 2)
        pygame.draw.line(self.screen, color, (x + 2, y + self.CELL_HEIGHT // 2), (x + self.CELL_WIDTH - 2, y + self.CELL_HEIGHT // 2), 1)
        pygame.draw.line(self.screen, color, (x + self.CELL_WIDTH // 2, y + 2), (x + self.CELL_WIDTH // 2, y + self.CELL_HEIGHT - 2), 1)

    def _render_ui(self):
        total_damage = sum(100 - city['health'] for city in self.cities)
        total_max_damage = len(self.cities) * 100 if self.cities else 0
        damage_percent = (total_damage / total_max_damage) * 100 if total_max_damage > 0 else 0

        texts = [
            f"SCORE: {int(self.score)}",
            f"MISSILES: {self.missiles_remaining}/{self.TOTAL_MISSILES}",
            f"DAMAGE: {damage_percent:.1f}%"
        ]

        for i, text in enumerate(texts):
            self._draw_text(text, (10 + i * 200, 10), self.font_small)

        if self.game_phase == "PLANNING" and self.missiles_remaining > 0:
            self._draw_text("AWAITING LAUNCH", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30), self.font_large, center=True)
        elif self.game_over:
            end_text = "MISSION COMPLETE" if self.missiles_remaining <= 0 else "MISSION FAILED"
            self._draw_text(end_text, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.font_large, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos

        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        total_damage = sum(100 - city['health'] for city in self.cities)
        total_max_damage = len(self.cities) * 100 if self.cities else 0
        damage_percent = (total_damage / total_max_damage) * 100 if total_max_damage > 0 else 0
        return {
            "score": self.score,
            "steps": self.steps,
            "missiles_remaining": self.missiles_remaining,
            "total_damage_percent": damage_percent,
            "game_phase": self.game_phase
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False

    # Manual play loop
    # Use arrow keys to move cursor, LSHIFT + UP/DOWN to change pressure
    # SPACE to launch, LSHIFT + SPACE to reset pressure at cursor

    action = [0, 0, 0]  # no-op, released, released

    # Create a display for human play
    pygame.display.set_caption("Wind Commander")
    display_surf = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()

        # Movement
        action[0] = 0  # None
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4

        # Space
        action[1] = 1 if keys[pygame.K_SPACE] else 0

        # Shift
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # For human play, render to screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_surf.blit(surf, (0, 0))

        pygame.display.flip()
        env.clock.tick(30)  # Limit to 30 FPS for human play

    env.close()