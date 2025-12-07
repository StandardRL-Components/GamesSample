
# Generated: 2025-08-27T23:32:09.647115
# Source Brief: brief_03494.md
# Brief Index: 3494

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold space to mine nearby asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a spaceship through an asteroid field, mining valuable minerals while managing limited fuel."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.WIN_MINERALS = 50
        self.MAX_STEPS = 1000
        self.INITIAL_FUEL = 250
        self.NUM_ASTEROIDS = 25
        self.SHIP_SPEED = 5
        self.MINING_RANGE = 70
        self.MINING_RATE = 1
        self.FUEL_COST_MOVE = 1

        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_STARS = (200, 200, 220)
        self.COLOR_SHIP = (60, 180, 255)
        self.COLOR_SHIP_GLOW = (60, 180, 255, 50)
        self.COLOR_ASTEROID_BASE = (80, 80, 90)
        self.COLOR_ASTEROID_RICH = (220, 220, 240)
        self.COLOR_LASER = (255, 220, 0)
        self.COLOR_LASER_INNER = (255, 255, 150)
        self.COLOR_PARTICLE = (255, 180, 50)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_FUEL_BAR = (255, 150, 0)
        self.COLOR_FUEL_LOW = (255, 50, 50)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- Game State (initialized in reset) ---
        self.ship_pos = None
        self.fuel = None
        self.minerals_collected = None
        self.asteroids = None
        self.stars = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.mining_info = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.ship_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.fuel = self.INITIAL_FUEL
        self.minerals_collected = 0

        self._generate_stars()
        self._generate_asteroids()

        self.particles = []
        self.mining_info = {'target': None, 'active': False}

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # 1. Handle Movement
        move_vec = np.array([0, 0], dtype=float)
        if movement == 1: move_vec[1] = -1  # UP
        elif movement == 2: move_vec[1] = 1   # DOWN
        elif movement == 3: move_vec[0] = -1  # LEFT
        elif movement == 4: move_vec[0] = 1   # RIGHT

        if np.any(move_vec):
            self.ship_pos += move_vec * self.SHIP_SPEED
            self.fuel = max(0, self.fuel - self.FUEL_COST_MOVE)
            reward -= 0.01  # Small penalty for fuel use
            # World wrap
            self.ship_pos[0] %= self.WIDTH
            self.ship_pos[1] %= self.HEIGHT

        # 2. Handle Mining
        self.mining_info['active'] = False
        if space_held:
            closest_asteroid, min_dist = None, self.MINING_RANGE
            for asteroid in self.asteroids:
                dist = np.linalg.norm(self.ship_pos - asteroid['pos'])
                if dist < min_dist:
                    min_dist = dist
                    closest_asteroid = asteroid

            if closest_asteroid:
                self.mining_info['active'] = True
                self.mining_info['target'] = closest_asteroid

                mined_this_step = min(closest_asteroid['minerals'], self.MINING_RATE)
                if mined_this_step > 0:
                    reward += 1.0  # Event reward for successful mining action
                    reward += mined_this_step * 0.1  # Per-mineral reward

                    self.minerals_collected += mined_this_step
                    closest_asteroid['minerals'] -= mined_this_step

                    self._create_mining_particles(closest_asteroid['pos'])
                    # sound: mining_hit.wav

        # 3. Update Game State
        self.asteroids = [a for a in self.asteroids if a['minerals'] > 0]
        self._update_particles()
        self.steps += 1

        # 4. Check Termination
        terminated = False
        if self.minerals_collected >= self.WIN_MINERALS:
            terminated = True
            reward += 100
            self.game_over = True
        elif self.fuel <= 0:
            terminated = True
            reward -= 100
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        # --- Clear Screen ---
        self.screen.fill(self.COLOR_BG)

        # --- Render Background ---
        for star in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STARS, star['pos'], star['radius'])

        # --- Render Game Objects ---
        for asteroid in self.asteroids:
            points = asteroid['pos'] + asteroid['shape']
            mineral_ratio = asteroid['minerals'] / asteroid['max_minerals']
            color = [int(a + (b - a) * mineral_ratio) for a, b in zip(self.COLOR_ASTEROID_BASE, self.COLOR_ASTEROID_RICH)]
            
            offset_points = points + np.array([3, 3])
            pygame.draw.polygon(self.screen, tuple(int(c * 0.6) for c in color), offset_points.astype(int))
            pygame.draw.polygon(self.screen, color, points.astype(int))
            pygame.gfxdraw.aapolygon(self.screen, points.astype(int), color)
        
        self._render_particles()
        self._render_ship()

        if self.mining_info['active'] and self.mining_info['target']:
            # sound: laser_beam.wav
            start_pos = self.ship_pos.astype(int)
            end_pos = self.mining_info['target']['pos'].astype(int)
            end_pos[0] += self.np_random.integers(-2, 3)
            end_pos[1] += self.np_random.integers(-2, 3)
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, 5)
            pygame.draw.aaline(self.screen, self.COLOR_LASER_INNER, start_pos, end_pos, 2)

        # --- Render UI ---
        self._render_ui()

        # --- Render Game Over Screen ---
        if self.game_over:
            self._render_game_over()

        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.fuel,
            "minerals": self.minerals_collected,
        }

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            pos = (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT))
            radius = self.np_random.uniform(0.5, 1.5)
            self.stars.append({'pos': pos, 'radius': radius})

    def _generate_asteroids(self):
        self.asteroids = []
        start_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2])
        for _ in range(self.NUM_ASTEROIDS):
            while True:
                pos = self.np_random.uniform(0, [self.WIDTH, self.HEIGHT], size=2)
                if np.linalg.norm(pos - start_pos) > 100:
                    break
            
            radius = self.np_random.integers(15, 30)
            minerals = self.np_random.integers(5, 25)
            
            num_vertices = self.np_random.integers(7, 12)
            points = []
            for i in range(num_vertices):
                angle = 2 * math.pi * i / num_vertices
                dist = self.np_random.uniform(0.7, 1.0) * radius
                points.append([math.cos(angle) * dist, math.sin(angle) * dist])
            
            self.asteroids.append({
                'pos': pos, 'radius': radius, 'minerals': minerals,
                'max_minerals': minerals, 'shape': np.array(points)
            })

    def _create_mining_particles(self, origin_pos):
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': origin_pos.copy(), 'vel': vel, 'lifetime': lifetime})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95  # drag
            p['lifetime'] -= 1
            if p['lifetime'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _render_particles(self):
        for p in self.particles:
            size = max(1, p['lifetime'] / 5)
            alpha = max(0, min(255, p['lifetime'] * 15))
            particle_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            color = (*self.COLOR_PARTICLE, alpha)
            pygame.draw.circle(particle_surf, color, (size, size), size)
            self.screen.blit(particle_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

    def _render_ship(self):
        p1 = np.array([0, -12])
        p2 = np.array([-7, 8])
        p3 = np.array([7, 8])
        points = np.array([p1, p2, p3]) + self.ship_pos

        glow_surface = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_SHIP_GLOW, (15, 15), 15)
        self.screen.blit(glow_surface, (int(self.ship_pos[0] - 15), int(self.ship_pos[1] - 15)))

        pygame.draw.polygon(self.screen, self.COLOR_SHIP, points.astype(int))
        pygame.gfxdraw.aapolygon(self.screen, points.astype(int), self.COLOR_SHIP)

    def _render_ui(self):
        fuel_ratio = max(0, self.fuel / self.INITIAL_FUEL)
        bar_width, bar_height = 200, 20
        fuel_width = int(bar_width * fuel_ratio)
        fuel_color = self.COLOR_FUEL_BAR if fuel_ratio > 0.25 else self.COLOR_FUEL_LOW

        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, fuel_color, (10, 10, fuel_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, 10, bar_width, bar_height), 1)

        mineral_text = self.font_ui.render(f"MINERALS: {self.minerals_collected} / {self.WIN_MINERALS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(mineral_text, (10, 40))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))

        if self.minerals_collected >= self.WIN_MINERALS:
            text, color = "MISSION COMPLETE", self.COLOR_WIN
        else:
            text, color = "GAME OVER", self.COLOR_LOSE

        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))

        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()