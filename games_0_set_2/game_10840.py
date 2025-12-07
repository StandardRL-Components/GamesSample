import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent controls two gravity wells to sort
    splitting asteroids into designated zones.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control two gravity wells to sort splitting asteroids into their designated color-coded zones before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the blue gravity well. Hold space to move the orange well up and shift to move it down."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    ZONE_WIDTH = 80
    MAX_ASTEROIDS = 100
    GAME_DURATION_SECONDS = 60

    # --- Colors ---
    COLOR_BG = (10, 15, 30)
    COLOR_STARS = (200, 200, 220)
    COLOR_GREEN_ASTEROID = (80, 255, 150)
    COLOR_RED_ASTEROID = (255, 80, 120)
    COLOR_WELL_1 = (100, 180, 255)
    COLOR_WELL_2 = (255, 180, 100)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_ZONE_GREEN = (80, 255, 150, 30)
    COLOR_ZONE_RED = (255, 80, 120, 30)

    class Asteroid:
        def __init__(self, pos, vel, size, type):
            self.pos = pygame.Vector2(pos)
            self.vel = pygame.Vector2(vel)
            self.size = size
            self.type = type  # 'green' or 'red'
            self.color = GameEnv.COLOR_GREEN_ASTEROID if type == 'green' else GameEnv.COLOR_RED_ASTEROID
            self.target_x = GameEnv.ZONE_WIDTH / 2 if type == 'green' else GameEnv.SCREEN_WIDTH - GameEnv.ZONE_WIDTH / 2
            self.collided_this_frame = False

    class Particle:
        def __init__(self, pos, vel, size, color, lifetime):
            self.pos = pygame.Vector2(pos)
            self.vel = pygame.Vector2(vel)
            self.size = size
            self.color = color
            self.lifetime = lifetime
            self.initial_lifetime = lifetime

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # --- State Variables ---
        self.asteroids = []
        self.particles = []
        self.gravity_well_1 = pygame.Vector2(0, 0)
        self.gravity_well_2 = pygame.Vector2(0, 0)
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.initial_green_count = 0
        self.initial_red_count = 0
        self.stars = []
        self.np_random = None

        self._generate_stars()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.time_remaining = self.FPS * self.GAME_DURATION_SECONDS

        self.gravity_well_1.update(self.SCREEN_WIDTH * 0.25, self.SCREEN_HEIGHT / 2)
        self.gravity_well_2.update(self.SCREEN_WIDTH * 0.75, self.SCREEN_HEIGHT / 2)

        self.asteroids.clear()
        self.particles.clear()

        self.initial_green_count = 0
        self.initial_red_count = 0
        num_initial_asteroids = 15

        for _ in range(num_initial_asteroids):
            pos = (
                self.np_random.uniform(self.ZONE_WIDTH * 1.5, self.SCREEN_WIDTH - self.ZONE_WIDTH * 1.5),
                self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
            )
            vel = (self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5))
            size = self.np_random.uniform(12, 16)
            type = 'green' if self.np_random.random() < 0.5 else 'red'

            if type == 'green': self.initial_green_count += 1
            else: self.initial_red_count += 1

            self.asteroids.append(self.Asteroid(pos, vel, size, type))

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        self.time_remaining -= 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        step_reward = self._update_physics()

        # Time penalty
        step_reward -= 0.001

        terminated = False
        win = len(self.asteroids) == 0 and (self.initial_green_count + self.initial_red_count > 0)
        timeout = self.time_remaining <= 0
        overpopulation = len(self.asteroids) > self.MAX_ASTEROIDS

        if win:
            terminated = True
            step_reward += 100
        elif timeout or overpopulation:
            terminated = True
            step_reward -= 100

        truncated = False # No truncation condition in this game
        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        well_speed = 5.0

        # Gravity Well 1 (controlled by arrows)
        if movement == 1: self.gravity_well_1.y -= well_speed
        if movement == 2: self.gravity_well_1.y += well_speed
        if movement == 3: self.gravity_well_1.x -= well_speed
        if movement == 4: self.gravity_well_1.x += well_speed

        # Gravity Well 2 (controlled by space/shift)
        if space_held: self.gravity_well_2.y -= well_speed
        if shift_held: self.gravity_well_2.y += well_speed

        # Clamp well positions to screen
        self.gravity_well_1.x = np.clip(self.gravity_well_1.x, 0, self.SCREEN_WIDTH)
        self.gravity_well_1.y = np.clip(self.gravity_well_1.y, 0, self.SCREEN_HEIGHT)
        self.gravity_well_2.x = np.clip(self.gravity_well_2.x, 0, self.SCREEN_WIDTH)
        self.gravity_well_2.y = np.clip(self.gravity_well_2.y, 0, self.SCREEN_HEIGHT)

    def _update_physics(self):
        reward = 0

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.pos += p.vel
            p.lifetime -= 1
            p.size = max(0, p.size - 0.1)

        # --- Update Asteroids ---
        for a in self.asteroids: a.collided_this_frame = False

        asteroids_to_add = []
        asteroids_to_remove = []

        for i, a1 in enumerate(self.asteroids):
            # Continuous reward for moving closer
            old_dist = a1.pos.distance_to(pygame.Vector2(a1.target_x, a1.pos.y))

            # Apply gravity
            gravity_strength = 600
            for well_pos in [self.gravity_well_1, self.gravity_well_2]:
                vec_to_well = well_pos - a1.pos
                dist_sq = max(100, vec_to_well.length_squared()) # Avoid division by zero and extreme forces
                force_dir = vec_to_well.normalize()
                force_mag = gravity_strength / dist_sq
                a1.vel += force_dir * force_mag / a1.size

            # Update position
            a1.vel *= 0.99 # Dampening
            a1.pos += a1.vel

            new_dist = a1.pos.distance_to(pygame.Vector2(a1.target_x, a1.pos.y))
            reward += (old_dist - new_dist) * 0.005 # Scaled continuous reward

            # --- Boundary Collisions ---
            # Top/Bottom Bounce
            if a1.pos.y < a1.size or a1.pos.y > self.SCREEN_HEIGHT - a1.size:
                a1.vel.y *= -0.8
                a1.pos.y = np.clip(a1.pos.y, a1.size, self.SCREEN_HEIGHT - a1.size)

            # Left/Right Zone Handling
            is_in_green_zone = a1.pos.x < self.ZONE_WIDTH
            is_in_red_zone = a1.pos.x > self.SCREEN_WIDTH - self.ZONE_WIDTH

            if is_in_green_zone:
                if a1.type == 'green':
                    asteroids_to_remove.append(a1)
                    reward += 1.0
                    self.score += 1
                    self._create_particles(a1.pos, a1.color, 30, 2.5, 40)
                else: # Wrong zone
                    reward -= 0.5
                    a1.vel.x *= -0.8
                    a1.pos.x = self.ZONE_WIDTH + a1.size
                    self._create_particles(a1.pos, (255, 255, 255), 10, 1.5, 20)
            elif is_in_red_zone:
                if a1.type == 'red':
                    asteroids_to_remove.append(a1)
                    reward += 1.0
                    self.score += 1
                    self._create_particles(a1.pos, a1.color, 30, 2.5, 40)
                else: # Wrong zone
                    reward -= 0.5
                    a1.vel.x *= -0.8
                    a1.pos.x = self.SCREEN_WIDTH - self.ZONE_WIDTH - a1.size
                    self._create_particles(a1.pos, (255, 255, 255), 10, 1.5, 20)

            # --- Asteroid-Asteroid Collision ---
            for j in range(i + 1, len(self.asteroids)):
                a2 = self.asteroids[j]
                if a1.collided_this_frame or a2.collided_this_frame: continue

                dist = a1.pos.distance_to(a2.pos)
                if dist < a1.size + a2.size:
                    a1.collided_this_frame = True
                    a2.collided_this_frame = True
                    asteroids_to_remove.extend([a1, a2])
                    self._create_particles((a1.pos + a2.pos) / 2, (255,255,255), 15, 2.0, 25)

                    if a1.size > 4: asteroids_to_add.extend(self._split_asteroid(a1))
                    if a2.size > 4: asteroids_to_add.extend(self._split_asteroid(a2))
                    break

        # Apply list changes
        self.asteroids = [a for a in self.asteroids if a not in asteroids_to_remove]
        self.asteroids.extend(asteroids_to_add)

        return reward

    def _split_asteroid(self, asteroid):
        new_size = asteroid.size / math.sqrt(2)

        # Create two new asteroids moving apart
        angle1 = self.np_random.uniform(0, 2 * math.pi)
        vel1 = pygame.Vector2(math.cos(angle1), math.sin(angle1)) * 1.5
        pos1 = asteroid.pos + vel1 * new_size

        angle2 = angle1 + math.pi
        vel2 = pygame.Vector2(math.cos(angle2), math.sin(angle2)) * 1.5
        pos2 = asteroid.pos + vel2 * new_size

        return [
            self.Asteroid(pos1, asteroid.vel + vel1, new_size, asteroid.type),
            self.Asteroid(pos2, asteroid.vel + vel2, new_size, asteroid.type)
        ]

    def _create_particles(self, pos, color, count, speed_range, life_range):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(speed_range * 0.5, speed_range)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            size = self.np_random.uniform(1, 4)
            lifetime = self.np_random.integers(life_range // 2, life_range + 1)
            self.particles.append(self.Particle(pos, vel, size, color, lifetime))

    def _get_observation(self):
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        green_remaining = sum(1 for a in self.asteroids if a.type == 'green')
        red_remaining = sum(1 for a in self.asteroids if a.type == 'red')
        return {"score": self.score, "steps": self.steps, "green_remaining": green_remaining, "red_remaining": red_remaining}

    def _render_frame(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_zones()
        self._render_particles()
        self._render_asteroids()
        self._render_gravity_wells()
        self._render_ui()

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append((
                (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)),
                random.uniform(0.5, 1.5) # size
            ))

    def _render_stars(self):
        for pos, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STARS, pos, size)

    def _render_zones(self):
        # Use a separate surface for alpha blending
        zone_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(zone_surface, self.COLOR_ZONE_GREEN, (0, 0, self.ZONE_WIDTH, self.SCREEN_HEIGHT))
        pygame.draw.rect(zone_surface, self.COLOR_ZONE_RED, (self.SCREEN_WIDTH - self.ZONE_WIDTH, 0, self.ZONE_WIDTH, self.SCREEN_HEIGHT))
        self.screen.blit(zone_surface, (0,0))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p.lifetime / p.initial_lifetime))
            color = (*p.color, alpha)
            # Simple rect particle is faster than circle for many particles
            pygame.draw.rect(self.screen, color, (p.pos.x, p.pos.y, max(1, p.size), max(1, p.size)))

    def _render_asteroids(self):
        for a in self.asteroids:
            # Glow effect
            glow_radius = int(a.size * 1.5)
            glow_color = (*a.color, 70)
            self._render_antialiased_circle(self.screen, glow_color, (int(a.pos.x), int(a.pos.y)), glow_radius)
            # Main body
            self._render_antialiased_circle(self.screen, a.color, (int(a.pos.x), int(a.pos.y)), int(a.size))

    def _render_gravity_wells(self):
        wells = [(self.gravity_well_1, self.COLOR_WELL_1), (self.gravity_well_2, self.COLOR_WELL_2)]
        for pos, color in wells:
            # Influence radius
            influence_color = (*color, 20)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 100, influence_color)
            # Core glow
            glow_color = (*color, 100)
            self._render_antialiased_circle(self.screen, glow_color, (int(pos.x), int(pos.y)), 15)
            # Core
            self._render_antialiased_circle(self.screen, color, (int(pos.x), int(pos.y)), 8)

    def _render_ui(self):
        # Timer
        time_text = f"{self.time_remaining / self.FPS:.1f}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 15, 10))

        # Asteroid counts
        green_rem = sum(1 for a in self.asteroids if a.type == 'green')
        red_rem = sum(1 for a in self.asteroids if a.type == 'red')

        green_text = f"GREEN: {green_rem}"
        red_text = f"RED: {red_rem}"

        green_surf = self.font_small.render(green_text, True, self.COLOR_GREEN_ASTEROID)
        red_surf = self.font_small.render(red_text, True, self.COLOR_RED_ASTEROID)

        self.screen.blit(green_surf, (15, 10))
        self.screen.blit(red_surf, (15, 30))

    def _render_antialiased_circle(self, surface, color, center, radius):
        if radius > 0:
            pygame.gfxdraw.aacircle(surface, center[0], center[1], int(radius), color)
            pygame.gfxdraw.filled_circle(surface, center[0], center[1], int(radius), color)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is not used by the evaluation environment.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display for manual play
    env = GameEnv()
    obs, info = env.reset(seed=42)

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Astro-Sorter")
    clock = pygame.time.Clock()

    running = True
    terminated = False

    while running:
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()
            terminated = False

        # --- Manual Control Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    env.close()