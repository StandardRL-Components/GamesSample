import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
import os
import pygame


# Set the SDL video driver to dummy to run Pygame headless
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player pilots a spaceship through an asteroid field.
    The goal is to collect fuel orbs to survive while managing momentum and avoiding collisions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- Metadata for Gymnasium integration ---
    game_description = (
        "Pilot a spaceship through a dangerous asteroid field. Collect energy orbs to refuel and survive, "
        "but watch out for collisions!"
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to apply thrust and navigate your ship."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_SHIP = (255, 255, 255)
        self.COLOR_SHIP_GLOW = (200, 200, 255)
        self.COLOR_ORB = (0, 200, 255)
        self.COLOR_ORB_GLOW = (100, 220, 255)
        self.COLOR_ASTEROID = (100, 110, 120)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_FUEL_HIGH = (0, 255, 128)
        self.COLOR_FUEL_LOW = (255, 70, 70)

        # Game parameters
        self.MAX_STEPS = 2000
        self.NUM_ASTEROIDS = 25
        self.NUM_ORBS = 40
        self.WIN_PERCENTAGE = 0.3

        # Player physics
        self.THRUST_STRENGTH = 0.35
        self.MAX_SPEED = 6.0
        self.FRICTION = 0.985
        self.PLAYER_RADIUS = 12
        self.COLLISION_SPEED_LOSS = 0.5

        # Fuel
        self.MAX_FUEL = 100.0
        self.FUEL_DEPLETION_RATE = 0.05

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_angle = -90.0
        self.last_thrust_direction = np.array([0, -1])
        self.fuel = 0.0
        self.orbs_collected = 0
        self.win_condition_orbs = 0

        self.asteroids = []
        self.orbs = []
        self.stars = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player_pos = np.array([self.WIDTH / 2.0, self.HEIGHT / 2.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_angle = -90.0
        self.last_thrust_direction = np.array([0, -1])

        # Resources
        self.fuel = self.MAX_FUEL
        self.orbs_collected = 0
        self.win_condition_orbs = math.ceil(self.NUM_ORBS * self.WIN_PERCENTAGE)

        # Procedural generation
        self._generate_level()
        self.particles.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1 # Unused in this brief
        # shift_held = action[2] == 1 # Unused in this brief

        reward = 0.0
        self.steps += 1

        # --- Update Game Logic ---
        self._apply_thrust(movement)
        self._update_player_state()

        # Deplete fuel and add survival reward
        self.fuel = max(0, self.fuel - self.FUEL_DEPLETION_RATE)
        reward += 0.01  # Small reward for surviving a step

        # Handle collisions and update rewards
        reward += self._handle_collisions()

        # Update particles
        self._update_particles()

        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.fuel <= 0:
            reward -= 100.0
            terminated = True
        elif self.orbs_collected >= self.win_condition_orbs:
            reward += 100.0
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            terminated = True

        self.game_over = terminated
        self.score = self.orbs_collected  # Score is number of orbs collected

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _generate_level(self):
        # Generate starfield
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
            for _ in range(100)
        ]

        # Generate asteroids
        self.asteroids.clear()
        for _ in range(self.NUM_ASTEROIDS):
            while True:
                pos = self.np_random.uniform(0, [self.WIDTH, self.HEIGHT])
                if np.linalg.norm(pos - self.player_pos) > 100:  # Don't spawn on player
                    radius = self.np_random.uniform(15, 35)
                    self.asteroids.append({'pos': pos, 'radius': radius, 'shape': self._create_asteroid_shape(radius)})
                    break

        # Generate orbs
        self.orbs.clear()
        for _ in range(self.NUM_ORBS):
            while True:
                pos = self.np_random.uniform([20, 20], [self.WIDTH - 20, self.HEIGHT - 20])
                # Ensure orbs don't spawn inside asteroids
                if not any(np.linalg.norm(pos - ast['pos']) < ast['radius'] + 15 for ast in self.asteroids):
                    self.orbs.append({'pos': pos, 'radius': 8})
                    break

    def _create_asteroid_shape(self, radius):
        points = []
        num_vertices = self.np_random.integers(7, 13)
        for i in range(num_vertices):
            angle = i * (2 * math.pi / num_vertices)
            dist = self.np_random.uniform(radius * 0.7, radius * 1.3)
            points.append((math.cos(angle) * dist, math.sin(angle) * dist))
        return points

    def _apply_thrust(self, movement):
        thrust_vector = np.array([0.0, 0.0])
        if movement == 1:  # Up
            thrust_vector = np.array([0, -1])
        elif movement == 2:  # Down
            thrust_vector = np.array([0, 1])
        elif movement == 3:  # Left
            thrust_vector = np.array([-1, 0])
        elif movement == 4:  # Right
            thrust_vector = np.array([1, 0])

        if np.any(thrust_vector):
            self.player_vel += thrust_vector * self.THRUST_STRENGTH
            self.last_thrust_direction = thrust_vector

            # Create thruster particles
            angle = math.atan2(-thrust_vector[1], -thrust_vector[0]) + self.np_random.uniform(-0.2, 0.2)
            speed = self.np_random.uniform(2, 4)
            particle_vel = np.array([math.cos(angle), math.sin(angle)]) * speed

            back_offset = -self.last_thrust_direction * self.PLAYER_RADIUS
            emitter_pos = self.player_pos + back_offset

            for _ in range(2):
                self.particles.append({
                    'pos': emitter_pos.copy(),
                    'vel': particle_vel * self.np_random.uniform(0.8, 1.2),
                    'lifespan': self.np_random.integers(10, 21),
                    'color': random.choice([(255, 180, 50), (255, 100, 0)]),
                    'radius': self.np_random.uniform(1, 3)
                })

    def _update_player_state(self):
        self.player_vel *= self.FRICTION
        speed = np.linalg.norm(self.player_vel)
        if speed > self.MAX_SPEED:
            self.player_vel = (self.player_vel / speed) * self.MAX_SPEED
        self.player_pos += self.player_vel
        target_angle = math.degrees(math.atan2(self.last_thrust_direction[1], self.last_thrust_direction[0])) + 90
        angle_diff = (target_angle - self.player_angle + 180) % 360 - 180
        self.player_angle += angle_diff * 0.2
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)

    def _handle_collisions(self):
        reward = 0.0

        # Asteroid collisions
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_RADIUS + asteroid['radius']:
                self.player_vel *= self.COLLISION_SPEED_LOSS
                reward -= 5.0
                for _ in range(20):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 5)
                    vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                    self.particles.append({
                        'pos': self.player_pos.copy(),
                        'vel': vel,
                        'lifespan': self.np_random.integers(20, 41),
                        'color': random.choice([(180, 180, 180), (255, 255, 100)]),
                        'radius': self.np_random.uniform(1, 4)
                    })
                overlap = (self.PLAYER_RADIUS + asteroid['radius']) - dist
                direction = (self.player_pos - asteroid['pos']) / dist
                self.player_pos += direction * overlap

        # Orb collisions
        orbs_to_keep = []
        for orb in self.orbs:
            dist = np.linalg.norm(self.player_pos - orb['pos'])
            if dist < self.PLAYER_RADIUS + orb['radius']:
                self.orbs_collected += 1
                reward += 1.0
            else:
                orbs_to_keep.append(orb)
        self.orbs = orbs_to_keep

        return reward

    def _update_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                particles_to_keep.append(p)
        self.particles = particles_to_keep

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, (200, 200, 220), (x, y, size, size))

        for ast in self.asteroids:
            points = [(p[0] + ast['pos'][0], p[1] + ast['pos'][1]) for p in ast['shape']]
            pygame.draw.polygon(self.screen, self.COLOR_ASTEROID, points)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)

        for orb in self.orbs:
            pos = (int(orb['pos'][0]), int(orb['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], orb['radius'] + 4, self.COLOR_ORB_GLOW + (30,))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], orb['radius'] + 2, self.COLOR_ORB_GLOW + (60,))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], orb['radius'], self.COLOR_ORB)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], orb['radius'], self.COLOR_ORB)

        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 20.0))
            color = p['color'] + (max(0, min(255, alpha)),)
            temp_surf = pygame.Surface((p['radius'] * 2, p['radius'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (p['pos'][0] - p['radius'], p['pos'][1] - p['radius']))

        self._render_player()

    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS + 5, self.COLOR_SHIP_GLOW + (30,))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS + 3, self.COLOR_SHIP_GLOW + (50,))

        rotated_points = []
        points = [
            (0, -self.PLAYER_RADIUS),
            (-self.PLAYER_RADIUS * 0.7, self.PLAYER_RADIUS * 0.7),
            (self.PLAYER_RADIUS * 0.7, self.PLAYER_RADIUS * 0.7)
        ]
        angle_rad = math.radians(self.player_angle)
        for x, y in points:
            rx = x * math.cos(angle_rad) - y * math.sin(angle_rad) + self.player_pos[0]
            ry = x * math.sin(angle_rad) + y * math.cos(angle_rad) + self.player_pos[1]
            rotated_points.append((int(rx), int(ry)))

        pygame.draw.polygon(self.screen, self.COLOR_SHIP, rotated_points)
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_SHIP)

    def _render_ui(self):
        fuel_ratio = self.fuel / self.MAX_FUEL
        bar_width = 200
        bar_height = 20
        fuel_width = int(bar_width * fuel_ratio)
        fuel_color = self.COLOR_FUEL_LOW if fuel_ratio < 0.3 else self.COLOR_FUEL_HIGH

        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, fuel_color, (10, 10, fuel_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, 10, bar_width, bar_height), 2)
        fuel_text = self.font_small.render("FUEL", True, self.COLOR_TEXT)
        self.screen.blit(fuel_text, (15, 12))

        orb_text_str = f"ORBS: {self.orbs_collected} / {self.win_condition_orbs}"
        orb_text = self.font_large.render(orb_text_str, True, self.COLOR_TEXT)
        text_rect = orb_text.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(orb_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.fuel,
            "orbs_collected": self.orbs_collected,
            "player_pos": self.player_pos.tolist(),
            "player_vel": self.player_vel.tolist(),
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Block ---
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    truncated = False

    # Use a separate display for human play
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Collector")

    total_reward = 0

    # Game loop
    running = True
    while running:
        # Action defaults
        movement = 0  # no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(
                f"Step: {info['steps']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, "
                f"Terminated: {terminated}, Truncated: {truncated}, Orbs: {info['orbs_collected']}"
            )

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))

        if terminated or truncated:
            end_font = pygame.font.SysFont("monospace", 50, bold=True)
            if info['orbs_collected'] >= env.win_condition_orbs:
                end_text = end_font.render("YOU WIN!", True, (100, 255, 100))
            else:
                end_text = end_font.render("GAME OVER", True, (255, 100, 100))

            text_rect = end_text.get_rect(center=(env.WIDTH / 2, env.HEIGHT / 2 - 30))
            display_screen.blit(end_text, text_rect)

            restart_font = pygame.font.SysFont("monospace", 20, bold=True)
            restart_text = restart_font.render("Press 'R' to restart", True, (200, 200, 200))
            restart_rect = restart_text.get_rect(center=(env.WIDTH / 2, env.HEIGHT / 2 + 30))
            display_screen.blit(restart_text, restart_rect)

            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False
                truncated = False
                total_reward = 0

        pygame.display.flip()
        env.clock.tick(env.FPS)

    env.close()