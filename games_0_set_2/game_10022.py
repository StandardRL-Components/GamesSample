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
    Singularity Shrink: A Gymnasium environment where a player pilots a ship through a debris field towards a black hole.
    The ship can change size, affecting its movement, gravity pull, and ability to destroy debris.
    """
    game_description = (
        "Pilot a ship through a dangerous debris field towards a singularity. "
        "Change your ship's size to destroy obstacles or navigate tight spaces."
    )
    user_guide = (
        "Controls: ←→ to move left and right. Hold space to grow your ship and hold shift to shrink it."
    )
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 200)
        self.COLOR_DEBRIS = (255, 50, 50)
        self.COLOR_WORMHOLE = (50, 255, 50)
        self.COLOR_WORMHOLE_GLOW = (30, 150, 30)
        self.COLOR_BLACK_HOLE = (0, 0, 0)
        self.COLOR_ACCRETION_DISK = (150, 50, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_PARTICLE_EXPLOSION = (255, 180, 50)

        # Player settings
        self.PLAYER_MIN_SIZE = 8
        self.PLAYER_MAX_SIZE = 30
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = 0.9
        self.PLAYER_SIZE_CHANGE_RATE = 0.5
        self.PLAYER_SIZE_LERP_RATE = 0.15

        # Game settings
        self.DEBRIS_SPAWN_RATE = 20
        self.WORMHOLE_SPAWN_RATE = 250
        self.WORMHOLE_LIFESPAN = 300
        self.INITIAL_DEBRIS_SPEED = 1.0
        self.SPEED_INCREASE_INTERVAL = 500
        self.SPEED_INCREASE_AMOUNT = 0.2

        # Termination zones
        self.SINGULARITY_Y = self.HEIGHT - 15
        self.EVENT_HORIZON_Y = self.HEIGHT - 50

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18, bold=True)

        # --- State Variables ---
        self.player_pos = None
        self.player_vel = None
        self.player_size = None
        self.player_target_size = None
        self.debris = []
        self.wormholes = []
        self.particles = []
        self.stars = []
        self.debris_base_speed = None
        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_stars()

        # Initialize state by calling reset
        # self.reset() # This is called by the test harness

    def _generate_stars(self):
        self.stars = []
        for _ in range(200):
            self.stars.append({
                'pos': pygame.math.Vector2(random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT)),
                'size': random.uniform(0.5, 1.5),
                'brightness': random.randint(50, 150),
                'parallax_factor': random.uniform(0.1, 0.4)
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, 50)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_size = self.PLAYER_MIN_SIZE
        self.player_target_size = self.PLAYER_MIN_SIZE

        self.debris = []
        self.wormholes = []
        self.particles = []

        self.debris_base_speed = self.INITIAL_DEBRIS_SPEED

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward

        # --- 1. Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        prev_target_size = self.player_target_size
        if space_held:
            self.player_target_size += self.PLAYER_SIZE_CHANGE_RATE
        if shift_held:
            self.player_target_size -= self.PLAYER_SIZE_CHANGE_RATE
        self.player_target_size = np.clip(self.player_target_size, self.PLAYER_MIN_SIZE, self.PLAYER_MAX_SIZE)

        if abs(self.player_target_size - prev_target_size) > 0.1 and self.steps % 3 == 0:
            self._spawn_particles(self.player_pos, self.COLOR_PLAYER_GLOW, 1, self.player_size * 0.5, 0.5)

        # --- 2. Update Player State ---
        self.player_size += (self.player_target_size - self.player_size) * self.PLAYER_SIZE_LERP_RATE

        if movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_ACCEL

        self.player_vel *= self.PLAYER_FRICTION
        self.player_pos += self.player_vel

        # Gravity pull increases with size and proximity to bottom
        gravity_pull = 0.005 * self.player_size * (self.player_pos.y / self.HEIGHT) ** 2
        self.player_pos.y += gravity_pull

        self.player_pos.x = np.clip(self.player_pos.x, self.player_size, self.WIDTH - self.player_size)
        self.player_pos.y = np.clip(self.player_pos.y, self.player_size, self.HEIGHT)

        # --- 3. Update Game World ---
        self._update_debris()
        self._update_wormholes()
        self._update_particles()

        # --- 4. Handle Collisions & Events ---
        temp_reward, terminated_by_event = self._handle_collisions_and_events()
        reward += temp_reward
        self.game_over = terminated_by_event

        # --- 5. Check Termination Conditions ---
        terminated = self.game_over
        truncated = False
        if not terminated:
            if self.player_pos.y >= self.SINGULARITY_Y:
                terminated = True
                reward += 100
                self.score += 100
            elif self.player_pos.y > self.EVENT_HORIZON_Y:
                terminated = True
                reward = -100  # Override other rewards
                self.score -= 100
                self._spawn_particles(self.player_pos, self.COLOR_PLAYER, 50, self.player_size)
            elif self.steps >= self.MAX_STEPS:
                truncated = True

        self.game_over = terminated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_collisions_and_events(self):
        reward = 0
        terminated = False

        # Debris Collision
        for d in self.debris[:]:
            dist = self.player_pos.distance_to(d['pos'])
            if dist < self.player_size + d['size']:
                if self.player_size > d['size'] + 2:  # Must be slightly larger to destroy
                    self.score += 1
                    reward += 1
                    self._spawn_particles(d['pos'], self.COLOR_PARTICLE_EXPLOSION, 15, d['size'])
                    self.debris.remove(d)
                else:
                    terminated = True
                    reward = -100
                    self.score -= 100
                    self._spawn_particles(self.player_pos, self.COLOR_PLAYER, 50, self.player_size)
                return reward, terminated

        # Wormhole Collision
        for w in self.wormholes[:]:
            dist = self.player_pos.distance_to(w['pos'])
            if dist < self.player_size + w['size']:
                self.score += 5
                reward += 5
                self.player_pos.x = self.np_random.uniform(50, self.WIDTH - 50)
                self.player_pos.y = min(self.HEIGHT - 100, self.player_pos.y + 100)
                self._spawn_particles(w['pos'], self.COLOR_WORMHOLE, 40, w['size'])
                self.wormholes.remove(w)
                break

        return reward, terminated

    def _update_debris(self):
        if self.steps % self.DEBRIS_SPAWN_RATE == 0:
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), -20)
            size = self.np_random.uniform(self.PLAYER_MIN_SIZE - 4, self.PLAYER_MAX_SIZE + 4)
            self.debris.append({'pos': pos, 'size': size})

        for d in self.debris[:]:
            d['pos'].y += self.debris_base_speed
            if d['pos'].y > self.HEIGHT + d['size']:
                self.debris.remove(d)

        if self.steps > 0 and self.steps % self.SPEED_INCREASE_INTERVAL == 0:
            self.debris_base_speed += self.SPEED_INCREASE_AMOUNT

    def _update_wormholes(self):
        if self.steps > 0 and self.steps % self.WORMHOLE_SPAWN_RATE == 0 and len(self.wormholes) < 3:
            pos = pygame.math.Vector2(self.np_random.uniform(50, self.WIDTH - 50),
                                      self.np_random.uniform(100, self.HEIGHT - 100))
            self.wormholes.append(
                {'pos': pos, 'size': 20, 'life': self.WORMHOLE_LIFESPAN, 'initial_life': self.WORMHOLE_LIFESPAN})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.98  # friction
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_particles(self, pos, color, count, base_speed_scale, speed_multiplier=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * (base_speed_scale / 10.0) * speed_multiplier
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy() + vel * self.np_random.uniform(1, 3),  # burst outwards
                'vel': vel,
                'life': self.np_random.integers(20, 41),
                'color': color,
                'size': self.np_random.uniform(1, 3.5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Stars
        for star in self.stars:
            star['pos'].y += star['parallax_factor']
            if star['pos'].y > self.HEIGHT:
                star['pos'].y = 0
                star['pos'].x = random.uniform(0, self.WIDTH)
            c = star['brightness']
            pygame.draw.circle(self.screen, (c, c, c), star['pos'], star['size'])

        # Black Hole
        bh_center = (self.WIDTH // 2, self.HEIGHT + self.EVENT_HORIZON_Y * 1.2)

        # Accretion Disk (visual effect)
        for i in range(10):
            angle_offset = i * 36
            radius_factor = (i + 1) / 10.0

            width = int(self.WIDTH * 0.7 * radius_factor + 30 * math.sin(self.steps * 0.02 + i))
            height = int(60 * radius_factor + 15 * math.cos(self.steps * 0.03 + i))
            alpha = int(20 + (1 - radius_factor) * 80)

            if width <= 0 or height <= 0:
                continue

            temp_surf = pygame.Surface((width, height), pygame.SRCALPHA)
            pygame.draw.ellipse(temp_surf, (*self.COLOR_ACCRETION_DISK, alpha), (0, 0, width, height),
                                max(1, int(4 * radius_factor)))

            angle = (self.steps * (0.2 + radius_factor * 0.3)) % 360
            rotated_surf = pygame.transform.rotate(temp_surf, angle + angle_offset)
            rect = rotated_surf.get_rect(center=bh_center)
            self.screen.blit(rotated_surf, rect)

        pygame.draw.circle(self.screen, self.COLOR_BLACK_HOLE, bh_center, self.EVENT_HORIZON_Y * 2)

    def _render_game(self):
        # Particles
        for p in self.particles:
            life_ratio = p['life'] / 40.0
            size = p['size'] * life_ratio
            if size < 1: continue
            color = tuple(min(255, max(0, c * life_ratio)) for c in p['color'])
            pygame.draw.circle(self.screen, color, p['pos'], size)

        # Wormholes
        for w in self.wormholes:
            pulse = math.sin(self.steps * 0.2) * 5
            life_fade = w['life'] / w['initial_life']

            glow_size = int((w['size'] + pulse + 5) * life_fade)
            core_size = int((w['size'] + pulse) * life_fade)

            if glow_size <= 0 or core_size <= 0: continue

            glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_WORMHOLE_GLOW, 80), (glow_size, glow_size), glow_size)
            self.screen.blit(glow_surf, (int(w['pos'].x) - glow_size, int(w['pos'].y) - glow_size))

            pygame.gfxdraw.filled_circle(self.screen, int(w['pos'].x), int(w['pos'].y), core_size, self.COLOR_WORMHOLE)
            pygame.gfxdraw.aacircle(self.screen, int(w['pos'].x), int(w['pos'].y), core_size, self.COLOR_WORMHOLE)

        # Debris
        for d in self.debris:
            p1 = (int(d['pos'].x), int(d['pos'].y - d['size']))
            p2 = (int(d['pos'].x - d['size'] * 0.866), int(d['pos'].y + d['size'] * 0.5))
            p3 = (int(d['pos'].x + d['size'] * 0.866), int(d['pos'].y + d['size'] * 0.5))
            points = [p1, p2, p3]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_DEBRIS)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_DEBRIS)

        # Player
        if not self.game_over:
            glow_size = int(self.player_size * 1.8)
            if glow_size > 0:
                glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 80), (glow_size, glow_size), glow_size)
                self.screen.blit(glow_surf, (int(self.player_pos.x) - glow_size, int(self.player_pos.y) - glow_size))

            player_size_int = int(self.player_size)
            if player_size_int > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y),
                                            player_size_int, self.COLOR_PLAYER)
                pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), player_size_int,
                                        self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        distance = max(0, int(self.SINGULARITY_Y - self.player_pos.y))
        dist_text = self.font.render(f"DISTANCE: {distance}", True, self.COLOR_UI_TEXT)
        dist_rect = dist_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(dist_text, dist_rect)

        bar_height, bar_width, bar_x, bar_y = 100, 15, 15, self.HEIGHT - 115
        size_ratio = np.clip(
            (self.player_size - self.PLAYER_MIN_SIZE) / (self.PLAYER_MAX_SIZE - self.PLAYER_MIN_SIZE), 0, 1)
        fill_height = bar_height * size_ratio

        pygame.draw.rect(self.screen, (50, 50, 80), (bar_x, bar_y, bar_width, bar_height), 2, border_radius=3)
        if fill_height > 0:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER,
                             (bar_x + 2, bar_y + bar_height - fill_height - 2, bar_width - 4, fill_height), 0,
                             border_radius=2)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will create a window and render the environment
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Singularity Shrink")
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0

    while not terminated:
        movement = 0  # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(env.metadata["render_fps"])

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Allow a moment to see the final state before closing
            pygame.time.wait(2000)
            break

    env.close()