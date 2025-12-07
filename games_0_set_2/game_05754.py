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
        "Controls: Use arrow keys to aim your jump. Press space to launch."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a spaceship through an asteroid field by hopping over obstacles to reach the end."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_Y = 350
    MAX_STEPS = 1500
    WIN_X = 590

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_GROUND = (40, 40, 50)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_ASTEROID = (120, 120, 130)
    COLOR_UI_TEXT = (100, 255, 100)
    COLOR_AIM = (255, 255, 0, 100)  # Yellow, semi-transparent

    # Physics
    GRAVITY = 0.25
    PLAYER_START_X = 50
    JUMP_AIM_ADJUST = 0.2
    JUMP_H_MIN, JUMP_H_MAX = 2.0, 7.0
    JUMP_V_MIN, JUMP_V_MAX = -12.0, -5.0

    # Asteroid properties
    ASTEROID_MIN_W, ASTEROID_MAX_W = 20, 40
    ASTEROID_MIN_H, ASTEROID_MAX_H = 20, 60
    ASTEROID_SPAWN_RATE_MIN, ASTEROID_SPAWN_RATE_MAX = 60, 120
    INITIAL_ASTEROID_SPEED = 2.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.rng = np.random.default_rng()

        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player_pos = pygame.Vector2(self.PLAYER_START_X, self.GROUND_Y)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_state = "grounded"  # "grounded" or "jumping"
        self.player_angle = 0

        # Jump aiming
        self.jump_power_h = (self.JUMP_H_MIN + self.JUMP_H_MAX) / 2
        self.jump_power_v = (self.JUMP_V_MIN + self.JUMP_V_MAX) / 2
        self.prev_space_held = False

        # World state
        self.asteroids = []
        self.asteroid_speed = self.INITIAL_ASTEROID_SPEED
        self.asteroid_spawn_timer = self.rng.integers(30, 60)
        self.passed_asteroids = set()

        # Visuals
        self.particles = []
        self.stars = [
            (self.rng.integers(0, self.SCREEN_WIDTH), self.rng.integers(0, self.GROUND_Y), self.rng.random() * 0.5 + 0.25)
            for _ in range(150)
        ]
        self.screen_shake = 0

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean (unused)
        reward = 0.0

        # Handle game over state by just updating visual effects
        if self.game_over:
            self._update_particles()
            if self.screen_shake > 0:
                self.screen_shake -= 1
            terminated = True
            return self._get_observation(), 0, terminated, False, self._get_info()

        # --- Handle Input ---
        if self.player_state == "grounded":
            # Adjust jump aim
            if movement == 1:  # Up
                self.jump_power_v = max(self.JUMP_V_MIN, self.jump_power_v - self.JUMP_AIM_ADJUST)
            elif movement == 2:  # Down
                self.jump_power_v = min(self.JUMP_V_MAX, self.jump_power_v + self.JUMP_AIM_ADJUST)
            elif movement == 3:  # Left
                self.jump_power_h = max(self.JUMP_H_MIN, self.jump_power_h - self.JUMP_AIM_ADJUST)
            elif movement == 4:  # Right
                self.jump_power_h = min(self.JUMP_H_MAX, self.jump_power_h + self.JUMP_AIM_ADJUST)

            # Initiate jump on space PRESS (not hold)
            if space_held and not self.prev_space_held:
                self.player_state = "jumping"
                self.player_vel.x = self.jump_power_h
                self.player_vel.y = self.jump_power_v
                # sfx: jump_launch.wav
                self._create_thruster_burst(20)

        self.prev_space_held = space_held

        # --- Update Game Logic ---
        self.steps += 1
        reward += 0.01  # Small reward for surviving

        self._update_player()
        reward += self._update_asteroids()

        if self.steps > 0 and self.steps % 500 == 0:
            self.asteroid_speed += 0.1

        self._update_particles()
        if self.screen_shake > 0:
            self.screen_shake -= 1

        # --- Check for Termination ---
        terminated = False
        player_rect = self._get_player_rect()

        # 1. Collision with asteroid
        for asteroid in self.asteroids:
            if player_rect.colliderect(asteroid['rect']):
                self.game_over = True
                terminated = True
                reward = -100.0
                self.screen_shake = 15
                self._create_explosion(self.player_pos, 50)
                # sfx: explosion.wav
                break

        if not terminated:
            # 2. Reached the end
            if self.player_pos.x > self.WIN_X:
                self.game_over = True
                terminated = True
                reward = 100.0
                self.score += 1000
                # sfx: win_level.wav

            # 3. Fell off the left
            elif self.player_pos.x < 0:
                self.game_over = True
                terminated = True
                reward = -100.0

        # 4. Max steps
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Per Gymnasium API, terminated should be True if truncated is True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self):
        if self.player_state == "jumping":
            self.player_vel.y += self.GRAVITY
            self.player_pos += self.player_vel
            self.player_angle = math.degrees(math.atan2(self.player_vel.y, self.player_vel.x))

            # Land on ground
            if self.player_pos.y >= self.GROUND_Y:
                self.player_pos.y = self.GROUND_Y
                self.player_vel = pygame.Vector2(0, 0)
                self.player_state = "grounded"
                self.player_angle = 0
                self.screen_shake = 5
                # sfx: landing_thud.wav

    def _update_asteroids(self):
        reward_from_asteroids = 0
        # Move and remove off-screen asteroids
        new_asteroids = []
        for asteroid in self.asteroids:
            asteroid['rect'].x -= self.asteroid_speed
            if asteroid['rect'].right > 0:
                new_asteroids.append(asteroid)
            else:
                # Asteroid successfully passed
                if asteroid['id'] not in self.passed_asteroids:
                    self.passed_asteroids.add(asteroid['id'])
                    reward_from_asteroids += 1.0
                    self.score += 10
                    # sfx: pass_asteroid.wav
        self.asteroids = new_asteroids

        # Spawn new asteroids
        self.asteroid_spawn_timer -= 1
        if self.asteroid_spawn_timer <= 0:
            self._spawn_asteroid()
            spawn_delay = self.rng.integers(self.ASTEROID_SPAWN_RATE_MIN, self.ASTEROID_SPAWN_RATE_MAX)
            self.asteroid_spawn_timer = spawn_delay / max(1, self.asteroid_speed / self.INITIAL_ASTEROID_SPEED)

        return reward_from_asteroids

    def _spawn_asteroid(self):
        width = self.rng.integers(self.ASTEROID_MIN_W, self.ASTEROID_MAX_W)
        height = self.rng.integers(self.ASTEROID_MIN_H, self.ASTEROID_MAX_H)
        x = self.SCREEN_WIDTH + self.rng.integers(0, 50)
        y = self.GROUND_Y - height
        rect = pygame.Rect(x, y, width, height)

        num_points = 8
        center_x, center_y = rect.center
        radius = max(width, height) / 2
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            rand_radius = radius * self.rng.uniform(0.7, 1.1)
            px = center_x + rand_radius * math.cos(angle) - x
            py = center_y + rand_radius * math.sin(angle) - y
            points.append((px, py))

        self.asteroids.append({'rect': rect, 'points': points, 'id': self.steps + x})

    def _get_observation(self):
        # Apply screen shake
        render_offset_x, render_offset_y = 0, 0
        if self.screen_shake > 0:
            render_offset_x = self.rng.integers(-self.screen_shake, self.screen_shake + 1)
            render_offset_y = self.rng.integers(-self.screen_shake, self.screen_shake + 1)

        self.screen.fill(self.COLOR_BG)
        self._render_stars(render_offset_x, render_offset_y)
        self._render_ground(render_offset_x, render_offset_y)
        self._render_particles(render_offset_x, render_offset_y)
        self._render_asteroids(render_offset_x, render_offset_y)
        if not (self.game_over and self.screen_shake > 10):
            self._render_player(render_offset_x, render_offset_y)
        if self.player_state == "grounded" and not self.game_over:
            self._render_aim_trajectory(render_offset_x, render_offset_y)
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self, ox, oy):
        for x, y, speed in self.stars:
            px = int((x - self.player_pos.x * speed * 0.1) % self.SCREEN_WIDTH) + ox
            py = int(y) + oy
            lum = int(speed * 150)
            pygame.draw.circle(self.screen, (lum, lum, lum), (px, py), 1)

    def _render_ground(self, ox, oy):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (ox, oy + self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

    def _render_asteroids(self, ox, oy):
        for asteroid in self.asteroids:
            points = [(p[0] + asteroid['rect'].x + ox, p[1] + asteroid['rect'].y + oy) for p in asteroid['points']]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

    def _get_player_rect(self):
        return pygame.Rect(self.player_pos.x - 8, self.player_pos.y - 8, 16, 16)

    def _render_player(self, ox, oy):
        p1, p2, p3 = pygame.Vector2(10, 0), pygame.Vector2(-7, -7), pygame.Vector2(-7, 7)
        angle_rad = math.radians(self.player_angle)
        points = [
            self.player_pos + p1.rotate_rad(angle_rad) + (ox, oy),
            self.player_pos + p2.rotate_rad(angle_rad) + (ox, oy),
            self.player_pos + p3.rotate_rad(angle_rad) + (ox, oy),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_aim_trajectory(self, ox, oy):
        pos, vel = pygame.Vector2(self.player_pos), pygame.Vector2(self.jump_power_h, self.jump_power_v)
        points = []
        for i in range(25):
            vel.y += self.GRAVITY
            pos += vel
            if i % 2 == 0: points.append(pos + (ox, oy))
            if pos.y > self.GROUND_Y + 10: break
        if len(points) > 1: pygame.draw.aalines(self.screen, self.COLOR_AIM, False, points)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        if self.game_over:
            status_str = "MISSION FAILED"
            if self.player_pos.x > self.WIN_X: status_str = "MISSION COMPLETE!"
            status_text = self.font.render(status_str, True, self.COLOR_UI_TEXT)
            text_rect = status_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(status_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _create_thruster_burst(self, count):
        for _ in range(count):
            angle = math.radians(self.player_angle + 180 + self.rng.uniform(-20, 20))
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.rng.uniform(1, 4)
            self.particles.append({'pos': pygame.Vector2(self.player_pos), 'vel': vel, 'life': self.rng.integers(15, 30), 'color': (255, self.rng.integers(150, 255), 0)})

    def _create_explosion(self, pos, count):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.rng.uniform(1, 8)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'life': self.rng.integers(30, 60), 'color': random.choice([(255, 255, 255), (255, 150, 0), (200, 200, 200)])})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1

    def _render_particles(self, ox, oy):
        for p in self.particles:
            size = max(1, p['life'] / 10)
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x + ox), int(p['pos'].y + oy)), int(size))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # Re-enable the video driver for direct play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    pygame.display.set_caption("Asteroid Hopper")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    terminated = False
    truncated = False
    total_reward = 0
    action = env.action_space.sample()
    action.fill(0)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action.fill(0)
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}")
            total_reward = 0
            obs, info = env.reset()
            terminated = False
            truncated = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)

    env.close()