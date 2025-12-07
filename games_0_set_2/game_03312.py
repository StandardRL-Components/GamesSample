import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


# Set Pygame to run in headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys draw line segments to guide the rider. "
        "↑→ draws an up-right line, ↓→ draws a down-right line, etc."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a sledding rider down procedurally generated slopes by drawing lines. "
        "Aim for the finish line while performing risky tricks for bonus points."
    )

    # Frames auto-advance for smooth graphics and time limits.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 20 * FPS  # 20 seconds

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (30, 50, 80)
    COLOR_RIDER = (255, 60, 60)
    COLOR_RIDER_GLOW = (255, 120, 120)
    COLOR_LINE = (255, 255, 255)
    COLOR_FINISH = (80, 255, 80)
    COLOR_TEXT = (240, 240, 240)
    COLOR_PARTICLE = (200, 200, 255)

    # Physics & Gameplay
    GRAVITY = 0.18
    FRICTION = 0.99
    BOUNCE_ELASTICITY = 0.6
    RIDER_RADIUS = 8
    LINE_DRAW_SIZE = 50
    LINE_DRAW_COOLDOWN = 10
    MAX_LINES = 30
    FINISH_LINE_Y_WORLD = 5000
    SLOW_COLLISION_THRESHOLD = 1.0

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

        self.font_ui = pygame.font.SysFont("consola", 24, bold=True)

        # State variables are initialized in reset()
        self.rider_pos = None
        self.rider_vel = None
        self.rider_on_ground = None
        self.was_on_ground = None
        self.lines = None
        self.particles = None
        self.world_scroll_y = None
        self.steps = None
        self.score = None
        self.timer = None
        self.line_draw_timer = None
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.timer = self.MAX_STEPS

        self.rider_pos = np.array([self.SCREEN_WIDTH / 2, 100.0])
        self.rider_vel = np.array([0.0, 0.0])
        self.rider_on_ground = False
        self.was_on_ground = False

        self.lines = deque(maxlen=self.MAX_LINES)
        self.particles = []

        self.world_scroll_y = 0
        self.line_draw_timer = 0

        self._generate_initial_terrain()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]

        self.steps += 1
        self.timer -= 1
        if self.line_draw_timer > 0:
            self.line_draw_timer -= 1

        reward = 0.1  # Survival reward

        self._handle_input(movement)

        collision_reward = self._update_physics()
        reward += collision_reward

        self._update_world_scroll()
        self._update_particles()

        # Jump reward
        if self.was_on_ground and not self.rider_on_ground:
            reward += 5.0  # sound: whoosh!
            self.score += 5

        self.score += reward

        terminated = self._check_termination()
        truncated = False # No truncation condition in this game

        if terminated and (self.rider_pos[1] + self.world_scroll_y) >= self.FINISH_LINE_Y_WORLD:
            reward += 100.0  # sound: success_fanfare!
            self.score += 100

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 0 or self.line_draw_timer > 0:
            return

        # sound: line_draw_swoop!
        self.line_draw_timer = self.LINE_DRAW_COOLDOWN

        x, y = self.rider_pos
        s = self.LINE_DRAW_SIZE / 2

        # Coordinates are in world space
        y += self.world_scroll_y

        if movement == 1:  # Up-Right
            p1 = (x - s, y + s)
            p2 = (x + s, y - s)
        elif movement == 2:  # Down-Right
            p1 = (x - s, y - s)
            p2 = (x + s, y + s)
        elif movement == 3:  # Down-Left
            p1 = (x + s, y - s)
            p2 = (x - s, y + s)
        elif movement == 4:  # Up-Left
            p1 = (x + s, y + s)
            p2 = (x - s, y - s)
        else:
            return

        self.lines.append((np.array(p1), np.array(p2)))

    def _update_physics(self):
        self.was_on_ground = self.rider_on_ground
        self.rider_on_ground = False

        # Apply gravity
        self.rider_vel[1] += self.GRAVITY
        self.rider_pos += self.rider_vel

        collision_reward = 0

        # Collision detection and response
        for p1, p2 in self.lines:
            rider_world_pos = self.rider_pos + np.array([0, self.world_scroll_y])

            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue

            t = np.dot(rider_world_pos - p1, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)

            closest_point = p1 + t * line_vec
            dist_vec = rider_world_pos - closest_point
            dist_sq = np.dot(dist_vec, dist_vec)

            if dist_sq < self.RIDER_RADIUS ** 2:
                # Collision occurred
                self.rider_on_ground = True
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 1e-6

                # Resolve penetration
                penetration = self.RIDER_RADIUS - dist
                self.rider_pos -= (penetration * dist_vec / dist)

                # Collision response
                normal = dist_vec / dist
                velocity_component = np.dot(self.rider_vel, normal)

                if velocity_component > 0:  # Moving away from surface
                    continue

                if abs(velocity_component) < self.SLOW_COLLISION_THRESHOLD:
                    collision_reward -= 1.0  # Penalty for slow/awkward collision
                    # sound: thud!
                else:
                    self._create_particles(self.rider_pos, 5 + int(abs(velocity_component)))
                    # sound: impact_scrape!

                # Reflect velocity
                self.rider_vel -= (1 + self.BOUNCE_ELASTICITY) * velocity_component * normal

                # Apply friction
                self.rider_vel *= self.FRICTION

        return collision_reward

    def _update_world_scroll(self):
        target_y = self.SCREEN_HEIGHT / 2.5
        scroll_delta = self.rider_pos[1] - target_y

        self.rider_pos[1] -= scroll_delta
        self.world_scroll_y += scroll_delta

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0] += p[1]  # Update position
            p[2] -= 1  # Decrease lifetime

    def _check_termination(self):
        off_screen = (self.rider_pos[1] > self.SCREEN_HEIGHT + self.RIDER_RADIUS or
                      self.rider_pos[0] < -self.RIDER_RADIUS or
                      self.rider_pos[0] > self.SCREEN_WIDTH + self.RIDER_RADIUS)

        time_up = self.timer <= 0

        finished = (self.rider_pos[1] + self.world_scroll_y) >= self.FINISH_LINE_Y_WORLD

        # Cast to a standard Python bool to satisfy the Gymnasium API check.
        # np.bool_ is not an instance of bool.
        return bool(off_screen or time_up or finished)

    def _generate_initial_terrain(self):
        x = self.SCREEN_WIDTH / 2
        y = 100.0

        for _ in range(20):
            next_x = x + self.np_random.uniform(-100, 100)
            next_y = y + self.np_random.uniform(50, 100)
            self.lines.append((np.array([x, y]), np.array([next_x, next_y])))
            x, y = next_x, next_y

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([pos.copy(), vel, lifetime])

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            t = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - t) + self.COLOR_BG_BOTTOM[0] * t),
                int(self.COLOR_BG_TOP[1] * (1 - t) + self.COLOR_BG_BOTTOM[1] * t),
                int(self.COLOR_BG_TOP[2] * (1 - t) + self.COLOR_BG_BOTTOM[2] * t),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw finish line
        finish_y_screen = self.FINISH_LINE_Y_WORLD - self.world_scroll_y
        if 0 < finish_y_screen < self.SCREEN_HEIGHT:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (0, int(finish_y_screen)),
                             (self.SCREEN_WIDTH, int(finish_y_screen)), 5)

        # Draw lines
        for p1, p2 in self.lines:
            p1_screen = p1 - np.array([0, self.world_scroll_y])
            p2_screen = p2 - np.array([0, self.world_scroll_y])
            pygame.draw.aaline(self.screen, self.COLOR_LINE, p1_screen, p2_screen, 2)

        # Draw particles
        for pos, vel, life in self.particles:
            alpha = max(0, min(255, int(255 * (life / 30.0))))
            color = (*self.COLOR_PARTICLE, alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 4, 4))
            self.screen.blit(temp_surf, (int(pos[0] - 2), int(pos[1] - 2)))

        # Draw rider
        rider_x, rider_y = int(self.rider_pos[0]), int(self.rider_pos[1])

        # Glow effect
        glow_radius = self.RIDER_RADIUS + 5 + int(3 * math.sin(self.steps * 0.1))
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_RIDER_GLOW, 80))
        self.screen.blit(glow_surf, (rider_x - glow_radius, rider_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Rider body
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score):05d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, self.timer / self.FPS)
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "rider_y_world": self.rider_pos[1] + self.world_scroll_y
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    import sys

    # Un-set the headless environment variable to allow display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen_main = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Sled Rider")

    running = True
    total_reward = 0

    action = np.array([0, 0, 0])  # No-op

    print("--- Sled Rider ---")
    print(env.user_guide)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement_action = 0
        if keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
            movement_action = 1
        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
            movement_action = 2
        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
            movement_action = 4

        action[0] = movement_action
        # action[1] and action[2] are unused in this game

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the main screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_main.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode Finished! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(env.FPS)

    env.close()
    pygame.quit()
    sys.exit()