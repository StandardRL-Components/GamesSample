import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to accelerate, ↓ to brake, ←→ to turn. "
        "Press space to activate your current power-up."
    )

    game_description = (
        "A retro-futuristic top-down racer. Navigate procedurally generated neon tracks, "
        "collect power-ups, and aim for the fastest time. Watch out for the walls!"
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (255, 255, 255)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_TRAIL = (0, 150, 150)
    COLOR_TEXT = (240, 240, 240)
    COLOR_FINISH_LINE = (0, 255, 0)
    COLOR_SPARK = (255, 255, 0)
    COLOR_POWERUP_SPEED = (50, 255, 50)
    COLOR_POWERUP_SHIELD = (50, 50, 255)
    COLOR_POWERUP_TURBO = (255, 100, 0)

    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game
    FPS = 30
    MAX_STEPS = 1800  # 60 seconds * 30 FPS
    NUM_LAPS = 3

    # Player Physics
    ACCELERATION = 0.4
    BRAKE_FORCE = 0.6
    FRICTION = 0.04
    TURN_SPEED = 0.08
    MAX_SPEED = 10
    NEAR_WALL_THRESHOLD = 25

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)

        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_angle = 0.0
        self.player_radius = 8
        self.trail = deque(maxlen=20)

        self.track_waypoints = []
        self.track_walls = []
        self.track_width = 80

        self.powerups = []
        self.held_powerup = None
        self.active_powerup = None
        self.powerup_timer = 0

        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.race_won = False

        self.lap_count = 0
        self.last_waypoint_index = 0

        # self.reset() # This is called by the test harness, no need to call it here.

    def _generate_track(self):
        self.track_waypoints = []
        self.track_walls = []

        # Center of the world, not screen
        world_center = np.array([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])

        num_points = self.np_random.integers(10, 15)
        min_rad, max_rad = 200, 400

        # Generate waypoints in a rough circle
        last_angle = 0
        for i in range(num_points):
            angle_offset = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            angle = last_angle + (2 * math.pi / num_points) + angle_offset
            radius = self.np_random.uniform(min_rad, max_rad)

            point = world_center + np.array([math.cos(angle) * radius, math.sin(angle) * radius])
            self.track_waypoints.append(point)
            last_angle = angle

        # Create wall segments from waypoints
        for i in range(num_points):
            p1 = self.track_waypoints[i]
            p2 = self.track_waypoints[(i + 1) % num_points]

            direction = p2 - p1
            if np.linalg.norm(direction) == 0: continue

            perp = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)

            w1_start = p1 - perp * self.track_width / 2
            w1_end = p2 - perp * self.track_width / 2
            self.track_walls.append((w1_start, w1_end))

            w2_start = p1 + perp * self.track_width / 2
            w2_end = p2 + perp * self.track_width / 2
            self.track_walls.append((w2_start, w2_end))

    def _spawn_powerups(self):
        self.powerups = []
        num_powerups = 5
        powerup_types = ["speed", "shield", "turbo"]
        for _ in range(num_powerups):
            waypoint_idx = self.np_random.integers(0, len(self.track_waypoints))
            p1 = self.track_waypoints[waypoint_idx]
            p2 = self.track_waypoints[(waypoint_idx + 1) % len(self.track_waypoints)]

            t = self.np_random.uniform(0.1, 0.9)
            pos = p1 * (1 - t) + p2 * t

            offset_dir = self.np_random.uniform(-1, 1)
            direction = p2 - p1
            if np.linalg.norm(direction) == 0: continue
            perp = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
            pos += perp * offset_dir * self.np_random.uniform(0, self.track_width / 3)

            ptype = self.np_random.choice(powerup_types)
            self.powerups.append({"pos": pos, "type": ptype, "radius": 10})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.race_won = False

        self._generate_track()

        start_pos_base = (self.track_waypoints[0] + self.track_waypoints[-1]) / 2
        start_dir = self.track_waypoints[0] - self.track_waypoints[-1]
        self.player_angle = math.atan2(start_dir[1], start_dir[0])
        self.player_pos = start_pos_base.copy()
        self.player_vel = np.array([0.0, 0.0])

        self.trail.clear()
        self.particles = []
        self._spawn_powerups()

        self.held_powerup = None
        self.active_powerup = None
        self.powerup_timer = 0

        self.lap_count = 0
        self.last_waypoint_index = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0.0
        terminated = False

        if not self.game_over:
            movement, space_pressed, _ = action

            # --- Update Player State ---
            self._handle_input(movement)
            self._update_physics()
            self._handle_powerups(space_pressed)

            # --- Continuous Reward ---
            reward += 0.1  # Reward for surviving

            # --- Interactions and Rewards ---
            collision_reward = self._check_wall_collisions()
            reward += collision_reward
            if collision_reward < 0 and self.active_powerup != "shield":
                self.game_over = True
                reward -= 100  # Crash penalty

            reward += self._check_powerup_collection()
            reward += self._check_lap_completion()

        self.steps += 1

        # --- Check Termination Conditions ---
        if self.lap_count >= self.NUM_LAPS and not self.game_over:
            self.game_over = True
            self.race_won = True
            time_taken = self.steps / self.FPS
            if time_taken < 60.0:
                reward += 50  # Win bonus
            else:
                reward -= 50  # Win but too slow penalty

        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            reward -= 50  # Timeout penalty

        if self.game_over:
            terminated = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        forward_vec = np.array([math.cos(self.player_angle), math.sin(self.player_angle)])

        if movement == 1:  # Up
            self.player_vel += forward_vec * self.ACCELERATION
        if movement == 2:  # Down
            self.player_vel -= forward_vec * self.BRAKE_FORCE
        if movement == 3:  # Left
            self.player_angle -= self.TURN_SPEED
        if movement == 4:  # Right
            self.player_angle += self.TURN_SPEED

    def _update_physics(self):
        # Apply friction
        self.player_vel *= (1 - self.FRICTION)

        # Cap speed
        speed = np.linalg.norm(self.player_vel)
        max_speed = self.MAX_SPEED
        if self.active_powerup == "speed":
            max_speed *= 1.5

        if speed > max_speed:
            self.player_vel = self.player_vel / speed * max_speed

        self.player_pos += self.player_vel
        self.trail.append(self.player_pos.copy())

    def _handle_powerups(self, space_pressed):
        # Activate held powerup
        if space_pressed and self.held_powerup and not self.active_powerup:
            self.active_powerup = self.held_powerup
            self.held_powerup = None
            if self.active_powerup == "speed":
                self.powerup_timer = 5 * self.FPS  # 5 seconds
            elif self.active_powerup == "shield":
                self.powerup_timer = 5 * self.FPS  # 5 seconds
            elif self.active_powerup == "turbo":
                # Instant burst
                forward_vec = np.array([math.cos(self.player_angle), math.sin(self.player_angle)])
                self.player_vel += forward_vec * 20
                self.active_powerup = None  # Instant effect

        # Update active powerup
        if self.active_powerup:
            self.powerup_timer -= 1
            if self.powerup_timer <= 0:
                self.active_powerup = None

    def _check_wall_collisions(self):
        reward = 0
        min_dist = float('inf')

        for wall_start, wall_end in self.track_walls:
            dist = self._dist_point_to_segment(self.player_pos, wall_start, wall_end)
            min_dist = min(min_dist, dist)

            if dist < self.player_radius:
                if self.active_powerup == "shield":
                    # Bounce off wall
                    self.player_vel *= -0.8
                    self.active_powerup = None  # Shield breaks on impact
                    self.powerup_timer = 0
                    for _ in range(20):
                        self._create_particle(self.player_pos, self.COLOR_POWERUP_SHIELD, 15)
                    return 0  # No penalty, but shield is lost
                else:
                    self.player_vel = np.array([0.0, 0.0])
                    return -1.0  # Collision detected

        if min_dist < self.NEAR_WALL_THRESHOLD:
            reward -= 0.5  # Near wall penalty
            if self.np_random.random() < 0.5:  # Create sparks intermittently
                self._create_particle(self.player_pos, self.COLOR_SPARK, 5)

        return reward

    def _check_powerup_collection(self):
        for i, pu in reversed(list(enumerate(self.powerups))):
            dist = np.linalg.norm(self.player_pos - pu["pos"])
            if dist < self.player_radius + pu["radius"]:
                if not self.held_powerup:
                    self.held_powerup = pu["type"]
                    del self.powerups[i]
                    return 5.0  # Collected powerup reward
        return 0.0

    def _check_lap_completion(self):
        num_waypoints = len(self.track_waypoints)
        next_waypoint_idx = (self.last_waypoint_index + 1) % num_waypoints
        next_waypoint_pos = self.track_waypoints[next_waypoint_idx]

        dist_to_next = np.linalg.norm(self.player_pos - next_waypoint_pos)

        if dist_to_next < self.track_width:
            if next_waypoint_idx == 0 and self.last_waypoint_index == num_waypoints - 1:
                self.lap_count += 1
                self.last_waypoint_index = 0
                return 10.0  # Lap completion reward
            else:
                self.last_waypoint_index = next_waypoint_idx
        return 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Camera offset to center player
        cam_offset = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2]) - self.player_pos

        # Render trail
        if len(self.trail) > 1:
            faded_trail = []
            for i, pos in enumerate(self.trail):
                alpha = int(255 * (i / len(self.trail)))
                faded_trail.append((pos + cam_offset, alpha))

            for i in range(len(faded_trail) - 1):
                p1, alpha1 = faded_trail[i]
                p2, alpha2 = faded_trail[i + 1]
                color = self.COLOR_TRAIL
                if self.active_powerup == "speed":
                    color = self.COLOR_POWERUP_SPEED
                elif self.active_powerup == "shield":
                    color = self.COLOR_POWERUP_SHIELD

                pygame.draw.line(self.screen, color, p1.astype(int), p2.astype(int), max(1, int(i / len(faded_trail) * 5)))

        # Render track walls
        for start, end in self.track_walls:
            pygame.draw.aaline(self.screen, self.COLOR_WALL, start + cam_offset, end + cam_offset)

        # Render finish line (pulsating)
        p1 = self.track_waypoints[0]
        p_last = self.track_waypoints[-1]
        direction = p1 - p_last
        if np.linalg.norm(direction) > 0:
            perp = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
            line_start = p_last + direction * 0.5 - perp * self.track_width / 2
            line_end = p_last + direction * 0.5 + perp * self.track_width / 2

            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            color = (
                int(self.COLOR_FINISH_LINE[0] * pulse),
                min(255, int(self.COLOR_FINISH_LINE[1] * pulse + 50)),
                int(self.COLOR_FINISH_LINE[2] * pulse)
            )
            pygame.draw.line(self.screen, color, line_start + cam_offset, line_end + cam_offset, 5)

        # Render powerups
        for pu in self.powerups:
            pos = (pu["pos"] + cam_offset).astype(int)
            radius = pu["radius"]
            color = self._get_powerup_color(pu["type"])

            angle = (self.steps * 0.1) % (2 * math.pi)
            points = []
            num_sides = 3 if pu["type"] == "speed" else 5 if pu["type"] == "turbo" else 6
            for i in range(num_sides):
                a = angle + (2 * math.pi * i / num_sides)
                p = pos + np.array([math.cos(a), math.sin(a)]) * radius
                points.append(p)
            pygame.draw.aalines(self.screen, color, True, points)

        # Render particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
                self.screen.blit(temp_surf, (p['pos'] + cam_offset - np.array([p['size'], p['size']])).astype(int),
                                 special_flags=pygame.BLEND_RGBA_ADD)

        # Render player
        player_screen_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])

        # Shield effect
        if self.active_powerup == "shield":
            pulse = (math.sin(self.steps * 0.4) + 1) / 2
            radius = self.player_radius + 5 + pulse * 3
            alpha = 100 + pulse * 50
            pygame.gfxdraw.filled_circle(self.screen, int(player_screen_pos[0]), int(player_screen_pos[1]), int(radius),
                                         (*self.COLOR_POWERUP_SHIELD, int(alpha)))

        # Player body
        p_color = self.COLOR_PLAYER
        pygame.gfxdraw.filled_circle(self.screen, int(player_screen_pos[0]), int(player_screen_pos[1]), self.player_radius,
                                     p_color)
        pygame.gfxdraw.aacircle(self.screen, int(player_screen_pos[0]), int(player_screen_pos[1]), self.player_radius,
                                p_color)

        # Player direction indicator
        forward_vec = np.array([math.cos(self.player_angle), math.sin(self.player_angle)])
        indicator_end = player_screen_pos + forward_vec * (self.player_radius + 2)
        pygame.draw.line(self.screen, (255, 255, 255), player_screen_pos, indicator_end, 2)

    def _render_ui(self):
        time_str = f"TIME: {self.steps / self.FPS:.2f}s"
        lap_str = f"LAP: {min(self.lap_count + 1, self.NUM_LAPS)}/{self.NUM_LAPS}"
        speed = np.linalg.norm(self.player_vel)
        speed_str = f"SPEED: {speed * 10:.0f}"

        time_surf = self.font_small.render(time_str, True, self.COLOR_TEXT)
        lap_surf = self.font_small.render(lap_str, True, self.COLOR_TEXT)
        speed_surf = self.font_small.render(speed_str, True, self.COLOR_TEXT)

        self.screen.blit(time_surf, (10, 10))
        self.screen.blit(lap_surf, (self.SCREEN_WIDTH - lap_surf.get_width() - 10, 10))
        self.screen.blit(speed_surf, (self.SCREEN_WIDTH - speed_surf.get_width() - 10, self.SCREEN_HEIGHT - 30))

        # Powerup indicator
        if self.held_powerup:
            pu_text = f"ITEM: {self.held_powerup.upper()}"
            color = self._get_powerup_color(self.held_powerup)
            pu_surf = self.font_small.render(pu_text, True, color)
            self.screen.blit(pu_surf, (10, self.SCREEN_HEIGHT - 30))

        # Game Over / Win message
        if self.game_over:
            msg = "RACE FINISHED" if self.race_won else "CRASHED"
            color = self.COLOR_FINISH_LINE if self.race_won else (255, 50, 50)
            msg_surf = self.font_large.render(msg, True, color)
            pos = (self.SCREEN_WIDTH / 2 - msg_surf.get_width() / 2, self.SCREEN_HEIGHT / 2 - msg_surf.get_height() / 2)
            self.screen.blit(msg_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": self.lap_count,
            "time_seconds": self.steps / self.FPS,
            "held_powerup": self.held_powerup or "None",
        }

    # --- Helper Methods ---
    def _create_particle(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = self.np_random.integers(10, 20)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'life': life, 'max_life': life,
                'color': color, 'size': self.np_random.integers(1, 4)
            })

    def _get_powerup_color(self, ptype):
        if ptype == "speed": return self.COLOR_POWERUP_SPEED
        if ptype == "shield": return self.COLOR_POWERUP_SHIELD
        if ptype == "turbo": return self.COLOR_POWERUP_TURBO
        return self.COLOR_TEXT

    @staticmethod
    def _dist_point_to_segment(p, a, b):
        ap = p - a
        ab = b - a
        ab_squared = np.dot(ab, ab)
        if ab_squared == 0:
            return np.linalg.norm(ap)

        t = np.dot(ap, ab) / ab_squared
        t = np.clip(t, 0, 1)

        closest_point = a + t * ab
        return np.linalg.norm(p - closest_point)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()

    # Override screen to be a display surface for manual play
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Arcade Racer")

    terminated = False
    total_reward = 0

    # Game loop
    while not terminated:
        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        movement = 0  # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Render to Display ---
        # The observation is already the rendered frame, so we just need to display it
        # But _get_observation returns a transposed array, so we need to fix it for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()