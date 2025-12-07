
# Generated: 2025-08-28T03:03:40.709444
# Source Brief: brief_04806.md
# Brief Index: 4806

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to steer, ↑ to accelerate, ↓ to brake. Hold Shift for a drift-boosted turn."
    )

    game_description = (
        "Race a sleek, procedurally generated line-car through a twisting, top-down track, dodging obstacles for the fastest lap time."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_STEPS = 2500

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_TRACK = (100, 100, 120)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_SPARK = (255, 255, 0)
        self.COLOR_UI = (255, 255, 255)
        self.COLOR_FINISH_LINE = (255, 255, 255)

        # Car Physics
        self.MAX_SPEED = 12.0
        self.MIN_SPEED = 4.0
        self.ACCELERATION = 0.25
        self.BRAKING = 0.5
        self.FRICTION = 0.985
        self.TURN_SPEED = 3.5
        self.DRIFT_TURN_SPEED = 5.5
        self.TURN_DAMPING = 0.85
        
        # Game constants
        self.NUM_LAPS_TO_WIN = 3
        self.NEAR_MISS_DISTANCE = 40
        self.NEAR_MISS_REWARD = 1.0
        self.NEAR_MISS_COOLDOWN = 15 # frames
        self.LAP_REWARD = 10.0
        self.WIN_REWARD = 50.0
        self.COLLISION_REWARD = -100.0
        self.SURVIVAL_REWARD = 0.1
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- State Initialization ---
        self.car = {}
        self.track_points = []
        self.obstacles = []
        self.checkpoints = []
        self.particles = []
        self.near_miss_trackers = {}
        self.np_random = None
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self._generate_track_and_obstacles()
        
        start_pos = self.track_points[0]
        start_angle = self._get_angle(self.track_points[0], self.track_points[1])
        
        self.car = {
            "pos": pygame.math.Vector2(start_pos),
            "angle": start_angle,
            "speed": self.MIN_SPEED,
            "turn_rate": 0.0,
            "width": 12,
            "height": 24,
        }
        
        self.lap_count = 0
        self.next_checkpoint = 0
        self.lap_start_time = 0
        self.best_lap_time = float('inf')
        
        self.particles = []
        self.near_miss_trackers = {i: 0 for i in range(len(self.obstacles))}
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._update_car(movement, shift_held)
        self._update_particles()
        
        reward = self.SURVIVAL_REWARD
        terminated = False

        # --- Event Checking ---
        collision = self._check_collisions()
        if collision:
            # sfx: explosion
            self.game_over = True
            terminated = True
            reward += self.COLLISION_REWARD
        else:
            near_miss_count = self._check_near_misses()
            reward += near_miss_count * self.NEAR_MISS_REWARD
            
            lap_completed = self._check_laps()
            if lap_completed:
                # sfx: lap_complete_ fanfare
                self.lap_count += 1
                reward += self.LAP_REWARD
                
                current_lap_time = (self.steps - self.lap_start_time) / self.FPS
                if current_lap_time < self.best_lap_time:
                    self.best_lap_time = current_lap_time
                self.lap_start_time = self.steps

                if self.lap_count >= self.NUM_LAPS_TO_WIN:
                    self.game_over = True
                    self.game_won = True
                    terminated = True
                    reward += self.WIN_REWARD
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.steps += 1
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        cam_offset = self.car["pos"] - pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        self._render_world(cam_offset)
        self._render_particles(cam_offset)
        self._render_car()
        self._render_ui()
        
        if self.game_over:
            self._render_end_message()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.lap_count,
        }
        
    def _generate_track_and_obstacles(self):
        self.track_points = []
        num_points = 150
        radius = self.SCREEN_HEIGHT * 1.5
        center = pygame.math.Vector2(0, 0)
        
        # Use seeded random values for deterministic track generation
        shape_freq1 = self.np_random.uniform(2.5, 4.5)
        shape_freq2 = self.np_random.uniform(1.5, 2.5)
        shape_amp1 = self.np_random.uniform(0.15, 0.35) * radius
        shape_amp2 = self.np_random.uniform(0.1, 0.25) * radius

        for i in range(num_points + 1):
            angle = (i / num_points) * 2 * math.pi
            r = radius + shape_amp1 * math.sin(angle * shape_freq1) + shape_amp2 * math.cos(angle * shape_freq2)
            x = center.x + r * math.cos(angle)
            y = center.y + r * math.sin(angle)
            self.track_points.append(pygame.math.Vector2(x, y))

        self.checkpoints = []
        num_checkpoints = 10
        for i in range(num_checkpoints):
            idx = int((i / num_checkpoints) * len(self.track_points))
            p = self.track_points[idx]
            self.checkpoints.append(pygame.Rect(p.x - 50, p.y - 50, 100, 100))
        self.total_checkpoints = len(self.checkpoints)
        
        self.obstacles = []
        num_obstacles = 30
        track_width = 80
        for i in range(num_obstacles):
            idx = self.np_random.integers(10, len(self.track_points) - 10)
            p1 = self.track_points[idx]
            p2 = self.track_points[idx + 1]
            
            angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
            perp_angle = angle + math.pi / 2
            
            offset = self.np_random.uniform(-track_width * 0.6, track_width * 0.6)
            pos_x = p1.x + offset * math.cos(perp_angle)
            pos_y = p1.y + offset * math.sin(perp_angle)
            
            size = self.np_random.uniform(15, 30)
            obstacle_rect = pygame.Rect(pos_x - size / 2, pos_y - size / 2, size, size)
            
            # Ensure obstacle doesn't block finish line
            if not obstacle_rect.colliderect(self.checkpoints[0]):
                self.obstacles.append(obstacle_rect)

    def _update_car(self, movement, shift_held):
        # --- Turning ---
        turn_input = 0
        if movement == 3: # Left
            turn_input = -1
        elif movement == 4: # Right
            turn_input = 1
        
        turn_speed = self.DRIFT_TURN_SPEED if shift_held else self.TURN_SPEED
        self.car["turn_rate"] += turn_input * turn_speed * 0.2
        self.car["turn_rate"] = np.clip(self.car["turn_rate"], -turn_speed, turn_speed)
        
        if turn_input == 0:
            self.car["turn_rate"] *= self.TURN_DAMPING

        self.car["angle"] += self.car["turn_rate"]

        # --- Speed ---
        if movement == 1: # Accelerate
            # sfx: engine_rev
            self.car["speed"] += self.ACCELERATION
        elif movement == 2: # Brake
            # sfx: brake_squeal
            self.car["speed"] -= self.BRAKING
        
        self.car["speed"] *= self.FRICTION
        self.car["speed"] = np.clip(self.car["speed"], self.MIN_SPEED, self.MAX_SPEED)

        # --- Position ---
        rad_angle = math.radians(self.car["angle"])
        velocity = pygame.math.Vector2(math.cos(rad_angle), math.sin(rad_angle)) * self.car["speed"]
        self.car["pos"] += velocity
        
        # --- Particles ---
        if self.car["speed"] > self.MIN_SPEED + 1:
            self._spawn_speed_lines(shift_held)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_car_corners(self):
        pos, angle, w, h = self.car["pos"], self.car["angle"], self.car["width"], self.car["height"]
        rad_angle = math.radians(angle)
        cos_a, sin_a = math.cos(rad_angle), math.sin(rad_angle)
        
        half_w, half_h = w / 2, h / 2
        corners = [
            pygame.math.Vector2(-half_w, -half_h), pygame.math.Vector2(half_w, -half_h),
            pygame.math.Vector2(half_w, half_h), pygame.math.Vector2(-half_w, half_h)
        ]
        
        rotated_corners = []
        for c in corners:
            x = c.x * cos_a - c.y * sin_a + pos.x
            y = c.x * sin_a + c.y * cos_a + pos.y
            rotated_corners.append(pygame.math.Vector2(x, y))
        return rotated_corners

    def _check_collisions(self):
        corners = self._get_car_corners()
        for obstacle in self.obstacles:
            for corner in corners:
                if obstacle.collidepoint(corner):
                    return True
        return False

    def _check_near_misses(self):
        miss_count = 0
        car_pos = self.car["pos"]
        
        for i, obstacle in enumerate(self.obstacles):
            if self.near_miss_trackers[i] > 0:
                self.near_miss_trackers[i] -= 1
                continue

            dist = car_pos.distance_to(obstacle.center)
            if dist < self.NEAR_MISS_DISTANCE + obstacle.width / 2:
                # sfx: near_miss_zap
                self.near_miss_trackers[i] = self.NEAR_MISS_COOLDOWN
                miss_count += 1
                self._spawn_sparks(obstacle.center)
        return miss_count

    def _check_laps(self):
        car_rect = pygame.Rect(self.car["pos"].x - 2, self.car["pos"].y - 2, 4, 4)
        
        # Check for next checkpoint
        if self.next_checkpoint < self.total_checkpoints:
            if car_rect.colliderect(self.checkpoints[self.next_checkpoint]):
                # sfx: checkpoint_ding
                self.next_checkpoint += 1
        
        # Check for finish line crossing after all checkpoints
        elif self.next_checkpoint == self.total_checkpoints:
            if car_rect.colliderect(self.checkpoints[0]):
                self.next_checkpoint = 1 # Reset for next lap
                return True
        return False

    def _render_world(self, cam_offset):
        # Track
        if len(self.track_points) > 1:
            track_width = 160
            inner_width = 150
            
            screen_points = [p - cam_offset for p in self.track_points]
            pygame.draw.aalines(self.screen, self.COLOR_BG, True, screen_points, blend=0)
            pygame.draw.lines(self.screen, self.COLOR_TRACK, True, screen_points, width=track_width)
            pygame.draw.lines(self.screen, self.COLOR_BG, True, screen_points, width=inner_width)
        
        # Finish Line
        finish_line_rect = self.checkpoints[0]
        p1 = pygame.math.Vector2(finish_line_rect.midleft) - cam_offset
        p2 = pygame.math.Vector2(finish_line_rect.midright) - cam_offset
        pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, p1, p2, 5)

        # Obstacles
        for obstacle in self.obstacles:
            screen_rect = obstacle.move(-cam_offset.x, -cam_offset.y)
            if self.screen.get_rect().colliderect(screen_rect):
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)
                pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.COLOR_OBSTACLE), screen_rect, 2)
    
    def _render_car(self):
        w, h = self.car["width"], self.car["height"]
        
        # Create a surface for the car
        car_surf = pygame.Surface((h, h), pygame.SRCALPHA)
        
        # Glow effect
        glow_color = self.COLOR_PLAYER
        for i in range(4):
            alpha = 60 - i * 15
            radius = int(h / 2 + i * 2)
            pygame.gfxdraw.filled_circle(car_surf, int(h/2), int(h/2), radius, (*glow_color, alpha))

        # Car body (as a rotated line)
        body_points = [
            (h / 2 - h / 2, h / 2),
            (h / 2 + h / 2, h / 2)
        ]
        
        # Car body
        car_rect = pygame.Rect(h/2 - h/2.5, h/2 - w/2, h/1.25, w)
        pygame.draw.rect(car_surf, self.COLOR_PLAYER, car_rect, border_radius=4)
        
        # "Windshield"
        windshield_rect = pygame.Rect(h/2 + h/5, h/2 - w/3, h/6, w*2/3)
        pygame.draw.rect(car_surf, self.COLOR_BG, windshield_rect, border_radius=2)

        rotated_surf = pygame.transform.rotate(car_surf, -self.car["angle"] - 90)
        
        # Center the rotated surface
        center_pos = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        car_draw_rect = rotated_surf.get_rect(center=center_pos)
        
        self.screen.blit(rotated_surf, car_draw_rect)

    def _render_particles(self, cam_offset):
        for p in self.particles:
            pos = p["pos"] - cam_offset
            if p["type"] == "spark":
                radius = p["life"] / p["max_life"] * 3
                pygame.draw.circle(self.screen, p["color"], pos, radius)
            elif p["type"] == "speed_line":
                end_pos = pos + p["vel"] * 0.5
                alpha = int(p["life"] / p["max_life"] * 200)
                color = (*p["color"], alpha)
                pygame.draw.line(self.screen, color, pos, end_pos, width=max(1, int(p["life"] / p["max_life"] * 3)))
    
    def _render_ui(self):
        lap_text = f"LAP: {min(self.lap_count + 1, self.NUM_LAPS_TO_WIN)}/{self.NUM_LAPS_TO_WIN}"
        time_text = f"TIME: {(self.steps / self.FPS):.2f}"
        
        best_lap_str = f"BEST: {self.best_lap_time:.2f}" if self.best_lap_time != float('inf') else "BEST: --.--"
        
        self._draw_text(lap_text, (10, 10), self.COLOR_UI)
        self._draw_text(time_text, (self.SCREEN_WIDTH - 150, 10), self.COLOR_UI)
        self._draw_text(best_lap_str, (self.SCREEN_WIDTH - 150, 30), self.COLOR_UI)
        
    def _render_end_message(self):
        msg = "RACE COMPLETE" if self.game_won else "CRASHED"
        color = self.COLOR_PLAYER if self.game_won else self.COLOR_OBSTACLE
        
        text_surf = self.font_msg.render(msg, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        bg_rect = text_rect.inflate(20, 20)
        s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        s.fill((*self.COLOR_BG, 200))
        self.screen.blit(s, bg_rect)
        
        self.screen.blit(text_surf, text_rect)

    def _draw_text(self, text, pos, color):
        text_surface = self.font_ui.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _get_angle(self, p1, p2):
        return math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x))
        
    def _spawn_sparks(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            life = self.np_random.integers(10, 20)
            self.particles.append({
                "pos": pygame.math.Vector2(pos),
                "vel": pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "life": life, "max_life": life,
                "color": self.COLOR_SPARK, "type": "spark"
            })
            
    def _spawn_speed_lines(self, is_drifting):
        car_angle_rad = math.radians(self.car["angle"])
        
        # Spawn behind the car
        back_offset = pygame.math.Vector2(-math.cos(car_angle_rad), -math.sin(car_angle_rad)) * (self.car["height"] / 2)
        
        for i in [-1, 1]:
            side_offset_angle = car_angle_rad + math.pi / 2
            side_offset = pygame.math.Vector2(math.cos(side_offset_angle), math.sin(side_offset_angle)) * (self.car["width"] / 2 * i)
            
            spawn_pos = self.car["pos"] + back_offset + side_offset
            
            drift_angle_mod = 0
            if is_drifting:
                drift_angle_mod = math.radians(self.car["turn_rate"] * 2 * -i)

            vel_angle = car_angle_rad + math.pi + drift_angle_mod
            vel = pygame.math.Vector2(math.cos(vel_angle), math.sin(vel_angle)) * self.car["speed"] * 0.5
            
            life = 8
            self.particles.append({
                "pos": spawn_pos, "vel": vel,
                "life": life, "max_life": life,
                "color": self.COLOR_PLAYER, "type": "speed_line"
            })

    def close(self):
        pygame.quit()