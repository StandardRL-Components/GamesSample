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

    user_guide = (
        "Controls: ↑ to accelerate, ↓ to brake/reverse, ←→ to turn. "
        "Hold Shift to drift. Press Space to boost."
    )

    game_description = (
        "A fast-paced, top-down arcade racer on a procedurally generated track. "
        "Dodge obstacles, drift through corners, and boost to beat the clock. "
        "Complete 3 laps, each under 60 seconds, to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60

    # Colors
    COLOR_BG = (30, 35, 40)
    COLOR_TRACK = (70, 80, 90)
    COLOR_TRACK_BORDER = (180, 190, 200)
    COLOR_PLAYER = (255, 60, 60)
    COLOR_PLAYER_GLOW = (255, 100, 100)
    COLOR_OBSTACLE = (60, 160, 255)
    COLOR_CHECKPOINT = (80, 220, 120)
    COLOR_PARTICLE_DRIFT = (200, 200, 200)
    COLOR_PARTICLE_CRASH = (255, 200, 80)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_VALUE = (255, 255, 255)
    COLOR_BOOST_BAR_BG = (50, 50, 60)
    COLOR_BOOST_BAR_FG = (100, 200, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game state variables are initialized in reset()
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = 0.0
        self.player_angle = 0.0
        self.player_movement_angle = 0.0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.lap = 0
        self.lap_timer = 0.0
        self.current_checkpoint = 0
        self.track_waypoints = []
        self.track_centerline = []
        self.checkpoints = []
        self.obstacles = []
        self.particles = []
        self.player_trail = []
        self.boost_meter = 100.0
        self.boost_active = False
        self.game_message = ""
        self.message_timer = 0
        self.base_obstacle_count = 0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.lap = 0
        self.lap_timer = 0.0
        self.boost_meter = 100.0
        self.boost_active = False
        self.game_message = ""
        self.message_timer = 0

        self._generate_track(self.np_random)

        self.player_pos = pygame.Vector2(self.track_waypoints[0])
        p1 = self.track_waypoints[0]
        p2 = self.track_waypoints[1]
        self.player_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
        self.player_movement_angle = self.player_angle
        self.player_vel = 0.0

        self.current_checkpoint = 1

        self.base_obstacle_count = 5
        self._generate_obstacles()

        self.particles = []
        self.player_trail = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0

        # --- Update Game Logic ---
        self._handle_input(action)
        self._update_player_physics()
        self._update_particles()

        # On-track reward
        reward += 0.01

        # Off-track check and penalty
        if self._is_off_track():
            self.player_vel *= 0.95  # Slow down off-track
            reward -= 0.02

        # Checkpoint and lap logic
        lap_reward, lap_done = self._check_laps_and_checkpoints()
        reward += lap_reward
        if lap_done:
            self._generate_obstacles()

        # Collision check
        collision_reward, collided = self._check_collisions()
        reward += collision_reward

        # Update timers
        self.lap_timer += 1 / self.FPS
        self.steps += 1

        # --- Termination Conditions ---
        terminated = False
        truncated = False
        if collided:
            terminated = True
            self.game_message = "CRASHED!"
            self.message_timer = self.FPS * 2
        elif self.lap_timer > 60:
            terminated = True
            reward -= 20
            self.game_message = "TIME OUT!"
            self.message_timer = self.FPS * 2
        elif self.lap >= 3:
            terminated = True
            reward += 100
            self.game_message = "FINISH!"
            self.message_timer = self.FPS * 2
        elif self.steps >= 9000:  # Max episode length (60s * 3 laps * 50fps)
            truncated = True

        if terminated or truncated:
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self.inputs = {
            "accelerate": 1.0 if movement == 1 else 0.0,
            "brake": 1.0 if movement == 2 else 0.0,
            "left": 1.0 if movement == 3 else 0.0,
            "right": 1.0 if movement == 4 else 0.0,
            "boost": space_held,
            "drift": shift_held,
        }

    def _update_player_physics(self):
        dt = 1 / self.FPS

        # --- Physics Constants ---
        ACCEL = 250.0
        BRAKE = 400.0
        FRICTION = 0.9
        TURN_SPEED = 2.5
        MAX_VEL = 200.0
        MAX_REVERSE_VEL = -50.0

        # Drift & Boost modifiers
        is_drifting = self.inputs["drift"] and self.player_vel > 50
        self.boost_active = self.inputs["boost"] and self.boost_meter > 0

        if self.boost_active:
            self.boost_meter -= 2.0
            current_max_vel = MAX_VEL * 1.5
            current_accel = ACCEL * 2.0
        else:
            self.boost_meter = min(100, self.boost_meter + 0.5)
            current_max_vel = MAX_VEL
            current_accel = ACCEL

        turn_mod = 1.75 if is_drifting else 1.0
        friction_mod = 0.3 if is_drifting else 1.0

        # --- Update Angle ---
        turn_input = self.inputs["right"] - self.inputs["left"]
        if self.player_vel != 0:
            flip = -1 if self.player_vel < 0 else 1
            self.player_angle += turn_input * TURN_SPEED * turn_mod * flip * dt

        # --- Update Velocity ---
        accel_input = self.inputs["accelerate"] - self.inputs["brake"]
        if accel_input > 0:
            self.player_vel += current_accel * accel_input * dt
        elif accel_input < 0:
            self.player_vel += BRAKE * accel_input * dt

        self.player_vel *= 1 - FRICTION * friction_mod * dt
        self.player_vel = max(MAX_REVERSE_VEL, min(current_max_vel, self.player_vel))
        if abs(self.player_vel) < 0.5:
            self.player_vel = 0

        # --- Update Position ---
        if is_drifting:
            # Slide effect
            self.player_movement_angle = self._lerp_angle(
                self.player_movement_angle, self.player_angle, 0.1
            )
            # Drift particles
            if self.steps % 2 == 0:
                self._create_drift_particles()
        else:
            self.player_movement_angle = self.player_angle

        move_vec = pygame.Vector2(
            math.cos(self.player_movement_angle), math.sin(self.player_movement_angle)
        )
        self.player_pos += move_vec * self.player_vel * dt

        # --- Trail ---
        self.player_trail.append(pygame.Vector2(self.player_pos))
        if len(self.player_trail) > 20:
            self.player_trail.pop(0)

    def _is_off_track(self):
        min_dist_sq = float("inf")
        for i in range(len(self.track_centerline) - 1):
            p1 = self.track_centerline[i]
            p2 = self.track_centerline[i + 1]
            dist_sq = self._dist_to_segment_sq(self.player_pos, p1, p2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        return min_dist_sq > (60**2)  # Track width is 120

    def _check_laps_and_checkpoints(self):
        reward = 0
        lap_completed = False
        next_checkpoint_idx = self.current_checkpoint % len(self.checkpoints)
        checkpoint_line = self.checkpoints[next_checkpoint_idx]

        # Simple line crossing check
        prev_pos_relative = (
            self.player_trail[-2] - checkpoint_line[0]
            if len(self.player_trail) > 1
            else self.player_pos - checkpoint_line[0]
        )
        current_pos_relative = self.player_pos - checkpoint_line[0]
        line_vec = checkpoint_line[1] - checkpoint_line[0]

        prev_dot = prev_pos_relative.dot(line_vec.rotate(90))
        current_dot = current_pos_relative.dot(line_vec.rotate(90))

        if prev_dot * current_dot < 0:  # Crossed the line
            self.current_checkpoint += 1
            if next_checkpoint_idx == 0:  # Crossed start/finish line
                reward += 50 + max(0, 60 - self.lap_timer) * 0.5
                self.lap += 1
                self.lap_timer = 0.0
                lap_completed = True
                self.game_message = f"LAP {self.lap}"
                self.message_timer = self.FPS * 1.5
            else:
                reward += 5

        return reward, lap_completed

    def _check_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x - 8, self.player_pos.y - 4, 16, 8)
        player_radius = 10  # Simplified circular collision for car

        for obs_pos, obs_radius in self.obstacles:
            dist = self.player_pos.distance_to(obs_pos)
            if dist < player_radius + obs_radius:
                # sfx: crash
                self._create_collision_particles(obs_pos)
                return -10, True
        return 0, False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)

        cam_offset = (
            pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
            - self.player_pos
        )

        self._render_track(cam_offset)
        self._render_obstacles(cam_offset)
        self._render_particles(cam_offset)
        self._render_player(cam_offset)
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_track(self, offset):
        # Draw track fill
        pygame.draw.polygon(
            self.screen, self.COLOR_TRACK, [(p + offset) for p in self.track_polygon_outer], 0
        )
        pygame.draw.polygon(
            self.screen, self.COLOR_BG, [(p + offset) for p in self.track_polygon_inner], 0
        )

        # Draw track borders
        pygame.draw.aalines(
            self.screen,
            self.COLOR_TRACK_BORDER,
            True,
            [(p + offset) for p in self.track_polygon_outer],
            1,
        )
        pygame.draw.aalines(
            self.screen,
            self.COLOR_TRACK_BORDER,
            True,
            [(p + offset) for p in self.track_polygon_inner],
            1,
        )

        # Draw next checkpoint
        next_checkpoint_idx = self.current_checkpoint % len(self.checkpoints)
        p1, p2 = self.checkpoints[next_checkpoint_idx]
        pygame.draw.aaline(self.screen, self.COLOR_CHECKPOINT, p1 + offset, p2 + offset, 3)

    def _render_obstacles(self, offset):
        for pos, radius in self.obstacles:
            pygame.gfxdraw.filled_circle(
                self.screen, int(pos.x + offset.x), int(pos.y + offset.y), int(radius), self.COLOR_OBSTACLE
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(pos.x + offset.x), int(pos.y + offset.y), int(radius), self.COLOR_OBSTACLE
            )

    def _render_player(self, offset):
        center_x, center_y = self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2

        # Trail
        if len(self.player_trail) > 1:
            for i in range(len(self.player_trail) - 1):
                alpha = int(200 * (i / len(self.player_trail)))
                color = (
                    (*self.COLOR_PLAYER_GLOW, alpha)
                    if self.boost_active
                    else (*self.COLOR_TRACK_BORDER, alpha)
                )
                p1 = self.player_trail[i] + offset
                p2 = self.player_trail[i + 1] + offset
                pygame.draw.line(self.screen, color, p1, p2, 3)

        # Car body
        car_surf = pygame.Surface((24, 12), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, self.COLOR_PLAYER, (0, 0, 24, 12), border_radius=3)

        # Rotate and blit
        rotated_surf = pygame.transform.rotate(car_surf, -math.degrees(self.player_angle))
        rect = rotated_surf.get_rect(center=(center_x, center_y))
        self.screen.blit(rotated_surf, rect.topleft)

    def _render_particles(self, offset):
        for p in self.particles:
            p_pos, p_vel, p_life, p_color = p
            pygame.draw.circle(self.screen, p_color, p_pos + offset, int(p_life))

    def _render_ui(self):
        # Lap Time
        time_text = self.font_small.render("TIME", True, self.COLOR_UI_TEXT)
        time_val = self.font_small.render(f"{self.lap_timer:05.2f}", True, self.COLOR_UI_VALUE)
        self.screen.blit(time_text, (10, 10))
        self.screen.blit(time_val, (10, 30))

        # Score
        score_text = self.font_small.render("SCORE", True, self.COLOR_UI_TEXT)
        score_val = self.font_small.render(f"{int(self.score):06d}", True, self.COLOR_UI_VALUE)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_val.get_width() - 10, 10))
        self.screen.blit(score_val, (self.SCREEN_WIDTH - score_val.get_width() - 10, 30))

        # Lap Counter
        lap_text = self.font_large.render(
            f"LAP {min(self.lap + 1, 3)}/3", True, self.COLOR_UI_TEXT
        )
        self.screen.blit(
            lap_text, lap_text.get_rect(centerx=self.SCREEN_WIDTH / 2, bottom=self.SCREEN_HEIGHT - 10)
        )

        # Boost Meter
        bar_width = 200
        bar_height = 10
        x = (self.SCREEN_WIDTH - bar_width) / 2
        y = self.SCREEN_HEIGHT - 60
        fill_width = (self.boost_meter / 100) * bar_width
        pygame.draw.rect(
            self.screen, self.COLOR_BOOST_BAR_BG, (x, y, bar_width, bar_height), border_radius=3
        )
        if fill_width > 0:
            pygame.draw.rect(
                self.screen, self.COLOR_BOOST_BAR_FG, (x, y, fill_width, bar_height), border_radius=3
            )

        # Game Message
        if self.message_timer > 0:
            self.message_timer -= 1
            alpha = min(255, int(255 * (self.message_timer / (self.FPS * 0.5))))
            msg_surf = self.font_large.render(self.game_message, True, self.COLOR_UI_VALUE)
            msg_surf.set_alpha(alpha)
            self.screen.blit(
                msg_surf, msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 50))
            )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": self.lap,
            "lap_time": self.lap_timer,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    # --- Generation and Particle Helpers ---
    def _generate_track(self, rng):
        num_waypoints = 12
        center = pygame.Vector2(0, 0)
        radius = 400
        angle_step = 2 * math.pi / num_waypoints

        self.track_waypoints = []
        for i in range(num_waypoints):
            angle = i * angle_step
            r = radius * rng.uniform(0.8, 1.2)
            x = center.x + r * math.cos(angle)
            y = center.y + r * math.sin(angle)
            self.track_waypoints.append(pygame.Vector2(x, y))

        # Create a smooth centerline using Catmull-Rom splines
        self.track_centerline = []
        for i in range(num_waypoints):
            p0 = self.track_waypoints[(i - 1 + num_waypoints) % num_waypoints]
            p1 = self.track_waypoints[i]
            p2 = self.track_waypoints[(i + 1) % num_waypoints]
            p3 = self.track_waypoints[(i + 2) % num_waypoints]
            for t in np.linspace(0, 1, 20):
                self.track_centerline.append(self._get_spline_point(t, p0, p1, p2, p3))

        # Create geometry for rendering and checkpoints
        self.track_polygon_outer = []
        self.track_polygon_inner = []
        self.checkpoints = []
        track_width = 60

        for i, p1 in enumerate(self.track_centerline):
            p2 = self.track_centerline[(i + 1) % len(self.track_centerline)]
            angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
            perp_vec = pygame.Vector2(-math.sin(angle), math.cos(angle))
            self.track_polygon_outer.append(p1 + perp_vec * track_width)
            self.track_polygon_inner.append(p1 - perp_vec * track_width)

        for i in range(num_waypoints):
            p1 = self.track_waypoints[i]
            p2 = self.track_waypoints[(i + 1) % num_waypoints]
            angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
            perp_vec = pygame.Vector2(-math.sin(angle), math.cos(angle))
            cp_p1 = p1 + perp_vec * (track_width + 10)
            cp_p2 = p1 - perp_vec * (track_width + 10)
            self.checkpoints.append((cp_p1, cp_p2))

    def _generate_obstacles(self):
        self.obstacles = []
        num_obstacles = self.base_obstacle_count + self.lap
        track_width = 60

        for _ in range(num_obstacles):
            segment_idx = self.np_random.integers(0, len(self.track_centerline) - 1)
            point_on_centerline = self.track_centerline[segment_idx]

            angle = math.atan2(
                self.track_centerline[segment_idx + 1].y - point_on_centerline.y,
                self.track_centerline[segment_idx + 1].x - point_on_centerline.x,
            )
            perp_vec = pygame.Vector2(-math.sin(angle), math.cos(angle))

            offset_dist = self.np_random.uniform(-track_width * 0.7, track_width * 0.7)
            radius = self.np_random.uniform(8, 15)
            pos = point_on_centerline + perp_vec * offset_dist

            # Avoid placing on start line
            if pos.distance_to(self.track_waypoints[0]) < 100:
                continue

            self.obstacles.append((pos, radius))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]
        for i, (pos, vel, life, color) in enumerate(self.particles):
            pos += vel
            life -= 0.2
            self.particles[i] = (pos, vel, life, color)

    def _create_drift_particles(self):
        # sfx: tire screech
        for _ in range(2):
            angle_offset = self.np_random.uniform(-0.2, 0.2)
            vel_angle = self.player_angle + math.pi + angle_offset
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector2(math.cos(vel_angle), math.sin(vel_angle)) * speed

            start_pos_offset = (
                pygame.Vector2(math.cos(self.player_angle + math.pi), math.sin(self.player_angle + math.pi))
                * 10
            )
            pos = self.player_pos + start_pos_offset
            life = self.np_random.uniform(1, 4)
            self.particles.append((pos, vel, life, self.COLOR_PARTICLE_DRIFT))

    def _create_collision_particles(self, collision_point):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.uniform(2, 6)
            self.particles.append((pygame.Vector2(collision_point), vel, life, self.COLOR_PARTICLE_CRASH))

    # --- Math Helpers ---
    @staticmethod
    def _get_spline_point(t, p0, p1, p2, p3):
        t2, t3 = t * t, t * t * t
        return 0.5 * (
            (2 * p1)
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )

    @staticmethod
    def _lerp_angle(a, b, t):
        diff = (b - a + math.pi) % (2 * math.pi) - math.pi
        return a + diff * t

    @staticmethod
    def _dist_to_segment_sq(p, v, w):
        l2 = v.distance_squared_to(w)
        if l2 == 0.0:
            return p.distance_squared_to(v)
        t = max(0, min(1, (p - v).dot(w - v) / l2))
        projection = v + t * (w - v)
        return p.distance_squared_to(projection)


# --- Example Usage ---
if __name__ == "__main__":
    # To run with display, comment out the os.environ line at the top of the file
    # and uncomment the following line.
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")

    # --- Pygame window for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()

    obs, info = env.reset()
    terminated = False
    truncated = False

    print("\n" + "=" * 30)
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("=" * 30 + "\n")

    while not (terminated or truncated):
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        move = 0  # none
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            move = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            move = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            move = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            move = 4

        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [move, space, shift]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False
                truncated = False

        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']:.2f}, Laps: {info['lap']}")
    env.close()
    pygame.quit()