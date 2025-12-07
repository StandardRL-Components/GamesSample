
# Generated: 2025-08-28T05:36:31.571092
# Source Brief: brief_02671.md
# Brief Index: 2671

        
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
        "Controls: ↑ to accelerate, ↓ to brake/reverse, ←→ to turn. "
        "Hold Space for a risky speed boost."
    )

    game_description = (
        "Control a snail in a fast-paced isometric race against two AI opponents. "
        "Stay on the track, manage your speed, and use boosts wisely to finish first!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.TIME_LIMIT = 60.0  # seconds
        self.MAX_STEPS = int(self.TIME_LIMIT * self.FPS)
        self.NUM_LAPS = 2

        # --- Colors ---
        self.COLOR_BG = (135, 206, 235)  # Sky Blue
        self.COLOR_TRACK = (188, 143, 143)  # Rosy Brown
        self.COLOR_TRACK_BORDER = (112, 84, 84)
        self.COLOR_FINISH_LINE = (255, 255, 0) # Yellow
        self.COLOR_PLAYER = (50, 205, 50)  # Lime Green
        self.COLOR_AI1 = (220, 20, 60)      # Crimson
        self.COLOR_AI2 = (65, 105, 225)     # Royal Blue
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)
        self.COLOR_BOOST = (255, 215, 0) # Gold

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # --- Isometric Projection ---
        self.iso_scale = 16
        self.iso_offset_x = self.WIDTH // 2
        self.iso_offset_y = self.HEIGHT // 5

        # --- Track Definition ---
        self.track_waypoints = self._create_track_waypoints()
        self.track_poly, self.track_border_poly = self._create_track_polygons(self.track_waypoints, 1.8)
        self.total_track_length = self._calculate_track_length(self.track_waypoints)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_elapsed = 0.0
        self.player = None
        self.ais = []
        self.all_snails = []
        self.particles = []
        self.end_message = ""
        self.race_results = []
        
        # --- Initialize state variables ---
        self.reset()
        
        # --- Implementation Validation ---
        self.validate_implementation()

    def _create_track_waypoints(self):
        points = []
        num_points = 60
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            x = math.cos(angle) * 10
            y = math.sin(angle) * 5 + math.sin(angle * 2) * 2.5
            points.append(pygame.Vector2(x, y))
        return points

    def _create_track_polygons(self, waypoints, width):
        outer_points = []
        inner_points = []
        for i in range(len(waypoints)):
            p1 = waypoints[i]
            p2 = waypoints[(i + 1) % len(waypoints)]
            
            direction = (p2 - p1).normalize()
            perp = pygame.Vector2(-direction.y, direction.x) * width
            
            outer_points.append(p1 + perp)
            inner_points.append(p1 - perp)

        return outer_points, inner_points

    def _calculate_track_length(self, waypoints):
        length = 0
        for i in range(len(waypoints)):
            p1 = waypoints[i]
            p2 = waypoints[(i + 1) % len(waypoints)]
            length += p1.distance_to(p2)
        return length

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_elapsed = 0.0
        self.end_message = ""
        self.race_results = []
        self.particles = []
        
        start_pos = self.track_waypoints[0] + pygame.Vector2(0, -0.5)
        self.player = Snail(self, start_pos, self.COLOR_PLAYER, is_ai=False, name="Player")

        ai1_start_pos = self.track_waypoints[0] + pygame.Vector2(0, 0.5)
        self.ais = [
            Snail(self, ai1_start_pos, self.COLOR_AI1, is_ai=True, name="AI 1"),
            Snail(self, start_pos + pygame.Vector2(0.3, 0), self.COLOR_AI2, is_ai=True, name="AI 2"),
        ]
        self.all_snails = [self.player] + self.ais
        
        for snail in self.all_snails:
            snail.reset()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False

        if not self.game_over:
            # --- Previous State for Rewards ---
            prev_player_progress = self.player.get_progress()
            prev_ranks = {snail.name: rank for rank, snail in enumerate(self._get_ranks())}
            
            # --- Action Handling ---
            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            self.player.handle_input(movement, space_held)

            # --- Update Game Logic ---
            for snail in self.all_snails:
                lap_completed = snail.update()
                if lap_completed and snail == self.player:
                    reward += 5.0 # Lap completion reward
                    # Increase AI speed
                    for ai in self.ais:
                        ai.speed_multiplier += 0.05
            
            self._update_particles()
            
            # --- Reward Calculation ---
            # Progress reward
            current_player_progress = self.player.get_progress()
            progress_delta = current_player_progress - prev_player_progress
            if progress_delta < -self.total_track_length / 2: # Lap crossover
                progress_delta += self.total_track_length
            reward += progress_delta * 0.1

            # Overtake reward
            current_ranks = {snail.name: rank for rank, snail in enumerate(self._get_ranks())}
            if current_ranks[self.player.name] < prev_ranks[self.player.name]:
                reward += 10.0
            
            # Off-track penalty
            if not self.player.is_on_track:
                reward -= 0.2

            # --- Time and Termination ---
            self.steps += 1
            self.time_elapsed += 1.0 / self.FPS
            
            if self._check_termination():
                terminated = True
                self.game_over = True
                self._calculate_terminal_rewards()
                reward += self.player.terminal_reward

        self.score += reward
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_termination(self):
        if self.player.fallen_off:
            self.end_message = "You fell off!"
            return True
        if self.time_elapsed >= self.TIME_LIMIT:
            self.end_message = "Time's Up!"
            return True
        if len(self.race_results) == len(self.all_snails):
            return True
        if self.player.lap > self.NUM_LAPS:
            return True
        return False

    def _calculate_terminal_rewards(self):
        final_ranks = self._get_ranks()
        
        # Add any remaining snails to results
        for snail in self.all_snails:
            if snail not in self.race_results:
                self.race_results.append(snail)

        for i, snail in enumerate(self.race_results):
            if snail == self.player:
                if i == 0:
                    snail.terminal_reward = 50.0
                    self.end_message = "You Win! 1st Place"
                elif i == 1:
                    snail.terminal_reward = 25.0
                    self.end_message = "2nd Place!"
                else:
                    snail.terminal_reward = 10.0
                    self.end_message = "3rd Place"
        
        if self.player.fallen_off:
            self.player.terminal_reward = -100.0 # Changed from -1.0 to be in range [-100, 100]


    def _get_ranks(self):
        return sorted(self.all_snails, key=lambda s: s.get_progress_total(), reverse=True)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Camera ---
        cam_pos = self.player.pos * 0.8 + self.ais[0].pos * 0.1 + self.ais[1].pos * 0.1
        
        # --- Render Track ---
        def project_poly(poly, offset):
            return [self._iso(p.x - offset.x, p.y - offset.y) for p in poly]

        track_screen_poly = project_poly(self.track_poly, cam_pos)
        border_screen_poly = project_poly(self.track_border_poly, cam_pos)
        
        pygame.gfxdraw.filled_polygon(self.screen, border_screen_poly, self.COLOR_TRACK_BORDER)
        pygame.gfxdraw.filled_polygon(self.screen, track_screen_poly, self.COLOR_TRACK)
        
        # --- Render Finish Line ---
        p1 = self.track_waypoints[0] + pygame.Vector2(0, 1.8)
        p2 = self.track_waypoints[0] - pygame.Vector2(0, 1.8)
        iso_p1 = self._iso(p1.x - cam_pos.x, p1.y - cam_pos.y)
        iso_p2 = self._iso(p2.x - cam_pos.x, p2.y - cam_pos.y)
        pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, iso_p1, iso_p2, 5)

        # --- Render Snails and Particles (sorted by Z-order) ---
        render_list = self.particles + self.all_snails
        render_list.sort(key=lambda obj: obj.pos.x + obj.pos.y)

        for obj in render_list:
            obj.render(self.screen, cam_pos)

    def _render_ui(self):
        # Race Timer
        time_left = max(0, self.TIME_LIMIT - self.time_elapsed)
        timer_text = f"{int(time_left // 60):02}:{int(time_left % 60):02}"
        self._draw_text(timer_text, (self.WIDTH - 10, 10), self.font_small, "topright")

        # Lap Counter
        lap_text = f"Lap: {min(self.player.lap, self.NUM_LAPS)} / {self.NUM_LAPS}"
        self._draw_text(lap_text, (10, self.HEIGHT - 10), self.font_small, "bottomleft")

        # Ranks
        ranks = self._get_ranks()
        for i, snail in enumerate(ranks):
            pygame.draw.circle(self.screen, snail.color, (20 + i * 30, 20), 10)
            if not snail.is_ai:
                pygame.draw.circle(self.screen, self.COLOR_TEXT, (20 + i * 30, 20), 10, 2)

        # Game Over Message
        if self.game_over:
            self._draw_text(self.end_message, (self.WIDTH // 2, self.HEIGHT // 2), self.font_large, "center")

    def _draw_text(self, text, pos, font, anchor="topleft"):
        text_surf = font.render(text, True, self.COLOR_TEXT)
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect()
        setattr(text_rect, anchor, pos)
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed,
            "player_lap": self.player.lap,
            "player_rank": self._get_ranks().index(self.player) + 1,
        }

    def _iso(self, x, y):
        return (
            (x - y) * self.iso_scale + self.iso_offset_x,
            (x + y) * self.iso_scale * 0.5 + self.iso_offset_y
        )
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def is_point_on_track(self, point):
        # Ray-casting algorithm
        x, y = point.x, point.y
        n = len(self.track_border_poly)
        inside = False
        p1x, p1y = self.track_border_poly[0].x, self.track_border_poly[0].y
        for i in range(n + 1):
            p2x, p2y = self.track_border_poly[i % n].x, self.track_border_poly[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

class Snail:
    def __init__(self, env, pos, color, is_ai=False, name=""):
        self.env = env
        self.start_pos = pos.copy()
        self.color = color
        self.is_ai = is_ai
        self.name = name
        self.reset()

    def reset(self):
        self.pos = self.start_pos.copy()
        self.vel = pygame.Vector2(0, 0)
        self.angle = -math.pi / 2  # Pointing "up" in world space
        self.lap = 1
        self.waypoint_index = 0
        self.is_on_track = True
        self.fallen_off = False
        self.fall_timer = 0
        self.terminal_reward = 0.0
        
        # Physics
        self.max_speed = 0.08
        self.accel = 0.002
        self.brake = 0.004
        self.turn_speed = 0.07
        self.friction = 0.97
        self.boost_cooldown = 0
        self.boost_duration = 0
        
        # AI specific
        self.ai_target_offset = pygame.Vector2(0, 0)
        self.speed_multiplier = 1.0

    def handle_input(self, movement, space_held):
        if self.is_ai: return

        # Movement: 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: # Accelerate
            speed = self.vel.length()
            accel_vec = pygame.Vector2(math.cos(self.angle), math.sin(self.angle)) * self.accel
            self.vel += accel_vec
        elif movement == 2: # Brake
            self.vel *= 0.92
        if movement == 3: # Turn Left
            self.angle -= self.turn_speed
        if movement == 4: # Turn Right
            self.angle += self.turn_speed
        
        # Boost
        self.boost_cooldown = max(0, self.boost_cooldown - 1)
        self.boost_duration = max(0, self.boost_duration - 1)
        if space_held and self.boost_cooldown == 0:
            self.boost_duration = 15 # frames
            self.boost_cooldown = 90 # frames
            # sfx: boost sound
            for _ in range(15):
                self.env.particles.append(Particle(self.env, self.pos.copy(), self.color))

    def update(self):
        if self.fallen_off: return False
        
        if self.is_ai:
            self._update_ai()

        # Apply friction
        self.vel *= self.friction
        
        # Apply boost
        if self.boost_duration > 0:
            boost_vec = pygame.Vector2(math.cos(self.angle), math.sin(self.angle)) * 0.015
            self.vel += boost_vec
        
        # Clamp speed
        speed = self.vel.length()
        if speed > self.max_speed:
            self.vel.scale_to_length(self.max_speed)
        
        # Update position
        self.pos += self.vel

        # Check track boundaries
        was_on_track = self.is_on_track
        self.is_on_track = self.env.is_point_on_track(self.pos)
        if not self.is_on_track:
            self.vel *= 0.95 # Slow down off-track
            if was_on_track:
                self.fall_timer = 30 # 0.5 seconds to get back on track
            else:
                self.fall_timer -= 1
                if self.fall_timer <= 0:
                    self.fallen_off = True
                    if self not in self.env.race_results:
                        self.env.race_results.append(self)
        
        # Check lap completion
        return self._update_progress()

    def _update_ai(self):
        # Target the next waypoint with some randomness
        if self.env.np_random.random() < 0.05:
            self.ai_target_offset = pygame.Vector2(
                self.env.np_random.uniform(-0.5, 0.5),
                self.env.np_random.uniform(-0.5, 0.5)
            )
        
        target_waypoint = self.env.track_waypoints[(self.waypoint_index + 1) % len(self.env.track_waypoints)]
        target_pos = target_waypoint + self.ai_target_offset
        
        # Steer towards target
        target_angle = math.atan2(target_pos.y - self.pos.y, target_pos.x - self.pos.x)
        angle_diff = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        self.angle += np.clip(angle_diff, -self.turn_speed * 0.8, self.turn_speed * 0.8)

        # Accelerate
        accel_vec = pygame.Vector2(math.cos(self.angle), math.sin(self.angle)) * self.accel * self.speed_multiplier
        self.vel += accel_vec

    def _update_progress(self):
        # Find closest waypoint
        min_dist = float('inf')
        closest_index = self.waypoint_index
        # Search a window around the current waypoint for efficiency
        for i in range(-5, 6):
            idx = (self.waypoint_index + i) % len(self.env.track_waypoints)
            dist = self.pos.distance_to(self.env.track_waypoints[idx])
            if dist < min_dist:
                min_dist = dist
                closest_index = idx

        prev_waypoint_index = self.waypoint_index
        self.waypoint_index = closest_index

        # Check for lap completion
        if prev_waypoint_index > len(self.env.track_waypoints) - 5 and self.waypoint_index < 5:
            if self.lap <= self.env.NUM_LAPS:
                self.lap += 1
                if self.lap > self.env.NUM_LAPS:
                    if self not in self.env.race_results:
                        self.env.race_results.append(self)
                return True # Lap completed
        return False

    def get_progress(self):
        # Distance from start of track segment to snail
        p1 = self.env.track_waypoints[self.waypoint_index]
        p2 = self.env.track_waypoints[(self.waypoint_index + 1) % len(self.env.track_waypoints)]
        
        segment_vec = p2 - p1
        snail_vec = self.pos - p1
        
        # Project snail_vec onto segment_vec
        if segment_vec.length() > 0:
            proj = snail_vec.dot(segment_vec) / segment_vec.length_squared()
            dist_on_segment = np.clip(proj, 0, 1) * segment_vec.length()
        else:
            dist_on_segment = 0
            
        # Add lengths of previous segments
        progress = dist_on_segment
        for i in range(self.waypoint_index):
            progress += self.env.track_waypoints[i].distance_to(self.env.track_waypoints[(i + 1) % len(self.env.track_waypoints)])
        
        return progress

    def get_progress_total(self):
        if self.fallen_off:
            return -1
        return (self.lap - 1) * self.env.total_track_length + self.get_progress()

    def render(self, screen, cam_pos):
        iso_pos = self.env._iso(self.pos.x - cam_pos.x, self.pos.y - cam_pos.y)
        
        # Simple snail shape
        shell_radius = 10
        body_len = 12
        
        # Body
        body_offset = pygame.Vector2(math.cos(self.angle), math.sin(self.angle)) * body_len
        iso_body_start = self.env._iso(self.pos.x - body_offset.x*0.2 - cam_pos.x, self.pos.y - body_offset.y*0.2 - cam_pos.y)
        iso_body_end = self.env._iso(self.pos.x + body_offset.x*0.5 - cam_pos.x, self.pos.y + body_offset.y*0.5 - cam_pos.y)
        
        body_color = tuple(np.clip(np.array(self.color) * 0.7, 0, 255))
        pygame.draw.line(screen, body_color, iso_body_start, iso_body_end, 8)
        
        # Shell
        pygame.gfxdraw.filled_circle(screen, int(iso_pos[0]), int(iso_pos[1]), shell_radius, self.color)
        pygame.gfxdraw.aacircle(screen, int(iso_pos[0]), int(iso_pos[1]), shell_radius, tuple(np.array(self.color)*0.5))

        # Eyes
        eye_base_x = iso_body_end[0]
        eye_base_y = iso_body_end[1]
        pygame.draw.circle(screen, (255,255,255), (int(eye_base_x-2), int(eye_base_y-2)), 3)
        pygame.draw.circle(screen, (0,0,0), (int(eye_base_x-2), int(eye_base_y-2)), 1)
        pygame.draw.circle(screen, (255,255,255), (int(eye_base_x+2), int(eye_base_y-2)), 3)
        pygame.draw.circle(screen, (0,0,0), (int(eye_base_x+2), int(eye_base_y-2)), 1)

class Particle:
    def __init__(self, env, pos, color):
        self.env = env
        self.pos = pos
        self.color = env.COLOR_BOOST
        angle = env.np_random.uniform(0, 2 * math.pi)
        speed = env.np_random.uniform(0.05, 0.1)
        self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.lifespan = 20

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        return self.lifespan > 0

    def render(self, screen, cam_pos):
        if self.lifespan > 0:
            iso_pos = self.env._iso(self.pos.x - cam_pos.x, self.pos.y - cam_pos.y)
            radius = int(self.lifespan / 20 * 4)
            if radius > 0:
                pygame.draw.circle(screen, self.color, iso_pos, radius)