
# Generated: 2025-08-27T15:45:03.399849
# Source Brief: brief_01062.md
# Brief Index: 1062

        
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
        "Controls: ↑ to accelerate, ←→ to turn, ↓ to brake. Hold Space to drift and build boost, then press Shift to use it."
    )

    game_description = (
        "Fast-paced arcade racer. Drift through corners to build up your boost meter, then unleash it on the straights to overtake your opponents and set the fastest lap time."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.W, self.H = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000
        self.TIME_LIMIT = 60.0
        self.NUM_OPPONENTS = 5
        self.NUM_LAPS = 2
        self.OFF_TRACK_LIMIT = 2

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_TRACK = (50, 55, 60)
        self.COLOR_RUMBLE = (200, 200, 200)
        self.COLOR_GRASS = (30, 40, 35)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLAYER_GLOW = (255, 100, 100, 100)
        self.COLOR_OPPONENTS = [
            (50, 150, 255), (50, 255, 150), (255, 255, 50),
            (255, 150, 50), (150, 50, 255)
        ]
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_DRIFT = (100, 150, 255, 150)
        self.COLOR_BOOST = (255, 200, 50, 200)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.winner = False
        self.timer = 0.0
        self.player = None
        self.opponents = []
        self.all_karts = []
        self.particles = []
        self.track_center = []
        self.track_poly = []
        self.checkpoints = []
        self.camera_pos = pygame.math.Vector2(0, 0)
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.winner = False
        self.timer = self.TIME_LIMIT
        self.particles = []

        self._generate_track()
        self._initialize_karts()

        self.camera_pos = self.player["pos"].copy()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Calculate pre-update state for reward
        prev_rank = self._get_player_rank()

        # Update game logic
        self._update_player(movement, space_held, shift_held)
        self._update_opponents()
        self._update_particles()
        self._update_game_state()

        # Calculate reward
        reward = self._calculate_reward(prev_rank)
        self.score += reward

        # Check for termination
        terminated, terminal_reward = self._check_termination()
        self.score += terminal_reward
        if terminated:
            self.game_over = True
            reward += terminal_reward

        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self._update_camera()

        self.screen.fill(self.COLOR_GRASS)
        self._render_track()
        self._render_scenery()
        
        self.all_karts.sort(key=lambda k: self._project_iso(k["pos"])[1])
        
        for p in self.particles:
            self._render_particle(p)
            
        for kart in self.all_karts:
            self._render_kart(kart)
            
        self._render_ui()

        if self.game_over:
            self._render_end_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.player["lap"],
            "timer": self.timer,
            "rank": self._get_player_rank(),
        }

    # --- Game Logic ---

    def _generate_track(self):
        num_points = 12
        center = pygame.math.Vector2(0, 0)
        min_radius, max_radius = 800, 1200
        
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            radius = self.rng.uniform(min_radius, max_radius)
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            points.append(pygame.math.Vector2(x, y))

        self.track_center = self._catmull_rom_chain(points + points[:3], 20)
        self.checkpoints = list(range(0, len(self.track_center), len(self.track_center) // 10))
        
        track_width = 150
        rumble_width = 10
        self.track_poly = self._create_track_poly(self.track_center, track_width)
        self.rumble_poly_inner = self._create_track_poly(self.track_center, track_width - rumble_width)
        self.rumble_poly_outer = self._create_track_poly(self.track_center, track_width + rumble_width)

    def _initialize_karts(self):
        start_pos = self.track_center[0]
        start_dir = (self.track_center[1] - self.track_center[0]).normalize()
        side_dir = start_dir.rotate(90)

        self.player = self._create_kart(
            pos=start_pos,
            angle=start_dir.angle_to(pygame.math.Vector2(1, 0)),
            color=self.COLOR_PLAYER,
            is_player=True
        )

        self.opponents = []
        positions = [-2, -1, 1, 2, 3] # Relative positions around player
        for i in range(self.NUM_OPPONENTS):
            offset = side_dir * positions[i] * 40 - start_dir * (abs(positions[i]) * 40)
            self.opponents.append(
                self._create_kart(
                    pos=start_pos + offset,
                    angle=start_dir.angle_to(pygame.math.Vector2(1, 0)),
                    color=self.COLOR_OPPONENTS[i]
                )
            )
        
        self.all_karts = [self.player] + self.opponents

    def _create_kart(self, pos, angle, color, is_player=False):
        return {
            "pos": pos, "vel": pygame.math.Vector2(0, 0), "angle": angle,
            "color": color, "is_player": is_player, "lap": 0,
            "next_checkpoint": 1, "is_drifting": False, "drift_direction": 0,
            "boost_charge": 0.0, "is_boosting": False, "off_track_timer": 0,
            "rank_distance": 0.0, "steer": 0,
            "ai_offset": self.rng.uniform(-20, 20) if not is_player else 0
        }

    def _update_player(self, movement, space_held, shift_held):
        kart = self.player
        
        # --- Steering ---
        steer_input = 0
        if movement == 3: steer_input = 1   # Left
        if movement == 4: steer_input = -1  # Right
        kart["steer"] = steer_input * 3.5

        # --- Drifting ---
        was_drifting = kart["is_drifting"]
        if space_held and abs(steer_input) > 0 and kart["vel"].length() > 2:
            if not was_drifting:
                kart["is_drifting"] = True
                kart["drift_direction"] = steer_input
                # Sound: *Tire screech start*
        else:
            kart["is_drifting"] = False

        # --- Acceleration ---
        accel_input = 0
        if movement == 1: accel_input = 1   # Up
        if movement == 2: accel_input = -0.5 # Down

        # --- Boosting ---
        if shift_held and kart["boost_charge"] > 0 and not kart["is_boosting"]:
            kart["is_boosting"] = True
            # Sound: *Boost activate*
        
        if kart["is_boosting"]:
            kart["boost_charge"] -= 3.0
            if kart["boost_charge"] <= 0:
                kart["is_boosting"] = False
                kart["boost_charge"] = 0
            boost_force = 1.5
        else:
            boost_force = 0

        # --- Physics ---
        max_speed = 8.0
        accel_rate = 0.3
        friction = 0.96
        drift_friction = 0.98
        drift_turn_factor = 1.5
        
        if kart["is_drifting"]:
            kart["angle"] += kart["steer"] * drift_turn_factor
            kart["boost_charge"] = min(100, kart["boost_charge"] + 1.5)
            # Spawn drift particles
            if self.steps % 2 == 0:
                self._spawn_particles(kart, 2, self.COLOR_DRIFT, 20, 2, 5)
        else:
            kart["angle"] += kart["steer"] * (1 - min(0.8, kart["vel"].length() / max_speed))

        # Apply forces
        force = pygame.math.Vector2(1, 0).rotate(-kart["angle"]) * (accel_input * accel_rate + boost_force)
        kart["vel"] += force
        
        # Apply friction
        if kart["is_drifting"]:
            kart["vel"] *= drift_friction
        else:
            kart["vel"] *= friction
        
        # Cap speed
        if kart["vel"].length() > max_speed + (5 if kart["is_boosting"] else 0):
            kart["vel"].scale_to_length(max_speed + (5 if kart["is_boosting"] else 0))

        kart["pos"] += kart["vel"]

        # Spawn boost particles
        if kart["is_boosting"]:
            self._spawn_particles(kart, 3, self.COLOR_BOOST, 30, 3, 7)

    def _update_opponents(self):
        for kart in self.opponents:
            target_point_index = kart["next_checkpoint"] % len(self.track_center)
            target_pos = self.track_center[target_point_index]
            
            # Add lateral offset for variety
            target_dir = (self.track_center[(target_point_index + 1) % len(self.track_center)] - target_pos).normalize()
            target_pos += target_dir.rotate(90) * kart["ai_offset"]

            dir_to_target = (target_pos - kart["pos"])
            
            if dir_to_target.length() < 100:
                kart["next_checkpoint"] = (kart["next_checkpoint"] + 1) % len(self.checkpoints)

            # --- AI Steering & Speed ---
            desired_angle = -dir_to_target.angle_to(pygame.math.Vector2(1, 0))
            angle_diff = (desired_angle - kart["angle"] + 180) % 360 - 180
            kart["angle"] += np.clip(angle_diff, -5, 5)

            # Slow down on turns
            look_ahead_index = (target_point_index + 5) % len(self.track_center)
            p1 = self.track_center[target_point_index]
            p2 = self.track_center[(target_point_index + 2) % len(self.track_center)]
            p3 = self.track_center[look_ahead_index]
            v1 = p2 - p1
            v2 = p3 - p2
            turn_severity = v1.angle_to(v2)
            
            target_speed = 6.0 - abs(turn_severity) * 0.05
            
            # Apply physics
            current_speed = kart["vel"].length()
            if current_speed < target_speed:
                kart["vel"] += pygame.math.Vector2(1,0).rotate(-kart["angle"]) * 0.2
            
            kart["vel"] *= 0.97
            kart["pos"] += kart["vel"]

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] -= p["decay"]
    
    def _update_game_state(self):
        self.timer = max(0, self.timer - 1.0 / self.FPS)

        for kart in self.all_karts:
            # Checkpoint logic
            next_cp_index = self.checkpoints[kart["next_checkpoint"] % len(self.checkpoints)]
            dist_to_cp = kart["pos"].distance_to(self.track_center[next_cp_index])
            if dist_to_cp < 100:
                if kart["next_checkpoint"] == 0: # Passed start/finish line
                    kart["lap"] += 1
                kart["next_checkpoint"] = (kart["next_checkpoint"] + 1) % len(self.checkpoints)
            
            # Off-track logic
            if not self._is_on_track(kart["pos"]):
                kart["off_track_timer"] += 1
                kart["vel"] *= 0.8 # Slow down
                if kart["off_track_timer"] > 60: # 2 seconds
                    if kart["is_player"]:
                        self.player["off_track_incidents"] += 1
                        self.score -= 100 # Penalty
                    # Reset to last checkpoint
                    last_cp_index = self.checkpoints[(kart["next_checkpoint"] -1 + len(self.checkpoints)) % len(self.checkpoints)]
                    kart["pos"] = self.track_center[last_cp_index].copy()
                    kart["vel"] = pygame.math.Vector2(0,0)
                    kart["off_track_timer"] = 0
            else:
                kart["off_track_timer"] = 0

            # Update rank distance for sorting
            lap_dist = kart["lap"] * len(self.track_center)
            cp_dist = kart["next_checkpoint"]
            dist_to_next_cp = kart["pos"].distance_to(self.track_center[self.checkpoints[kart["next_checkpoint"] % len(self.checkpoints)]])
            kart["rank_distance"] = lap_dist + cp_dist - dist_to_next_cp * 0.001

    def _calculate_reward(self, prev_rank):
        reward = 0
        
        # On-track reward
        if self._is_on_track(self.player["pos"]):
            reward += 0.1

        # Drift initiation reward
        if self.player["is_drifting"] and not self.player.get("_was_drifting_prev_frame", False):
            reward += 0.5
        self.player["_was_drifting_prev_frame"] = self.player["is_drifting"]
        
        # Overtake reward
        current_rank = self._get_player_rank()
        if current_rank < prev_rank:
            reward += 1.0 * (prev_rank - current_rank)
            
        return reward

    def _check_termination(self):
        terminated = False
        terminal_reward = 0

        if self.player["lap"] >= self.NUM_LAPS:
            terminated = True
            self.winner = True
            terminal_reward = 100
        elif self.timer <= 0:
            terminated = True
            terminal_reward = -50
        elif self.player.get("off_track_incidents", 0) >= self.OFF_TRACK_LIMIT:
            terminated = True
            terminal_reward = -100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            terminal_reward = -50

        return terminated, terminal_reward

    # --- Rendering ---

    def _update_camera(self):
        lead_factor = 10
        camera_target = self.player["pos"] + self.player["vel"] * lead_factor
        self.camera_pos = self.camera_pos.lerp(camera_target, 0.1)

    def _project_iso(self, pos):
        screen_pos = pos - self.camera_pos + pygame.math.Vector2(self.W / 2, self.H / 2)
        iso_x = screen_pos.x - screen_pos.y
        iso_y = (screen_pos.x + screen_pos.y) * 0.5
        return pygame.math.Vector2(iso_x, iso_y)

    def _render_track(self):
        to_screen = lambda p: self._project_iso(p)
        
        # Draw grass -> rumble strip -> track
        pygame.gfxdraw.filled_polygon(self.screen, [to_screen(p) for p in self.rumble_poly_outer], self.COLOR_RUMBLE)
        pygame.gfxdraw.filled_polygon(self.screen, [to_screen(p) for p in self.track_poly], self.COLOR_TRACK)
        pygame.gfxdraw.filled_polygon(self.screen, [to_screen(p) for p in self.rumble_poly_inner], self.COLOR_RUMBLE)
        pygame.gfxdraw.filled_polygon(self.screen, [to_screen(p) for p in self.track_poly], self.COLOR_TRACK)

        # Draw start/finish line
        p1 = self.track_center[0]
        p2 = self.track_center[1]
        p_mid = (p1 + p2) / 2
        direction = (p2 - p1).normalize()
        perp = direction.rotate(90)
        line_width = 80
        for i in range(-4, 5):
            color = (255, 255, 255) if i % 2 == 0 else self.COLOR_TRACK
            start = p_mid + perp * (i * line_width / 8)
            end = start + direction * 10
            pygame.draw.line(self.screen, color, to_screen(start), to_screen(end), 3)

    def _render_scenery(self):
        # Simple moving grid for speed effect
        for i in range(-10, 11):
            start_world = pygame.math.Vector2(i * 200, -2000)
            end_world = pygame.math.Vector2(i * 200, 2000)
            start_screen = self._project_iso(start_world)
            end_screen = self._project_iso(end_world)
            pygame.draw.line(self.screen, self.COLOR_BG, start_screen, end_screen, 1)

            start_world = pygame.math.Vector2(-2000, i * 200)
            end_world = pygame.math.Vector2(2000, i * 200)
            start_screen = self._project_iso(start_world)
            end_screen = self._project_iso(end_world)
            pygame.draw.line(self.screen, self.COLOR_BG, start_screen, end_screen, 1)

    def _render_kart(self, kart):
        iso_pos = self._project_iso(kart["pos"])
        
        # Kart body
        w, h = 20, 40
        body_points = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/4), (0, h/2), (-w/2, h/4)]
        
        # Rotate points
        angle_rad = math.radians(kart["angle"] - 90)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rotated_points = []
        for x, y in body_points:
            rot_x = x * cos_a - y * sin_a
            rot_y = x * sin_a + y * cos_a
            rotated_points.append((iso_pos.x + rot_x, iso_pos.y + rot_y * 0.5))

        # Draw shadow
        shadow_points = [(p[0], p[1] + 5) for p in rotated_points]
        pygame.gfxdraw.filled_polygon(self.screen, shadow_points, (0, 0, 0, 100))

        # Draw kart
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, kart["color"])
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, kart["color"])

        # Player glow
        if kart["is_player"]:
            pygame.gfxdraw.filled_circle(self.screen, int(iso_pos.x), int(iso_pos.y), 25, self.COLOR_PLAYER_GLOW)
    
    def _render_particle(self, p):
        iso_pos = self._project_iso(p["pos"])
        if p["radius"] > 1:
            alpha = p["color"][3] * (p["life"] / p["max_life"])
            color = (*p["color"][:3], int(alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(iso_pos.x), int(iso_pos.y), int(p["radius"]), color)

    def _render_ui(self):
        # Lap counter
        lap_text = f"LAP: {min(self.player['lap'] + 1, self.NUM_LAPS)}/{self.NUM_LAPS}"
        self._draw_text(lap_text, (10, 10), self.font_medium)
        
        # Timer
        timer_text = f"TIME: {self.timer:.1f}"
        self._draw_text(timer_text, (self.W - 150, 10), self.font_medium)
        
        # Rank
        rank = self._get_player_rank()
        rank_text = f"POS: {rank}/{len(self.all_karts)}"
        self._draw_text(rank_text, (10, 40), self.font_medium)
        
        # Boost meter
        boost_rect = pygame.Rect(self.W // 2 - 100, self.H - 30, 200, 20)
        pygame.draw.rect(self.screen, (80, 80, 80), boost_rect)
        fill_width = (self.player["boost_charge"] / 100) * 200
        fill_rect = pygame.Rect(self.W // 2 - 100, self.H - 30, fill_width, 20)
        pygame.draw.rect(self.screen, self.COLOR_BOOST[:3], fill_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, boost_rect, 2)
        self._draw_text("BOOST", (self.W // 2 - 28, self.H - 32), self.font_small)

    def _render_end_screen(self):
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        end_text = "RACE FINISHED!" if self.winner else "GAME OVER"
        text_surf = self.font_large.render(end_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.W / 2, self.H / 2 - 20))
        self.screen.blit(text_surf, text_rect)

        score_text = f"Final Score: {self.score:.0f}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(center=(self.W / 2, self.H / 2 + 30))
        self.screen.blit(score_surf, score_rect)

    # --- Helpers ---
    
    def _catmull_rom_chain(self, points, num_between):
        chain = []
        for i in range(len(points) - 3):
            p0, p1, p2, p3 = points[i], points[i+1], points[i+2], points[i+3]
            for t in np.linspace(0, 1, num_between, endpoint=False):
                x = 0.5 * ((2 * p1.x) + (-p0.x + p2.x) * t + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t**2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t**3)
                y = 0.5 * ((2 * p1.y) + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t**2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t**3)
                chain.append(pygame.math.Vector2(x, y))
        return chain

    def _create_track_poly(self, centerline, width):
        path1, path2 = [], []
        for i in range(len(centerline)):
            p1 = centerline[i]
            p2 = centerline[(i + 1) % len(centerline)]
            direction = (p2 - p1).normalize()
            perp = direction.rotate(90)
            path1.append(p1 + perp * width / 2)
            path2.append(p1 - perp * width / 2)
        return path1 + path2[::-1]

    def _is_on_track(self, pos):
        # Simple but fast bounding box check first
        min_x = min(p.x for p in self.track_poly)
        max_x = max(p.x for p in self.track_poly)
        min_y = min(p.y for p in self.track_poly)
        max_y = max(p.y for p in self.track_poly)
        if not (min_x < pos.x < max_x and min_y < pos.y < max_y):
            return False
        
        # Ray casting algorithm for point in polygon
        n = len(self.track_poly)
        inside = False
        p1x, p1y = self.track_poly[0]
        for i in range(n + 1):
            p2x, p2y = self.track_poly[i % n]
            if pos.y > min(p1y, p2y):
                if pos.y <= max(p1y, p2y):
                    if pos.x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (pos.y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or pos.x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _spawn_particles(self, kart, num, color, life, min_speed, max_speed):
        for _ in range(num):
            vel_angle = -kart["angle"] + 180 + self.rng.uniform(-30, 30)
            vel_mag = self.rng.uniform(min_speed, max_speed)
            vel = pygame.math.Vector2(1, 0).rotate(vel_angle) * vel_mag
            self.particles.append({
                "pos": kart["pos"].copy(),
                "vel": vel,
                "life": self.rng.integers(life // 2, life),
                "max_life": life,
                "radius": self.rng.uniform(3, 6),
                "decay": 0.1,
                "color": color
            })

    def _get_player_rank(self):
        sorted_karts = sorted(self.all_karts, key=lambda k: k["rank_distance"], reverse=True)
        try:
            return sorted_karts.index(self.player) + 1
        except ValueError:
            return len(self.all_karts)

    def _draw_text(self, text, pos, font, color=None):
        color = color or self.COLOR_TEXT
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    game_over_screen_timer = 120 # Show end screen for 4 seconds

    while running:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # The action space is MultiDiscrete, so we need to handle left/right + up/down
        # For human play, we prioritize steering over acceleration if both are pressed
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the game
        # Pygame screen is transposed, so we need to fix it for display
        screen_for_display = pygame.display.set_mode((env.W, env.H))
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_for_display.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            if game_over_screen_timer > 0:
                game_over_screen_timer -= 1
            else:
                running = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                game_over_screen_timer = 120

        env.clock.tick(env.FPS)
        
    env.close()