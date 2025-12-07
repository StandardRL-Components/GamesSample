
# Generated: 2025-08-27T18:22:42.828915
# Source Brief: brief_01812.md
# Brief Index: 1812

        
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
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 3600 # 2 minutes at 30fps
    MAX_CRASHES = 5
    LAPS_TO_WIN = 3

    # Colors
    COLOR_BG = (50, 50, 60)
    COLOR_TRACK = (100, 100, 110)
    COLOR_BOUNDARY = (180, 180, 190)
    COLOR_FINISH_LIGHT = (255, 255, 255)
    COLOR_FINISH_DARK = (200, 200, 200)

    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 64)
    COLOR_OPPONENTS = [(255, 80, 80), (80, 150, 255), (255, 255, 80)]

    COLOR_TEXT = (240, 240, 240)
    COLOR_CRASH = (255, 0, 0, 150)
    
    # Physics
    ACCELERATION = 0.15
    BRAKING = 0.3
    MAX_SPEED = 5.0
    FRICTION = 0.98
    TURN_SPEED = 2.5
    
    BOOST_AMOUNT = 5.0
    BOOST_DURATION = 30 # steps
    BOOST_COOLDOWN = 150 # steps
    
    DRIFT_FRICTION = 0.95
    DRIFT_TURN_MOD = 1.5
    DRIFT_SLIDE_MOD = 0.8

    # Track
    TRACK_WIDTH = 80
    NUM_CHECKPOINTS = 16

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
        
        self.font_small = pygame.font.SysFont("sans", 18)
        self.font_large = pygame.font.SysFont("sans", 32)
        
        self._init_track()

        self.player = {}
        self.opponents = []
        self.particles = []
        self.finish_line_offset = 0
        self.crash_flash_timer = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.race_rankings = []
        self.lap_times = []
        self.current_lap_start_time = 0

        self.reset()
        self.validate_implementation()

    def _init_track(self):
        self.track_centerline = []
        track_radius_x, track_radius_y = 300, 150
        center_x, center_y = 0, 0
        
        for i in range(self.NUM_CHECKPOINTS):
            angle = (i / self.NUM_CHECKPOINTS) * 2 * math.pi
            # Create a slightly squashed oval shape
            x = center_x + track_radius_x * math.cos(angle)
            y = center_y + track_radius_y * math.sin(angle) * 1.5
            self.track_centerline.append(pygame.math.Vector2(x, y))

        self.track_polygons = []
        for i in range(len(self.track_centerline)):
            p1 = self.track_centerline[i]
            p2 = self.track_centerline[(i + 1) % len(self.track_centerline)]
            
            direction = (p2 - p1).normalize()
            perp = pygame.math.Vector2(-direction.y, direction.x)
            
            o1 = p1 + perp * self.TRACK_WIDTH / 2
            o2 = p2 + perp * self.TRACK_WIDTH / 2
            i1 = p1 - perp * self.TRACK_WIDTH / 2
            i2 = p2 - perp * self.TRACK_WIDTH / 2
            self.track_polygons.append((o1, o2, i2, i1))

    def _world_to_screen(self, pos, camera_pos):
        iso_x = pos.x - pos.y
        iso_y = (pos.x + pos.y) * 0.5
        
        cam_iso_x = camera_pos.x - camera_pos.y
        cam_iso_y = (camera_pos.x + camera_pos.y) * 0.5
        
        screen_x = self.SCREEN_WIDTH / 2 + (iso_x - cam_iso_x)
        screen_y = self.SCREEN_HEIGHT / 2 + (iso_y - cam_iso_y)
        return pygame.math.Vector2(int(screen_x), int(screen_y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        start_pos = self.track_centerline[0] + pygame.math.Vector2(0, -20)
        self.player = {
            "pos": start_pos.copy(), "angle": 0.0, "speed": 0.0, "vel": pygame.math.Vector2(0,0),
            "laps": 0, "crashes": 0, "next_checkpoint": 1, "is_drifting": False,
            "boost_timer": 0, "boost_cooldown": 0, "rank": None, "finish_time": float('inf'),
            "name": "Player", "color": self.COLOR_PLAYER
        }
        
        self.opponents = []
        for i in range(3):
            offset = (i - 1) * self.TRACK_WIDTH / 3
            self.opponents.append({
                "pos": start_pos + pygame.math.Vector2(offset, 0), "angle": 0.0, "speed": 3.0 + self.np_random.uniform(-0.2, 0.2), 
                "vel": pygame.math.Vector2(0,0), "laps": 0, "next_checkpoint": 1, "target_speed": 3.0,
                "rank": None, "finish_time": float('inf'), "name": f"AI {i+1}", "color": self.COLOR_OPPONENTS[i]
            })

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.race_rankings = []
        self.lap_times = []
        self.current_lap_start_time = 0
        self.crash_flash_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        self.steps += 1
        reward = 0

        self._handle_player_input(action)
        self._update_karts()
        self._update_particles()
        
        reward += self._calculate_reward()
        terminated = self._check_termination()

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Turning
        if movement == 3: # Left
            turn_rate = self.TURN_SPEED * (self.DRIFT_TURN_MOD if shift_held else 1.0)
            self.player["angle"] += turn_rate
        if movement == 4: # Right
            turn_rate = self.TURN_SPEED * (self.DRIFT_TURN_MOD if shift_held else 1.0)
            self.player["angle"] -= turn_rate

        # Acceleration/Braking
        if movement == 1: # Up
            self.player["speed"] += self.ACCELERATION
            # sfx: engine_accelerate
        elif movement == 2: # Down
            self.player["speed"] -= self.BRAKING
            # sfx: brake_screech
        
        self.player["speed"] = max(-self.MAX_SPEED/2, min(self.MAX_SPEED, self.player["speed"]))

        # Boost
        if space_held and self.player["boost_cooldown"] == 0:
            self.player["boost_timer"] = self.BOOST_DURATION
            self.player["boost_cooldown"] = self.BOOST_COOLDOWN
            # sfx: boost_activate
        
        # Drifting
        self.player["is_drifting"] = shift_held and self.player["speed"] > 2.0

    def _update_karts(self):
        all_karts = [self.player] + self.opponents

        # Update Player
        p = self.player
        if p["boost_timer"] > 0:
            p["speed"] = min(p["speed"] + self.BOOST_AMOUNT, self.MAX_SPEED * 1.5)
            p["boost_timer"] -= 1
            if p["boost_timer"] % 2 == 0:
                self._add_particles(p["pos"], p["angle"], 5, (255, 255, 255), 1.5)
        
        if p["boost_cooldown"] > 0:
            p["boost_cooldown"] -= 1

        # Apply friction
        friction = self.DRIFT_FRICTION if p["is_drifting"] else self.FRICTION
        p["speed"] *= friction
        if abs(p["speed"]) < 0.05: p["speed"] = 0

        # Update velocity and position
        heading = pygame.math.Vector2(1, 0).rotate(-p["angle"])
        if p["is_drifting"]:
            # sfx: tire_skid
            p["vel"] = p["vel"].lerp(heading * p["speed"], 0.1)
            p["vel"] *= self.DRIFT_SLIDE_MOD
            if self.steps % 3 == 0:
                self._add_particles(p["pos"], p["angle"] + 180, 2, (150, 150, 150), 0.5)
        else:
            p["vel"].from_polar((p["speed"], -p["angle"]))
        
        p["pos"] += p["vel"]

        # Update Opponents
        for kart in self.opponents:
            if kart["rank"] is not None: continue

            # AI Navigation
            target_point = self.track_centerline[kart["next_checkpoint"]]
            direction_to_target = (target_point - kart["pos"])
            
            if direction_to_target.length() < self.TRACK_WIDTH * 1.5:
                 kart["target_speed"] = (3.0 + kart["laps"] * 0.2) + self.np_random.uniform(-0.1, 0.1)
            else: # Slow down on tight turns
                 kart["target_speed"] = (2.5 + kart["laps"] * 0.2)
            
            kart["speed"] = kart["speed"] * 0.95 + kart["target_speed"] * 0.05
            
            target_angle = direction_to_target.angle_to(pygame.math.Vector2(1, 0))
            angle_diff = (target_angle - kart["angle"] + 180) % 360 - 180
            kart["angle"] += np.clip(angle_diff, -self.TURN_SPEED*1.5, self.TURN_SPEED*1.5)

            kart["vel"].from_polar((kart["speed"], -kart["angle"]))
            kart["pos"] += kart["vel"]

        # Collision and Lap checks
        for kart in all_karts:
            if kart["rank"] is not None: continue
            self._check_wall_collision(kart)
            self._check_checkpoints(kart)

    def _check_wall_collision(self, kart):
        # Simplified collision: check distance to centerline
        min_dist_sq = float('inf')
        closest_segment = (None, None)
        
        for i in range(len(self.track_centerline)):
            p1 = self.track_centerline[i]
            p2 = self.track_centerline[(i + 1) % len(self.track_centerline)]
            dist_sq = self._point_segment_distance_sq(kart["pos"], p1, p2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq

        if min_dist_sq > (self.TRACK_WIDTH / 2) ** 2:
            kart["speed"] *= 0.1
            kart["vel"] *= -0.5 # Bounce
            if kart is self.player:
                self.player["crashes"] += 1
                self.crash_flash_timer = 10
                self.score -= 1 # Immediate penalty
                # sfx: crash
            return True
        return False
        
    def _point_segment_distance_sq(self, p, a, b):
        l2 = (a - b).length_squared()
        if l2 == 0.0: return (p - a).length_squared()
        t = max(0, min(1, (p - a).dot(b - a) / l2))
        projection = a + t * (b - a)
        return (p - projection).length_squared()

    def _check_checkpoints(self, kart):
        next_checkpoint_pos = self.track_centerline[kart["next_checkpoint"]]
        if (kart["pos"] - next_checkpoint_pos).length() < self.TRACK_WIDTH * 0.75:
            if kart["next_checkpoint"] == 0: # Crossed finish line
                kart["laps"] += 1
                if kart is self.player:
                    self.score += 5
                    lap_time = (self.steps - self.current_lap_start_time) / self.FPS
                    if lap_time > 1: self.lap_times.append(lap_time)
                    self.current_lap_start_time = self.steps
                    # sfx: lap_complete
                if kart["laps"] >= self.LAPS_TO_WIN and kart["rank"] is None:
                    kart["finish_time"] = self.steps
                    kart["rank"] = len(self.race_rankings) + 1
                    self.race_rankings.append(kart)
            
            kart["next_checkpoint"] = (kart["next_checkpoint"] + 1) % len(self.track_centerline)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _add_particles(self, pos, angle, count, color, speed_mult):
        for _ in range(count):
            p_angle = angle + self.np_random.uniform(-45, 45)
            p_speed = self.np_random.uniform(0.5, 1.5) * speed_mult
            p_vel = pygame.math.Vector2(1, 0).rotate(-p_angle) * p_speed
            self.particles.append({
                "pos": pos.copy(), "vel": p_vel,
                "life": self.np_random.integers(10, 20), "color": color
            })

    def _calculate_reward(self):
        reward = 0
        # Reward for forward velocity component along track direction
        current_checkpoint_idx = (self.player["next_checkpoint"] -1 + self.NUM_CHECKPOINTS) % self.NUM_CHECKPOINTS
        p1 = self.track_centerline[current_checkpoint_idx]
        p2 = self.track_centerline[self.player["next_checkpoint"]]
        track_direction = (p2 - p1).normalize()
        
        forward_velocity = self.player["vel"].dot(track_direction)
        reward += forward_velocity * 0.1
        
        return reward

    def _check_termination(self):
        if self.player["laps"] >= self.LAPS_TO_WIN:
            self.game_over = True
            # Calculate final bonus based on rank
            if self.player["rank"] == 1: self.score += 50
            elif self.player["rank"] == 2: self.score += 25
            elif self.player["rank"] == 3: self.score += 10
        if self.player["crashes"] >= self.MAX_CRASHES:
            self.game_over = True
            self.score = -100 # Hard penalty for crashing out
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_pos = self.player["pos"]
        
        # Draw track polygons
        for poly_points in self.track_polygons:
            screen_points = [self._world_to_screen(p, cam_pos) for p in poly_points]
            pygame.gfxdraw.aapolygon(self.screen, screen_points, self.COLOR_BOUNDARY)
            pygame.gfxdraw.filled_polygon(self.screen, screen_points, self.COLOR_TRACK)

        # Draw finish line
        self.finish_line_offset = (self.finish_line_offset + 1) % 20
        p1 = self.track_centerline[0]
        p2 = self.track_centerline[-1]
        direction = (p1 - p2).normalize()
        perp = pygame.math.Vector2(-direction.y, direction.x)
        for i in range(-self.TRACK_WIDTH // 10, self.TRACK_WIDTH // 10):
            start = p1 + perp * i * 5
            end = start + direction * 10
            color = self.COLOR_FINISH_LIGHT if (i + self.finish_line_offset//10) % 2 == 0 else self.COLOR_FINISH_DARK
            pygame.draw.line(self.screen, color, self._world_to_screen(start, cam_pos), self._world_to_screen(end, cam_pos), 3)

        # Draw particles
        for p in self.particles:
            screen_pos = self._world_to_screen(p["pos"], cam_pos)
            alpha = int(255 * (p["life"] / 20.0))
            color = p["color"] + (alpha,)
            pygame.gfxdraw.pixel(self.screen, int(screen_pos.x), int(screen_pos.y), color)

        # Draw karts
        all_karts = sorted(self.opponents + [self.player], key=lambda k: k["pos"].x + k["pos"].y)
        for kart in all_karts:
            self._render_kart(kart, cam_pos)
            
    def _render_kart(self, kart, cam_pos):
        screen_pos = self._world_to_screen(kart["pos"], cam_pos)
        
        w, h = 12, 22 # Kart dimensions
        points = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        
        angle_rad = math.radians(kart["angle"])
        rotated_points = []
        for x, y in points:
            rx = x * math.cos(angle_rad) - y * math.sin(angle_rad)
            ry = x * math.sin(angle_rad) + y * math.cos(angle_rad)
            rotated_points.append((screen_pos.x + rx, screen_pos.y + ry))

        if kart is self.player:
            # Glow effect
            glow_points = []
            for x, y in points:
                rx = x * 1.5 * math.cos(angle_rad) - y * 1.5 * math.sin(angle_rad)
                ry = x * 1.5 * math.sin(angle_rad) + y * 1.5 * math.cos(angle_rad)
                glow_points.append((screen_pos.x + rx, screen_pos.y + ry))
            pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, kart["color"])
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, kart["color"])
    
    def _render_ui(self):
        # Lap Counter
        lap_text = self.font_large.render(f"LAP: {min(self.player['laps']+1, self.LAPS_TO_WIN)}/{self.LAPS_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(lap_text, (10, 10))

        # Crash Counter
        crash_text = self.font_large.render(f"CRASHES: {self.player['crashes']}/{self.MAX_CRASHES}", True, self.COLOR_TEXT)
        self.screen.blit(crash_text, (self.SCREEN_WIDTH - crash_text.get_width() - 10, 10))
        
        # Lap Times
        for i, t in enumerate(self.lap_times[-3:]):
            time_text = self.font_small.render(f"Lap {i+1}: {t:.2f}s", True, self.COLOR_TEXT)
            self.screen.blit(time_text, (10, 50 + i * 20))

        # Current Lap Time
        if not self.game_over:
            current_time = (self.steps - self.current_lap_start_time) / self.FPS
            time_str = f"{current_time:.2f}"
            time_surf = self.font_large.render(time_str, True, self.COLOR_TEXT)
            self.screen.blit(time_surf, (self.SCREEN_WIDTH/2 - time_surf.get_width()/2, self.SCREEN_HEIGHT - 40))
        
        # Boost Meter
        boost_rect = pygame.Rect(self.SCREEN_WIDTH/2 - 50, 15, 100, 15)
        pygame.draw.rect(self.screen, (40,40,40), boost_rect)
        if self.player["boost_cooldown"] > 0:
            fill_ratio = 1 - (self.player["boost_cooldown"] / self.BOOST_COOLDOWN)
            fill_rect = pygame.Rect(boost_rect.x, boost_rect.y, boost_rect.width * fill_ratio, boost_rect.height)
            pygame.draw.rect(self.screen, (80, 120, 200), fill_rect)
        else:
            pygame.draw.rect(self.screen, (150, 200, 255), boost_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, boost_rect, 1)

        # Crash flash
        if self.crash_flash_timer > 0:
            self.crash_flash_timer -= 1
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = 150 * (self.crash_flash_timer / 10.0)
            flash_surface.fill((255, 0, 0, alpha))
            self.screen.blit(flash_surface, (0,0))
            
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0,0))
            
            if self.player["laps"] >= self.LAPS_TO_WIN:
                msg = f"FINISH! RANK: {self.player['rank']}"
            elif self.player["crashes"] >= self.MAX_CRASHES:
                msg = "CRASHED OUT!"
            else:
                msg = "TIME UP!"

            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.player["laps"],
            "crashes": self.player["crashes"],
            "rank": self.player["rank"],
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to test the environment
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # To display the game, we need a screen
    pygame.display.set_caption("Arcade Racer")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    total_reward = 0
    
    # Mapping keyboard keys to actions for human play
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    action = np.array([0, 0, 0]) # No-op, no space, no shift
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        action[0] = 0 # Default to no movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        # Handle simultaneous left/right with up/down (prioritize turning)
        if keys[pygame.K_LEFT]:
            action[0] = 3
        if keys[pygame.K_RIGHT]:
            action[0] = 4
        if keys[pygame.K_UP] and not (keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]):
             action[0] = 1
        if keys[pygame.K_DOWN] and not (keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]):
             action[0] = 2

        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print(f"Game Over! Final Score: {total_reward:.2f}, Info: {info}")
            # Wait a bit before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
            done = False
            total_reward = 0
            
    env.close()