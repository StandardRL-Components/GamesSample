
# Generated: 2025-08-28T04:43:16.845818
# Source Brief: brief_02403.md
# Brief Index: 2403

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to accelerate, ←→ to turn, ↓ to brake. Hold Shift to drift. Press Space to boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric arcade racer. Drift around corners to build up boost, then blast past your opponents. Complete 3 laps before time runs out!"
    )

    # Frames auto-advance at 30fps.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_SCALE = 1.5
        self.WORLD_WIDTH = int(self.WIDTH * self.WORLD_SCALE)
        self.WORLD_HEIGHT = int(self.HEIGHT * self.WORLD_SCALE)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_TRACK = (70, 80, 90)
        self.COLOR_WALL = (100, 110, 120)
        self.COLOR_FINISH_1 = (200, 200, 200)
        self.COLOR_FINISH_2 = (170, 20, 20)
        self.COLOR_PLAYER = (50, 220, 50)
        self.COLOR_PLAYER_GLOW = (150, 255, 150)
        self.COLOR_BOOST_PAD = (255, 220, 0)
        self.COLOR_BOOST_PAD_GLOW = (255, 255, 150)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_BOOST_BAR = (0, 150, 255)
        self.COLOR_DRIFT_SMOKE = (200, 200, 200)
        self.COLOR_BOOST_FLAME = (255, 100, 0)

        # Game constants
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        self.LAPS_TO_WIN = 3
        
        # Track definition
        self._define_track()

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.is_drifting = None
        self.drift_direction = None
        self.drift_power = None
        self.is_boosting = None
        self.boost_timer = None
        self.boost_meter = None
        self.camera_pos = None
        self.particles = None
        self.active_boost_pads = None
        self.current_checkpoint = None
        self.lap_count = None
        self.time_remaining = None
        self.last_progress = None
        self.steps = None
        self.score = None
        self.game_over_message = None

        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def _define_track(self):
        self.track_centerline = [
            (self.WORLD_WIDTH * 0.2, self.WORLD_HEIGHT * 0.5),
            (self.WORLD_WIDTH * 0.8, self.WORLD_HEIGHT * 0.5),
            (self.WORLD_WIDTH * 0.8, self.WORLD_HEIGHT * 0.8),
            (self.WORLD_WIDTH * 0.5, self.WORLD_HEIGHT * 0.8),
            (self.WORLD_WIDTH * 0.5, self.WORLD_HEIGHT * 0.2),
            (self.WORLD_WIDTH * 0.2, self.WORLD_HEIGHT * 0.2),
        ]
        self.track_width = 120
        self.walls = []
        self.checkpoints = []
        for i in range(len(self.track_centerline)):
            p1 = self.track_centerline[i]
            p2 = self.track_centerline[(i + 1) % len(self.track_centerline)]
            self.checkpoints.append((p1, p2))
        
        self.boost_pads_initial = [
            (self.WORLD_WIDTH * 0.7, self.WORLD_HEIGHT * 0.5),
            (self.WORLD_WIDTH * 0.5, self.WORLD_HEIGHT * 0.3),
            (self.WORLD_WIDTH * 0.3, self.WORLD_HEIGHT * 0.5),
            (self.WORLD_WIDTH * 0.65, self.WORLD_HEIGHT * 0.8),
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        start_pos = self.track_centerline[0]
        self.player_pos = pygame.Vector2(start_pos[0], start_pos[1])
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = 0
        self.is_drifting = False
        self.drift_direction = 0
        self.drift_power = 0
        self.is_boosting = False
        self.boost_timer = 0
        self.boost_meter = 0
        self.camera_pos = self.player_pos.copy()
        self.particles = []
        self.active_boost_pads = [pygame.Vector2(p) for p in self.boost_pads_initial]

        self.current_checkpoint = 0
        self.lap_count = 0
        self.time_remaining = self.MAX_STEPS
        self.last_progress = 0

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if not self.game_over:
            # Unpack factorized action
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

            # Update game logic
            self._handle_input(movement, space_held, shift_held)
            reward += self._update_physics()
            reward += self._handle_collections()
            reward += self._handle_checkpoints()

            self.time_remaining -= 1
        
        self._update_particles()

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over: # First frame of termination
            if self.lap_count >= self.LAPS_TO_WIN:
                reward += 100 # Win bonus
                self.game_over_message = "YOU WIN!"
            else:
                reward -= 100 # Loss penalty
                self.game_over_message = "TIME UP!"
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Boosting
        if space_held and self.boost_meter > 0 and not self.is_boosting:
            self.is_boosting = True
            self.boost_timer = 60  # 2 seconds
            # sfx: boost_start
        
        if self.is_boosting:
            self.boost_meter = max(0, self.boost_meter - 2)
            if self.boost_timer > 0:
                self.boost_timer -= 1
            else:
                self.is_boosting = False
            if self.boost_meter <= 0:
                self.is_boosting = False
                # sfx: boost_end

        # Turning
        turn_speed = 0.08
        if self.is_drifting:
            turn_speed *= 1.5

        if movement == 3:  # Left
            self.player_angle += turn_speed
        if movement == 4:  # Right
            self.player_angle -= turn_speed

        # Acceleration/Braking
        accel = pygame.Vector2(0, 0)
        if movement == 1:  # Up
            accel = pygame.Vector2(math.cos(self.player_angle), -math.sin(self.player_angle)) * 0.4
        if movement == 2:  # Down
            accel = pygame.Vector2(math.cos(self.player_angle), -math.sin(self.player_angle)) * -0.2

        if self.is_boosting:
            accel *= 3.0
            # sfx: boost_loop

        self.player_vel += accel

        # Drifting
        is_turning = movement in [3, 4]
        if shift_held and is_turning and self.player_vel.length() > 3:
            if not self.is_drifting:
                self.is_drifting = True
                self.drift_direction = 1 if movement == 3 else -1
                # sfx: drift_start
            self.drift_power = min(100, self.drift_power + 1)
        else:
            if self.is_drifting:
                self.is_drifting = False
                self.boost_meter = min(100, self.boost_meter + self.drift_power / 2)
                self.drift_power = 0
                # sfx: drift_end

    def _update_physics(self):
        # Drifting physics
        if self.is_drifting:
            # Reduce forward friction, increase sideways friction
            forward_vec = pygame.Vector2(math.cos(self.player_angle), -math.sin(self.player_angle))
            sideways_vec = forward_vec.rotate(90)
            
            forward_speed = self.player_vel.dot(forward_vec)
            sideways_speed = self.player_vel.dot(sideways_vec)

            self.player_vel = forward_vec * forward_speed * 0.98 + sideways_vec * sideways_speed * 0.9
            
            # Add drift smoke
            if self.steps % 3 == 0:
                for _ in range(2):
                    offset = (random.uniform(-10, 10), random.uniform(-10, 10))
                    p_pos = self.player_pos + offset
                    p_vel = pygame.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
                    self.particles.append({'pos': p_pos, 'vel': p_vel, 'life': 20, 'color': self.COLOR_DRIFT_SMOKE, 'radius': random.randint(3, 6)})
        else:
            # Normal friction
            self.player_vel *= 0.95
        
        # Speed limit
        max_speed = 6 if not self.is_boosting else 12
        if self.player_vel.length() > max_speed:
            self.player_vel.scale_to_length(max_speed)

        # Update position
        self.player_pos += self.player_vel
        
        # Wall collisions
        self._handle_wall_collisions()

        # Calculate forward progress reward
        current_segment_vec = pygame.Vector2(self.checkpoints[self.current_checkpoint][1]) - pygame.Vector2(self.checkpoints[self.current_checkpoint][0])
        if current_segment_vec.length() > 0:
            track_direction = current_segment_vec.normalize()
            progress_reward = self.player_vel.dot(track_direction) * 0.05
            return progress_reward
        return 0

    def _handle_wall_collisions(self):
        # Simplified wall collision based on distance from centerline
        p1 = pygame.Vector2(self.track_centerline[self.current_checkpoint])
        p2 = pygame.Vector2(self.track_centerline[(self.current_checkpoint + 1) % len(self.track_centerline)])
        
        # Project player pos onto the line segment
        line_vec = p2 - p1
        if line_vec.length_squared() == 0: return # Avoid division by zero
        
        t = max(0, min(1, (self.player_pos - p1).dot(line_vec) / line_vec.length_squared()))
        closest_point = p1 + t * line_vec
        
        dist_to_center = self.player_pos.distance_to(closest_point)
        
        if dist_to_center > self.track_width / 2:
            # Collision occurred
            # sfx: wall_hit
            normal = (self.player_pos - closest_point).normalize()
            self.player_pos = closest_point + normal * (self.track_width / 2)
            self.player_vel = self.player_vel.reflect(normal) * 0.5 # Lose speed on hit

    def _handle_collections(self):
        reward = 0
        pads_to_remove = []
        for pad in self.active_boost_pads:
            if self.player_pos.distance_to(pad) < 20:
                self.boost_meter = min(100, self.boost_meter + 50)
                pads_to_remove.append(pad)
                reward += 1
                # sfx: collect_boost
        self.active_boost_pads = [p for p in self.active_boost_pads if p not in pads_to_remove]
        return reward

    def _handle_checkpoints(self):
        reward = 0
        next_checkpoint_idx = (self.current_checkpoint + 1) % len(self.checkpoints)
        p1, p2 = self.checkpoints[next_checkpoint_idx]
        p1, p2 = pygame.Vector2(p1), pygame.Vector2(p2)
        
        # Check if player crossed the checkpoint line
        # Using a simple distance check for robustness
        if self.player_pos.distance_to((p1 + p2) / 2) < self.track_width:
            prev_pos = self.player_pos - self.player_vel
            
            # Simple line segment intersection check
            v1 = p2 - p1
            v2 = self.player_pos - prev_pos
            v3 = prev_pos - p1
            
            cross_product = v1.x * v2.y - v1.y * v2.x
            if abs(cross_product) > 1e-6:
                t1 = (v3.x * v2.y - v3.y * v2.x) / cross_product
                t2 = (v3.x * v1.y - v3.y * v1.x) / cross_product
                if 0 <= t1 <= 1 and 0 <= t2 <= 1:
                    self.current_checkpoint = next_checkpoint_idx
                    if self.current_checkpoint == 0: # Crossed finish line
                        self.lap_count += 1
                        reward += 5
                        self.active_boost_pads = [pygame.Vector2(p) for p in self.boost_pads_initial]
                        # sfx: lap_complete
                    else:
                        # sfx: checkpoint
                        pass
        return reward

    def _check_termination(self):
        return self.lap_count >= self.LAPS_TO_WIN or self.time_remaining <= 0 or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.lap_count,
            "time_remaining": self.time_remaining / self.FPS,
        }

    def _update_particles(self):
        if self.is_boosting:
            # sfx: boost_flame
            for _ in range(3):
                angle = self.player_angle + random.uniform(-0.2, 0.2)
                vel = pygame.Vector2(math.cos(angle), -math.sin(angle)) * -self.player_vel.length() * 0.5
                pos = self.player_pos + vel.normalize() * -15
                self.particles.append({'pos': pos, 'vel': vel, 'life': 15, 'color': self.COLOR_BOOST_FLAME, 'radius': random.randint(5, 10)})
        
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.5)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _world_to_iso(self, x, y):
        iso_x = (x - y) * 0.707
        iso_y = (x + y) * 0.4
        return iso_x, iso_y

    def _update_camera(self):
        self.camera_pos.move_towards_ip(self.player_pos, self.player_vel.length() + 5)

    def _get_observation(self):
        self._update_camera()
        
        cam_x, cam_y = self.camera_pos.x, self.camera_pos.y
        offset_x = self.WIDTH / 2 - cam_x
        offset_y = self.HEIGHT / 2 - cam_y

        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_track(offset_x, offset_y)
        self._render_boost_pads(offset_x, offset_y)
        self._render_particles(offset_x, offset_y)
        self._render_player(offset_x, offset_y)
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_track(self, ox, oy):
        # Draw track surface
        track_poly = []
        for p in self.track_centerline:
            iso_x, iso_y = self._world_to_iso(p[0], p[1])
            track_poly.append((iso_x + ox, iso_y + oy))
        
        # Draw track segments
        for i in range(len(self.track_centerline)):
            p1_w = self.track_centerline[i]
            p2_w = self.track_centerline[(i + 1) % len(self.track_centerline)]
            
            p1_iso = self._world_to_iso(*p1_w)
            p2_iso = self._world_to_iso(*p2_w)

            pygame.draw.line(self.screen, self.COLOR_TRACK, (p1_iso[0] + ox, p1_iso[1] + oy), (p2_iso[0] + ox, p2_iso[1] + oy), int(self.track_width * 0.8))
        
        # Draw finish line
        p1_w, p2_w = self.checkpoints[0]
        p1_iso = self._world_to_iso(*p1_w)
        p2_iso = self._world_to_iso(*p2_w)
        
        line_vec = pygame.Vector2(p2_iso) - pygame.Vector2(p1_iso)
        perp_vec = line_vec.rotate(90).normalize()
        
        for i in range(10):
            start = pygame.Vector2(p1_iso) + line_vec * (i / 10.0)
            end = start + perp_vec * self.track_width * 0.4
            start -= perp_vec * self.track_width * 0.4
            color = self.COLOR_FINISH_1 if i % 2 == 0 else self.COLOR_FINISH_2
            pygame.draw.line(self.screen, color, (start.x + ox, start.y + oy), (end.x + ox, end.y + oy), int(self.track_width / 10))

    def _render_boost_pads(self, ox, oy):
        for pad_pos in self.active_boost_pads:
            iso_x, iso_y = self._world_to_iso(pad_pos.x, pad_pos.y)
            screen_pos = (int(iso_x + ox), int(iso_y + oy))
            radius = 12
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius + 3, (*self.COLOR_BOOST_PAD_GLOW, 100))
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, self.COLOR_BOOST_PAD)
            pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], radius, self.COLOR_BOOST_PAD)

    def _render_particles(self, ox, oy):
        for p in self.particles:
            iso_x, iso_y = self._world_to_iso(p['pos'].x, p['pos'].y)
            screen_pos = (int(iso_x + ox), int(iso_y + oy))
            alpha = int(255 * (p['life'] / 20.0))
            color = (*p['color'], alpha)
            if p['radius'] > 0:
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], int(p['radius']), color)

    def _render_player(self, ox, oy):
        iso_x, iso_y = self._world_to_iso(self.player_pos.x, self.player_pos.y)
        screen_pos = (int(iso_x + ox), int(iso_y + oy))
        size = 12

        # Create rotated points for the kart
        points = [(-size, -size/2), (size, -size/2), (size, size/2), (-size, size/2)]
        rotated_points = []
        for p in points:
            x_rot = p[0] * math.cos(-self.player_angle) - p[1] * math.sin(-self.player_angle)
            y_rot = p[0] * math.sin(-self.player_angle) + p[1] * math.cos(-self.player_angle)
            rotated_points.append((screen_pos[0] + x_rot, screen_pos[1] + y_rot))
        
        # Glow effect
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, (*self.COLOR_PLAYER_GLOW, 100))
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, (*self.COLOR_PLAYER_GLOW, 100))
        
        # Main body
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PLAYER)
        
    def _render_ui(self):
        # Lap counter
        lap_text = self.font_small.render(f"LAP: {min(self.lap_count + 1, self.LAPS_TO_WIN)}/{self.LAPS_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(lap_text, (10, 10))

        # Timer
        time_str = f"{self.time_remaining / self.FPS:.1f}"
        time_text = self.font_small.render(f"TIME: {time_str}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Boost meter
        bar_width, bar_height = 150, 15
        bar_x, bar_y = self.WIDTH - bar_width - 10, self.HEIGHT - bar_height - 10
        fill_width = int((self.boost_meter / 100) * bar_width)
        
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (bar_x, bar_y, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_BOOST_BAR, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Game Over Message
        if self.game_over:
            msg_surf = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import sys

    # Set SDL to dummy to run headless, or remove this line for visible window
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    
    # --- To run with a visible window ---
    pygame.display.set_caption("Arcade Racer")
    real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    action = env.action_space.sample() # Start with a no-op
    action[0] = 0
    action[1] = 0
    action[2] = 0

    while not done:
        # Human input
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Display the frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']:.2f}, Laps: {info['laps']}")
    
    env.close()
    pygame.quit()
    sys.exit()