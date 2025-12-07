
# Generated: 2025-08-27T19:41:53.277075
# Source Brief: brief_02228.md
# Brief Index: 2228

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG = (30, 35, 40)
        self.COLOR_TRACK = (80, 80, 90)
        self.COLOR_TRACK_BORDER = (120, 120, 130)
        self.COLOR_START_LINE = (220, 220, 220)
        self.COLOR_KART = (255, 50, 50)
        self.COLOR_KART_GLOW = (255, 100, 100, 50)
        self.COLOR_BOOST = (50, 150, 255)
        self.COLOR_DRIFT = (200, 200, 200)
        self.COLOR_OBSTACLE = (255, 150, 0)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_UI_BOOST_BG = (50, 50, 60)
        self.COLOR_UI_BOOST_FG = self.COLOR_BOOST

        # Game constants
        self.FPS = 30
        self.MAX_STEPS = 1500 # 50 seconds
        self.LAPS_TO_WIN = 3
        self.TRACK_WIDTH = 100
        self.TRACK_POINTS = 200
        
        # Kart physics
        self.ACCELERATION = 0.2
        self.BRAKING = 0.4
        self.FRICTION = 0.04
        self.MAX_SPEED = 5.0
        self.TURN_SPEED = 0.05
        self.DRIFT_TURN_MOD = 1.8
        self.DRIFT_FRICTION_MOD = 1.5
        self.BOOST_SPEED = 12.0
        self.BOOST_DRAIN = 4.0
        self.BOOST_RECHARGE = 0.5
        self.OFF_TRACK_PENALTY = 0.5
        self.COLLISION_SPEED_PENALTY = 0.8
        self.COLLISION_COOLDOWN = 15 # frames

        # Initialize state variables
        self.rng = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.kart_pos = None
        self.kart_angle = None
        self.kart_speed = None
        self.kart_velocity_angle = None
        self.is_drifting = False
        self.is_boosting = False
        self.boost_level = None
        self.collision_timer = 0
        self.track_centerline = []
        self.track_polygons = []
        self.current_segment_index = 0
        self.laps_completed = 0
        self.total_time = 0.0
        self.lap_times = []
        self.obstacles = []
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_time = 0.0
        self.lap_times = []
        self.laps_completed = 0
        
        self._generate_track()
        
        self.kart_pos = list(self.track_centerline[0])
        self.kart_angle = math.atan2(self.track_centerline[1][1] - self.kart_pos[1], self.track_centerline[1][0] - self.kart_pos[0])
        self.kart_speed = 0.0
        self.kart_velocity_angle = self.kart_angle
        self.is_drifting = False
        self.is_boosting = False
        self.boost_level = 100.0
        self.collision_timer = 0
        self.current_segment_index = 0
        
        self.particles.clear()
        self._spawn_obstacles()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            shift_held = action[2] == 1

            # Update game logic
            reward += self._update_kart_state(movement, space_held, shift_held)
            self._update_particles()
            
            self.total_time += 1.0 / self.FPS
            self.boost_level = min(100.0, self.boost_level + self.BOOST_RECHARGE)
            if self.collision_timer > 0:
                self.collision_timer -= 1

            lap_completed = self._check_lap_completion()
            if lap_completed:
                reward += 5.0
                self.laps_completed += 1
                self.lap_times.append(self.total_time - sum(self.lap_times))
                self._spawn_obstacles() # Add more obstacles
                # sfx: lap_complete_sound()

        self.steps += 1
        self.score += reward
        terminated = self._check_termination()
        if terminated and not self.game_over:
            # Final reward for finishing the race
            time_bonus = 100 * (self.MAX_STEPS - self.steps) / self.MAX_STEPS
            reward += max(0, time_bonus)
            self.score += max(0, time_bonus)
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_kart_state(self, movement, space_held, shift_held):
        reward = 0
        
        # Handle boost
        self.is_boosting = space_held and self.boost_level > 0
        if self.is_boosting:
            self.kart_speed = self.BOOST_SPEED
            self.boost_level -= self.BOOST_DRAIN
            reward += 0.02 # Small reward for boosting
            # sfx: boost_sound()
            if self.rng.random() < 0.8:
                self._create_particles(self.kart_pos, 5, self.COLOR_BOOST, 2, 0.5, self.kart_angle + math.pi, 0.5)

        # Handle turning and drifting
        self.is_drifting = shift_held and self.kart_speed > 2.0
        turn_mod = self.DRIFT_TURN_MOD if self.is_drifting else 1.0
        if movement == 3: # Left
            self.kart_angle -= self.TURN_SPEED * turn_mod
        if movement == 4: # Right
            self.kart_angle += self.TURN_SPEED * turn_mod

        if self.is_drifting:
            # sfx: drift_sound()
            if self.rng.random() < 0.5:
                self._create_particles(self.kart_pos, 2, self.COLOR_DRIFT, 3, 0.3, self.kart_velocity_angle + math.pi, 1.0)

        # Handle acceleration/braking
        if not self.is_boosting:
            if movement == 1: # Accelerate
                self.kart_speed += self.ACCELERATION
            elif movement == 2: # Brake
                self.kart_speed -= self.BRAKING
        
        # Apply friction
        drift_friction = self.DRIFT_FRICTION_MOD if self.is_drifting else 1.0
        self.kart_speed = max(0, self.kart_speed - self.FRICTION * drift_friction)
        self.kart_speed = min(self.MAX_SPEED, self.kart_speed)

        # Update velocity angle (drifting effect)
        angle_diff = (self.kart_angle - self.kart_velocity_angle + math.pi) % (2 * math.pi) - math.pi
        slip_factor = 0.1 if self.is_drifting else 0.5
        self.kart_velocity_angle += angle_diff * slip_factor

        # Update position
        self.kart_pos[0] += math.cos(self.kart_velocity_angle) * self.kart_speed
        self.kart_pos[1] += math.sin(self.kart_velocity_angle) * self.kart_speed

        # Forward movement reward
        if self.kart_speed > 0.1:
            reward += 0.01 * (self.kart_speed / self.MAX_SPEED)

        # Off-track check
        if self._is_off_track():
            self.kart_speed *= self.OFF_TRACK_PENALTY
            reward -= 0.1

        # Obstacle collision check
        collided_obstacle = self._check_obstacle_collision()
        if collided_obstacle and self.collision_timer == 0:
            reward -= 1.0
            self.kart_speed *= self.COLLISION_SPEED_PENALTY
            self.collision_timer = self.COLLISION_COOLDOWN
            # sfx: collision_sound()
            self._create_particles(collided_obstacle.center, 10, self.COLOR_OBSTACLE, 5, 1.0, self.kart_velocity_angle, 2.0)
            self.obstacles.remove(collided_obstacle)
        
        return reward

    def _check_lap_completion(self):
        passed_finish_line = False
        kart_segment_index = self._get_closest_segment_index(self.kart_pos)

        # Detect crossing from the last segment to the first
        if self.current_segment_index > len(self.track_centerline) * 0.8 and kart_segment_index < len(self.track_centerline) * 0.2:
            passed_finish_line = True
        
        self.current_segment_index = kart_segment_index
        return passed_finish_line

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.laps_completed,
            "lap_times": self.lap_times,
            "total_time": self.total_time,
            "boost_level": self.boost_level
        }

    def _check_termination(self):
        return self.laps_completed >= self.LAPS_TO_WIN or self.steps >= self.MAX_STEPS

    def _generate_track(self):
        self.track_centerline.clear()
        self.track_polygons.clear()
        
        # Generate a procedural, somewhat randomized oval/racetrack shape
        center_x, center_y = 0, 0
        main_radius_x = self.rng.uniform(600, 800)
        main_radius_y = self.rng.uniform(400, 500)
        wobble_freq = self.rng.integers(3, 7)
        wobble_amp = self.rng.uniform(50, 150)

        for i in range(self.TRACK_POINTS):
            angle = 2 * math.pi * i / self.TRACK_POINTS
            radius_offset = wobble_amp * math.sin(angle * wobble_freq)
            x = center_x + (main_radius_x + radius_offset) * math.cos(angle)
            y = center_y + (main_radius_y + radius_offset) * math.sin(angle)
            self.track_centerline.append((x, y))

        # Create renderable polygons for the track
        for i in range(self.TRACK_POINTS):
            p1 = self.track_centerline[i]
            p2 = self.track_centerline[(i + 1) % self.TRACK_POINTS]
            
            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            perp_angle = angle + math.pi / 2
            
            w = self.TRACK_WIDTH / 2
            p1_l = (p1[0] + w * math.cos(perp_angle), p1[1] + w * math.sin(perp_angle))
            p1_r = (p1[0] - w * math.cos(perp_angle), p1[1] - w * math.sin(perp_angle))
            p2_l = (p2[0] + w * math.cos(perp_angle), p2[1] + w * math.sin(perp_angle))
            p2_r = (p2[0] - w * math.cos(perp_angle), p2[1] - w * math.sin(perp_angle))
            
            self.track_polygons.append((p1_l, p2_l, p2_r, p1_r))

    def _spawn_obstacles(self):
        self.obstacles.clear()
        num_obstacles = 1 + self.laps_completed
        for _ in range(num_obstacles):
            segment_idx = self.rng.integers(10, len(self.track_centerline) - 10)
            point_on_centerline = self.track_centerline[segment_idx]
            
            offset_angle = self.rng.uniform(0, 2 * math.pi)
            offset_dist = self.rng.uniform(0, self.TRACK_WIDTH / 2 - 10)
            
            ox = point_on_centerline[0] + offset_dist * math.cos(offset_angle)
            oy = point_on_centerline[1] + offset_dist * math.sin(offset_angle)
            
            self.obstacles.append(pygame.Rect(ox - 5, oy - 5, 10, 10))

    def _get_closest_segment_index(self, pos):
        min_dist_sq = float('inf')
        closest_idx = 0
        for i, p in enumerate(self.track_centerline):
            dist_sq = (pos[0] - p[0])**2 + (pos[1] - p[1])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_idx = i
        return closest_idx

    def _is_off_track(self):
        idx = self._get_closest_segment_index(self.kart_pos)
        closest_point = self.track_centerline[idx]
        dist_sq = (self.kart_pos[0] - closest_point[0])**2 + (self.kart_pos[1] - closest_point[1])**2
        return dist_sq > (self.TRACK_WIDTH / 2)**2

    def _check_obstacle_collision(self):
        kart_rect = pygame.Rect(self.kart_pos[0] - 5, self.kart_pos[1] - 5, 10, 10)
        for obs in self.obstacles:
            if kart_rect.colliderect(obs):
                return obs
        return None

    def _create_particles(self, pos, count, color, size, life, angle, spread):
        for _ in range(count):
            p_angle = angle + self.rng.uniform(-spread, spread)
            p_speed = self.rng.uniform(1, 5)
            p_vel = [math.cos(p_angle) * p_speed, math.sin(p_angle) * p_speed]
            self.particles.append({'pos': list(pos), 'vel': p_vel, 'life': life * self.FPS, 'color': color, 'size': size})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] = max(0, p['size'] * 0.95)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _world_to_screen(self, x, y):
        # Camera is centered on the kart
        rel_x = x - self.kart_pos[0]
        rel_y = y - self.kart_pos[1]
        
        # Isometric projection
        iso_x = (rel_x - rel_y)
        iso_y = (rel_x + rel_y) * 0.5
        
        # Center on screen
        screen_x = self.WIDTH / 2 + iso_x
        screen_y = self.HEIGHT * 0.75 + iso_y # Place horizon higher
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Sort all renderable objects by their Y world coordinate for correct isometric layering
        render_queue = []

        # Add track polygons
        for i, poly in enumerate(self.track_polygons):
            avg_y = sum(p[1] for p in poly) / 4
            render_queue.append(('poly', avg_y, self.COLOR_TRACK, poly, 0))
            if i == 0: # Start/Finish line
                p1, p2, p3, p4 = poly
                render_queue.append(('line', avg_y, self.COLOR_START_LINE, p1, p4, 5))

        # Add obstacles
        for obs in self.obstacles:
            render_queue.append(('obstacle', obs.centery, obs))

        # Add particles
        for p in self.particles:
            render_queue.append(('particle', p['pos'][1], p))
        
        # Add kart
        render_queue.append(('kart', self.kart_pos[1]))

        # Sort and render
        render_queue.sort(key=lambda item: item[1])

        for item in render_queue:
            if item[0] == 'poly':
                _, _, color, points, width = item
                screen_points = [self._world_to_screen(p[0], p[1]) for p in points]
                pygame.gfxdraw.aapolygon(self.screen, screen_points, self.COLOR_TRACK_BORDER)
                pygame.gfxdraw.filled_polygon(self.screen, screen_points, color)
            elif item[0] == 'line':
                 _, _, color, p1, p2, width = item
                 sp1 = self._world_to_screen(p1[0], p1[1])
                 sp2 = self._world_to_screen(p2[0], p2[1])
                 pygame.draw.line(self.screen, color, sp1, sp2, width)
            elif item[0] == 'obstacle':
                _, _, obs = item
                sx, sy = self._world_to_screen(obs.centerx, obs.centery)
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, 8, self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.screen, sx, sy, 8, self.COLOR_OBSTACLE)
            elif item[0] == 'particle':
                _, _, p = item
                sx, sy = self._world_to_screen(p['pos'][0], p['pos'][1])
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, int(p['size']), p['color'])
            elif item[0] == 'kart':
                self._render_kart()

    def _render_kart(self):
        screen_pos = (int(self.WIDTH / 2), int(self.HEIGHT * 0.75))
        
        # Draw glow
        if self.is_boosting:
            glow_color = self.COLOR_BOOST
            glow_radius = 25
        else:
            glow_color = self.COLOR_KART_GLOW
            glow_radius = 20
        
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (screen_pos[0] - glow_radius, screen_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Kart body
        kart_size = 10
        angle = self.kart_angle
        points = [
            (kart_size, 0),
            (-kart_size * 0.5, -kart_size * 0.6),
            (-kart_size * 0.3, 0),
            (-kart_size * 0.5, kart_size * 0.6),
        ]
        
        rotated_points = []
        for x, y in points:
            rx = x * math.cos(angle) - y * math.sin(angle)
            ry = x * math.sin(angle) + y * math.cos(angle)
            rotated_points.append((screen_pos[0] + rx, screen_pos[1] + ry))
        
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_KART)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_KART)

    def _render_ui(self):
        # Laps
        lap_text = self.font_ui.render(f"LAP: {min(self.laps_completed + 1, self.LAPS_TO_WIN)}/{self.LAPS_TO_WIN}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lap_text, (10, 10))

        # Time
        minutes = int(self.total_time) // 60
        seconds = int(self.total_time) % 60
        millis = int((self.total_time * 100) % 100)
        time_text = self.font_ui.render(f"TIME: {minutes:02d}:{seconds:02d}:{millis:02d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        # Boost Gauge
        bar_width, bar_height = 200, 20
        bar_x, bar_y = self.WIDTH / 2 - bar_width / 2, self.HEIGHT - bar_height - 10
        pygame.draw.rect(self.screen, self.COLOR_UI_BOOST_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        fill_width = (self.boost_level / 100) * (bar_width - 4)
        pygame.draw.rect(self.screen, self.COLOR_UI_BOOST_FG, (bar_x + 2, bar_y + 2, fill_width, bar_height - 4), border_radius=3)

        # Game Over / Start Message
        if self.game_over:
            text = "FINISH"
            if self.steps >= self.MAX_STEPS:
                text = "TIME UP"
            msg = self.font_big.render(text, True, self.COLOR_UI_TEXT)
            self.screen.blit(msg, (self.WIDTH/2 - msg.get_width()/2, self.HEIGHT/2 - msg.get_height()/2))
        elif self.steps < self.FPS * 3:
            countdown = 3 - (self.steps // self.FPS)
            text = str(countdown) if countdown > 0 else "GO!"
            msg = self.font_big.render(text, True, self.COLOR_UI_TEXT)
            self.screen.blit(msg, (self.WIDTH/2 - msg.get_width()/2, self.HEIGHT/2 - msg.get_height()/2))

    def close(self):
        pygame.quit()

    def render(self):
        return self._get_observation()
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Kart Racer")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # --- Action selection for human play ---
        # Default action is no-op
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # --- Frame rate ---
        clock.tick(env.FPS)

    print(f"Game Over. Final Score: {info['score']:.2f}, Laps: {info['laps']}, Time: {info['total_time']:.2f}s")
    env.close()