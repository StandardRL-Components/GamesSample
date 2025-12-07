
# Generated: 2025-08-28T05:17:13.531577
# Source Brief: brief_05522.md
# Brief Index: 5522

        
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
        "Hold Shift to drift. Fill the boost meter by drifting, then press Space to boost!"
    )

    game_description = (
        "A fast-paced, top-down arcade racer. Drift through corners on a "
        "procedurally generated track to build up boost. Complete a lap in under "
        "30 seconds to win, but watch out for obstacles!"
    )

    auto_advance = True

    # --- Colors ---
    COLOR_BG = (20, 30, 25)
    COLOR_TRACK = (80, 100, 90)
    COLOR_GRASS = (50, 120, 60)
    COLOR_RUMBLE = (200, 200, 200)
    COLOR_FINISH_DARK = (40, 40, 40)
    COLOR_FINISH_LIGHT = (220, 220, 220)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 200, 255, 50)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (255, 50, 50, 100)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SUCCESS = (100, 255, 100)
    COLOR_UI_FAIL = (255, 100, 100)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        self.screen_center = (self.screen.get_width() // 2, self.screen.get_height() // 2)

        # Game state variables are initialized in reset()
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.terminated_reason = ""

        # Player state
        self.player_pos = np.array([0.0, 0.0])
        self.player_angle = -math.pi / 2
        self.player_speed = 0.0
        self.is_drifting = False
        self.drift_angle_offset = 0.0
        self.drift_sparks_cooldown = 0
        self.on_grass = False

        # Game mechanics
        self.lap_time = 0.0
        self.collisions = 0
        self.boost_meter = 0.0
        self.is_boosting = False
        self.boost_timer = 0
        self.last_space_press = False
        
        self.camera_shake = 0

        # Track generation
        self._generate_track()
        self.player_pos = self.track_points[0].copy() + np.array([0.0, -50.0])
        self.checkpoints_passed = 0
        self.distance_to_next_checkpoint = self._get_dist_to_checkpoint(0)

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_press, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.lap_time += 1 / 30.0
        self.steps += 1
        reward = -0.01  # Time penalty

        self._handle_input(movement, space_press, shift_held)
        self._update_physics()
        self._update_particles()
        
        # Order matters: check checkpoints before collisions to get correct progress reward
        progress_reward = self._update_lap_progress()
        collision_penalty = self._check_collisions()
        
        reward += progress_reward
        reward += collision_penalty
        self.score += reward

        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.win:
                self.score += 100
                reward += 100
            else:
                self.score -= 100
                reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_track(self):
        num_points = 12
        radius = 500
        center = np.array([0.0, 0.0])
        self.track_points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            base_point = center + np.array([math.cos(angle) * radius, math.sin(angle) * radius])
            noise = self.np_random.uniform(-radius * 0.4, radius * 0.4, size=2)
            self.track_points.append(base_point + noise)
        
        self.track_width = 120
        self.rumble_width = 10
        self.obstacles = []
        for i in range(len(self.track_points)):
            p1 = self.track_points[i]
            p2 = self.track_points[(i + 1) % len(self.track_points)]
            
            # Place 2 obstacles per segment
            for _ in range(2):
                t = self.np_random.uniform(0.2, 0.8)
                pos = p1 + t * (p2 - p1)
                offset_dir = (p2 - p1)
                offset_dir = np.array([-offset_dir[1], offset_dir[0]])
                offset_dir /= np.linalg.norm(offset_dir)
                
                offset_mag = self.np_random.uniform(-self.track_width/2 * 0.8, self.track_width/2 * 0.8)
                obstacle_pos = pos + offset_dir * offset_mag
                self.obstacles.append({'pos': obstacle_pos, 'radius': 15, 'hit': False})

    def _handle_input(self, movement, space_press, shift_held):
        # --- Turning ---
        turn_speed = 0.08 if not self.is_drifting else 0.12
        if movement == 3:  # Left
            self.player_angle -= turn_speed
        if movement == 4:  # Right
            self.player_angle += turn_speed

        # --- Acceleration ---
        if movement == 1:  # Up
            self.player_speed += 0.25
        if movement == 2:  # Down
            self.player_speed -= 0.3

        # --- Drifting ---
        self.is_drifting = shift_held and abs(self.player_speed) > 2.0 and (movement in [3, 4])
        if self.is_drifting:
            self.boost_meter = min(100, self.boost_meter + 0.7) # Charge boost meter
            self._create_drift_sparks()
        
        # --- Boosting ---
        if space_press and not self.last_space_press and self.boost_meter >= 100 and not self.is_boosting:
            self.is_boosting = True
            self.boost_timer = 60 # 2 seconds
            self.boost_meter = 0
            self.score += 5 # Boost activation reward
            # sfx: boost_activate.wav
        self.last_space_press = space_press

    def _update_physics(self):
        # --- Boost Logic ---
        if self.is_boosting:
            self.boost_timer -= 1
            if self.boost_timer <= 0:
                self.is_boosting = False
            max_speed = 15.0
            self._create_boost_particles()
        else:
            max_speed = 8.0

        # --- Friction & Speed Clamp ---
        if self.on_grass:
            self.player_speed *= 0.90 # High friction on grass
        else:
            self.player_speed *= 0.98 # Normal friction
        self.player_speed = np.clip(self.player_speed, -3.0, max_speed)
        if abs(self.player_speed) < 0.01: self.player_speed = 0

        # --- Drift Physics ---
        motion_angle = self.player_angle
        if self.is_drifting:
            # Angle of motion lags behind car's visual angle
            self.drift_angle_offset = (self.drift_angle_offset * 5 + (self.player_angle - motion_angle)) / 6
            motion_angle += self.drift_angle_offset * 0.5
            self.player_speed *= 0.99 # Slight speed loss during drift
        else:
            self.drift_angle_offset *= 0.9 # Smoothly return to normal

        # --- Update Position ---
        velocity = np.array([math.cos(motion_angle), math.sin(motion_angle)]) * self.player_speed
        self.player_pos += velocity
        
        # --- Camera Shake ---
        if self.camera_shake > 0:
            self.camera_shake -= 1

    def _create_drift_sparks(self):
        if self.drift_sparks_cooldown > 0:
            self.drift_sparks_cooldown -= 1
            return
        self.drift_sparks_cooldown = 2

        for i in [-1, 1]:
            angle = self.player_angle + math.pi/2 * i
            offset = np.array([math.cos(angle), math.sin(angle)]) * 15
            
            # Sparks should fly out opposite to motion
            motion_angle = math.atan2(math.sin(self.player_angle + self.drift_angle_offset), math.cos(self.player_angle + self.drift_angle_offset))
            vel_angle = motion_angle + math.pi + self.np_random.uniform(-0.3, 0.3)
            speed = self.np_random.uniform(2, 4)
            vel = np.array([math.cos(vel_angle), math.sin(vel_angle)]) * speed

            self.particles.append({
                'pos': self.player_pos + offset,
                'vel': vel,
                'lifespan': self.np_random.integers(10, 20),
                'color': (255, 255, 200),
                'radius': self.np_random.uniform(1, 3)
            })

    def _create_boost_particles(self):
        for _ in range(3):
            angle = self.player_angle + math.pi + self.np_random.uniform(-0.4, 0.4)
            speed = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            offset = np.array([math.cos(self.player_angle + math.pi), math.sin(self.player_angle + math.pi)]) * 20
            self.particles.append({
                'pos': self.player_pos + offset,
                'vel': vel,
                'lifespan': self.np_random.integers(15, 25),
                'color': random.choice([(255, 200, 0), (255, 150, 0), (255, 255, 100)]),
                'radius': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['vel'] *= 0.95

    def _get_dist_to_checkpoint(self, checkpoint_idx):
        checkpoint_pos = self.track_points[checkpoint_idx % len(self.track_points)]
        return np.linalg.norm(self.player_pos - checkpoint_pos)

    def _update_lap_progress(self):
        reward = 0
        next_checkpoint_idx = self.checkpoints_passed % len(self.track_points)
        
        # Reward for getting closer to the next checkpoint
        dist = self._get_dist_to_checkpoint(next_checkpoint_idx)
        reward += (self.distance_to_next_checkpoint - dist) * 0.05
        self.distance_to_next_checkpoint = dist

        # Check if we passed the checkpoint
        if dist < self.track_width * 0.8:
            self.checkpoints_passed += 1
            self.distance_to_next_checkpoint = self._get_dist_to_checkpoint(self.checkpoints_passed)
            # sfx: checkpoint.wav
            if self.checkpoints_passed >= len(self.track_points):
                # Lap complete!
                self.win = True
                self.terminated_reason = "LAP COMPLETED!"
        return reward

    def _check_collisions(self):
        penalty = 0
        
        # --- Obstacle Collisions ---
        for obs in self.obstacles:
            if not obs['hit']:
                dist = np.linalg.norm(self.player_pos - obs['pos'])
                if dist < obs['radius'] + 10: # 10 is approx half player width
                    obs['hit'] = True
                    self.collisions += 1
                    self.player_speed *= 0.2
                    penalty -= 1
                    self.camera_shake = 10
                    # sfx: crash.wav
        
        # --- Track Boundary Collisions ---
        min_dist_to_centerline = float('inf')
        for i in range(len(self.track_points)):
            p1 = self.track_points[i]
            p2 = self.track_points[(i + 1) % len(self.track_points)]
            l2 = np.linalg.norm(p2 - p1)**2
            if l2 == 0: continue
            t = max(0, min(1, np.dot(self.player_pos - p1, p2 - p1) / l2))
            projection = p1 + t * (p2 - p1)
            dist = np.linalg.norm(self.player_pos - projection)
            min_dist_to_centerline = min(min_dist_to_centerline, dist)

        if min_dist_to_centerline > self.track_width / 2:
            self.on_grass = True
            if min_dist_to_centerline > self.track_width / 2 + self.rumble_width:
                 # Heavy penalty for going far off track
                self.player_speed *= 0.5 
                penalty -= 0.1
        else:
            self.on_grass = False
            
        return penalty

    def _check_termination(self):
        if self.win:
            return True
        if self.collisions >= 4:
            self.terminated_reason = "TOO MANY COLLISIONS"
            return True
        if self.lap_time > 30.0:
            self.terminated_reason = "TIME'S UP!"
            return True
        if self.steps >= 1500: # Increased from 1000 to allow for slower laps
            self.terminated_reason = "MAX STEPS REACHED"
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_GRASS)
        
        cam_offset = self.screen_center - self.player_pos
        if self.camera_shake > 0:
            cam_offset += self.np_random.uniform(-5, 5, size=2)

        self._render_track(cam_offset)
        self._render_obstacles(cam_offset)
        self._render_particles(cam_offset)
        self._render_player()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_track(self, offset):
        # Draw track base
        for i in range(len(self.track_points)):
            p1 = self.track_points[i] + offset
            p2 = self.track_points[(i + 1) % len(self.track_points)] + offset
            pygame.draw.line(self.screen, self.COLOR_TRACK, p1.astype(int), p2.astype(int), int(self.track_width))
        
        # Draw rumble strips
        for i in range(len(self.track_points)):
            p1 = self.track_points[i] + offset
            p2 = self.track_points[(i + 1) % len(self.track_points)] + offset
            pygame.draw.line(self.screen, self.COLOR_RUMBLE, p1.astype(int), p2.astype(int), int(self.track_width + self.rumble_width))
            pygame.draw.line(self.screen, self.COLOR_TRACK, p1.astype(int), p2.astype(int), int(self.track_width))
        
        # Draw finish line
        p1 = self.track_points[0] + offset
        p2 = self.track_points[1] + offset
        mid = (p1 + p2) / 2
        
        line_dir = p2 - p1
        perp_dir = np.array([-line_dir[1], line_dir[0]])
        perp_dir /= np.linalg.norm(perp_dir)

        start_pos = self.track_points[0] - perp_dir * (self.track_width/2 + self.rumble_width) + offset
        end_pos = self.track_points[0] + perp_dir * (self.track_width/2 + self.rumble_width) + offset
        
        for i in range(12):
            color = self.COLOR_FINISH_LIGHT if i % 2 == 0 else self.COLOR_FINISH_DARK
            t = i / 11.0
            p_start = start_pos + t * (end_pos - start_pos)
            p_end = p_start + (end_pos - start_pos) / 11.0
            pygame.draw.line(self.screen, color, p_start.astype(int), p_end.astype(int), 15)

    def _render_obstacles(self, offset):
        for obs in self.obstacles:
            pos = obs['pos'] + offset
            # Draw glow
            glow_surf = pygame.Surface((obs['radius']*4, obs['radius']*4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_OBSTACLE_GLOW, (obs['radius']*2, obs['radius']*2), obs['radius']*2)
            self.screen.blit(glow_surf, (int(pos[0]-obs['radius']*2), int(pos[1]-obs['radius']*2)))
            # Draw obstacle
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(obs['radius']), self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(obs['radius']), self.COLOR_OBSTACLE)

    def _render_particles(self, offset):
        for p in self.particles:
            pos = p['pos'] + offset
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(p['radius']), p['color'])

    def _render_player(self):
        w, h = 30, 15
        center = self.screen_center
        
        # Create a surface for the kart
        kart_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(kart_surf, self.COLOR_PLAYER, (0, 0, w, h), border_radius=4)
        
        # Rotate kart
        visual_angle = self.player_angle + self.drift_angle_offset
        rotated_surf = pygame.transform.rotate(kart_surf, -math.degrees(visual_angle))
        rect = rotated_surf.get_rect(center=center)
        
        # Draw glow
        glow_surf = pygame.Surface((w*2, w*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (w, w), w)
        rotated_glow = pygame.transform.rotate(glow_surf, -math.degrees(visual_angle))
        glow_rect = rotated_glow.get_rect(center=center)
        self.screen.blit(rotated_glow, glow_rect)

        self.screen.blit(rotated_surf, rect)

    def _render_ui(self):
        # Lap Time
        time_text = self.font_small.render(f"TIME: {self.lap_time:.2f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (10, 10))

        # Collisions
        col_color = self.COLOR_UI_TEXT if self.collisions < 2 else self.COLOR_UI_FAIL
        col_text = self.font_small.render(f"HITS: {self.collisions}/4", True, col_color)
        self.screen.blit(col_text, (self.screen.get_width() - col_text.get_width() - 10, 10))
        
        # Boost Meter
        boost_w = 100
        boost_h = 10
        boost_x = self.screen_center[0] - boost_w // 2
        boost_y = self.screen_center[1] + 40
        
        # Draw meter background
        pygame.draw.rect(self.screen, (50, 50, 50), (boost_x, boost_y, boost_w, boost_h), border_radius=3)
        # Draw meter fill
        fill_w = (self.boost_meter / 100) * boost_w
        fill_color = (255, 255, 0) if self.boost_meter < 100 else (0, 255, 0)
        if fill_w > 0:
            pygame.draw.rect(self.screen, fill_color, (boost_x, boost_y, fill_w, boost_h), border_radius=3)
        # Draw meter border
        pygame.draw.rect(self.screen, (150, 150, 150), (boost_x, boost_y, boost_w, boost_h), 1, border_radius=3)
        
        # Game Over Text
        if self.game_over:
            color = self.COLOR_UI_SUCCESS if self.win else self.COLOR_UI_FAIL
            end_text = self.font_large.render(self.terminated_reason, True, color)
            text_rect = end_text.get_rect(center=self.screen_center)
            
            # Draw text with shadow
            shadow_text = self.font_large.render(self.terminated_reason, True, (0,0,0))
            shadow_rect = shadow_text.get_rect(center=(self.screen_center[0]+3, self.screen_center[1]+3))
            self.screen.blit(shadow_text, shadow_rect)
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap_time": self.lap_time,
            "collisions": self.collisions,
            "boost_meter": self.boost_meter,
            "is_boosting": self.is_boosting,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0

    # Main game loop
    running = True
    pygame.display.set_caption("Kart Kingdom")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False
                total_reward = 0

        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Step Environment ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()