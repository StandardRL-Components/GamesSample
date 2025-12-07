
# Generated: 2025-08-27T21:35:07.920168
# Source Brief: brief_02834.md
# Brief Index: 2834

        
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
    user_guide = "Controls: Use arrow keys to draw a track. Press Shift to clear your drawing. Press Space to run the simulation."

    # Must be a short, user-facing description of the game:
    game_description = "Draw a track for your sled to ride on. Get to the finish line as fast as you can without crashing!"

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.SIM_FPS = 30
        self.MAX_STEPS = 1000
        
        # Colors
        self.COLOR_BG = (135, 206, 235) # Light Blue
        self.COLOR_TERRAIN = (139, 69, 19) # Brown
        self.COLOR_TRACK = (20, 20, 20) # Almost Black
        self.COLOR_TRACK_DRAWING = (100, 100, 100, 150)
        self.COLOR_SLED = (220, 20, 60) # Crimson Red
        self.COLOR_FINISH = (0, 128, 0) # Green
        self.COLOR_PARTICLE = (255, 255, 255)
        self.COLOR_CRASH = (255, 165, 0) # Orange
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)

        # Physics & Gameplay Constants
        self.GRAVITY = 0.4
        self.FRICTION = 0.995
        self.DRAW_SEGMENT_LENGTH = 20
        self.SLED_RADIUS = 8
        self.STOP_VELOCITY_THRESHOLD = 0.5
        self.STOP_FRAMES_REQUIRED = 15

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.reset()
        
        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_sim_time_steps = 0
        
        self._generate_terrain()
        
        self.start_pos = pygame.Vector2(50, self.terrain_points[1][1] - 30)
        self.finish_rect = pygame.Rect(self.WIDTH - 40, 0, 10, self.HEIGHT)
        
        self.sled = {
            'pos': pygame.Vector2(self.start_pos),
            'vel': pygame.Vector2(0, 0),
        }
        
        self.tracks = []
        self.current_track_segment = []
        
        self.game_phase = "DRAWING" # "DRAWING" or "SIMULATING"
        self.low_velocity_frames = 0
        
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        shift_pressed = action[2] == 1  # Boolean
        
        self.steps += 1
        reward = 0
        terminated = False
        
        if self.game_phase == "DRAWING":
            reward -= 0.01 # Cost for drawing/thinking
            if movement in [1, 2, 3, 4]:
                self._extend_track(movement)
            
            if shift_pressed:
                self.current_track_segment = []
                # Sound: "erase.wav"
                
            if space_pressed and len(self.current_track_segment) > 1:
                self.tracks.extend(self._segment_to_lines(self.current_track_segment))
                self.current_track_segment = []
                self.game_phase = "SIMULATING"
                self.low_velocity_frames = 0
                # Sound: "whoosh_start.wav"

        elif self.game_phase == "SIMULATING":
            self.total_sim_time_steps += 1
            reward_this_step, terminated_this_step, terminal_reason = self._update_simulation()
            reward += reward_this_step
            if terminated_this_step:
                terminated = True
                self.game_over = True
                if terminal_reason == "win":
                    # Sound: "win_jingle.wav"
                    time_penalty = self.total_sim_time_steps / 100.0
                    win_bonus = 10.0 - time_penalty
                    reward += 10.0 + max(0, win_bonus) # Total win reward: [10, 20]
                elif terminal_reason == "crash":
                    # Sound: "crash.wav"
                    reward -= 10.0
                    self._create_crash_particles()

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        self.score += reward
        self._update_particles()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _extend_track(self, movement):
        if not self.current_track_segment:
            start_point = pygame.Vector2(self.sled['pos'])
            self.current_track_segment.append(start_point)
        
        last_point = self.current_track_segment[-1]
        new_point = pygame.Vector2(last_point)
        
        if movement == 1: new_point.y -= self.DRAW_SEGMENT_LENGTH # Up
        elif movement == 2: new_point.y += self.DRAW_SEGMENT_LENGTH # Down
        elif movement == 3: new_point.x -= self.DRAW_SEGMENT_LENGTH # Left
        elif movement == 4: new_point.x += self.DRAW_SEGMENT_LENGTH # Right

        new_point.x = max(0, min(self.WIDTH, new_point.x))
        new_point.y = max(0, min(self.HEIGHT, new_point.y))
        
        if (new_point - last_point).length() > 1:
            self.current_track_segment.append(new_point)

    def _segment_to_lines(self, segment_points):
        return [(segment_points[i], segment_points[i+1]) for i in range(len(segment_points) - 1)]

    def _update_simulation(self):
        old_pos_x = self.sled['pos'].x
        
        self.sled['vel'].y += self.GRAVITY
        self.sled['pos'] += self.sled['vel']
        
        collided_track = None
        for p1, p2 in self.tracks:
            closest_point, dist_sq = self._closest_point_on_segment(self.sled['pos'], p1, p2)
            if dist_sq < self.SLED_RADIUS**2:
                collided_track = (p1, p2, closest_point, dist_sq)
                break
        
        if collided_track:
            p1, p2, closest_point, dist_sq = collided_track
            dist = math.sqrt(dist_sq) if dist_sq > 0 else 0
            overlap = self.SLED_RADIUS - dist
            collision_normal = (self.sled['pos'] - closest_point).normalize() if dist > 0 else pygame.Vector2(0, -1)
            self.sled['pos'] += collision_normal * overlap
            
            vel_dot_normal = self.sled['vel'].dot(collision_normal)
            if vel_dot_normal < 0:
                restitution = 0.3 # Bounciness
                impulse = -(1 + restitution) * vel_dot_normal * collision_normal
                self.sled['vel'] += impulse
            
            self.sled['vel'] *= self.FRICTION
            if self.sled['vel'].length() > 1.0:
                 self._create_trail_particles()
                 # Sound: "sled_grind.wav" (looping)
        
        if self.finish_rect.collidepoint(self.sled['pos']): return 0, True, "win"
        if not self.screen.get_rect().inflate(20, 20).collidepoint(self.sled['pos']): return 0, True, "crash"
        if self._check_terrain_collision(): return 0, True, "crash"
            
        if self.sled['vel'].length() < self.STOP_VELOCITY_THRESHOLD:
            self.low_velocity_frames += 1
        else:
            self.low_velocity_frames = 0
            
        if self.low_velocity_frames > self.STOP_FRAMES_REQUIRED:
            self.game_phase = "DRAWING"
            # Sound: "stop.wav"
            self.sled['vel'] = pygame.Vector2(0, 0)
        
        progress = self.sled['pos'].x - old_pos_x
        reward = progress * 0.1
        
        return reward, False, None

    def _closest_point_on_segment(self, p, a, b):
        ap = p - a
        ab = b - a
        ab_len_sq = ab.length_squared()
        if ab_len_sq == 0: return a, (p - a).length_squared()
        t = max(0, min(1, ap.dot(ab) / ab_len_sq))
        closest = a + ab * t
        return closest, (p - closest).length_squared()

    def _generate_terrain(self):
        points = [(-10, self.HEIGHT + 10)]
        y = self.np_random.uniform(self.HEIGHT - 80, self.HEIGHT - 40)
        points.append((-10, y))
        
        num_points = 10
        for i in range(num_points + 1):
            x = (self.WIDTH / num_points) * i
            y_offset = self.np_random.uniform(-30, 30)
            y = np.clip(y + y_offset, self.HEIGHT * 0.6, self.HEIGHT - 20)
            points.append((x, y))
            
        points.append((self.WIDTH + 10, y))
        points.append((self.WIDTH + 10, self.HEIGHT + 10))
        self.terrain_points = points

    def _check_terrain_collision(self):
        x, y = int(self.sled['pos'].x), int(self.sled['pos'].y + self.SLED_RADIUS)
        n = len(self.terrain_points)
        inside = False
        p1x, p1y = self.terrain_points[0]
        for i in range(n + 1):
            p2x, p2y = self.terrain_points[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        pygame.draw.polygon(self.screen, self.COLOR_TERRAIN, self.terrain_points)
        
        pygame.draw.rect(self.screen, self.COLOR_FINISH, self.finish_rect)
        for i in range(0, self.HEIGHT, 20):
            c = (255,255,255) if (i // 20) % 2 == 0 else self.COLOR_FINISH
            pygame.draw.rect(self.screen, c, (self.finish_rect.x, i, self.finish_rect.width, 10))

        for p1, p2 in self.tracks: pygame.draw.line(self.screen, self.COLOR_TRACK, p1, p2, 4)
            
        if len(self.current_track_segment) > 1:
            temp_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.lines(temp_surface, self.COLOR_TRACK_DRAWING, False, self.current_track_segment, 4)
            self.screen.blit(temp_surface, (0,0))
            
        for p in self.particles: pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['size']))
            
        sled_pos = (int(self.sled['pos'].x), int(self.sled['pos'].y))
        glow_radius = int(self.SLED_RADIUS * 1.5)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.COLOR_SLED, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (sled_pos[0] - glow_radius, sled_pos[1] - glow_radius))

        pygame.draw.circle(self.screen, self.COLOR_SLED, sled_pos, self.SLED_RADIUS)
        pygame.draw.circle(self.screen, (255,255,255), sled_pos, self.SLED_RADIUS, 1)

    def _render_ui(self):
        sim_time_sec = self.total_sim_time_steps / self.SIM_FPS
        self._draw_text(f"Time: {sim_time_sec:.2f}s", (10, 10), self.font_large)
        
        speed = self.sled['vel'].length() * 5
        self._draw_text(f"Speed: {speed:.1f}", (10, 40), self.font_small)

        self._draw_text(f"Mode: {self.game_phase}", (self.WIDTH - 150, 10), self.font_small)

    def _draw_text(self, text, pos, font):
        text_surface_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(text_surface_shadow, (pos[0] + 1, pos[1] + 1))
        text_surface = font.render(text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, pos)
        
    def _create_trail_particles(self):
        if len(self.particles) < 100:
            p_vel = self.sled['vel'].normalize().rotate(180) + pygame.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5))
            p_vel *= self.np_random.uniform(1, 3)
            self.particles.append({'pos': pygame.Vector2(self.sled['pos']), 'vel': p_vel, 'size': self.np_random.uniform(2, 4), 'life': self.np_random.integers(10, 20), 'color': self.COLOR_PARTICLE})

    def _create_crash_particles(self):
        for _ in range(50):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(2, 8)
            p_vel = pygame.Vector2(speed, 0).rotate(angle)
            self.particles.append({'pos': pygame.Vector2(self.sled['pos']), 'vel': p_vel, 'size': self.np_random.uniform(3, 7), 'life': self.np_random.integers(30, 60), 'color': self.COLOR_CRASH})
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sim_time_steps": self.total_sim_time_steps,
            "phase": self.game_phase
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