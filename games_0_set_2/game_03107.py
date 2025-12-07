
# Generated: 2025-08-28T07:00:36.180655
# Source Brief: brief_03107.md
# Brief Index: 3107

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move the cursor. Press Space to draw a track segment. Hold Shift to brake."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a track in real-time for your sledder to ride. Reach the finish as fast as you can, but be careful not to crash!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 20 * self.FPS  # 20-second time limit

        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (30, 35, 58)
        self.COLOR_TRACK = (255, 64, 64)
        self.COLOR_TRACK_INVALID = (100, 30, 30)
        self.COLOR_RIDER = (230, 230, 255)
        self.COLOR_TRAIL = (230, 230, 255)
        self.COLOR_CURSOR = (255, 220, 0)
        self.COLOR_FINISH = (64, 255, 128)
        self.COLOR_TEXT = (240, 240, 240)
        
        # Physics
        self.GRAVITY = 0.25
        self.FRICTION = 0.005
        self.BRAKE_FRICTION = 0.04
        self.RIDER_RADIUS = 7
        self.CURSOR_SPEED = 6
        self.MAX_SEGMENT_LENGTH = 150
        self.MIN_SEGMENT_LENGTH = 10
        self.MAX_ANGLE_CHANGE = math.pi / 2  # 90 degrees

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
        self.ui_font = pygame.font.SysFont("monospace", 18, bold=True)

        # Etc...
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.rider_pos = None
        self.rider_vel = None
        self.rider_angle = None
        self.rider_on_track = False
        self.rider_trail = None
        self.track_points = None
        self.cursor_pos = None
        self.particles = None
        self.finish_x = self.WIDTH - 40
        self.prev_space_state = 0
        self.time_elapsed = 0.0
        
        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state, for example:
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0.0

        # Rider state
        self.rider_pos = np.array([70.0, 100.0])
        self.rider_vel = np.array([4.0, 0.0])
        self.rider_angle = 0.0
        self.rider_on_track = True
        self.rider_trail = deque(maxlen=20)
        
        # Track state
        self.track_points = [[30, 100], [180, 100]]

        # Cursor state
        self.cursor_pos = np.array([250.0, 150.0])
        
        # Effects
        self.particles = []

        # Input state
        self.prev_space_state = 0

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_button = action[1]
        shift_held = action[2] == 1  # Boolean
        
        # Detect a press event for space
        space_pressed = space_button == 1 and self.prev_space_state == 0
        self.prev_space_state = space_button

        # Update game logic
        self.steps += 1
        self.time_elapsed = self.steps / self.FPS

        self._handle_input(movement, space_pressed)
        self._update_physics(shift_held)
        self._update_particles()
        
        self.rider_trail.append(self.rider_pos.copy())

        reward, terminated = self._calculate_rewards_and_termination()
        self.score += reward
        
        if terminated:
            self.game_over = True
            if self.rider_pos[0] < self.finish_x and self.steps < self.MAX_STEPS: # Crash
                self._create_particles(self.rider_pos, 30, self.COLOR_RIDER)
                # Sound effect placeholder: play_crash_sound()

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        # Place track segment
        if space_pressed and len(self.track_points) > 0:
            last_point = np.array(self.track_points[-1])
            if self._is_valid_segment(last_point, self.cursor_pos):
                self.track_points.append(list(self.cursor_pos))
                # Sound effect placeholder: play_place_track_sound()

    def _is_valid_segment(self, start_point, end_point):
        # Length check
        segment_vec = end_point - start_point
        dist = np.linalg.norm(segment_vec)
        if not (self.MIN_SEGMENT_LENGTH < dist < self.MAX_SEGMENT_LENGTH):
            return False
        
        # Angle check
        if len(self.track_points) >= 2:
            prev_segment_vec = start_point - np.array(self.track_points[-2])
            angle_new = math.atan2(segment_vec[1], segment_vec[0])
            angle_prev = math.atan2(prev_segment_vec[1], prev_segment_vec[0])
            angle_diff = abs(angle_new - angle_prev)
            if angle_diff > math.pi: angle_diff = 2 * math.pi - angle_diff # handle wrap-around
            if angle_diff > self.MAX_ANGLE_CHANGE:
                return False
        
        return True

    def _update_physics(self, shift_held):
        # Apply gravity
        if not self.rider_on_track:
            self.rider_vel[1] += self.GRAVITY

        # Find closest track segment
        min_dist = float('inf')
        closest_segment_idx = -1
        projected_pos = None

        for i in range(len(self.track_points) - 1):
            p1 = np.array(self.track_points[i])
            p2 = np.array(self.track_points[i+1])
            
            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue
            
            point_vec = self.rider_pos - p1
            t = np.dot(point_vec, line_vec) / line_len_sq
            t_clamped = np.clip(t, 0, 1)
            
            closest_point = p1 + t_clamped * line_vec
            dist = np.linalg.norm(self.rider_pos - closest_point)

            if dist < min_dist:
                min_dist = dist
                closest_segment_idx = i
                projected_pos = closest_point

        # Rider physics on track
        if closest_segment_idx != -1 and min_dist < self.RIDER_RADIUS * 3:
            self.rider_on_track = True
            self.rider_pos = projected_pos

            p1 = np.array(self.track_points[closest_segment_idx])
            p2 = np.array(self.track_points[closest_segment_idx+1])
            segment_vec = p2 - p1
            
            self.rider_angle = math.atan2(segment_vec[1], segment_vec[0])
            
            slope_force = self.GRAVITY * math.sin(self.rider_angle)
            vel_mag = np.linalg.norm(self.rider_vel)
            
            track_dir = segment_vec / (np.linalg.norm(segment_vec) + 1e-6)
            new_vel_mag = vel_mag + slope_force
            
            friction = (self.BRAKE_FRICTION if shift_held else self.FRICTION)
            new_vel_mag *= (1 - friction)
            new_vel_mag = max(0, new_vel_mag)
            
            self.rider_vel = track_dir * new_vel_mag

        else: # Rider is in freefall
            self.rider_on_track = False
            self.rider_vel *= 0.99
            self.rider_angle = math.atan2(self.rider_vel[1], self.rider_vel[0])

        self.rider_pos += self.rider_vel

    def _calculate_rewards_and_termination(self):
        terminated = False
        reward = 0.0

        if self.rider_pos[0] >= self.finish_x:
            terminated = True
            reward = 50.0 - (self.time_elapsed * 0.5) # Base reward + speed penalty
            # Sound effect placeholder: play_win_sound()

        elif not (0 < self.rider_pos[0] < self.WIDTH and -50 < self.rider_pos[1] < self.HEIGHT):
            terminated = True
            reward = -100.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward = -10.0
        
        if not terminated and self.rider_on_track:
            reward += 0.1
        
        return reward, terminated

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[3] > 0]
        for p in self.particles:
            p[0] += p[2][0]; p[1] += p[2][1]; p[3] -= 1

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([pos[0], pos[1], vel, lifetime, color])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background()
        self._render_finish_line()
        self._render_track()
        self._render_rider()
        self._render_cursor()

    def _render_background(self):
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)
            
    def _render_finish_line(self):
        for i in range(0, self.HEIGHT, 10):
            color = self.COLOR_FINISH if (i // 10) % 2 == 0 else self.COLOR_BG
            pygame.draw.rect(self.screen, color, (self.finish_x, i, 10, 10))

    def _render_track(self):
        if len(self.track_points) >= 2:
            pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, self.track_points, 2)
            for p in self.track_points:
                pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), 3, self.COLOR_TRACK)

    def _render_rider(self):
        # Trail
        for i, pos in enumerate(self.rider_trail):
            alpha = int(255 * (i / len(self.rider_trail)))
            if alpha > 10:
                s = pygame.Surface((self.RIDER_RADIUS, self.RIDER_RADIUS), pygame.SRCALPHA)
                radius = int(self.RIDER_RADIUS * 0.5 * (i / len(self.rider_trail)))
                pygame.draw.circle(s, (*self.COLOR_TRAIL[:3], alpha), (self.RIDER_RADIUS//2, self.RIDER_RADIUS//2), radius)
                self.screen.blit(s, (int(pos[0] - self.RIDER_RADIUS//2), int(pos[1] - self.RIDER_RADIUS//2)))

        # Rider body
        p1 = (self.rider_pos[0] + math.cos(self.rider_angle) * self.RIDER_RADIUS, self.rider_pos[1] + math.sin(self.rider_angle) * self.RIDER_RADIUS)
        p2 = (self.rider_pos[0] + math.cos(self.rider_angle + 2.2) * self.RIDER_RADIUS, self.rider_pos[1] + math.sin(self.rider_angle + 2.2) * self.RIDER_RADIUS)
        p3 = (self.rider_pos[0] + math.cos(self.rider_angle - 2.2) * self.RIDER_RADIUS, self.rider_pos[1] + math.sin(self.rider_angle - 2.2) * self.RIDER_RADIUS)
        int_points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_RIDER)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_RIDER)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p[3] / 30.0))))
            s = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p[4][:3], alpha), (2,2), 2)
            self.screen.blit(s, (int(p[0])-2, int(p[1])-2))

    def _render_cursor(self):
        if self.game_over: return
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        
        last_point = np.array(self.track_points[-1])
        color = self.COLOR_CURSOR if self._is_valid_segment(last_point, self.cursor_pos) else self.COLOR_TRACK_INVALID
        pygame.draw.aaline(self.screen, color, (int(last_point[0]), int(last_point[1])), (x,y))

        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x - 8, y), (x + 8, y), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x, y - 8), (x, y + 8), 2)

    def _render_ui(self):
        time_text = f"TIME: {self.time_elapsed:.2f}"
        text_surface = self.ui_font.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (self.WIDTH - text_surface.get_width() - 10, 10))

        if self.game_over:
            status = "FINISH!" if self.rider_pos[0] >= self.finish_x else "CRASHED!"
            if self.steps >= self.MAX_STEPS and self.rider_pos[0] < self.finish_x: status = "TIME UP!"
            
            status_font = pygame.font.SysFont("monospace", 48, bold=True)
            status_surface = status_font.render(status, True, self.COLOR_CURSOR)
            pos = (self.WIDTH // 2 - status_surface.get_width() // 2, self.HEIGHT // 2 - status_surface.get_height() // 2)
            self.screen.blit(status_surface, pos)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
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

if __name__ == "__main__":
    env = GameEnv()
    env.reset()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Sled Rider")
    running = True
    terminated = False
    
    while running:
        if terminated:
            env.reset()
            terminated = False

        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        
        env.clock.tick(env.FPS)
    env.close()