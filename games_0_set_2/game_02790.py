# Generated: 2025-08-27T21:27:32.226599
# Source Brief: brief_02790.md
# Brief Index: 2790

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys draw the track. No-op (0) draws straight ahead."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a track in real-time for a sled to ride, balancing speed and stability to reach the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRAVITY = 0.3
        self.FRICTION = 0.005
        self.DRAW_SEGMENT_LENGTH = 15
        self.MAX_TRACK_POINTS = 300
        self.CRASH_DISTANCE_THRESHOLD = 25
        self.CRASH_ANGLE_THRESHOLD = math.pi / 2  # 90 degrees
        self.STUCK_THRESHOLD_STEPS = 90
        self.STUCK_VELOCITY_LIMIT = 0.1
        self.MAX_EPISODE_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_TRACK = (0, 160, 255)
        self.COLOR_TRACK_CASING = (0, 80, 128)
        self.COLOR_SLED = (255, 65, 54)
        self.COLOR_SLED_GLOW = (255, 100, 90, 100)
        self.COLOR_START = (46, 204, 64)
        self.COLOR_FINISH = (255, 220, 0)
        self.COLOR_PARTICLE = (255, 133, 27)
        self.COLOR_TEXT = (240, 240, 240)

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
        self.font_title = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Initialize state variables
        self.sled_pos = np.array([0.0, 0.0])
        self.sled_vel = np.array([0.0, 0.0])
        self.sled_angle = 0.0
        self.track_points = []
        self.particles = []
        self.crashes_left = 0
        self.stage = 1
        self.finish_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stuck_counter = 0
        self.just_crashed = False

        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.crashes_left = 2
        self.stage = 1
        self.particles = []
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()
    
    def _setup_stage(self):
        stage_lengths = {1: 500, 2: 750, 3: 1000}
        self.finish_x = 40 + stage_lengths.get(self.stage, 1000)
        start_y = self.HEIGHT / 2
        
        self.sled_pos = np.array([60.0, start_y - 20.0])
        self.sled_vel = np.array([0.0, 0.0])
        self.sled_angle = 0.0
        
        # Create a starting platform
        self.track_points = [np.array([0, start_y]), np.array([100, start_y])]
        
        self.stuck_counter = 0
        self.just_crashed = False

    def step(self, action):
        reward = 0
        self.just_crashed = False
        
        self._handle_input(action)
        
        if not self.game_over:
            crashed = self._update_physics_and_check_crash()
            if crashed:
                reward -= 5.0
                self.just_crashed = True
                self._handle_crash()

        # Calculate rewards based on movement
        reward += self.sled_vel[0] * 0.1
        reward -= abs(self.sled_vel[1]) * 0.01
        
        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()
        
        if terminated:
            if self.sled_pos[0] >= self.finish_x and not self.game_over:
                reward += 50.0
                self.score += 50.0
            else: # Failure from timeout, crashes, or getting stuck
                reward -= 50.0
                self.score -= 50.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        last_point = self.track_points[-1]
        
        angle_map = {
            0: 0,          # right
            1: -math.pi / 4, # up-right
            2: math.pi / 4,  # down-right
            3: 3 * math.pi / 4, # down-left
            4: -3 * math.pi / 4, # up-left
        }
        angle = angle_map[movement]
        
        new_point = last_point + np.array([
            math.cos(angle) * self.DRAW_SEGMENT_LENGTH,
            math.sin(angle) * self.DRAW_SEGMENT_LENGTH
        ])
        
        # Clamp to screen bounds to prevent drawing off-screen
        new_point[0] = np.clip(new_point[0], 0, self.WIDTH)
        new_point[1] = np.clip(new_point[1], 0, self.HEIGHT)

        self.track_points.append(new_point)
        
        if len(self.track_points) > self.MAX_TRACK_POINTS:
            self.track_points.pop(0)

    def _update_physics_and_check_crash(self):
        # 1. Find closest point on track
        min_dist_sq = float('inf')
        closest_segment_idx = -1
        
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i+1]
            line_vec = p2 - p1
            p_vec = self.sled_pos - p1
            
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue

            t = np.dot(p_vec, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)
            
            closest_point = p1 + t * line_vec
            dist_sq = np.sum((self.sled_pos - closest_point)**2)
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_segment_idx = i

        # 2. Apply forces and update state if on track
        if closest_segment_idx != -1 and min_dist_sq < self.CRASH_DISTANCE_THRESHOLD**2:
            p1 = self.track_points[closest_segment_idx]
            p2 = self.track_points[closest_segment_idx+1]
            track_vec = p2 - p1
            
            # Apply gravity
            self.sled_vel[1] += self.GRAVITY
            
            # Apply reaction force
            if np.linalg.norm(track_vec) > 0:
                normal_vec = np.array([-track_vec[1], track_vec[0]])
                norm_mag = np.linalg.norm(normal_vec)
                if norm_mag > 0:
                    normal_vec = normal_vec / norm_mag
                
                vel_dot_normal = np.dot(self.sled_vel, normal_vec)
                # FIX: The condition was incorrect. It should be > 0 to react when the sled moves into the track from above.
                if vel_dot_normal > 0:
                    self.sled_vel -= vel_dot_normal * normal_vec
            
            # Apply friction
            self.sled_vel *= (1.0 - self.FRICTION)
            
            # Update angle
            if np.linalg.norm(track_vec) > 0:
                track_angle = math.atan2(track_vec[1], track_vec[0])
                self.sled_angle = self._lerp_angle(self.sled_angle, track_angle, 0.2)
        else: # Off track
            self.sled_vel[1] += self.GRAVITY
            self.sled_vel *= (1.0 - self.FRICTION)
            return True # Crash

        # 3. Update position
        self.sled_pos += self.sled_vel

        # 4. Check for angle crash
        if abs(self._angle_diff(self.sled_angle, 0)) > self.CRASH_ANGLE_THRESHOLD:
            return True # Crash
        
        return False

    def _handle_crash(self):
        # Create particles
        # sfx: explosion_small.wav
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": self.sled_pos.copy(),
                "vel": vel,
                "life": random.randint(20, 40)
            })

        self.crashes_left -= 1
        if self.crashes_left < 0:
            self.game_over = True
        else:
            # Reset sled to start of stage
            start_y = self.HEIGHT / 2
            self.sled_pos = np.array([60.0, start_y - 20.0])
            self.sled_vel = np.array([0.0, 0.0])
            self.sled_angle = 0.0

    def _check_termination(self):
        # Success condition
        if self.sled_pos[0] >= self.finish_x:
            if self.stage < 3:
                self.stage += 1
                self._setup_stage()
                # sfx: level_up.wav
                return False # Continue to next stage
            else:
                self.game_over = True
                # sfx: victory.wav
                return True # Game won
        
        # Failure conditions
        if self.game_over: # from crashes
            # sfx: game_over.wav
            return True
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            return True
        
        # Stuck condition
        if np.linalg.norm(self.sled_vel) < self.STUCK_VELOCITY_LIMIT:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            
        if self.stuck_counter >= self.STUCK_THRESHOLD_STEPS:
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_background()
        self._render_track()
        self._render_markers()
        self._render_particles()
        if not self.just_crashed:
            self._render_sled()

    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_track(self):
        if len(self.track_points) > 1:
            points_int = [p.astype(int).tolist() for p in self.track_points]
            pygame.draw.lines(self.screen, self.COLOR_TRACK_CASING, False, points_int, 8)
            pygame.draw.lines(self.screen, self.COLOR_TRACK, False, points_int, 4)

    def _render_markers(self):
        start_y = self.HEIGHT / 2
        pygame.draw.line(self.screen, self.COLOR_START, (100, start_y - 20), (100, start_y + 20), 5)
        
        finish_y_top = 0
        finish_y_bottom = self.HEIGHT
        if len(self.track_points) > 1:
            # Find track height at finish line for better marker placement
            for i in range(len(self.track_points) - 1):
                p1 = self.track_points[i]
                p2 = self.track_points[i+1]
                if (p1[0] <= self.finish_x and self.finish_x <= p2[0]) or \
                   (p2[0] <= self.finish_x and self.finish_x <= p1[0]):
                    t = (self.finish_x - p1[0]) / (p2[0] - p1[0]) if (p2[0] - p1[0]) != 0 else 0
                    y_at_finish = p1[1] + t * (p2[1] - p1[1])
                    finish_y_top = y_at_finish - 20
                    finish_y_bottom = y_at_finish + 20
                    break
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_x, finish_y_top), (self.finish_x, finish_y_bottom), 5)

    def _render_sled(self):
        x, y = int(self.sled_pos[0]), int(self.sled_pos[1])
        angle = self.sled_angle
        
        size = 12
        points = [
            (size, 0),
            (-size/2, -size/2),
            (-size/2, size/2)
        ]
        
        rotated_points = []
        for p_x, p_y in points:
            r_x = p_x * math.cos(angle) - p_y * math.sin(angle)
            r_y = p_x * math.sin(angle) + p_y * math.cos(angle)
            rotated_points.append((x + r_x, y + r_y))

        # Glow effect
        glow_surface = pygame.Surface((size*3, size*3), pygame.SRCALPHA)
        pygame.draw.polygon(glow_surface, self.COLOR_SLED_GLOW, [(p[0] - x + size*1.5, p[1] - y + size*1.5) for p in rotated_points])
        self.screen.blit(glow_surface, (x - size*1.5, y - size*1.5), special_flags=pygame.BLEND_RGBA_ADD)

        # Main sled body
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_SLED)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_SLED)

    def _render_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / 40))))
                size = max(1, int(5 * (p['life'] / 40)))
                rect = pygame.Rect(int(p['pos'][0] - size/2), int(p['pos'][1] - size/2), size, size)
                
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                temp_surf.fill((*self.COLOR_PARTICLE, alpha))
                self.screen.blit(temp_surf, rect.topleft)

    def _render_ui(self):
        # Stage Text
        stage_text = self.font_title.render(f"STAGE {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH // 2 - stage_text.get_width() // 2, 10))
        
        # Speed Text
        speed = np.linalg.norm(self.sled_vel) * 5
        speed_text = self.font_ui.render(f"SPEED: {speed:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (15, 10))
        
        # Crashes Left
        crash_text = self.font_ui.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(crash_text, (15, 35))
        for i in range(self.crashes_left):
            icon_rect = pygame.Rect(85 + i * 20, 38, 15, 10)
            pygame.draw.rect(self.screen, self.COLOR_SLED, icon_rect)

        # Progress bar
        progress = np.clip((self.sled_pos[0] - 100) / (self.finish_x - 100), 0, 1)
        bar_width = self.WIDTH - 40
        bar_x = 20
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, self.HEIGHT - 20, bar_width, 10))
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (bar_x, self.HEIGHT - 20, bar_width * progress, 10))
        # Draw player dot on progress bar
        dot_x = bar_x + bar_width * progress
        pygame.draw.circle(self.screen, self.COLOR_SLED, (int(dot_x), self.HEIGHT - 15), 7)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "crashes_left": self.crashes_left,
        }
        
    def _lerp_angle(self, a, b, t):
        diff = self._angle_diff(a, b)
        return a + diff * t

    def _angle_diff(self, a, b):
        return (b - a + math.pi) % (2 * math.pi) - math.pi

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    # In this mode, we want to see the screen.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Sled Drawer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("--- Human Controls ---")
    print(f"Description: {env.game_description}")
    print("Up/Down/Left/Right to draw the track.")
    print("Letting go of keys draws straight.")
    print("--------------------")

    while running:
        movement_action = 0 # Default is 0 (no-op / straight)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
             movement_action = 1 # up-right
        elif keys[pygame.K_DOWN]:
             movement_action = 2 # down-right
        elif keys[pygame.K_LEFT]:
             movement_action = 4 # up-left
        elif keys[pygame.K_RIGHT]:
             movement_action = 3 # down-left

        if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
            running = False

        # Action is always [movement, space, shift]
        action = [movement_action, 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            # Pause for a moment before restarting
            pygame.time.wait(1000)
            
        clock.tick(env.FPS)
        
    env.close()