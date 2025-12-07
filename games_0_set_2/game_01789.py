
# Generated: 2025-08-28T02:43:39.855306
# Source Brief: brief_01789.md
# Brief Index: 1789

        
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
        "Controls: ↑/↓ to aim. Press Space to draw a line. Guide the rider to the finish."
    )

    game_description = (
        "A physics-based sledding game. Draw lines to create a path for the rider. "
        "Reach the finish line as fast as possible while gaining style points for airtime."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_RIDER = (255, 255, 255)
        self.COLOR_RIDER_GLOW = (100, 150, 255)
        self.COLOR_LINE = (20, 20, 20)
        self.COLOR_TERRAIN = (40, 45, 60)
        self.COLOR_FINISH = (255, 50, 50)
        self.COLOR_START = (50, 255, 50)
        self.COLOR_PARTICLE = (180, 200, 255)
        self.COLOR_SPEEDLINE = (100, 150, 255, 150)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PREVIEW_LINE = (255, 255, 255)

        # Fonts
        try:
            self.font_large = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 16)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 22)
            
        # Game constants
        self.GRAVITY = 0.2
        self.MAX_STEPS = 2000
        self.RIDER_RADIUS = 8
        self.LINE_LENGTH = 120
        self.FINISH_X = 5000
        self.MAX_LINES = 15
        
        # Initialize state variables to be set in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.rider_pos = None
        self.rider_vel = None
        self.on_surface = None
        self.air_time = None
        self.lines = None
        self.terrain = None
        self.camera_x = None
        self.line_angle = None
        self.prev_space_held = None
        self.particles = None
        self.speed_lines = None
        self.last_rider_x = None
        self.last_speed = None
        self.terrain_slope_factor = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.rider_pos = np.array([100.0, 150.0])
        self.rider_vel = np.array([2.0, 0.0])
        self.last_rider_x = self.rider_pos[0]
        self.last_speed = np.linalg.norm(self.rider_vel)

        self.on_surface = True
        self.air_time = 0
        
        self.lines = []
        self.terrain_slope_factor = 0.1
        self.terrain = self._generate_terrain()

        self.camera_x = 0
        self.line_angle = -0.2  # Slightly downward
        self.prev_space_held = False

        self.particles = []
        self.speed_lines = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self._handle_input(action)
        reward = self._update_physics()
        self.score += reward
        
        self._update_effects()

        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            if self.rider_pos[0] >= self.FINISH_X:
                self.score += 100 # Win bonus
            else:
                self.score -= 5 # Crash penalty
            self.game_over = True
        
        # auto_advance=True means we manage the framerate
        self.clock.tick(30)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Adjust line angle
        if movement == 1: # Up
            self.line_angle -= 0.05
        elif movement == 2: # Down
            self.line_angle += 0.05
        self.line_angle = np.clip(self.line_angle, -math.pi / 2, math.pi / 2)

        # Place line on space press (rising edge)
        if space_held and not self.prev_space_held:
            # sfx: line_draw.wav
            start_pos = self.rider_pos
            end_pos = start_pos + np.array([math.cos(self.line_angle), math.sin(self.line_angle)]) * self.LINE_LENGTH
            
            # Prevent drawing offscreen
            if start_pos[0] > 0 and start_pos[1] > 0 and start_pos[1] < self.height:
                 self.lines.append((start_pos.copy(), end_pos.copy()))
                 if len(self.lines) > self.MAX_LINES:
                     self.lines.pop(0) # Memory leak prevention

        self.prev_space_held = space_held

    def _update_physics(self):
        # --- Update Rider ---
        if not self.on_surface:
            self.rider_vel[1] += self.GRAVITY
            self.air_time += 1
        
        self.rider_pos += self.rider_vel

        # --- Collision Detection ---
        collided_surface = None
        all_surfaces = self.lines + self.terrain

        self.on_surface = False
        min_dist = float('inf')
        closest_point = None
        
        for p1, p2 in all_surfaces:
            # Broad-phase check
            if self.rider_pos[0] + self.RIDER_RADIUS < min(p1[0], p2[0]) or \
               self.rider_pos[0] - self.RIDER_RADIUS > max(p1[0], p2[0]):
                continue

            p, dist_sq = self._get_closest_point_on_segment(self.rider_pos, p1, p2)
            if dist_sq < self.RIDER_RADIUS**2:
                self.on_surface = True
                dist = math.sqrt(dist_sq)
                if dist < min_dist:
                    min_dist = dist
                    collided_surface = (p1, p2)
                    closest_point = p

        reward = 0
        if self.on_surface:
            # --- Landing Reward ---
            if self.air_time > 10: # Min duration for a jump
                reward += 1.0 # Jump reward
                # sfx: land.wav
                self._create_particles(self.rider_pos, 15)
            self.air_time = 0

            # --- Collision Response ---
            p1, p2 = collided_surface
            surface_vec = p2 - p1
            surface_angle = math.atan2(surface_vec[1], surface_vec[0])
            
            # Positional correction
            overlap = self.RIDER_RADIUS - min_dist
            normal = (self.rider_pos - closest_point)
            normal_mag = np.linalg.norm(normal)
            if normal_mag > 1e-6:
                self.rider_pos += (normal / normal_mag) * overlap

            # Velocity response (sliding)
            gravity_force = np.array([0, self.GRAVITY * math.cos(surface_angle)])
            slide_force = math.sin(surface_angle) * self.GRAVITY
            
            current_speed = np.linalg.norm(self.rider_vel)
            new_speed = current_speed + slide_force
            
            dir_vec = surface_vec / np.linalg.norm(surface_vec)
            self.rider_vel = dir_vec * new_speed
            
            # Friction
            self.rider_vel *= 0.995

        # --- Reward Calculation ---
        # Forward progress reward
        progress = self.rider_pos[0] - self.last_rider_x
        reward += progress * 0.1
        self.last_rider_x = self.rider_pos[0]
        
        # Speed change reward
        current_speed = np.linalg.norm(self.rider_vel)
        if current_speed < self.last_speed:
            reward -= 0.01 # Small penalty for losing speed
        self.last_speed = current_speed

        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.terrain_slope_factor += 0.05

        return reward

    def _update_effects(self):
        # Update particles
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1 # lifetime

        # Update speed lines
        speed = np.linalg.norm(self.rider_vel)
        if speed > 5:
            start_pos = self.rider_pos.copy()
            line = [start_pos, speed, self.rider_vel.copy() / speed]
            self.speed_lines.append(line)
        
        self.speed_lines = [sl for sl in self.speed_lines if sl[1] > 0.1]
        for sl in self.speed_lines:
            sl[0] -= sl[2] * sl[1] * 0.2
            sl[1] *= 0.8 # length decay

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if self.rider_pos[0] >= self.FINISH_X:
            return True
        if not (0 < self.rider_pos[1] < self.height):
            return True
        # Check collision with terrain underside
        for p1, p2 in self.terrain:
             if self.rider_pos[0] > p1[0] and self.rider_pos[0] < p2[0]:
                 if self.rider_pos[1] > p1[1] and self.rider_pos[1] > p2[1]:
                     return True # Crashed by going under terrain
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Update camera
        self.camera_x = self.rider_pos[0] - self.width * 0.25

        # Render parallax background
        for i in range(3, 0, -1):
            color = (self.COLOR_BG[0] + i*5, self.COLOR_BG[1] + i*5, self.COLOR_BG[2] + i*10)
            offset = self.camera_x / (i * 2)
            # Simple repeating mountains
            for j in range(-5, 15):
                base_x = j * 300 - (offset % 300)
                pygame.gfxdraw.filled_trigon(self.screen, int(base_x), self.height, int(base_x + 150), self.height - 150 - i*20, int(base_x + 300), self.height, color)

        # Render terrain
        terrain_points_on_screen = []
        for p1, p2 in self.terrain:
            if p2[0] - self.camera_x > 0 and p1[0] - self.camera_x < self.width:
                terrain_points_on_screen.append((p1[0] - self.camera_x, p1[1]))
                terrain_points_on_screen.append((p2[0] - self.camera_x, p2[1]))
        
        if len(terrain_points_on_screen) > 1:
            # Create a closed polygon for filling
            poly_points = [(p[0], p[1]) for p in terrain_points_on_screen]
            if poly_points:
                poly_points.insert(0, (poly_points[0][0], self.height))
                poly_points.append((poly_points[-1][0], self.height))
                pygame.gfxdraw.filled_polygon(self.screen, poly_points, self.COLOR_TERRAIN)


        # Render start/finish lines
        start_x = 50 - self.camera_x
        finish_x = self.FINISH_X - self.camera_x
        pygame.draw.line(self.screen, self.COLOR_START, (start_x, 0), (start_x, self.height), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_x, 0), (finish_x, self.height), 5)
        
        # Render drawn lines
        for p1, p2 in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_LINE, (p1 - self.camera_x), (p2 - self.camera_x), 5)

        # Render speed lines
        for pos, length, direction in self.speed_lines:
            start = pos - self.camera_x
            end = start - direction * length
            alpha = max(0, min(255, int(length * 20)))
            pygame.draw.aaline(self.screen, (*self.COLOR_SPEEDLINE[:3], alpha), start, end, 2)

        # Render particles
        for x, y, vx, vy, life in self.particles:
            alpha = int(255 * (life / 20.0))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(x - self.camera_x), int(y), 2, (*self.COLOR_PARTICLE, alpha))

        # Render rider
        rider_x, rider_y = int(self.rider_pos[0] - self.camera_x), int(self.rider_pos[1])
        # Glow effect
        glow_radius = int(self.RIDER_RADIUS * (1.5 + math.sin(self.steps * 0.2) * 0.2))
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y, glow_radius, (*self.COLOR_RIDER_GLOW, 50))
        # Rider body
        pygame.gfxdraw.aacircle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)

        # Render line preview
        start_pos = self.rider_pos - self.camera_x
        end_pos = start_pos + np.array([math.cos(self.line_angle), math.sin(self.line_angle)]) * self.LINE_LENGTH
        self._draw_dashed_line(start_pos, end_pos, self.COLOR_PREVIEW_LINE)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font_large.render(f"Time: {self.steps/30.0:.1f}s", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.width - steps_text.get_width() - 10, 10))

        if self.game_over:
            status_text_str = "FINISH!" if self.rider_pos[0] >= self.FINISH_X else "CRASHED"
            status_text = self.font_large.render(status_text_str, True, self.COLOR_FINISH if status_text_str == "CRASHED" else self.COLOR_START)
            text_rect = status_text.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(status_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rider_pos": tuple(self.rider_pos),
            "rider_vel": tuple(self.rider_vel),
        }

    def _generate_terrain(self):
        points = []
        y = 250
        for x in range(-self.width, self.FINISH_X + self.width, 80):
            points.append(np.array([float(x), float(y)]))
            y += self.np_random.uniform(-40, 20) + self.terrain_slope_factor * 5
            y = np.clip(y, 150, self.height - 50)
        
        terrain_segments = []
        for i in range(len(points) - 1):
            terrain_segments.append((points[i], points[i+1]))
        return terrain_segments

    def _get_closest_point_on_segment(self, p, a, b):
        ap = p - a
        ab = b - a
        ab_sq = np.dot(ab, ab)
        if ab_sq == 0:
            return a, np.dot(ap, ap)
        
        t = np.dot(ap, ab) / ab_sq
        t = np.clip(t, 0, 1)
        
        closest_point = a + t * ab
        dist_sq = np.sum((p - closest_point)**2)
        return closest_point, dist_sq
    
    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(10, 25)
            self.particles.append([pos[0], pos[1], vx, vy, lifetime])

    def _draw_dashed_line(self, p1, p2, color, dash_length=5):
        p1 = np.array(p1)
        p2 = np.array(p2)
        line_vec = p2 - p1
        distance = np.linalg.norm(line_vec)
        if distance == 0: return
        
        unit_vec = line_vec / distance
        
        current_pos = p1
        for _ in range(int(distance / (dash_length * 2))):
            end_segment = current_pos + unit_vec * dash_length
            pygame.draw.aaline(self.screen, color, current_pos, end_segment)
            current_pos += unit_vec * dash_length * 2

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # To run and play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Mapping keyboard keys to actions for manual play
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Use a separate screen for display if running manually
    display_screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Sled Rider")

    running = True
    while running:
        # --- Create action from keyboard input ---
        movement_action = 0 # no-op
        space_action = 0
        shift_action = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render to display ---
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # We can just re-use the env's internal screen
        display_screen.blit(env.screen, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()
            total_reward = 0

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()