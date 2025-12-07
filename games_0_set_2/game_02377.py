
# Generated: 2025-08-28T04:38:16.731076
# Source Brief: brief_02377.md
# Brief Index: 2377

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Use arrow keys to position the drawing cursor. Press space to draw the line and watch the rider go. Shift resets the cursor."
    )

    game_description = (
        "A physics-based puzzle game. Draw lines for a sledder to ride on, guiding them across the level to reach the finish line. Plan your track carefully to build momentum and overcome obstacles."
    )

    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (240, 240, 250)
    COLOR_GRID = (220, 220, 230)
    COLOR_RIDER = (255, 50, 50)
    COLOR_RIDER_GLOW = (255, 150, 150)
    COLOR_LINE = (20, 20, 40)
    COLOR_START = (50, 200, 50)
    COLOR_FINISH = (200, 50, 50)
    COLOR_CURSOR = (50, 50, 255)
    COLOR_UI_TEXT = (10, 10, 20)
    
    # Screen and World
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WORLD_WIDTH = 3200  # Wider world for a longer track
    WORLD_HEIGHT = 800
    FINISH_LINE_X = WORLD_WIDTH - 200

    # Game Parameters
    MAX_STEPS = 500  # Max number of lines that can be drawn
    SIMULATION_SUBSTEPS = 150 # Physics ticks per line drawn
    
    # Rider Physics
    GRAVITY = 0.2
    FRICTION = 0.995
    RIDER_RADIUS = 10
    MIN_VELOCITY_FOR_SIM = 0.1

    # Drawing
    CURSOR_SPEED = 15
    MAX_CURSOR_OFFSET = 200

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)

        self.render_mode = render_mode
        
        # State variables are initialized in reset()
        self.rider_pos = pygame.Vector2(0, 0)
        self.rider_vel = pygame.Vector2(0, 0)
        self.lines = []
        self.particles = []
        self.cursor_offset = pygame.Vector2(0, 0)
        
        self.steps = 0
        self.score = 0
        self.max_x_reached = 0
        self.game_over = False
        self.win = False
        self.crash_reason = ""
        
        self.camera_pos = pygame.Vector2(0, 0)

        # This will be initialized in reset()
        self.np_random = None

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.crash_reason = ""

        # Initial rider state
        start_y = self.WORLD_HEIGHT / 2
        self.rider_pos = pygame.Vector2(100, start_y)
        self.rider_vel = pygame.Vector2(0, 0)
        self.max_x_reached = self.rider_pos.x
        
        # Initial terrain
        self.lines = []
        # Flat starting platform
        self.lines.append( (pygame.Vector2(20, start_y + self.RIDER_RADIUS), pygame.Vector2(150, start_y + self.RIDER_RADIUS)) )
        
        # Procedural terrain generation
        last_point = self.lines[0][1]
        while last_point.x < self.FINISH_LINE_X:
            segment_length = self.np_random.uniform(100, 250)
            angle = self.np_random.uniform(-math.pi / 6, math.pi / 6) # -30 to +30 degrees
            
            # Prevent terrain from going too high or too low
            if last_point.y < self.WORLD_HEIGHT * 0.2:
                angle = abs(angle) # Force downwards
            if last_point.y > self.WORLD_HEIGHT * 0.8:
                angle = -abs(angle) # Force upwards

            next_point = last_point + pygame.Vector2(math.cos(angle), math.sin(angle)) * segment_length
            self.lines.append((last_point, next_point))
            last_point = next_point

        self.cursor_offset = pygame.Vector2(80, 0)
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01  # Small penalty for each decision step
        
        if shift_pressed:
            self.cursor_offset = pygame.Vector2(80, 0)
        
        if movement == 1: self.cursor_offset.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_offset.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_offset.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_offset.x += self.CURSOR_SPEED
        
        self.cursor_offset.x = max(-self.MAX_CURSOR_OFFSET, min(self.MAX_CURSOR_OFFSET, self.cursor_offset.x))
        self.cursor_offset.y = max(-self.MAX_CURSOR_OFFSET, min(self.MAX_CURSOR_OFFSET, self.cursor_offset.y))

        if space_pressed and not self.game_over:
            # Finalize the line and run simulation
            start_point = self.rider_pos.copy()
            end_point = self.rider_pos + self.cursor_offset
            
            # Prevent drawing lines outside the world
            end_point.x = max(0, min(self.WORLD_WIDTH, end_point.x))
            end_point.y = max(0, min(self.WORLD_HEIGHT, end_point.y))
            
            # Add line, ensuring it's not a zero-length line
            if start_point.distance_to(end_point) > 1:
                self.lines.append((start_point, end_point))
            
            sim_reward = self._run_simulation()
            reward += sim_reward
            
            # Reset cursor for next turn
            self.cursor_offset = pygame.Vector2(80, 0)

        self.steps += 1
        terminated = self.game_over or (self.steps >= self.MAX_STEPS)
        if terminated and not self.win and not self.crash_reason:
            self.crash_reason = "Max steps reached"
            reward -= 25 # Penalty for running out of time

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _run_simulation(self):
        sim_reward = 0
        stuck_counter = 0

        for i in range(self.SIMULATION_SUBSTEPS):
            if self.game_over: break

            # --- Physics Update ---
            old_pos = self.rider_pos.copy()
            self.rider_vel.y += self.GRAVITY
            self.rider_pos += self.rider_vel
            
            # --- Collision Detection and Response ---
            collided = False
            for line_start, line_end in self.lines:
                closest_point, on_segment = self._get_closest_point_on_segment(self.rider_pos, line_start, line_end)
                dist_to_line = self.rider_pos.distance_to(closest_point)

                if dist_to_line < self.RIDER_RADIUS:
                    # Resolve penetration
                    penetration = self.RIDER_RADIUS - dist_to_line
                    normal = (self.rider_pos - closest_point).normalize()
                    self.rider_pos += normal * penetration
                    
                    # Project velocity onto line tangent for sliding effect
                    line_vec = (line_end - line_start)
                    # Ensure rider only interacts with the "top" of a line
                    if line_vec.x < 0: line_vec = -line_vec
                    line_normal = pygame.Vector2(-line_vec.y, line_vec.x).normalize()
                    
                    # Only collide if moving into the surface
                    if self.rider_vel.dot(line_normal) > 0:
                        tangent = line_vec.normalize()
                        vel_dot_tangent = self.rider_vel.dot(tangent)
                        self.rider_vel = tangent * vel_dot_tangent * self.FRICTION
                        collided = True
                        break
            
            # --- Check for Termination ---
            if self.rider_pos.x >= self.FINISH_LINE_X:
                self.game_over = True
                self.win = True
                sim_reward += 100
                break

            if not (0 < self.rider_pos.x < self.WORLD_WIDTH and 0 < self.rider_pos.y < self.WORLD_HEIGHT):
                self.game_over = True
                self.crash_reason = "Out of bounds"
                sim_reward -= 50
                self._create_particles(self.rider_pos, 50, (200, 20, 20)) # sound: crash
                break
            
            # Check if stuck
            if self.rider_vel.length() < self.MIN_VELOCITY_FOR_SIM:
                stuck_counter += 1
                if stuck_counter > 20: # Stuck for 20 substeps
                    break
            else:
                stuck_counter = 0

        # --- Reward for progress ---
        if self.rider_pos.x > self.max_x_reached:
            progress = self.rider_pos.x - self.max_x_reached
            sim_reward += progress * 0.1  # Reward for distance
            sim_reward += 5 # Bonus for new max distance
            self.max_x_reached = self.rider_pos.x
            
        return sim_reward

    def _get_closest_point_on_segment(self, p, a, b):
        ab = b - a
        ap = p - a
        if ab.length_squared() == 0:
            return a, True
        
        t = ap.dot(ab) / ab.length_squared()
        
        if t < 0.0:
            return a, False
        elif t > 1.0:
            return b, False
        
        return a + t * ab, True
        
    def _get_observation(self):
        self._update_camera()
        
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _update_camera(self):
        # Camera follows the rider's x, but is smoother and clamped.
        target_cam_x = self.rider_pos.x - self.SCREEN_WIDTH / 3
        target_cam_y = self.rider_pos.y - self.SCREEN_HEIGHT / 2
        
        # Smooth camera movement
        self.camera_pos.x += (target_cam_x - self.camera_pos.x) * 0.1
        self.camera_pos.y += (target_cam_y - self.camera_pos.y) * 0.1
        
        # Clamp camera
        self.camera_pos.x = max(0, min(self.WORLD_WIDTH - self.SCREEN_WIDTH, self.camera_pos.x))
        self.camera_pos.y = max(0, min(self.WORLD_HEIGHT - self.SCREEN_HEIGHT, self.camera_pos.y))

    def _world_to_screen(self, pos):
        return int(pos.x - self.camera_pos.x), int(pos.y - self.camera_pos.y)

    def _render_grid(self):
        grid_size = 50
        left = int(-self.camera_pos.x % grid_size)
        top = int(-self.camera_pos.y % grid_size)
        for x in range(left, self.SCREEN_WIDTH, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(top, self.SCREEN_HEIGHT, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game_elements(self):
        # Start and Finish lines
        start_screen_pos = self._world_to_screen(pygame.Vector2(20, 0))
        finish_screen_pos = self._world_to_screen(pygame.Vector2(self.FINISH_LINE_X, 0))
        pygame.draw.line(self.screen, self.COLOR_START, (start_screen_pos[0], 0), (start_screen_pos[0], self.SCREEN_HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_screen_pos[0], 0), (finish_screen_pos[0], self.SCREEN_HEIGHT), 3)

        # All drawn lines
        for start, end in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_LINE, self._world_to_screen(start), self._world_to_screen(end), 3)

        # Particles
        self._update_and_draw_particles()

        # Rider
        rider_screen_pos = self._world_to_screen(self.rider_pos)
        
        # Rider speed glow
        speed = self.rider_vel.length()
        glow_radius = int(self.RIDER_RADIUS + speed * 1.5)
        if glow_radius > self.RIDER_RADIUS:
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            alpha = min(255, int(50 + speed * 10))
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_RIDER_GLOW, alpha))
            self.screen.blit(glow_surf, (rider_screen_pos[0] - glow_radius, rider_screen_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.filled_circle(self.screen, rider_screen_pos[0], rider_screen_pos[1], self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, rider_screen_pos[0], rider_screen_pos[1], self.RIDER_RADIUS, self.COLOR_LINE)

        # Drawing cursor and preview line
        if not self.game_over:
            cursor_world_pos = self.rider_pos + self.cursor_offset
            cursor_screen_pos = self._world_to_screen(cursor_world_pos)
            pygame.draw.aaline(self.screen, self.COLOR_CURSOR, rider_screen_pos, cursor_screen_pos, 1)
            pygame.gfxdraw.filled_circle(self.screen, cursor_screen_pos[0], cursor_screen_pos[1], 5, self.COLOR_CURSOR)
            pygame.gfxdraw.aacircle(self.screen, cursor_screen_pos[0], cursor_screen_pos[1], 5, self.COLOR_CURSOR)

    def _render_ui(self):
        # UI Text
        score_text = self.font_small.render(f"SCORE: {self.score:.2f}", True, self.COLOR_UI_TEXT)
        steps_text = self.font_small.render(f"LINES: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        dist_text = self.font_small.render(f"PROGRESS: {int(self.max_x_reached / self.FINISH_LINE_X * 100)}%", True, self.COLOR_UI_TEXT)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))
        self.screen.blit(dist_text, (10, 30))

        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "FINISH!"
                color = self.COLOR_START
            else:
                msg = "CRASHED"
                color = self.COLOR_FINISH
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(end_text, text_rect)
            
            reason_text = self.font_small.render(self.crash_reason, True, self.COLOR_UI_TEXT)
            reason_rect = reason_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(reason_text, reason_rect)


    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append([pos.copy(), velocity, self.np_random.integers(20, 40), color])

    def _update_and_draw_particles(self):
        for p in self.particles:
            p[0] += p[1] # pos += vel
            p[1] *= 0.95 # friction
            p[2] -= 1 # lifetime
            
            if p[2] > 0:
                screen_pos = self._world_to_screen(p[0])
                size = max(1, int(p[2] / 10))
                pygame.draw.circle(self.screen, p[3], screen_pos, size)
        
        self.particles = [p for p in self.particles if p[2] > 0]

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "progress_percent": int(self.max_x_reached / self.FINISH_LINE_X * 100),
            "is_success": self.win,
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play ---
    # This requires a window, which is not standard for Gym envs, but useful for testing
    pygame.display.set_caption("Line Rider Gym")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # Map keyboard to MultiDiscrete action
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # Only step if an action is taken (for manual play)
        # In an RL loop, you'd step continuously
        if any(keys):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")

        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for manual play

    env.close()