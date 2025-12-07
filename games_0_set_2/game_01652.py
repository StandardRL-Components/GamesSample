import os
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import math
from collections import deque
import os
import pygame


# Set Pygame to run in headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import pygame.gfxdraw

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to draw ramps. Hold Space for longer ramps and Shift to make them horizontal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a sledder down a hazardous, procedurally generated slope. Draw ramps to bridge gaps and control your descent against the clock."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRAVITY = 0.2
        self.MAX_STEPS = 1500
        self.MAX_DRAWN_LINES = 8
        self.FINISH_Y = 5000

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (25, 30, 45)
        self.COLOR_TERRAIN = (180, 190, 200)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GLOW = (0, 200, 255, 50)
        self.COLOR_DRAWN_LINE = (255, 255, 0)
        self.COLOR_PREDICTION = (0, 100, 255, 100)
        self.COLOR_CHECKPOINT = (0, 255, 100)
        self.COLOR_START = (255, 50, 50)
        self.COLOR_FINISH = (50, 255, 50)
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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # State variables are initialized in reset()
        self.rider_pos = None
        self.rider_vel = None
        self.rider_radius = None
        self.camera_y = None
        self.terrain_segments = None
        self.drawn_lines = None
        self.checkpoints = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.timer = None
        self.max_y_reached = None
        self.last_action_frame = -10 # Cooldown for drawing lines
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 30.0 * self.FPS # 30 seconds
        self.max_y_reached = 0

        self.rider_pos = pygame.Vector2(self.WIDTH / 2, 50)
        self.rider_vel = pygame.Vector2(0, 0)
        self.rider_radius = 8
        self.camera_y = 0
        
        self.drawn_lines = deque(maxlen=self.MAX_DRAWN_LINES)
        
        self._procedural_generation()
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # Update game logic
        self._handle_action(movement, space_held, shift_held)
        reward = self._update_physics()
        
        self.steps += 1
        self.timer -= 1

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            # Calculate terminal reward only once
            if self.rider_pos.y >= self.FINISH_Y:
                reward += 100 # Victory
                self.score += 100
            elif self.timer <= 0:
                reward -= 25 # Timeout
                self.score -= 25
            else: # Fell off screen
                reward -= 50
                self.score -= 50
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _procedural_generation(self):
        self.terrain_segments = []
        self.checkpoints = []
        
        current_pos = pygame.Vector2(self.WIDTH / 2 - 100, 100)
        self.terrain_segments.append((pygame.Vector2(0, 100), current_pos.copy()))
        
        difficulty_angle_factor = min(4, self.steps // 500)
        difficulty_gap_factor = min(0.5, (self.steps // 500) * 0.05)

        y = 100
        while y < self.FINISH_Y + 200:
            segment_length = self.np_random.uniform(100, 250)
            # This angle always points downwards, ensuring the y-coordinate increases.
            base_angle_rad = math.radians(self.np_random.uniform(20, 45 + difficulty_angle_factor * 5))
            
            # Decide horizontal direction based on position to prevent going off-screen
            # and to create a varied path.
            if current_pos.x > self.WIDTH - 150: # Near right edge, must go left
                # Angle for left-downward slope (in range 90-180)
                angle_rad = math.pi - base_angle_rad
            elif current_pos.x < 150: # Near left edge, must go right
                angle_rad = base_angle_rad
            else: # In the middle, choose randomly
                if self.np_random.random() < 0.5:
                    angle_rad = base_angle_rad # Go right
                else:
                    angle_rad = math.pi - base_angle_rad # Go left

            direction_vec = pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad))
            next_pos = current_pos + direction_vec * segment_length
            
            self.terrain_segments.append((current_pos.copy(), next_pos.copy()))
            
            if self.np_random.random() > 0.3 - difficulty_gap_factor:
                gap_size = self.np_random.uniform(50, 100)
                gap_angle = math.radians(self.np_random.uniform(-10, 10))
                current_pos = next_pos + pygame.Vector2(math.cos(gap_angle), math.sin(gap_angle)) * gap_size
            else:
                current_pos = next_pos.copy()
            
            y = current_pos.y
            
            if len(self.checkpoints) == 0 or y > self.checkpoints[-1][0].y + 500:
                self.checkpoints.append((pygame.Vector2(0, y), pygame.Vector2(self.WIDTH, y), False))

    def _handle_action(self, movement, space_held, shift_held):
        if movement == 0 or self.steps < self.last_action_frame + 10: # Cooldown
            return

        self.last_action_frame = self.steps
        
        line_len = 80 if space_held else 40
        
        # Place line relative to player
        start_pos = self.rider_pos + pygame.Vector2(0, 20)

        angle_deg = 0
        if movement == 1: angle_deg = -45 # Up-Right
        elif movement == 2: angle_deg = 45 # Down-Right
        elif movement == 3: angle_deg = 135 # Down-Left
        elif movement == 4: angle_deg = -135 # Up-Left

        if shift_held: # Make line horizontal
            if angle_deg > 0: angle_deg = 90
            else: angle_deg = -90

        angle_rad = math.radians(angle_deg)
        end_pos = start_pos + pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * line_len

        self.drawn_lines.append((start_pos, end_pos))

    def _update_physics(self):
        reward = -0.01 # Time penalty
        
        # Apply gravity
        self.rider_vel.y += self.GRAVITY
        
        # Move rider
        self.rider_pos += self.rider_vel

        # Track progress for reward
        if self.rider_pos.y > self.max_y_reached:
            reward += 0.1 * (self.rider_pos.y - self.max_y_reached)
            self.score += 0.1 * (self.rider_pos.y - self.max_y_reached)
            self.max_y_reached = self.rider_pos.y

        # Collision detection and response
        all_lines = self.terrain_segments + list(self.drawn_lines)
        for line_start, line_end in all_lines:
            # Simple broad-phase check
            if not pygame.Rect(min(line_start.x, line_end.x) - self.rider_radius,
                               min(line_start.y, line_end.y) - self.rider_radius,
                               abs(line_end.x - line_start.x) + 2*self.rider_radius,
                               abs(line_end.y - line_start.y) + 2*self.rider_radius).collidepoint(self.rider_pos):
                continue

            line_vec = line_end - line_start
            if line_vec.length_squared() == 0: continue
            
            point_vec = self.rider_pos - line_start
            t = point_vec.dot(line_vec) / line_vec.length_squared()
            t = max(0, min(1, t))
            
            closest_point = line_start + t * line_vec
            dist_vec = self.rider_pos - closest_point
            
            if dist_vec.length_squared() < self.rider_radius**2:
                # Collision occurred
                dist = dist_vec.length()
                penetration = self.rider_radius - dist
                if dist == 0: continue
                
                normal = dist_vec.normalize()
                self.rider_pos += normal * penetration # Resolve penetration
                
                # Check for "risky" bounce
                if normal.y < -0.707: # Bounced mostly upwards (angle > 45 deg from vertical)
                    reward -= 1
                    self.score -= 1

                # Bounce physics
                restitution = 0.6
                friction = 0.98
                
                vel_normal_comp = self.rider_vel.dot(normal)
                self.rider_vel -= (1 + restitution) * vel_normal_comp * normal
                self.rider_vel *= friction

        # Checkpoints
        for i, (start, end, triggered) in enumerate(self.checkpoints):
            if not triggered and self.rider_pos.y > start.y:
                self.checkpoints[i] = (start, end, True)
                reward += 5
                self.score += 5
                self.timer += 5 * self.FPS # Add 5 seconds to timer
        
        return reward

    def _check_termination(self):
        off_screen_top = self.rider_pos.y < self.camera_y - self.rider_radius * 2
        off_screen_bottom = self.rider_pos.y > self.camera_y + self.HEIGHT + self.rider_radius * 2
        
        return (
            self.rider_pos.y >= self.FINISH_Y or
            off_screen_top or
            off_screen_bottom or
            self.timer <= 0
        )

    def _get_observation(self):
        # Update camera to keep rider in view
        self.camera_y = self.rider_pos.y - self.HEIGHT / 3
        
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_world()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        grid_size = 50
        # Vertical lines
        for x in range(0, self.WIDTH, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        # Horizontal lines (scrolling)
        start_y = - (self.camera_y % grid_size)
        for y in range(int(start_y), self.HEIGHT, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _world_to_screen(self, pos):
        return int(pos.x), int(pos.y - self.camera_y)

    def _render_world(self):
        # Render trajectory prediction
        if not self.game_over and self.rider_vel.length() > 0.5:
            pred_pos = self.rider_pos.copy()
            pred_vel = self.rider_vel.copy()
            points = [self._world_to_screen(pred_pos)]
            for _ in range(20):
                pred_vel.y += self.GRAVITY
                pred_pos += pred_vel
                points.append(self._world_to_screen(pred_pos))
            if len(points) > 1:
                pygame.draw.aalines(self.screen, self.COLOR_PREDICTION, False, points)

        # Render terrain
        for start, end in self.terrain_segments:
            pygame.draw.aaline(self.screen, self.COLOR_TERRAIN, self._world_to_screen(start), self._world_to_screen(end), 2)
        
        # Render drawn lines
        for start, end in self.drawn_lines:
            pygame.draw.aaline(self.screen, self.COLOR_DRAWN_LINE, self._world_to_screen(start), self._world_to_screen(end), 2)

        # Render checkpoints
        for start, end, triggered in self.checkpoints:
            color = self.COLOR_CHECKPOINT if not triggered else self.COLOR_GRID
            pygame.draw.line(self.screen, color, self._world_to_screen(start), self._world_to_screen(end), 1)

        # Render start/finish
        pygame.draw.line(self.screen, self.COLOR_START, self._world_to_screen(pygame.Vector2(0,100)), self._world_to_screen(pygame.Vector2(self.WIDTH,100)), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, self._world_to_screen(pygame.Vector2(0,self.FINISH_Y)), self._world_to_screen(pygame.Vector2(self.WIDTH,self.FINISH_Y)), 3)

        # Render player
        player_screen_pos = self._world_to_screen(self.rider_pos)
        pygame.gfxdraw.filled_circle(self.screen, player_screen_pos[0], player_screen_pos[1], self.rider_radius + 4, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, player_screen_pos[0], player_screen_pos[1], self.rider_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_screen_pos[0], player_screen_pos[1], self.rider_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        timer_sec = max(0, self.timer / self.FPS)
        timer_text = self.font_main.render(f"TIME: {timer_sec:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        if self.game_over:
            reason = ""
            if self.rider_pos.y >= self.FINISH_Y: reason = "FINISH!"
            elif self.timer <= 0: reason = "TIME UP"
            else: reason = "GAME OVER"
            
            end_text = self.font_main.render(reason, True, self.COLOR_FINISH if reason == "FINISH!" else self.COLOR_START)
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "y_position": self.rider_pos.y,
            "timer": self.timer / self.FPS
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
if __name__ == "__main__":
    # This block allows you to play the game directly
    # It requires a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Line Rider Gym Env")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Human controls
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        # The original code had a bug where up/down/left/right were not mutually exclusive.
        # This if/elif/.. structure fixes it.
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press R to reset
                    obs, info = env.reset()
                    total_reward = 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()