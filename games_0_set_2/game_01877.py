
# Generated: 2025-08-28T02:59:18.877986
# Source Brief: brief_01877.md
# Brief Index: 1877

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: Arrow keys to move the cursor. Hold Space to draw a line from where you started "
        "holding to the cursor's current position. Release Space to finalize the line. Press Shift to "
        "delete the last line drawn."
    )

    game_description = (
        "A physics-based puzzle game where you draw lines to guide a sled from the start to the finish. "
        "The sled moves in real-time, so you must build the track ahead of it. Reach the finish before "
        "time runs out, but be careful not to let the sled fall off the screen!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_SLED = (255, 215, 0)
        self.COLOR_LINE = (255, 80, 80)
        self.COLOR_PREVIEW = (100, 150, 255)
        self.COLOR_START = (80, 255, 80)
        self.COLOR_FINISH = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE = [(255, 80, 80), (255, 140, 0), (200, 200, 200)]

        # Game constants
        self.FPS = 30
        self.MAX_TIME = 20.0  # seconds
        self.MAX_STEPS = self.MAX_TIME * self.FPS
        self.GRAVITY = pygame.Vector2(0, 350)
        self.CURSOR_SPEED = 10
        self.START_POS = pygame.Vector2(100, 320)
        self.FINISH_X = self.WIDTH - 50
        
        # Initialize state variables
        self.sled_pos = pygame.Vector2(0, 0)
        self.sled_vel = pygame.Vector2(0, 0)
        self.sled_angle = 0
        self.on_ground = False
        self.lines = deque(maxlen=50) # Limit number of lines to prevent slowdown
        self.cursor_pos = pygame.Vector2(0, 0)
        self.drawing_start_pos = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.last_sled_x = 0
        self.time_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""

        # This call will fail if the implementation is incorrect.
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.sled_pos = self.START_POS.copy()
        self.sled_vel = pygame.Vector2(20, 0)
        self.sled_angle = 0
        self.on_ground = True
        
        self.lines.clear()
        # Initial flat platform
        self.lines.append((pygame.Vector2(self.START_POS.x - 50, self.START_POS.y + 20),
                           pygame.Vector2(self.START_POS.x + 50, self.START_POS.y + 20)))

        self.cursor_pos = self.START_POS + pygame.Vector2(100, -50)
        self.drawing_start_pos = None
        self.prev_space_held = False
        self.prev_shift_held = False

        self.particles = []
        self.last_sled_x = self.sled_pos.x
        
        self.time_remaining = self.MAX_TIME
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        if not self.game_over:
            self._handle_input(movement, space_held, shift_held)
            self._update_physics()
            
            # Calculate rewards
            reward += self._calculate_reward()
            self.score += reward
            
            # Update timers
            self.time_remaining -= 1.0 / self.FPS
            self.steps += 1

        terminated = self._check_termination()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        # Draw line
        if space_held and not self.prev_space_held:
            self.drawing_start_pos = self.cursor_pos.copy()
        elif not space_held and self.prev_space_held and self.drawing_start_pos:
            if self.drawing_start_pos.distance_to(self.cursor_pos) > 5: # Min line length
                self.lines.append((self.drawing_start_pos, self.cursor_pos.copy()))
            self.drawing_start_pos = None
        
        # Undo line
        if shift_held and not self.prev_shift_held and len(self.lines) > 1:
            self.lines.pop() # Sound: "undo"

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_physics(self):
        dt = 1.0 / self.FPS
        
        # Sled physics
        self.sled_vel += self.GRAVITY * dt
        
        # Collision detection and response
        self.on_ground = False
        min_dist = float('inf')
        closest_line = None
        closest_point_on_line = None

        sled_bottom = self.sled_pos + pygame.Vector2(0, 5) # Check from a point slightly below center

        for line in self.lines:
            p, dist_sq = self._point_segment_distance(sled_bottom, line[0], line[1])
            if dist_sq < min_dist:
                min_dist = dist_sq
                closest_line = line
                closest_point_on_line = p

        if closest_line and min_dist < 25**2: # 25px collision radius
            line_vec = closest_line[1] - closest_line[0]
            if line_vec.length() > 0:
                self.on_ground = True
                
                # Correct position to prevent sinking
                self.sled_pos.y = closest_point_on_line.y - 5
                
                # Align velocity with the surface
                line_normal = line_vec.rotate(90).normalize()
                if line_normal.y > 0: line_normal = -line_normal # Normal should point upwards
                
                proj = self.sled_vel.dot(line_normal)
                if proj > 0:
                    self.sled_vel -= line_normal * proj
                
                # Apply friction
                friction = 0.995
                self.sled_vel *= friction
                
                # Update angle
                self.sled_angle = line_vec.angle_to(pygame.Vector2(1, 0))
        
        self.sled_pos += self.sled_vel * dt
        
        # Update particles
        self.particles = [p for p in self.particles if p.update(dt)]

    def _calculate_reward(self):
        reward = 0
        # Reward for forward progress
        progress = self.sled_pos.x - self.last_sled_x
        reward += progress * 0.1
        self.last_sled_x = self.sled_pos.x
        
        # Time penalty
        reward -= 0.01

        return reward

    def _check_termination(self):
        if self.game_over:
            return True

        terminated = False
        # Win condition
        if self.sled_pos.x >= self.FINISH_X:
            self.score += 10.0 # +5 for finishing, +5 for within time
            self.game_over = True
            self.game_outcome = "FINISH!"
            terminated = True
            # Sound: "win_fanfare"

        # Lose conditions
        elif not (0 < self.sled_pos.x < self.WIDTH and -50 < self.sled_pos.y < self.HEIGHT):
            self.score -= 1.0
            self.game_over = True
            self.game_outcome = "CRASH!"
            terminated = True
            self._create_explosion(self.sled_pos)
            # Sound: "explosion"
        
        elif self.time_remaining <= 0:
            self.score -= 1.0
            self.game_over = True
            self.game_outcome = "TIME UP"
            terminated = True
            # Sound: "timeout_buzzer"

        return terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_lines()
        if not (self.game_over and self.game_outcome == "CRASH!"):
             self._render_sled()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "sled_speed": self.sled_vel.length(),
        }
        
    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)
            
        # Start and Finish Lines
        pygame.draw.line(self.screen, self.COLOR_START, (self.START_POS.x, 0), (self.START_POS.x, self.HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.FINISH_X, 0), (self.FINISH_X, self.HEIGHT), 2)
        # Checkered finish line
        for y in range(0, self.HEIGHT, 20):
             pygame.draw.rect(self.screen, self.COLOR_GRID, (self.FINISH_X - 5, y, 10, 10))

    def _render_lines(self):
        for start, end in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_LINE, start, end, 3)
            
        if self.drawing_start_pos:
            pygame.draw.aaline(self.screen, self.COLOR_PREVIEW, self.drawing_start_pos, self.cursor_pos, 2)
            pygame.gfxdraw.filled_circle(self.screen, int(self.drawing_start_pos.x), int(self.drawing_start_pos.y), 4, self.COLOR_PREVIEW)
            
    def _render_sled(self):
        # Sled as a simple triangle
        points = [
            pygame.Vector2(-12, 0),
            pygame.Vector2(12, 0),
            pygame.Vector2(0, -8)
        ]
        
        # Rotate points
        rotated_points = [p.rotate(-self.sled_angle) + self.sled_pos for p in points]
        
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_SLED)
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_SLED)

    def _render_cursor(self):
        x, y = int(self.cursor_pos.x), int(self.cursor_pos.y)
        pygame.gfxdraw.filled_circle(self.screen, x, y, 6, self.COLOR_PREVIEW)
        pygame.gfxdraw.aacircle(self.screen, x, y, 6, self.COLOR_FINISH)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)
            
    def _render_ui(self):
        # Time remaining
        time_text = f"TIME: {self.time_remaining:.1f}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))
        
        # Speed
        speed = self.sled_vel.length() / 10
        speed_text = f"SPEED: {speed:.1f}"
        speed_surf = self.font_small.render(speed_text, True, self.COLOR_TEXT)
        self.screen.blit(speed_surf, (self.WIDTH - speed_surf.get_width() - 10, 10))
        
        # Score
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH // 2 - score_surf.get_width() // 2, 10))
        
        # Game Over message
        if self.game_over:
            outcome_surf = self.font_large.render(self.game_outcome, True, self.COLOR_FINISH)
            pos = (self.WIDTH // 2 - outcome_surf.get_width() // 2, self.HEIGHT // 2 - outcome_surf.get_height() // 2)
            self.screen.blit(outcome_surf, pos)

    def _point_segment_distance(self, p, a, b):
        # Returns the closest point on segment ab to point p, and the squared distance
        if a == b:
            return a, (p - a).length_squared()
        
        l2 = (a - b).length_squared()
        t = max(0, min(1, (p - a).dot(b - a) / l2))
        projection = a + t * (b - a)
        return projection, (p - projection).length_squared()

    def _create_explosion(self, pos):
        for _ in range(30):
            self.particles.append(Particle(pos, self.np_random, self.COLOR_PARTICLE))
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


class Particle:
    def __init__(self, pos, rng, colors):
        self.pos = pos.copy()
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(50, 150)
        self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.lifespan = rng.uniform(0.5, 1.5)
        self.color = random.choice(colors)
        self.size = rng.uniform(2, 6)

    def update(self, dt):
        self.pos += self.vel * dt
        self.vel *= 0.95 # Damping
        self.lifespan -= dt
        self.size = max(0, self.size - 2 * dt)
        return self.lifespan > 0

    def draw(self, surface):
        if self.size > 1:
            pygame.draw.circle(surface, self.color, self.pos, int(self.size))