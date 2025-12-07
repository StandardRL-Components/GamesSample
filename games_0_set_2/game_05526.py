
# Generated: 2025-08-28T05:17:25.993295
# Source Brief: brief_05526.md
# Brief Index: 5526

        
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
        "Controls: Use arrow keys to select line direction. Hold space for a longer line, hold shift for a curved line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A physics-based puzzle game. Draw lines to guide the sled from the start (green) to the finish (black). Reach the goal quickly for a higher score, but don't fall off the screen!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    PHYSICS_SUBSTEPS = 8
    GRAVITY = 0.08
    FRICTION = 0.998
    SLED_RADIUS = 6
    NORMAL_LINE_LENGTH = 40
    LONG_LINE_LENGTH = 80
    CURVE_SEGMENTS = 10
    CURVE_RADIUS = 60

    # --- Colors ---
    COLOR_BG = (210, 218, 226)
    COLOR_TRACK = (200, 30, 30)
    COLOR_SLED = (0, 150, 255)
    COLOR_SLED_RIDER = (255, 200, 0)
    COLOR_START = (0, 200, 100)
    COLOR_FINISH = (20, 20, 20)
    COLOR_PREDICTION = (0, 150, 255, 100)
    COLOR_TEXT = (50, 50, 50)
    COLOR_MOUNTAIN_1 = (170, 180, 190)
    COLOR_MOUNTAIN_2 = (140, 150, 160)
    COLOR_PARTICLE = (255, 80, 80)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.SysFont("monospace", 16)
            self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 60)

        # Initialize state variables
        self.sled_pos = pygame.Vector2(0, 0)
        self.sled_vel = pygame.Vector2(0, 0)
        self.track_lines = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""
        self.dist_to_finish = 0.0
        self.start_pos = pygame.Vector2(100, 150)
        self.finish_line_x = self.SCREEN_WIDTH - 50
        self.mountains1 = []
        self.mountains2 = []
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.sled_pos = pygame.Vector2(self.start_pos)
        self.sled_vel = pygame.Vector2(0.5, 0)
        
        # Initial flat track segment
        self.track_lines = [
            (pygame.Vector2(self.start_pos.x - 50, self.start_pos.y + self.SLED_RADIUS),
             pygame.Vector2(self.start_pos.x + 50, self.start_pos.y + self.SLED_RADIUS))
        ]

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""
        self.particles = []
        
        self.dist_to_finish = abs(self.finish_line_x - self.sled_pos.x)

        # Generate background mountains
        self._generate_mountains()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        
        prev_dist_to_finish = abs(self.finish_line_x - self.sled_pos.x)

        # 1. Handle Action: Draw a new line
        self._draw_line_from_action(movement, space_held, shift_held)

        # 2. Update Physics
        for _ in range(self.PHYSICS_SUBSTEPS):
            self._update_physics()
            if self.game_over: break

        # 3. Calculate Reward
        reward = self._calculate_reward(prev_dist_to_finish)
        self.score += reward

        # 4. Check Termination
        terminated = self._check_termination()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _draw_line_from_action(self, movement, space_held, shift_held):
        length = self.LONG_LINE_LENGTH if space_held else self.NORMAL_LINE_LENGTH
        start_point = self.sled_pos + self.sled_vel.normalize() * (self.SLED_RADIUS + 2) if self.sled_vel.length() > 0.1 else self.sled_pos + pygame.Vector2(1,0) * (self.SLED_RADIUS + 2)

        if movement == 0: # No-op: draw stabilizing horizontal line
            end_point = start_point + pygame.Vector2(self.NORMAL_LINE_LENGTH, 0)
            self.track_lines.append((start_point, end_point))
            return

        directions = {
            1: pygame.Vector2(0, -1),  # Up
            2: pygame.Vector2(0, 1),   # Down
            3: pygame.Vector2(-1, 0), # Left
            4: pygame.Vector2(1, 0),  # Right
        }
        direction_vec = directions.get(movement)

        if direction_vec:
            if shift_held: # Draw Curve
                # sound: "swoosh_curve.wav"
                # Center of circle for arc
                perp_vec = direction_vec.rotate(90)
                center = start_point + direction_vec * self.CURVE_RADIUS
                
                start_angle_vec = start_point - center
                
                total_angle = (length / self.CURVE_RADIUS) * (180 / math.pi)
                
                points = [start_point]
                for i in range(1, self.CURVE_SEGMENTS + 1):
                    angle = (i / self.CURVE_SEGMENTS) * total_angle
                    rotated_vec = start_angle_vec.rotate(angle)
                    points.append(center + rotated_vec)
                
                for i in range(len(points) - 1):
                    self.track_lines.append((points[i], points[i+1]))

            else: # Draw Straight Line
                # sound: "swoosh_straight.wav"
                end_point = start_point + direction_vec * length
                self.track_lines.append((start_point, end_point))

    def _update_physics(self):
        ground_line, ground_y, ground_normal = self._find_ground()

        if ground_line:
            # On a track
            line_vec = ground_line[1] - ground_line[0]
            angle = math.atan2(line_vec.y, line_vec.x)
            
            # Gravity component along the slope
            gravity_force = self.GRAVITY * math.sin(angle)
            accel = pygame.Vector2(math.cos(angle), math.sin(angle)) * gravity_force
            self.sled_vel += accel
            
            # Friction
            self.sled_vel *= self.FRICTION

            # Snap to surface
            self.sled_pos.y = ground_y - self.SLED_RADIUS
            
            # Project velocity onto the line to prevent bouncing
            if line_vec.length_squared() > 0:
                self.sled_vel = self.sled_vel.project(line_vec)

        else:
            # In the air (freefall)
            self.sled_vel.y += self.GRAVITY

        self.sled_pos += self.sled_vel

    def _find_ground(self):
        for line_start, line_end in reversed(self.track_lines):
            # Ensure line has horizontal extent to avoid division by zero
            if abs(line_end.x - line_start.x) < 1e-6:
                if abs(self.sled_pos.x - line_start.x) < self.SLED_RADIUS:
                    y_on_line = (line_start.y + line_end.y) / 2
                    if line_start.y < self.sled_pos.y + self.SLED_RADIUS and line_end.y < self.sled_pos.y + self.SLED_RADIUS:
                        continue # Sled is below the vertical line
                    # Simplified check for vertical lines
                    if min(line_start.y, line_end.y) <= self.sled_pos.y <= max(line_start.y, line_end.y):
                         return (line_start, line_end), self.sled_pos.y, pygame.Vector2(-1, 0) if line_start.x > self.sled_pos.x else pygame.Vector2(1, 0)
                continue

            # Check if sled is horizontally within the line segment
            if not (min(line_start.x, line_end.x) - self.SLED_RADIUS <= self.sled_pos.x <= max(line_start.x, line_end.x) + self.SLED_RADIUS):
                continue
            
            # Calculate line's y at sled's x
            t = (self.sled_pos.x - line_start.x) / (line_end.x - line_start.x)
            y_on_line = line_start.y + t * (line_end.y - line_start.y)
            
            # Check if sled is on or just above the line
            if y_on_line >= self.sled_pos.y and y_on_line - self.sled_pos.y < self.SLED_RADIUS * 2:
                normal = (line_end - line_start).rotate(90).normalize()
                if normal.y > 0: normal = -normal # Ensure normal points up
                return (line_start, line_end), y_on_line, normal
        return None, None, None

    def _calculate_reward(self, prev_dist_to_finish):
        current_dist_to_finish = abs(self.finish_line_x - self.sled_pos.x)
        
        # Reward for making progress towards the finish line
        reward = (prev_dist_to_finish - current_dist_to_finish) * 0.1
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.01

        # Terminal rewards/penalties
        if self.sled_pos.x >= self.finish_line_x: # Reached finish
            # sound: "win_fanfare.wav"
            fast_bonus = max(0, 50 * (1 - self.steps / 300)) # Bonus for speed
            reward += 10 + fast_bonus
            self.win_message = "YOU WIN!"
            self.game_over = True
        elif not self.screen.get_rect().collidepoint(self.sled_pos): # Crashed
            # sound: "crash_explosion.wav"
            reward -= 5
            self.win_message = "CRASHED!"
            self.game_over = True
            self._spawn_particles(20)

        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win_message = "TIME UP!"
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Parallax background
        camera_offset_x = self.sled_pos.x / 4
        for mountain in self.mountains2:
            points = [(p[0] - camera_offset_x * 0.5, p[1]) for p in mountain]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_MOUNTAIN_2)
        for mountain in self.mountains1:
            points = [(p[0] - camera_offset_x, p[1]) for p in mountain]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_MOUNTAIN_1)

        # Start and Finish lines
        pygame.draw.line(self.screen, self.COLOR_START, (self.start_pos.x, 0), (self.start_pos.x, self.SCREEN_HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_line_x, 0), (self.finish_line_x, self.SCREEN_HEIGHT), 3)

        # Track lines
        for start, end in self.track_lines:
            pygame.draw.line(self.screen, self.COLOR_TRACK, start, end, 3)

        # Trajectory prediction
        if self.sled_vel.length() > 0.1:
            pred_surface = self.screen.copy()
            pred_surface.set_colorkey(self.COLOR_BG)
            pred_surface.set_alpha(self.COLOR_PREDICTION[3])
            pygame.draw.line(pred_surface, (self.COLOR_PREDICTION[0], self.COLOR_PREDICTION[1], self.COLOR_PREDICTION[2]), self.sled_pos, self.sled_pos + self.sled_vel.normalize() * 50, 2)
            self.screen.blit(pred_surface, (0,0))

        # Sled
        pygame.gfxdraw.filled_circle(self.screen, int(self.sled_pos.x), int(self.sled_pos.y), self.SLED_RADIUS, self.COLOR_SLED)
        pygame.gfxdraw.aacircle(self.screen, int(self.sled_pos.x), int(self.sled_pos.y), self.SLED_RADIUS, self.COLOR_SLED)
        # Rider
        rider_pos = self.sled_pos - pygame.Vector2(0, self.SLED_RADIUS + 2)
        pygame.draw.circle(self.screen, self.COLOR_SLED_RIDER, (int(rider_pos.x), int(rider_pos.y)), 3)

        # Particles
        self._update_and_draw_particles()

    def _render_ui(self):
        speed_text = self.font_small.render(f"Speed: {int(self.sled_vel.length() * 10):03d}", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (10, 10))

        time_text = self.font_small.render(f"Time: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))
        
        score_text = self.font_small.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))

        if self.game_over and self.win_message:
            end_text = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sled_pos": (self.sled_pos.x, self.sled_pos.y),
            "sled_vel": (self.sled_vel.x, self.sled_vel.y),
        }

    def _generate_mountains(self):
        self.mountains1, self.mountains2 = [], []
        for layer, color, y_base, y_range, num_peaks in [(self.mountains2, self.COLOR_MOUNTAIN_2, 350, 150, 15), (self.mountains1, self.COLOR_MOUNTAIN_1, 380, 100, 10)]:
            points = [(0, self.SCREEN_HEIGHT)]
            last_x = 0
            for i in range(num_peaks):
                x = last_x + self.np_random.integers(100, 300)
                y = y_base - self.np_random.integers(0, y_range)
                points.append((x, y))
                last_x = x
            points.append((last_x + 200, self.SCREEN_HEIGHT))
            layer.append(points)

    def _spawn_particles(self, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append([pygame.Vector2(self.sled_pos), vel, self.np_random.integers(20, 40)]) # [pos, vel, lifetime]

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p[0] += p[1]
            p[1].y += 0.1 # gravity on particles
            p[2] -= 1
            if p[2] <= 0:
                self.particles.remove(p)
            else:
                size = int(max(0, p[2] / 8))
                pygame.draw.circle(self.screen, self.COLOR_PARTICLE, (int(p[0].x), int(p[0].y)), size)

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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a real screen for manual play
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Line Rider Gym")
    clock = pygame.time.Clock()

    total_reward = 0
    
    # Action buffer
    current_action = [0, 0, 0] # move, space, shift

    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(GameEnv.user_guide)
    print("Press R to reset.")
    print("="*30 + "\n")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Environment Reset ---")

        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]

        # Since auto_advance is False, we only step when an action is taken.
        # For manual play, we can step on every key press or hold.
        # A simple way is to step every frame with the current key state.
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")

        # Render to the real screen
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(draw_surface, (0, 0))
        pygame.display.flip()

        clock.tick(15) # Control manual play speed

    env.close()