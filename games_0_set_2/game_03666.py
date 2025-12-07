import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in headless mode for the environment
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to draw different track slopes. "
        "Space for a steep incline, Shift for a steep decline."
    )

    game_description = (
        "Draw a track for the sledder to ride on. Guide them to the finish line "
        "while gaining speed and performing jumps."
    )

    auto_advance = False

    # Class-level state for difficulty progression
    finish_line_x_start = 200.0
    successful_episodes = 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)

        # Constants
        self.PHYSICS_SUBSTEPS = 10
        self.MAX_EPISODE_STEPS = 1000
        self.STOPPED_THRESHOLD = 60  # Terminate if stopped for 60 steps

        # Colors
        self.COLOR_BG = (15, 18, 26)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_TRACK = (230, 50, 75)
        self.COLOR_PREDICTION = (80, 150, 255)
        self.COLOR_RIDER = (0, 255, 255)
        self.COLOR_START = (80, 255, 80)
        self.COLOR_FINISH = (255, 220, 80)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_SPARK = (255, 180, 100)

        # Game parameters
        self.GRAVITY = 0.35
        self.FRICTION = 0.998
        self.LINE_LENGTH = 35.0
        self.RIDER_HEAD_RADIUS = 4
        self.RIDER_BODY_LENGTH = 12

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.rider_pos = None
        self.rider_vel = None
        self.rider_on_ground = False
        self.rider_angle = 0.0
        self.track = None
        self.drawing_cursor = None
        self.max_dist_x = 0.0
        self.stopped_steps = 0
        self.particles = []
        self.finish_line_x = 0.0

        # Initialize np_random
        self.reset(seed=0)
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        elif self.np_random is None:
            # Ensure np_random is initialized even without a seed
            self.np_random, _ = gym.utils.seeding.np_random(0)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        start_pos = pygame.Vector2(80, self.screen_height / 3)
        self.rider_pos = pygame.Vector2(start_pos)
        self.rider_vel = pygame.Vector2(0.5, 0)
        self.rider_on_ground = False
        self.rider_angle = 0.0

        start_platform = [
            (pygame.Vector2(start_pos.x - 40, start_pos.y + 20),
             pygame.Vector2(start_pos.x + 10, start_pos.y + 20))
        ]
        self.track = start_platform
        self.drawing_cursor = pygame.Vector2(start_platform[-1][1])

        self.max_dist_x = self.rider_pos.x
        self.stopped_steps = 0
        self.particles = []

        # Update difficulty based on class-level progression
        self.finish_line_x = self.finish_line_x_start + (self.successful_episodes // 5) * 10
        self.finish_line_x = min(self.finish_line_x, self.screen_width - 40)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1

        # 1. Handle Action: Draw a new line
        self._handle_action(action)

        # 2. Run Physics Simulation
        prev_on_ground = self.rider_on_ground
        self._run_physics()

        # 3. Calculate Reward
        reward = self._calculate_reward(prev_on_ground)

        # 4. Check for Termination
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward

        if terminated and not self.game_over:
            self.score += terminal_reward
            self.game_over = True
            if self.rider_pos.x >= self.finish_line_x:
                GameEnv.successful_episodes += 1

        self.score += reward

        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Reset drawing cursor if it's too far behind the rider
        if (self.drawing_cursor.x < self.rider_pos.x - 50):
            self.drawing_cursor = self.rider_pos + pygame.Vector2(20, 20)

        start_point = self.drawing_cursor

        angle_deg = 0
        if space_held:  # Steep up
            angle_deg = -45
        elif shift_held:  # Steep down
            angle_deg = 45
        else:
            if movement == 1:  # Up
                angle_deg = -20
            elif movement == 2:  # Down
                angle_deg = 20
            elif movement == 3:  # Left (flat backward)
                angle_deg = 180
            elif movement == 4:  # Right (flat forward)
                angle_deg = 0
            # movement == 0 is a no-op, no line is drawn

        if movement != 0 or space_held or shift_held:
            angle_rad = math.radians(angle_deg)
            end_point = start_point + pygame.Vector2(
                self.LINE_LENGTH * math.cos(angle_rad),
                self.LINE_LENGTH * math.sin(angle_rad)
            )

            # Prevent drawing off-screen
            end_point.x = np.clip(end_point.x, 0, self.screen_width)
            end_point.y = np.clip(end_point.y, 0, self.screen_height)

            if (start_point - end_point).length() > 1:
                self.track.append((start_point, end_point))
                self.drawing_cursor = end_point

    def _run_physics(self):
        dt = 1.0 / self.PHYSICS_SUBSTEPS
        for _ in range(self.PHYSICS_SUBSTEPS):
            # Apply gravity
            self.rider_vel.y += self.GRAVITY * dt

            # Store position before collision check
            old_pos = pygame.Vector2(self.rider_pos)
            self.rider_pos += self.rider_vel

            collided_this_substep = False
            best_line = None
            min_y_dist = float('inf')

            # Find the closest line segment directly below the rider
            for p1, p2 in self.track:
                # Bounding box check for efficiency
                if not (min(p1.x, p2.x) - self.RIDER_HEAD_RADIUS <= self.rider_pos.x <= max(p1.x, p2.x) + self.RIDER_HEAD_RADIUS):
                    continue

                line_vec = p2 - p1
                if line_vec.length_squared() == 0: continue

                # Project rider onto the line
                t = ((self.rider_pos - p1).dot(line_vec)) / line_vec.dot(line_vec)

                if 0 <= t <= 1:
                    line_point = p1.lerp(p2, t)
                    y_dist = self.rider_pos.y - line_point.y

                    if 0 <= y_dist < min_y_dist:
                        min_y_dist = y_dist
                        best_line = (p1, p2)

            if best_line and min_y_dist < self.RIDER_HEAD_RADIUS:
                # Collision occurred
                p1, p2 = best_line
                line_vec = p2 - p1
                line_angle_rad = math.atan2(line_vec.y, line_vec.x)

                # Correct position to sit on the line
                t = ((self.rider_pos - p1).dot(line_vec)) / line_vec.dot(line_vec)
                line_point = p1.lerp(p2, t)
                self.rider_pos.y = line_point.y

                # Project velocity onto the line tangent
                if line_vec.length_squared() > 0:
                    line_normal = line_vec.normalize()
                    speed = self.rider_vel.dot(line_normal)
                    self.rider_vel = line_normal * speed

                # Apply friction
                self.rider_vel *= self.FRICTION

                # Update rider visual angle
                self.rider_angle = math.degrees(line_angle_rad)
                collided_this_substep = True

                # Spawn particles on contact
                if self.np_random.random() < 0.5:
                    self._spawn_particles(self.rider_pos, 1)

            self.rider_on_ground = collided_this_substep
            if not collided_this_substep:
                # In air, rider tries to level out
                self.rider_angle *= 0.95

    def _calculate_reward(self, prev_on_ground):
        reward = 0.0

        # Reward for forward progress
        if self.rider_pos.x > self.max_dist_x:
            reward += 5.0  # Big reward for new furthest distance
            self.max_dist_x = self.rider_pos.x
        elif self.rider_pos.x > self.max_dist_x - 1.0: # Small reward for any forward movement
            reward += 0.1

        # Penalty for slow speed
        if self.rider_vel.length() < 0.5:
            reward -= 0.01

        # Reward for jumping (leaving the ground)
        if prev_on_ground and not self.rider_on_ground:
            reward += 1.0

        return reward

    def _check_termination(self):
        # Win condition
        if self.rider_pos.x >= self.finish_line_x:
            return True, 100.0

        # Max steps
        if self.steps >= self.MAX_EPISODE_STEPS:
            return True, -10.0

        # Out of bounds
        if not (0 < self.rider_pos.y < self.screen_height and 0 < self.rider_pos.x < self.screen_width):
            return True, -10.0

        # Rider stopped
        if self.rider_vel.length() < 0.1 and self.rider_on_ground:
            self.stopped_steps += 1
        else:
            self.stopped_steps = 0

        if self.stopped_steps > self.STOPPED_THRESHOLD:
            return True, -10.0

        return False, 0.0

    def _get_observation(self):
        # 1. Clear screen and draw background elements
        self.screen.fill(self.COLOR_BG)
        self._render_background_grid()
        self._render_start_finish_lines()

        # 2. Render game elements
        self._render_track()
        self._render_prediction_path()
        self._update_and_render_particles()
        self._render_rider()

        # 3. Render UI overlay
        self._render_ui()

        # 4. Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_grid(self):
        for i in range(0, self.screen_width, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.screen_height))
        for i in range(0, self.screen_height, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.screen_width, i))

    def _render_start_finish_lines(self):
        start_x = self.track[0][0].x if self.track else 40
        pygame.draw.line(self.screen, self.COLOR_START, (start_x, 0), (start_x, self.screen_height), 2)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_line_x, 0), (self.finish_line_x, self.screen_height), 2)

    def _render_track(self):
        for p1, p2 in self.track:
            pygame.gfxdraw.line(self.screen, int(p1.x), int(p1.y), int(p2.x), int(p2.y), self.COLOR_TRACK)
            pygame.gfxdraw.line(self.screen, int(p1.x), int(p1.y)+1, int(p2.x), int(p2.y)+1, self.COLOR_TRACK)

    def _render_prediction_path(self):
        # Simulate a "ghost" rider to predict path
        ghost_pos = pygame.Vector2(self.rider_pos)
        ghost_vel = pygame.Vector2(self.rider_vel)
        path_points = [ghost_pos]

        for _ in range(30): # Predict 30 physics steps ahead
            ghost_vel.y += self.GRAVITY * (1.0 / self.PHYSICS_SUBSTEPS)
            ghost_pos += ghost_vel
            path_points.append(pygame.Vector2(ghost_pos))

        if len(path_points) > 1:
            # Use aaline for anti-aliasing; it requires a special format
            pygame.draw.aalines(self.screen, self.COLOR_PREDICTION, False, [(int(p.x), int(p.y)) for p in path_points])

    def _render_rider(self):
        # Sled body
        p1 = self.rider_pos + pygame.Vector2(self.RIDER_BODY_LENGTH / 2, 0).rotate(-self.rider_angle)
        p2 = self.rider_pos - pygame.Vector2(self.RIDER_BODY_LENGTH / 2, 0).rotate(-self.rider_angle)
        pygame.draw.line(self.screen, self.COLOR_RIDER, p1, p2, 3)

        # Rider head
        head_pos = self.rider_pos - pygame.Vector2(0, self.RIDER_HEAD_RADIUS).rotate(-self.rider_angle)
        pygame.gfxdraw.filled_circle(self.screen, int(head_pos.x), int(head_pos.y), self.RIDER_HEAD_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, int(head_pos.x), int(head_pos.y), self.RIDER_HEAD_RADIUS, self.COLOR_RIDER)

    def _spawn_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(10, 20)
            })

    def _update_and_render_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.9
            p['life'] -= 1
            if p['life'] > 0:
                size = int(p['life'] / 4)
                if size > 0:
                    pygame.draw.circle(self.screen, self.COLOR_SPARK, (int(p['pos'].x), int(p['pos'].y)), size)
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_ui(self):
        speed_text = f"Speed: {self.rider_vel.length():.1f}"
        dist_text = f"Distance: {self.rider_pos.x:.0f} / {self.finish_line_x:.0f}"
        score_text = f"Score: {self.score:.0f}"

        speed_surf = self.font_small.render(speed_text, True, self.COLOR_TEXT)
        dist_surf = self.font_small.render(dist_text, True, self.COLOR_TEXT)
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)

        self.screen.blit(speed_surf, (10, 10))
        self.screen.blit(dist_surf, (self.screen_width - dist_surf.get_width() - 10, 10))
        self.screen.blit(score_surf, (self.screen_width // 2 - score_surf.get_width() // 2, 10))

        if self.game_over:
            msg = "GOAL!" if self.rider_pos.x >= self.finish_line_x else "CRASHED"
            color = self.COLOR_FINISH if msg == "GOAL!" else self.COLOR_TRACK
            end_surf = self.font_large.render(msg, True, color)
            self.screen.blit(end_surf, (self.screen_width // 2 - end_surf.get_width() // 2, self.screen_height // 2 - end_surf.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance": self.rider_pos.x,
            "max_distance": self.max_dist_x,
            "speed": self.rider_vel.length(),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Validating implementation...")
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Unset the dummy video driver to allow a display window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False

    # Pygame setup for manual play
    pygame.display.set_caption("Line Rider Gym")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))

    total_reward = 0

    action = [0, 0, 0] # Start with no-op

    # Game loop for manual play
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            # Map keyboard inputs to action space
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit FPS for manual play

    env.close()