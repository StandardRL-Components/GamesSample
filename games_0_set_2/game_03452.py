import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the drawing cursor. "
        "Hold Space to draw a line from the rider to the cursor."
    )

    game_description = (
        "Draw lines to guide a physics-based rider across a challenging, "
        "procedurally generated landscape to reach the finish line."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_RIDER = (255, 80, 80)
        self.COLOR_LINE = (220, 220, 220)
        self.COLOR_GROUND = (100, 120, 140)
        self.COLOR_CHECKPOINT = (255, 255, 0)
        self.COLOR_FINISH = (0, 255, 120)
        self.COLOR_PARTICLE = (255, 255, 255)
        self.COLOR_CURSOR = (255, 255, 255, 150)
        self.COLOR_TEXT = (240, 240, 240)

        # Physics constants
        self.GRAVITY = pygame.Vector2(0, 0.25)
        self.RIDER_RADIUS = 8
        self.MAX_VEL = 15
        self.CRASH_VEL = 14.5
        self.LINE_DAMPING = 0.95

        # Game constants
        self.MAX_STEPS = 2500
        self.STUCK_LIMIT = 300
        self.LEVEL_WIDTH = 4000
        self.MAX_LINES = 25

        # Initialize state variables
        self.rider_pos = pygame.Vector2(0, 0)
        self.rider_vel = pygame.Vector2(0, 0)
        self.lines = []
        self.ground_segments = []
        self.checkpoints = []
        self.particles = []
        self.draw_endpoint = pygame.Vector2(0, 0)
        self.camera_x = 0
        self.max_x_achieved = 0
        self.stuck_timer = 0
        self.passed_checkpoints = set()
        self.steps = 0
        self.score = 0
        self.game_over_message = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over_message = ""

        self.rider_pos = pygame.Vector2(100, 150)
        self.rider_vel = pygame.Vector2(4, 0)

        self.lines = []
        self.particles = []
        self._generate_level()

        self.draw_endpoint = self.rider_pos + pygame.Vector2(100, 0)

        self.camera_x = 0
        self.max_x_achieved = self.rider_pos.x
        self.stuck_timer = 0
        self.passed_checkpoints = set()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.ground_segments = []
        self.checkpoints = []

        rng = np.random.default_rng(self.np_random.integers(1e9))

        points = [pygame.Vector2(0, 250)]
        current_x = 0
        num_features = 5
        feature_spacing = self.LEVEL_WIDTH / (num_features + 1)

        for i in range(num_features):
            current_x += feature_spacing + rng.uniform(-100, 100)
            last_y = points[-1].y

            feature_type = rng.choice(["gap", "hill"])
            if feature_type == "gap":
                gap_width = rng.uniform(100, 250)
                points.append(pygame.Vector2(current_x, last_y))
                points.append(pygame.Vector2(current_x, self.HEIGHT + 200))
                points.append(pygame.Vector2(current_x + gap_width, self.HEIGHT + 200))
                points.append(pygame.Vector2(current_x + gap_width, last_y + rng.uniform(-50, 50)))
                current_x += gap_width
            elif feature_type == "hill":
                hill_width = rng.uniform(200, 400)
                hill_height = rng.uniform(80, 150) * rng.choice([-1, 1])
                points.append(pygame.Vector2(current_x, last_y))
                points.append(pygame.Vector2(current_x + hill_width / 2, last_y - hill_height))
                points.append(pygame.Vector2(current_x + hill_width, last_y))
                current_x += hill_width

        points.append(pygame.Vector2(self.LEVEL_WIDTH, points[-1].y))

        for i in range(len(points) - 1):
            self.ground_segments.append((points[i], points[i + 1]))

        self.checkpoints.append(self.LEVEL_WIDTH / 3)
        self.checkpoints.append(2 * self.LEVEL_WIDTH / 3)
        self.finish_line_x = self.LEVEL_WIDTH

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        reward = -0.01  # Time penalty
        terminated = False

        self._handle_input(movement, space_held)
        self._update_physics()

        # Check progress
        if self.rider_pos.x > self.max_x_achieved:
            reward += 0.05 * (self.rider_pos.x - self.max_x_achieved)
            self.max_x_achieved = self.rider_pos.x
            self.stuck_timer = 0
        else:
            self.stuck_timer += 1

        # Check events
        event_reward, event_termination, event_message = self._check_events()
        reward += event_reward
        if event_termination:
            terminated = True
            self.game_over_message = event_message

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 10
            self.game_over_message = "TIME UP"

        self.score += reward
        self.camera_x = self.rider_pos.x - self.WIDTH / 4

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        move_speed = 10
        if movement == 1: self.draw_endpoint.y -= move_speed
        elif movement == 2: self.draw_endpoint.y += move_speed
        elif movement == 3: self.draw_endpoint.x -= move_speed
        elif movement == 4: self.draw_endpoint.x += move_speed

        self.draw_endpoint.x = max(self.rider_pos.x - 50, self.draw_endpoint.x)
        self.draw_endpoint.x = min(self.rider_pos.x + 250, self.draw_endpoint.x)
        self.draw_endpoint.y = max(0, min(self.HEIGHT, self.draw_endpoint.y))

        if space_held and len(self.lines) < self.MAX_LINES:
            line_vec = self.draw_endpoint - self.rider_pos
            if line_vec.length() > self.RIDER_RADIUS * 2:
                # SFX: whoosh_draw.wav
                self.lines.append((pygame.Vector2(self.rider_pos), pygame.Vector2(self.draw_endpoint)))

    def _update_physics(self):
        self.rider_vel += self.GRAVITY
        if self.rider_vel.length() > self.MAX_VEL:
            self.rider_vel.scale_to_length(self.MAX_VEL)
        self.rider_pos += self.rider_vel

        collided_this_frame = False
        all_lines = self.lines + self.ground_segments
        for p1, p2 in all_lines:
            collided = self._resolve_line_circle_collision(p1, p2)
            if collided:
                collided_this_frame = True

        # Update and prune particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _resolve_line_circle_collision(self, p1, p2):
        line_vec = p2 - p1
        line_len_sq = line_vec.length_squared()
        if line_len_sq == 0: return False

        t = max(0, min(1, (self.rider_pos - p1).dot(line_vec) / line_len_sq))
        closest_point = p1 + t * line_vec

        dist_vec = self.rider_pos - closest_point
        dist_sq = dist_vec.length_squared()

        if dist_sq < self.RIDER_RADIUS ** 2:
            if self.rider_vel.length() > self.CRASH_VEL:
                # SFX: crash_hard.wav
                self.game_over_message = "CRASHED"
                return True

            # SFX: impact_thud.wav
            dist = math.sqrt(dist_sq)
            penetration = self.RIDER_RADIUS - dist
            normal = dist_vec.normalize()

            self.rider_pos += normal * penetration

            v_normal_comp = self.rider_vel.dot(normal)
            if v_normal_comp < 0:
                self.rider_vel -= 2 * v_normal_comp * normal
                self.rider_vel *= self.LINE_DAMPING

            self._create_impact_particles(self.rider_pos - normal * self.RIDER_RADIUS, normal)
            return True
        return False

    def _check_events(self):
        # Out of bounds
        if self.rider_pos.y > self.HEIGHT + 50 or self.rider_pos.y < -50:
            return -100, True, "FELL OFF"

        # Stuck
        if self.stuck_timer > self.STUCK_LIMIT:
            return -100, True, "STUCK"

        # Crash from high speed collision (checked in physics)
        if self.game_over_message == "CRASHED":
            return -100, True, "CRASHED"

        # Checkpoints
        for i, cp_x in enumerate(self.checkpoints):
            if i not in self.passed_checkpoints and self.rider_pos.x > cp_x:
                self.passed_checkpoints.add(i)
                # SFX: checkpoint.wav
                return 10, False, ""

        # Finish line
        if self.rider_pos.x > self.finish_line_x:
            # SFX: victory.wav
            return 100, True, "FINISH!"

        return 0, False, ""

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        grid_size = 50
        for x in range(int(self.camera_x) % grid_size - grid_size, self.WIDTH + grid_size, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw checkpoints and finish line
        for cp_x in self.checkpoints:
            screen_x = int(cp_x - self.camera_x)
            if 0 <= screen_x <= self.WIDTH:
                pygame.draw.line(self.screen, self.COLOR_CHECKPOINT, (screen_x, 0), (screen_x, self.HEIGHT), 2)

        finish_screen_x = int(self.finish_line_x - self.camera_x)
        if 0 <= finish_screen_x <= self.WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_screen_x, 0), (finish_screen_x, self.HEIGHT), 3)

        # Draw ground
        for p1, p2 in self.ground_segments:
            sp1 = (int(p1.x - self.camera_x), int(p1.y))
            sp2 = (int(p2.x - self.camera_x), int(p2.y))
            pygame.draw.aaline(self.screen, self.COLOR_GROUND, sp1, sp2, 3)

        # Draw user-drawn lines
        for p1, p2 in self.lines:
            sp1 = (int(p1.x - self.camera_x), int(p1.y))
            sp2 = (int(p2.x - self.camera_x), int(p2.y))
            pygame.draw.aaline(self.screen, self.COLOR_LINE, sp1, sp2, 2)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*self.COLOR_PARTICLE, alpha)
            pos = (int(p['pos'].x - self.camera_x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), color)

        # Draw cursor preview line
        rider_screen_pos = (int(self.rider_pos.x - self.camera_x), int(self.rider_pos.y))
        cursor_screen_pos = (int(self.draw_endpoint.x - self.camera_x), int(self.draw_endpoint.y))
        pygame.draw.aaline(self.screen, self.COLOR_CURSOR, rider_screen_pos, cursor_screen_pos)
        pygame.gfxdraw.filled_circle(self.screen, cursor_screen_pos[0], cursor_screen_pos[1], 4, self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(self.screen, cursor_screen_pos[0], cursor_screen_pos[1], 4, self.COLOR_CURSOR)

        # Draw rider
        pygame.gfxdraw.filled_circle(self.screen, rider_screen_pos[0], rider_screen_pos[1], self.RIDER_RADIUS,
                                      self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, rider_screen_pos[0], rider_screen_pos[1], self.RIDER_RADIUS,
                                self.COLOR_RIDER)

    def _render_ui(self):
        time_text = f"TIME: {self.steps / self.FPS:.1f}s"
        speed_text = f"SPEED: {self.rider_vel.length():.1f}"
        score_text = f"SCORE: {self.score:.0f}"

        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        speed_surf = self.font_small.render(speed_text, True, self.COLOR_TEXT)
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)

        self.screen.blit(time_surf, (10, 10))
        self.screen.blit(speed_surf, (self.WIDTH - speed_surf.get_width() - 10, 10))
        self.screen.blit(score_surf, (10, 35))

        if self.game_over_message:
            end_surf = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
            end_rect = end_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_surf, end_rect)

    def _create_impact_particles(self, pos, normal):
        num_particles = self.np_random.integers(5, 10)
        for _ in range(num_particles):
            angle = math.atan2(normal.y, normal.x) + self.np_random.uniform(-math.pi / 2, math.pi / 2)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(10, 25)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': life,
                'max_life': life,
                'size': self.np_random.uniform(1, 3)
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "rider_x": self.rider_pos.x,
            "rider_y": self.rider_pos.y,
            "lines_drawn": len(self.lines),
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # To play the game manually
    # This part requires a display. If you run this file directly,
    # comment out the `os.environ` line at the top.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    terminated = False
    total_reward = 0

    # Pygame window for human play
    try:
        render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Line Rider")
        human_play = True
    except pygame.error:
        print("Could not create display for human play. Running headless.")
        human_play = False


    action = np.array([0, 0, 0])

    while not terminated:
        if human_play:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()

            mov = 0
            if keys[pygame.K_UP]: mov = 1
            elif keys[pygame.K_DOWN]: mov = 2
            elif keys[pygame.K_LEFT]: mov = 3
            elif keys[pygame.K_RIGHT]: mov = 4

            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = np.array([mov, space, shift])
        else: # Simple bot for headless mode
            action = env.action_space.sample()


        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if human_play:
            # Render observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(env.FPS) # Control speed for human play

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            if human_play:
                pygame.time.wait(2000)  # Pause to see final screen
                obs, info = env.reset()
                terminated = False
            else:
                break

    env.close()