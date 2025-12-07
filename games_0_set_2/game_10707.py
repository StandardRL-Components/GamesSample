import gymnasium as gym
import os
import pygame
import numpy as np
import math
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import os
import pygame


# Set headless mode for Pygame, required for the environment to run in a server environment
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper class for particles to add visual flair
class Particle:
    def __init__(self, pos, vel, color, radius, lifetime):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.radius = radius
        self.lifetime = lifetime
        self.initial_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.radius = max(0, self.radius * (self.lifetime / self.initial_lifetime))

    def draw(self, surface):
        if self.lifetime > 0:
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), self.color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing metadata ---
    game_description = (
        "Trace a moving path with your cursor, matching its speed for bonus points. "
        "The better you trace, the higher your score."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to control the tracer and follow the green path."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WIN_SCORE = 500
    MAX_STEPS = 5000
    FPS = 30

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60)
    COLOR_TARGET_PATH = (0, 255, 150)
    COLOR_TRACER = (50, 150, 255)
    COLOR_TRACER_GLOW = (100, 200, 255, 50)
    COLOR_PERFECT_TRACE = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_SPEED_LOW = pygame.Color(0, 255, 100)
    COLOR_SPEED_HIGH = pygame.Color(255, 50, 50)

    # Game Mechanics
    TRACER_ACCELERATION = 0.2
    TRACER_MAX_SPEED = 10.0
    TRACE_TOLERANCE = 8.0  # pixels
    SPEED_SYNC_TOLERANCE = 0.5  # absolute speed difference
    INITIAL_PATH_LENGTH = 150
    INITIAL_MAX_TARGET_SPEED = 4.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        self.render_mode = render_mode
        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.tracer_pos = pygame.Vector2(0, 0)
        self.tracer_vel = pygame.Vector2(0, 0)
        self.tracer_history = []

        self.path_start = pygame.Vector2(0, 0)
        self.path_end = pygame.Vector2(0, 0)
        self.path_vector = pygame.Vector2(0, 0)
        self.path_length = 0
        self.target_speed = 0
        self.target_progress = 0.0  # 0.0 to 1.0

        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.tracer_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.tracer_vel = pygame.Vector2(0, 0)
        self.tracer_history = [self.tracer_pos.copy() for _ in range(20)]

        self.particles = []

        start_pos = pygame.Vector2(
            self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
            self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
        )
        self._generate_new_path_segment(start_pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1

        # --- 1. Handle Action & Update Tracer ---
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        if movement == 1:  # Up
            self.tracer_vel.y -= self.TRACER_ACCELERATION
        elif movement == 2:  # Down
            self.tracer_vel.y += self.TRACER_ACCELERATION
        elif movement == 3:  # Left
            self.tracer_vel.x -= self.TRACER_ACCELERATION
        elif movement == 4:  # Right
            self.tracer_vel.x += self.TRACER_ACCELERATION

        # Clamp speed
        speed = self.tracer_vel.length()
        if speed > self.TRACER_MAX_SPEED:
            self.tracer_vel.scale_to_length(self.TRACER_MAX_SPEED)

        self.tracer_pos += self.tracer_vel

        # Screen wrap
        self.tracer_pos.x %= self.SCREEN_WIDTH
        self.tracer_pos.y %= self.SCREEN_HEIGHT

        self.tracer_history.append(self.tracer_pos.copy())
        if len(self.tracer_history) > 50:
            self.tracer_history.pop(0)

        # --- 2. Update Target Path ---
        self.target_progress += self.target_speed / self.path_length if self.path_length > 0 else 0
        if self.target_progress >= 1.0:
            self._generate_new_path_segment(self.path_end)
            self.target_progress = 0.0

        # --- 3. Calculate Reward ---
        reward = 0.0

        # Find distance from tracer to the target path segment
        dist_to_path = self._distance_point_to_segment(self.tracer_pos, self.path_start, self.path_end)

        # Proximity/Deviation reward/penalty
        if dist_to_path < self.TRACE_TOLERANCE:
            # Positive reward for being close
            proximity_reward = 1.0 * (1 - (dist_to_path / self.TRACE_TOLERANCE)) ** 2
            # Add particles for visual feedback
            self._spawn_trace_particles()

            # Speed synchronization bonus
            tracer_speed = self.tracer_vel.length()
            speed_diff = abs(tracer_speed - self.target_speed)
            if speed_diff < self.SPEED_SYNC_TOLERANCE:
                proximity_reward *= 1.2  # 20% bonus

            reward += proximity_reward
        else:
            # Penalty for being far
            penalty = -0.1 * (dist_to_path - self.TRACE_TOLERANCE)
            reward += penalty

        self.score += reward

        # --- 4. Update Particles ---
        for p in self.particles[:]:
            p.update()
            if p.lifetime <= 0:
                self.particles.remove(p)

        # --- 5. Check Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100.0  # Goal-oriented reward
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated

        return self._get_observation(), float(reward), terminated, False, self._get_info()

    def _generate_new_path_segment(self, start_pos):
        self.path_start = start_pos.copy()

        # Difficulty scaling
        difficulty_level = self.score // 100
        path_length_scaler = 1.0 - (difficulty_level * 0.01)  # Length decreases
        speed_scaler = 1.0 + (difficulty_level * 0.02)  # Speed increases

        current_path_length = self.np_random.uniform(0.8, 1.2) * self.INITIAL_PATH_LENGTH * path_length_scaler
        current_max_target_speed = self.INITIAL_MAX_TARGET_SPEED * speed_scaler

        angle = self.np_random.uniform(0, 2 * math.pi)
        end_pos_candidate = self.path_start + pygame.Vector2(math.cos(angle), math.sin(angle)) * current_path_length

        # Ensure path stays reasonably on-screen
        end_pos_candidate.x = np.clip(end_pos_candidate.x, 50, self.SCREEN_WIDTH - 50)
        end_pos_candidate.y = np.clip(end_pos_candidate.y, 50, self.SCREEN_HEIGHT - 50)
        self.path_end = end_pos_candidate

        self.path_vector = self.path_end - self.path_start
        self.path_length = self.path_vector.length()
        if self.path_length == 0: self.path_length = 1  # Avoid division by zero
        self.target_speed = self.np_random.uniform(0.5, 1.0) * current_max_target_speed

    def _spawn_trace_particles(self):
        if len(self.particles) < 100:
            for _ in range(2):
                vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)) * 0.5
                p = Particle(self.tracer_pos, vel, self.COLOR_PERFECT_TRACE, self.np_random.uniform(1, 3), 20)
                self.particles.append(p)

    def _distance_point_to_segment(self, p, a, b):
        if a.distance_to(b) == 0:
            return p.distance_to(a)

        ab = b - a
        ap = p - a

        t = ap.dot(ab) / ab.dot(ab)
        t = np.clip(t, 0, 1)

        closest_point = a + ab * t
        return p.distance_to(closest_point)

    def _get_observation(self):
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tracer_speed": self.tracer_vel.length(),
            "target_speed": self.target_speed
        }

    def _render_frame(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_game()
        self._render_ui()

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Particles first, so they are behind other elements
        for p in self.particles:
            p.draw(self.screen)

        # Target path
        pygame.draw.line(self.screen, self.COLOR_TARGET_PATH, self.path_start, self.path_end, 3)
        target_pos = self.path_start.lerp(self.path_end, self.target_progress)
        self._draw_glow_circle(self.screen, self.COLOR_TARGET_PATH, target_pos, 6)

        # Tracer path history
        if len(self.tracer_history) > 2:
            pygame.draw.aalines(self.screen, self.COLOR_TRACER, False, self.tracer_history, 2)

        # Tracer
        self._draw_glow_circle(self.screen, self.COLOR_TRACER, self.tracer_pos, 8, glow_color=self.COLOR_TRACER_GLOW)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Speed Indicator
        speed_bar_width = 15
        speed_bar_height = 100
        speed_bar_x = self.SCREEN_WIDTH - speed_bar_width - 10
        speed_bar_y = 10

        pygame.draw.rect(self.screen, self.COLOR_GRID, (speed_bar_x, speed_bar_y, speed_bar_width, speed_bar_height), 2)

        tracer_speed_ratio = np.clip(self.tracer_vel.length() / self.TRACER_MAX_SPEED, 0, 1)
        target_speed_ratio = np.clip(self.target_speed / self.TRACER_MAX_SPEED, 0, 1)

        # Tracer speed bar
        if tracer_speed_ratio > 0:
            fill_height = int(speed_bar_height * tracer_speed_ratio)
            # FIX: Correctly use lerp to interpolate between two colors
            color = self.COLOR_SPEED_LOW.lerp(self.COLOR_SPEED_HIGH, tracer_speed_ratio)
            pygame.draw.rect(self.screen, color,
                             (speed_bar_x, speed_bar_y + speed_bar_height - fill_height, speed_bar_width, fill_height))

        # Target speed marker
        marker_y = speed_bar_y + speed_bar_height - int(speed_bar_height * target_speed_ratio)
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (speed_bar_x - 2, marker_y),
                         (speed_bar_x + speed_bar_width + 2, marker_y), 2)

        speed_text = self.font_small.render("SPEED", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (speed_bar_x, speed_bar_y + speed_bar_height + 5))

        # Termination message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            msg = "GOAL REACHED!" if self.score >= self.WIN_SCORE else "TIME UP"
            end_text = self.font_main.render(msg, True, self.COLOR_PERFECT_TRACE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _draw_glow_circle(self, surface, color, pos, radius, glow_color=None):
        if glow_color is None:
            glow_color = (*color, 50)

        int_pos = (int(pos.x), int(pos.y))

        # Draw glow layers
        for i in range(3):
            r = radius + i * 4
            alpha = glow_color[3] * (1 - i / 3)
            temp_glow_color = (*glow_color[:3], int(alpha))
            pygame.gfxdraw.filled_circle(surface, int_pos[0], int_pos[1], r, temp_glow_color)
            pygame.gfxdraw.aacircle(surface, int_pos[0], int_pos[1], r, temp_glow_color)

        # Draw main circle
        pygame.gfxdraw.filled_circle(surface, int_pos[0], int_pos[1], radius, color)
        pygame.gfxdraw.aacircle(surface, int_pos[0], int_pos[1], radius, color)

    def close(self):
        pygame.quit()

    def render(self):
        return self._get_observation()


if __name__ == "__main__":
    # This block is for manual play and debugging.
    # It will not be executed by the verification tests.
    # We need to unset the dummy video driver to see the window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")

    obs, info = env.reset()

    # Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tracer Path")
    clock = pygame.time.Clock()

    total_reward = 0.0

    running = True
    while running:
        movement_action = 0  # No-op by default

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement_action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement_action = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement_action = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement_action = 4

        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement_action, space_action, shift_action]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            total_reward = 0.0
            obs, info = env.reset()

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    env.close()