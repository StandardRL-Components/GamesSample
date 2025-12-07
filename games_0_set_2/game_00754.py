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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrows to draw track segments. Hold Shift for shorter "
        "segments and Space for longer ones. Guide the sled to the finish line!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A physics-based puzzle racer. Draw the track for your sled to ride on "
        "and reach the finish line as fast as possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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

        # Colors
        self.COLOR_BG = (210, 230, 255)  # Light Sky Blue
        self.COLOR_TRACK = (20, 20, 20)
        self.COLOR_SLED = (230, 50, 50)
        self.COLOR_SLED_GLOW = (255, 100, 100)
        self.COLOR_START = (50, 200, 50)
        self.COLOR_FINISH = (200, 50, 50)
        self.COLOR_CHECKPOINT = (255, 165, 0)  # Orange
        self.COLOR_PEN = (255, 215, 0, 128)  # Gold, semi-transparent
        self.COLOR_TEXT = (10, 10, 10)
        self.COLOR_PARTICLE = (255, 255, 255)

        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        # Game constants
        self.MAX_STEPS = 2000
        self.GRAVITY = pygame.math.Vector2(0, 0.15)
        self.SLED_SIZE = pygame.math.Vector2(16, 8)
        self.SLED_RADIUS = self.SLED_SIZE.x / 2  # For collision
        self.FRICTION = 0.995
        self.BOUNCE = 0.6
        self.LINE_LENGTH_NORMAL = 30
        self.LINE_LENGTH_SHORT = 15
        self.LINE_LENGTH_LONG = 60
        self.START_X = 50
        self.FINISH_X = self.WIDTH - 50

        # State variables will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_reason = ""
        self.sled_pos = pygame.math.Vector2(0, 0)
        self.sled_vel = pygame.math.Vector2(0, 0)
        self.track_segments = []
        self.pen_pos = pygame.math.Vector2(0, 0)
        self.particles = []
        self.checkpoints = []
        self.cleared_checkpoints = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_reason = ""

        # Sled state
        start_y = self.HEIGHT - 100
        self.sled_pos = pygame.math.Vector2(self.START_X, start_y - self.SLED_SIZE.y)
        self.sled_vel = pygame.math.Vector2(0, 0)

        # Track state
        self.track_segments = []
        initial_platform_start = pygame.math.Vector2(20, start_y)
        initial_platform_end = pygame.math.Vector2(self.START_X + 40, start_y)
        self.track_segments.append((initial_platform_start, initial_platform_end))
        self.pen_pos = initial_platform_end.copy()

        # Particles
        self.particles = []

        # Checkpoints
        self.checkpoints = [
            pygame.math.Vector2(self.WIDTH / 3, self.HEIGHT - 50),
            pygame.math.Vector2(2 * self.WIDTH / 3, self.HEIGHT - 150),
        ]
        self.cleared_checkpoints = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0

        # 1. Handle player input (drawing)
        self._handle_drawing(action)

        # 2. Update physics
        self._update_physics()

        # 3. Update particles
        self._update_particles()

        # 4. Check game state and calculate rewards
        terminated, term_reward = self._check_termination_and_get_terminal_reward()
        reward += term_reward
        reward += self._calculate_step_reward()

        self.score += reward
        self.steps += 1

        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True
            self.game_over_reason = "TIME UP"
            reward -= 10  # Small penalty for running out of time

        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_drawing(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement == 0:  # No-op for drawing
            return

        direction_map = {
            1: pygame.math.Vector2(0, -1),  # Up
            2: pygame.math.Vector2(0, 1),  # Down
            3: pygame.math.Vector2(-1, 0),  # Left
            4: pygame.math.Vector2(1, 0),  # Right
        }
        direction = direction_map[movement]

        if space_held:
            length = self.LINE_LENGTH_LONG
        elif shift_held:
            length = self.LINE_LENGTH_SHORT
        else:
            length = self.LINE_LENGTH_NORMAL

        new_end_point = self.pen_pos + direction * length

        # Clamp to screen bounds
        new_end_point.x = max(0, min(self.WIDTH, new_end_point.x))
        new_end_point.y = max(0, min(self.HEIGHT, new_end_point.y))

        # Add segment and update pen position
        self.track_segments.append((self.pen_pos.copy(), new_end_point))
        self.pen_pos = new_end_point

    def _update_physics(self):
        # Apply gravity
        self.sled_vel += self.GRAVITY

        # Store pre-collision position
        old_pos = self.sled_pos.copy()

        # Move sled
        self.sled_pos += self.sled_vel

        # Collision detection and response
        for p1, p2 in self.track_segments:
            closest_point, distance_sq, on_segment = self._get_closest_point_on_segment(
                self.sled_pos, p1, p2
            )

            if distance_sq < self.SLED_RADIUS**2:
                distance = math.sqrt(distance_sq)

                # Nudge sled out of collision
                overlap = self.SLED_RADIUS - distance
                collision_normal = (self.sled_pos - closest_point).normalize()
                self.sled_pos += collision_normal * overlap

                # Reflect velocity
                velocity_component = self.sled_vel.dot(collision_normal)
                self.sled_vel -= (1 + self.BOUNCE) * velocity_component * collision_normal

                # Apply friction
                self.sled_vel *= self.FRICTION
                # sfx: // sled scrape sound
                break  # Handle one collision per frame for stability

    def _get_closest_point_on_segment(self, point, p1, p2):
        line_vec = p2 - p1
        if line_vec.length_squared() == 0:
            return p1, point.distance_squared_to(p1), True

        point_vec = point - p1
        t = point_vec.dot(line_vec) / line_vec.length_squared()

        on_segment = True
        if t < 0:
            closest_point = p1
            on_segment = False
        elif t > 1:
            closest_point = p2
            on_segment = False
        else:
            closest_point = p1 + t * line_vec

        return closest_point, point.distance_squared_to(closest_point), on_segment

    def _update_particles(self):
        # Add new particles
        if self.sled_vel.length() > 1:
            for _ in range(2):
                self.particles.append(
                    {
                        "pos": self.sled_pos.copy()
                        + pygame.math.Vector2(
                            self.np_random.uniform(-4, 4), self.np_random.uniform(-4, 4)
                        ),
                        "vel": self.sled_vel.copy()
                        * self.np_random.uniform(0.1, 0.3)
                        * -1,
                        "life": self.np_random.integers(20, 40),
                        "size": self.np_random.uniform(2, 5),
                    }
                )

        # Update and remove old particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["size"] *= 0.95

    def _check_termination_and_get_terminal_reward(self):
        # Crash condition
        if not (0 < self.sled_pos.x < self.WIDTH and 0 < self.sled_pos.y < self.HEIGHT):
            self.game_over_reason = "CRASH!"
            # sfx: // explosion sound
            return True, -50

        # Win condition
        if self.sled_pos.x > self.FINISH_X:
            self.game_over_reason = "FINISH!"
            # sfx: // victory fanfare
            return True, 100

        return False, 0

    def _calculate_step_reward(self):
        reward = 0
        # Reward for forward movement, penalize backward
        reward += 0.1 if self.sled_vel.x > 0 else -0.1

        # Penalize for being slow
        if self.sled_vel.length() < 0.5 and self.steps > 50:
            reward -= 1.0

        # Reward for clearing checkpoints
        for i, cp_pos in enumerate(self.checkpoints):
            if i not in self.cleared_checkpoints and self.sled_pos.x > cp_pos.x:
                self.cleared_checkpoints.append(i)
                reward += 5
                # sfx: // checkpoint sound

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Start/Finish/Checkpoint lines
        pygame.draw.line(
            self.screen, self.COLOR_START, (self.START_X, 0), (self.START_X, self.HEIGHT), 3
        )
        pygame.draw.line(
            self.screen,
            self.COLOR_FINISH,
            (self.FINISH_X, 0),
            (self.FINISH_X, self.HEIGHT),
            3,
        )
        for i, cp_pos in enumerate(self.checkpoints):
            color = (
                self.COLOR_START
                if i in self.cleared_checkpoints
                else self.COLOR_CHECKPOINT
            )
            pygame.draw.line(
                self.screen, color, (cp_pos.x, 0), (cp_pos.x, self.HEIGHT), 2
            )

        # Track
        for p1, p2 in self.track_segments:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2, 3)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 40.0))
            color = self.COLOR_PARTICLE + (alpha,)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p["pos"].x), int(p["pos"].y), int(p["size"]), color
            )

        # Pen cursor
        pen_surface = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.circle(pen_surface, self.COLOR_PEN, (10, 10), 8)
        self.screen.blit(
            pen_surface, (int(self.pen_pos.x - 10), int(self.pen_pos.y - 10))
        )

        # Sled
        sled_rect = pygame.Rect(0, 0, self.SLED_SIZE.x, self.SLED_SIZE.y)
        sled_rect.center = self.sled_pos

        # Glow effect
        glow_radius = int(self.SLED_SIZE.x * 0.8)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            glow_surface, self.COLOR_SLED_GLOW + (50,), (glow_radius, glow_radius), glow_radius
        )
        self.screen.blit(
            glow_surface,
            (int(sled_rect.centerx - glow_radius), int(sled_rect.centery - glow_radius)),
        )

        pygame.draw.rect(self.screen, self.COLOR_SLED, sled_rect, border_radius=3)

    def _render_ui(self):
        speed_text = f"Speed: {self.sled_vel.length():.1f}"
        time_text = f"Time: {self.steps / 30:.1f}s"  # Assuming 30fps for display
        score_text = f"Score: {self.score:.1f}"

        speed_surf = self.font_ui.render(speed_text, True, self.COLOR_TEXT)
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)

        self.screen.blit(speed_surf, (10, 10))
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))
        self.screen.blit(score_surf, (10, 30))

        if self.game_over:
            reason_surf = self.font_big.render(
                self.game_over_reason, True, self.COLOR_TEXT
            )
            reason_rect = reason_surf.get_rect(
                center=(self.WIDTH / 2, self.HEIGHT / 2)
            )
            self.screen.blit(reason_surf, reason_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sled_pos": (self.sled_pos.x, self.sled_pos.y),
            "sled_vel": (self.sled_vel.x, self.sled_vel.y),
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # We need to create a display for manual play, which is not headless.
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Sled Drawer")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement, space, shift = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]

        # A no-op action [0,0,0] will just advance physics.
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            # Auto-reset after a delay
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30)  # Control the speed of manual play

    env.close()