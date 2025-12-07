import gymnasium as gym
import os
import pygame
import math
import os
import pygame


# Ensure Pygame runs headless for automated tests
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame.gfxdraw


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # --- Fixes for Test Failures ---
    game_description = (
        "Swing from a pendulum and release at the right moment to land on "
        "progressively challenging platforms."
    )
    user_guide = (
        "Controls: Press Space to release from the pendulum and attempt to land on a platform."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    GRAVITY_SWING = 0.4
    GRAVITY_FALL = 0.5
    VICTORY_PLATFORMS = 12
    MAX_STEPS = 2500

    # --- Colors ---
    COLOR_BG_TOP = (15, 10, 40)
    COLOR_BG_BOTTOM = (40, 20, 70)
    COLOR_PENDULUM = (0, 200, 255)
    COLOR_PENDULUM_GLOW = (0, 200, 255, 50)
    COLOR_PLATFORM = (255, 150, 0)
    COLOR_PLATFORM_GLOW = (255, 150, 0, 40)
    COLOR_TEXT = (240, 240, 255)
    COLOR_CHASM = (10, 5, 25)
    COLOR_PARTICLE = (255, 255, 180)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        self._create_background()

        # --- State Variables ---
        self.game_state = None
        self.pivot_point = None
        self.pendulum_length = None
        self.pendulum_angle = None
        self.pendulum_angular_velocity = None
        self.pendulum_bob_pos = None
        self.pendulum_fall_velocity = None
        self.momentum = None
        self.platforms = None
        self.platform_fall_speed = None
        self.landed_platform_count = None
        self.particles = None
        self.last_space_state = None
        self.steps = None
        self.game_over = None
        self.game_won = None
        self.event_reward = 0

        # The original code called a validation helper here.
        # It's kept for fidelity, though not part of the standard API.
        # self.validate_implementation() is called after the first reset in the original code.
        # We will initialize state by calling reset() once.
        # The validation helper is called at the end of __init__ in the original code, after reset.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game_state = "swinging"  # "swinging", "falling"
        self.pivot_point = pygame.Vector2(self.WIDTH // 2, 80)
        self.pendulum_length = 120
        self.pendulum_angle = math.pi / 3  # Start with an initial swing
        self.pendulum_angular_velocity = 0
        self.pendulum_bob_pos = pygame.Vector2(0, 0)
        self.pendulum_fall_velocity = pygame.Vector2(0, 0)
        self.momentum = 100.0

        self.platforms = []
        self._spawn_platform(initial=True)
        for _ in range(4):
            self._spawn_platform()

        self.platform_fall_speed = 1.0
        self.landed_platform_count = 0
        self.particles = []
        self.last_space_state = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.event_reward = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        self.event_reward = 0

        space_pressed = action[1] == 1 and self.last_space_state == 0
        self.last_space_state = action[1]

        if not self.game_over:
            self._update_game_state(space_pressed)
            self._update_platforms()
            self._update_particles()

        reward = self._calculate_reward()
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self.momentum < 20:
                reward -= 50  # Loss penalty
            elif self.game_won:
                reward += 100  # Victory bonus

        # Per Gymnasium API, truncated is for time limits, not game-over states.
        # This env uses MAX_STEPS for termination, so truncated is always False.
        truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _update_game_state(self, space_pressed):
        if self.game_state == "swinging":
            self._update_swinging_state(space_pressed)
        elif self.game_state == "falling":
            self._update_falling_state()

    def _update_swinging_state(self, space_pressed):
        # Physics: d²θ/dt² = -(g/L) * sin(θ)
        angular_acceleration = -(self.GRAVITY_SWING / self.pendulum_length) * math.sin(
            self.pendulum_angle
        )
        self.pendulum_angular_velocity += angular_acceleration
        self.pendulum_angle += self.pendulum_angular_velocity

        # Dampen swing over time
        self.pendulum_angular_velocity *= 0.998

        # FIX: The original code incorrectly recalculated momentum from velocity every frame,
        # causing immediate termination. Momentum should be a resource that decays over time.
        self.momentum = max(0, self.momentum - 0.1)

        # Calculate bob position
        self.pendulum_bob_pos.x = self.pivot_point.x + self.pendulum_length * math.sin(
            self.pendulum_angle
        )
        self.pendulum_bob_pos.y = self.pivot_point.y + self.pendulum_length * math.cos(
            self.pendulum_angle
        )

        if space_pressed:
            self.game_state = "falling"
            # Calculate initial velocity vector from angular velocity
            speed = self.pendulum_angular_velocity * self.pendulum_length
            self.pendulum_fall_velocity.x = speed * math.cos(self.pendulum_angle)
            self.pendulum_fall_velocity.y = -speed * math.sin(self.pendulum_angle)

    def _update_falling_state(self):
        # Physics: projectile motion
        self.pendulum_fall_velocity.y += self.GRAVITY_FALL
        self.pendulum_bob_pos += self.pendulum_fall_velocity

        # Collision check
        bob_rect = pygame.Rect(
            self.pendulum_bob_pos.x - 10, self.pendulum_bob_pos.y - 10, 20, 20
        )
        landed = False
        for plat in self.platforms:
            if plat["rect"].colliderect(bob_rect):
                self._handle_successful_landing(plat)
                landed = True
                break

        if not landed and self.pendulum_bob_pos.y > self.HEIGHT + 20:
            self._handle_miss()

    def _handle_successful_landing(self, platform):
        self.game_state = "swinging"
        self.landed_platform_count += 1
        self.event_reward += 5.0

        # Update pivot to new platform
        self.pivot_point = pygame.Vector2(
            platform["rect"].centerx, platform["rect"].centery
        )

        # Reset swing state
        initial_angle_offset = self.pendulum_fall_velocity.x * 0.01
        self.pendulum_angle = max(
            -math.pi / 2.1, min(math.pi / 2.1, initial_angle_offset)
        )
        self.pendulum_angular_velocity = 0
        self.momentum = 100.0

        for _ in range(30):
            self._spawn_particle(self.pendulum_bob_pos)

        if self.landed_platform_count > 0 and self.landed_platform_count % 3 == 0:
            self.platform_fall_speed += 0.05

        if self.landed_platform_count >= self.VICTORY_PLATFORMS:
            self.game_won = True

        self.platforms.remove(platform)
        while len(self.platforms) < 5:
            self._spawn_platform()

    def _handle_miss(self):
        self.game_state = "swinging"
        self.momentum = max(0, self.momentum - 15)
        # FIX: Original code set angle and velocity to 0, stalling the game.
        # This change restarts the swing to allow for recovery.
        self.pendulum_angle = math.pi / 4
        self.pendulum_angular_velocity = 0

    def _update_platforms(self):
        for plat in self.platforms:
            plat["rect"].y += self.platform_fall_speed

        self.platforms = [p for p in self.platforms if p["rect"].top < self.HEIGHT]
        while len(self.platforms) < 5:
            self._spawn_platform()

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _spawn_platform(self, initial=False):
        width = self.np_random.integers(60, 100)
        height = 15

        if initial:
            px = self.pivot_point.x + self.pendulum_length * math.sin(
                self.pendulum_angle
            )
            py = (
                self.pivot_point.y
                + self.pendulum_length * math.cos(self.pendulum_angle)
                + 100
            )
            x = px - width / 2
            y = py
        else:
            min_x = int(self.pivot_point.x - self.pendulum_length * 1.5)
            max_x = int(self.pivot_point.x + self.pendulum_length * 1.5)
            x = self.np_random.integers(
                max(20, min_x), min(self.WIDTH - width - 20, max_x)
            )

            min_y = -150
            if self.platforms:
                min_y = min(p["rect"].y for p in self.platforms) - height - 50
            y = self.np_random.integers(min_y - 150, min_y)

        self.platforms.append({"rect": pygame.Rect(x, y, width, height)})

    def _spawn_particle(self, pos):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        life = self.np_random.integers(20, 40)
        self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "life": life})

    def _calculate_reward(self):
        reward = self.event_reward
        if self.momentum > 80:
            reward += 0.1
        elif self.momentum < 80 and self.game_state == "swinging":
            reward -= 0.2
        return reward

    def _check_termination(self):
        if self.game_won:
            return True
        if self.momentum < 20 and self.game_state == "swinging":
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.blit(self.background_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {
            "score": self.landed_platform_count,
            "steps": self.steps,
            "momentum": self.momentum,
            "platform_speed": self.platform_fall_speed,
        }

    def _create_background(self):
        self.background_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.background_surface, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        pygame.draw.polygon(
            self.screen,
            self.COLOR_CHASM,
            [(0, 0), (20, 0), (70, self.HEIGHT), (0, self.HEIGHT)],
        )
        pygame.draw.polygon(
            self.screen,
            self.COLOR_CHASM,
            [
                (self.WIDTH, 0),
                (self.WIDTH - 20, 0),
                (self.WIDTH - 70, self.HEIGHT),
                (self.WIDTH, self.HEIGHT),
            ],
        )

        self._render_particles()
        self._render_platforms()
        self._render_pendulum()

        if self.game_over:
            self._render_game_over()

    def _render_pendulum(self):
        bob_pos_int = (int(self.pendulum_bob_pos.x), int(self.pendulum_bob_pos.y))

        if self.game_state == "swinging":
            pivot_pos_int = (int(self.pivot_point.x), int(self.pivot_point.y))
            pygame.draw.aaline(
                self.screen, self.COLOR_PENDULUM, pivot_pos_int, bob_pos_int, 1
            )
            try:
                max_angle_cos = (
                    self.pivot_point.y
                    - (
                        self.pivot_point.y
                        + self.pendulum_length * math.cos(self.pendulum_angle)
                    )
                    + 0.5
                    * self.pendulum_angular_velocity**2
                    * self.pendulum_length
                ) / self.pendulum_length
                if -1 <= max_angle_cos <= 1:
                    max_angle = math.acos(max_angle_cos)
                    arc_rect = pygame.Rect(
                        pivot_pos_int[0] - self.pendulum_length,
                        pivot_pos_int[1] - self.pendulum_length,
                        self.pendulum_length * 2,
                        self.pendulum_length * 2,
                    )
                    start_angle = math.pi / 2 - max_angle
                    end_angle = math.pi / 2 + max_angle
                    if start_angle < end_angle:
                        pygame.draw.arc(
                            self.screen,
                            self.COLOR_PENDULUM_GLOW,
                            arc_rect,
                            start_angle,
                            end_angle,
                            1,
                        )
            except (ValueError, TypeError):
                pass

        pygame.gfxdraw.filled_circle(
            self.screen, bob_pos_int[0], bob_pos_int[1], 16, self.COLOR_PENDULUM_GLOW
        )
        pygame.gfxdraw.filled_circle(
            self.screen, bob_pos_int[0], bob_pos_int[1], 12, self.COLOR_PENDULUM_GLOW
        )

        pygame.gfxdraw.filled_circle(
            self.screen, bob_pos_int[0], bob_pos_int[1], 10, self.COLOR_PENDULUM
        )
        pygame.gfxdraw.aacircle(
            self.screen, bob_pos_int[0], bob_pos_int[1], 10, self.COLOR_PENDULUM
        )

    def _render_platforms(self):
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat["rect"], border_radius=4)
            glow_rect = plat["rect"].inflate(8, 8)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, self.COLOR_PLATFORM_GLOW, s.get_rect(), border_radius=8)
            self.screen.blit(s, glow_rect.topleft)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 40))))
            color = (*self.COLOR_PARTICLE, alpha)
            size = int(max(1, 5 * (p["life"] / 40)))
            pos_int = (int(p["pos"].x), int(p["pos"].y))
            pygame.draw.circle(self.screen, color, pos_int, size)

    def _render_ui(self):
        mom_text = f"Momentum: {int(self.momentum)}%"
        mom_surf = self.font_ui.render(mom_text, True, self.COLOR_TEXT)
        self.screen.blit(mom_surf, (15, 10))

        score_text = f"Landed: {self.landed_platform_count}/{self.VICTORY_PLATFORMS}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 15, 10))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))

        text = "VICTORY!" if self.game_won else "GAME OVER"
        text_surf = self.font_game_over.render(text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))

        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()


# Example usage to run and visualize the game
if __name__ == "__main__":
    # To run with visualization, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()

    action = [0, 0, 0]

    pygame.display.set_caption("Pendulum Swing")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()

    running = True
    terminated = False

    print("\n--- Controls ---")
    print("Spacebar: Release Pendulum")
    print("R: Reset Environment")
    print("Q: Quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_SPACE:
                    action[1] = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    action[1] = 0

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()