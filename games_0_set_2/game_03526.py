import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, manage your boost, and complete three laps before you crash out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

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
        self.width, self.height = 640, 400
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()

        # Colors
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_TRACK = (100, 100, 110)
        self.COLOR_WALL = (60, 60, 70)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLAYER_GLOW = (255, 100, 100, 64)
        self.COLOR_BOOST = (255, 220, 0)
        self.COLOR_SKID = (20, 20, 20)
        self.COLOR_UI_TEXT = (240, 240, 240)

        # Fonts
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 32)

        # Game constants
        self.max_steps = 1800  # 60 seconds at 30fps
        self.laps_to_win = 3
        self.crashes_to_lose = 3

        # Track layout
        self.track_outer_rect = pygame.Rect(50, 50, self.width - 100, self.height - 100)
        self.track_inner_rect = pygame.Rect(150, 120, self.width - 300, self.height - 240)
        self.track_border_radius = 60

        # Checkpoints for lap counting
        self.checkpoints = [
            pygame.Rect(self.width / 2, self.track_inner_rect.bottom, 2, self.track_outer_rect.bottom - self.track_inner_rect.bottom),  # Start/Finish
            pygame.Rect(self.track_inner_rect.right, self.height / 2, self.track_outer_rect.right - self.track_inner_rect.right, 2),  # Right
            pygame.Rect(self.width / 2, self.track_outer_rect.top, 2, self.track_inner_rect.top - self.track_outer_rect.top),  # Top
            pygame.Rect(self.track_outer_rect.left, self.height / 2, self.track_inner_rect.left - self.track_outer_rect.left, 2)  # Left
        ]

        # Kart physics parameters
        self.car_dims = (12, 22)
        self.accel_rate = 0.15
        self.brake_rate = 0.3
        self.max_speed = 3.5
        self.max_boost_speed = 7.0
        self.turn_speed = 0.05
        self.friction = 0.98
        self.drift_friction = 0.96
        self.drift_turn_multiplier = 1.5
        self.drift_slide_factor = 0.6
        self.wall_bounce = -0.5

        # Initialize state variables
        self.particles = []
        self.skid_marks = []
        # self.reset() is called by the gym wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Seed python's random module
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = None  # "WIN" or "LOSE"

        self.laps = 0
        self.crashes = 0
        self.next_checkpoint = 0

        self.car_pos = pygame.Vector2(self.width / 2, self.track_inner_rect.bottom + 30)
        self.car_angle = -math.pi / 2  # Pointing up
        self.car_speed = 0
        self.car_velocity = pygame.Vector2(0, 0)

        self.particles.clear()
        self.skid_marks.clear()

        return self._get_observation(), self._get_info()

    def _reset_car_after_crash(self):
        # sfx: car_crash_sound()
        self.car_pos = pygame.Vector2(self.width / 2, self.track_inner_rect.bottom + 30)
        self.car_angle = -math.pi / 2
        self.car_speed = 0
        self.car_velocity.x = self.car_velocity.y = 0

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.1  # Survival reward

        # --- Input and Physics ---
        is_turning = movement in [3, 4]

        if shift_held:
            # sfx: tire_squeal_loop()
            current_friction = self.drift_friction
            turn_rate = self.turn_speed * self.drift_turn_multiplier
            if is_turning:
                reward += 0.05  # Small reward for active drifting
            if not space_held:
                reward -= 0.2  # Penalty for drifting without boost
        else:
            current_friction = self.friction
            turn_rate = self.turn_speed

        if is_turning:
            turn_direction = 1 if movement == 3 else -1  # 3=left, 4=right
            self.car_angle += turn_direction * turn_rate * (self.car_speed / self.max_speed)

        if movement == 1:  # Accelerate
            self.car_speed += self.accel_rate
        elif movement == 2:  # Brake
            self.car_speed -= self.brake_rate

        if space_held:
            # sfx: boost_sound()
            self.car_speed += self.accel_rate * 2.5
            max_speed = self.max_boost_speed
            self._create_particles(self.car_pos, self.car_angle, count=2, p_type='boost')
        else:
            max_speed = self.max_speed

        self.car_speed *= current_friction
        self.car_speed = max(0, min(self.car_speed, max_speed))

        # Update velocity and position
        heading_vector = pygame.Vector2(math.cos(self.car_angle), math.sin(self.car_angle))
        self.car_velocity = heading_vector * self.car_speed

        if shift_held and is_turning:
            slide_vector = pygame.Vector2(-heading_vector.y, heading_vector.x)
            self.car_velocity += slide_vector * turn_direction * self.drift_slide_factor * (self.car_speed / max_speed)

        self.car_pos += self.car_velocity

        # --- Effects ---
        if shift_held and self.car_speed > 1.0 and is_turning:
            self._add_skid_mark()
        self._update_particles()

        # --- Collisions ---
        car_rect = self._get_car_rect()
        if not self._is_on_track(car_rect):
            reward -= 5
            self.crashes += 1
            self._create_particles(self.car_pos, self.car_angle, count=20, p_type='spark')
            self._reset_car_after_crash()

        # --- Lap Logic ---
        if car_rect.colliderect(self.checkpoints[self.next_checkpoint]):
            self.next_checkpoint = (self.next_checkpoint + 1) % len(self.checkpoints)
            if self.next_checkpoint == 0:
                # sfx: lap_complete_sound()
                self.laps += 1
                reward += 10  # Changed from 1 to 10 for more significance

        # --- Termination ---
        self.steps += 1
        terminated = False
        if self.laps >= self.laps_to_win:
            reward += 50
            terminated = True
            self.win_state = "WIN"
        elif self.crashes >= self.crashes_to_lose:
            reward -= 50
            terminated = True
            self.win_state = "LOSE"
        elif self.steps >= self.max_steps:
            terminated = True
            self.win_state = "TIME UP"

        self.game_over = terminated
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_on_track(self, car_rect):
        # A simple but effective collision check using distance from rounded rect centers
        def check_poly(poly, rect, radius):
            for p in poly:
                # Check main body of the rect
                if rect.collidepoint(p):
                    continue
                # Check rounded corners
                if math.hypot(p[0] - rect.left, p[1] - rect.top) < radius or \
                   math.hypot(p[0] - rect.right, p[1] - rect.top) < radius or \
                   math.hypot(p[0] - rect.left, p[1] - rect.bottom) < radius or \
                   math.hypot(p[0] - rect.right, p[1] - rect.bottom) < radius:
                    continue
                return False
            return True

        car_poly = self._get_car_poly()
        on_outer = check_poly(car_poly, self.track_outer_rect, self.track_border_radius)
        on_inner = check_poly(car_poly, self.track_inner_rect, self.track_border_radius)

        return on_outer and not on_inner

    def _get_car_poly(self):
        w, h = self.car_dims[0] / 2, self.car_dims[1] / 2
        corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
        cos_a, sin_a = math.cos(self.car_angle), math.sin(self.car_angle)

        poly = []
        for x, y in corners:
            rotated_x = x * cos_a - y * sin_a + self.car_pos.x
            rotated_y = x * sin_a + y * cos_a + self.car_pos.y
            poly.append((rotated_x, rotated_y))
        return poly

    def _get_car_rect(self):
        poly = self._get_car_poly()
        min_x = min(p[0] for p in poly)
        max_x = max(p[0] for p in poly)
        min_y = min(p[1] for p in poly)
        max_y = max(p[1] for p in poly)
        return pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    def _add_skid_mark(self):
        w, h = self.car_dims[0] / 2, self.car_dims[1] / 2
        cos_a, sin_a = math.cos(self.car_angle), math.sin(self.car_angle)

        rear_axle_offset = h * 0.8

        for side in [-1, 1]:
            x, y = w * side, rear_axle_offset
            rotated_x = x * cos_a - y * sin_a + self.car_pos.x
            rotated_y = x * sin_a + y * cos_a + self.car_pos.y
            self.skid_marks.append([(rotated_x, rotated_y), 100])  # pos, life

        self.skid_marks = self.skid_marks[-200:]  # Limit total skid marks

    def _create_particles(self, pos, angle, count, p_type):
        for _ in range(count):
            if p_type == 'boost':
                vel = pygame.Vector2(-math.cos(angle), -math.sin(angle)) * random.uniform(2, 4)
                vel.rotate_ip(random.uniform(-15, 15))
                life = random.randint(10, 20)
                color = self.COLOR_BOOST
                radius = random.randint(3, 6)
            elif p_type == 'spark':
                vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * random.uniform(2, 8)
                life = random.randint(15, 30)
                color = random.choice([(255, 255, 255), (255, 200, 0), (255, 100, 0)])
                radius = random.randint(1, 3)
            # Create a copy of the position vector, as pygame.Vector2 is mutable
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'life': life, 'max_life': life, 'color': color, 'radius': radius})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Track
        pygame.draw.rect(self.screen, self.COLOR_TRACK, self.track_outer_rect, border_radius=self.track_border_radius)
        pygame.draw.rect(self.screen, self.COLOR_BG, self.track_inner_rect, border_radius=self.track_border_radius)

        # Skid marks
        for i in range(len(self.skid_marks)):
            self.skid_marks[i][1] -= 1  # Decrease life
            if i > 0 and self.skid_marks[i - 1][1] > 0 and math.hypot(self.skid_marks[i][0][0] - self.skid_marks[i - 1][0][0], self.skid_marks[i][0][1] - self.skid_marks[i - 1][0][1]) < 20:
                alpha = max(0, min(255, int(self.skid_marks[i][1] * 2.55)))
                temp_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.draw.line(temp_surf, self.COLOR_SKID + (alpha,), self.skid_marks[i - 1][0], self.skid_marks[i][0], width=3)
                self.screen.blit(temp_surf, (0, 0))
        self.skid_marks = [skid for skid in self.skid_marks if skid[1] > 0]

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color']
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), (*color, alpha))
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), (*color, alpha))

        # Car
        car_poly = self._get_car_poly()

        # Glow
        glow_poly = self._get_car_poly()  # Recalculate for potential scaling if needed
        pygame.gfxdraw.filled_polygon(self.screen, glow_poly, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aapolygon(self.screen, glow_poly, self.COLOR_PLAYER_GLOW)

        # Main body
        pygame.gfxdraw.filled_polygon(self.screen, car_poly, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, car_poly, self.COLOR_PLAYER)

    def _render_ui(self):
        # Laps
        lap_text = self.font_medium.render(f"LAP: {min(self.laps, self.laps_to_win)} / {self.laps_to_win}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lap_text, (20, 20))

        # Crashes
        crash_text = self.font_medium.render(f"CRASHES: {self.crashes} / {self.crashes_to_lose}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crash_text, (self.width - crash_text.get_width() - 20, 20))

        # Game Over Message
        if self.game_over:
            if self.win_state == "WIN":
                msg = "YOU WIN!"
                color = self.COLOR_BOOST
            else:
                msg = "GAME OVER"
                color = self.COLOR_PLAYER

            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.laps,
            "crashes": self.crashes,
        }

    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # To play the game manually, you can run this file.
    # This requires pygame to be installed with display support.
    # To do so, you may need to unset the SDL_VIDEODRIVER variable.
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)

    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Arcade Racer")
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0

    print(env.game_description)
    print(env.user_guide)

    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0  # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        clock.tick(30)  # Run at 30 FPS

    print(f"Game Over! Final Score: {total_reward:.2f}")
    print(f"Info: {info}")

    env.close()