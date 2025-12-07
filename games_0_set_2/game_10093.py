import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete
import os
import pygame


# Set the SDL_VIDEODRIVER to "dummy" to run Pygame headlessly
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Particle:
    """A simple particle class for visual effects."""
    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = random.randint(20, 40)  # Frames
        self.color = color
        self.radius = random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        self.radius *= 0.97 # Shrink over time

    def draw(self, surface):
        if self.lifetime > 0 and self.radius > 1:
            pos = (int(self.x), int(self.y))
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(self.radius), self.color)


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent must balance a scale using three levers.
    The goal is to achieve 10 "perfect balances" within 120 seconds.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Balance a delicate scale by applying force with three different levers. "
        "Achieve a set number of 'perfect balances' before time runs out to win."
    )
    user_guide = (
        "Controls: Use ↑/↓ for Lever 1, ←/→ for Lever 2, and Space/Shift for Lever 3. "
        "Keep the scale balanced to score points."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 120 * FPS  # 120 seconds

    # Colors
    COLOR_BG = (15, 18, 28)
    COLOR_SCALE = (200, 205, 220)
    COLOR_FULCRUM = (150, 155, 170)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_LEVER_BG = (30, 35, 50)
    COLOR_LEVER_FILL = (255, 190, 0) # Yellow
    COLOR_BALANCED = (0, 255, 120)
    COLOR_UNBALANCED = (255, 80, 80)
    COLOR_PARTICLE = (255, 215, 0) # Gold

    # Physics & Game Rules
    LEVER_ACCEL = 0.2
    LEVER_DAMPING = 0.97
    MAX_LEVER_SPEED = 5.0
    SCALE_TORQUE_FACTOR = 0.015
    SCALE_RESTORING_FORCE = 0.001
    SCALE_DAMPING = 0.985
    ANGLE_LIMIT = 60.0  # degrees
    COLLAPSE_ANGLE = 45.0  # degrees
    BALANCE_ANGLE_TOLERANCE = 5.0 # For continuous reward
    PERFECT_BALANCE_ANGLE = 1.0
    PERFECT_BALANCE_VELOCITY = 0.05
    WIN_CONDITION_BALANCES = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 20)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.balances_achieved = 0
        self.lever_speeds = [0.0, 0.0, 0.0]
        self.scale_angle = 0.0
        self.scale_angular_velocity = 0.0
        self.particles = []
        self.in_perfect_balance_zone = False
        self.just_collapsed = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS
        self.balances_achieved = 0
        self.lever_speeds = [0.0, 0.0, 0.0]
        # Start with a slight random tilt for challenge
        self.scale_angle = self.np_random.uniform(-5.0, 5.0)
        self.scale_angular_velocity = 0.0
        self.particles = []
        self.in_perfect_balance_zone = False
        self.just_collapsed = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1

        # 1. Update game state based on action
        self._apply_action(action)
        self._update_physics()
        self._update_particles()

        # 2. Calculate reward
        reward = self._calculate_reward()
        self.score += reward

        # 3. Check for termination
        terminated = self._check_termination()
        truncated = False # This env doesn't truncate
        if terminated:
            self.game_over = True
            # Apply terminal reward
            if self.balances_achieved >= self.WIN_CONDITION_BALANCES:
                reward += 100
                self.score += 100

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _apply_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Lever 1: Controlled by Up/Down
        if movement == 1: self.lever_speeds[0] += self.LEVER_ACCEL
        if movement == 2: self.lever_speeds[0] -= self.LEVER_ACCEL

        # Lever 2: Controlled by Left/Right
        if movement == 3: self.lever_speeds[1] += self.LEVER_ACCEL
        if movement == 4: self.lever_speeds[1] -= self.LEVER_ACCEL

        # Lever 3: Controlled by Space/Shift
        if space_held: self.lever_speeds[2] += self.LEVER_ACCEL
        if shift_held: self.lever_speeds[2] -= self.LEVER_ACCEL

    def _update_physics(self):
        # Apply damping and clamp lever speeds
        for i in range(3):
            self.lever_speeds[i] *= self.LEVER_DAMPING
            self.lever_speeds[i] = np.clip(self.lever_speeds[i], -self.MAX_LEVER_SPEED, self.MAX_LEVER_SPEED)

        # Calculate torque from levers
        # Lever 0 pushes left side down (negative angle), Lever 1 pushes right side down (positive angle)
        # Lever 2 adds a general rotational force
        torque_from_levers = (self.lever_speeds[1] - self.lever_speeds[0]) * self.SCALE_TORQUE_FACTOR
        torque_from_spin = self.lever_speeds[2] * self.SCALE_TORQUE_FACTOR * 0.5

        # Calculate restoring torque (gravity pulling it back to center)
        restoring_torque = -math.sin(math.radians(self.scale_angle)) * self.SCALE_RESTORING_FORCE

        # Update angular velocity
        net_torque = torque_from_levers + torque_from_spin + restoring_torque
        self.scale_angular_velocity += net_torque
        self.scale_angular_velocity *= self.SCALE_DAMPING

        # Update angle
        self.scale_angle += self.scale_angular_velocity
        self.scale_angle = np.clip(self.scale_angle, -self.ANGLE_LIMIT, self.ANGLE_LIMIT)

    def _calculate_reward(self):
        reward = 0

        # Continuous reward for being near balance
        if abs(self.scale_angle) < self.BALANCE_ANGLE_TOLERANCE:
            reward += 0.01

        # Event-based reward for "perfect balance"
        is_perfect = (abs(self.scale_angle) < self.PERFECT_BALANCE_ANGLE and
                      abs(self.scale_angular_velocity) < self.PERFECT_BALANCE_VELOCITY)

        if is_perfect and not self.in_perfect_balance_zone:
            # Entered a perfect balance state
            self.in_perfect_balance_zone = True
            self.balances_achieved += 1
            avg_speed = (abs(self.lever_speeds[0]) + abs(self.lever_speeds[1]) + abs(self.lever_speeds[2])) / 3.0
            balance_reward = 5.0 + (avg_speed / self.MAX_LEVER_SPEED) * 10.0
            reward += balance_reward
            self._spawn_particles(20)

        elif not is_perfect and self.in_perfect_balance_zone:
            # Exited the perfect balance state
            self.in_perfect_balance_zone = False

        # Event-based penalty for collapsing
        is_collapsed = abs(self.scale_angle) >= self.COLLAPSE_ANGLE
        if is_collapsed and not self.just_collapsed:
            reward -= 10.0
            self.just_collapsed = True
            # Push it back slightly and kill velocity to avoid continuous penalty
            self.scale_angle = math.copysign(self.COLLAPSE_ANGLE - 1, self.scale_angle)
            self.scale_angular_velocity = 0
        elif not is_collapsed:
            self.just_collapsed = False

        return reward

    def _check_termination(self):
        win_condition = self.balances_achieved >= self.WIN_CONDITION_BALANCES
        loss_condition = self.timer <= 0
        return win_condition or loss_condition

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

    def _spawn_particles(self, count):
        for _ in range(count):
            self.particles.append(Particle(self.WIDTH / 2, self.HEIGHT / 2 + 30, self.COLOR_PARTICLE))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": int(self.score),
            "steps": self.steps,
            "timer": self.timer,
            "balances_achieved": self.balances_achieved,
            "scale_angle": self.scale_angle,
        }

    def _render_game(self):
        # --- Draw Scale Base and Fulcrum ---
        base_points = [
            (self.WIDTH / 2 - 50, self.HEIGHT),
            (self.WIDTH / 2 + 50, self.HEIGHT),
            (self.WIDTH / 2, self.HEIGHT - 70)
        ]
        pygame.gfxdraw.filled_trigon(self.screen, int(base_points[0][0]), int(base_points[0][1]),
                                     int(base_points[1][0]), int(base_points[1][1]),
                                     int(base_points[2][0]), int(base_points[2][1]), self.COLOR_FULCRUM)

        fulcrum_pos = (int(self.WIDTH / 2), int(self.HEIGHT / 2 + 30))
        pygame.gfxdraw.filled_circle(self.screen, fulcrum_pos[0], fulcrum_pos[1], 10, self.COLOR_SCALE)
        pygame.gfxdraw.aacircle(self.screen, fulcrum_pos[0], fulcrum_pos[1], 10, self.COLOR_BG)

        # --- Draw Scale Beam ---
        beam_length = self.WIDTH * 0.7
        half_beam = beam_length / 2
        angle_rad = math.radians(self.scale_angle)
        
        start_pos = (
            fulcrum_pos[0] - half_beam * math.cos(angle_rad),
            fulcrum_pos[1] - half_beam * math.sin(angle_rad)
        )
        end_pos = (
            fulcrum_pos[0] + half_beam * math.cos(angle_rad),
            fulcrum_pos[1] + half_beam * math.sin(angle_rad)
        )

        # Draw thick anti-aliased line
        pygame.draw.line(self.screen, self.COLOR_SCALE, start_pos, end_pos, 12)
        
        # Draw end caps
        pygame.gfxdraw.filled_circle(self.screen, int(start_pos[0]), int(start_pos[1]), 10, self.COLOR_SCALE)
        pygame.gfxdraw.filled_circle(self.screen, int(end_pos[0]), int(end_pos[1]), 10, self.COLOR_SCALE)
        pygame.gfxdraw.aacircle(self.screen, int(start_pos[0]), int(start_pos[1]), 10, self.COLOR_BG)
        pygame.gfxdraw.aacircle(self.screen, int(end_pos[0]), int(end_pos[1]), 10, self.COLOR_BG)

        # --- Draw Particles ---
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # --- Balance Indicator ---
        is_balanced = abs(self.scale_angle) < self.PERFECT_BALANCE_ANGLE
        indicator_color = self.COLOR_BALANCED if is_balanced else self.COLOR_UNBALANCED
        
        # Glow effect
        glow_radius = 20 if is_balanced else 15
        glow_color = (*indicator_color, 60) # RGBA with alpha
        temp_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surface, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surface, (self.WIDTH/2 - glow_radius, 20 - glow_radius))

        pygame.gfxdraw.filled_circle(self.screen, int(self.WIDTH/2), 20, 10, indicator_color)
        pygame.gfxdraw.aacircle(self.screen, int(self.WIDTH/2), 20, 10, self.COLOR_BG)

        # --- Timer ---
        seconds_left = max(0, self.timer // self.FPS)
        timer_text = f"{seconds_left // 60:02d}:{seconds_left % 60:02d}"
        text_surface = self.font_large.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (self.WIDTH/2 - text_surface.get_width()/2, 50))

        # --- Score ---
        score_text = f"SCORE: {int(self.score)}"
        text_surface = self.font_medium.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (20, 20))

        # --- Balances Achieved ---
        balances_text = f"BALANCES: {self.balances_achieved} / {self.WIN_CONDITION_BALANCES}"
        text_surface = self.font_medium.render(balances_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (self.WIDTH - text_surface.get_width() - 20, 20))

        # --- Lever Indicators ---
        lever_width, lever_height = 20, 100
        lever_y = self.HEIGHT - lever_height - 20
        lever_spacing = 30

        for i in range(3):
            lever_x = 20 + i * (lever_width + lever_spacing)
            # Background
            pygame.draw.rect(self.screen, self.COLOR_LEVER_BG, (lever_x, lever_y, lever_width, lever_height), border_radius=4)
            # Fill
            fill_ratio = self.lever_speeds[i] / self.MAX_LEVER_SPEED
            fill_height = abs(fill_ratio * (lever_height / 2))
            
            if fill_ratio > 0:
                fill_y = lever_y + (lever_height / 2) - fill_height
            else:
                fill_y = lever_y + (lever_height / 2)
            
            fill_rect = (lever_x + 2, fill_y, lever_width - 4, fill_height)
            pygame.draw.rect(self.screen, self.COLOR_LEVER_FILL, fill_rect, border_radius=3)
            # Center line
            pygame.draw.line(self.screen, self.COLOR_LEVER_BG, (lever_x, lever_y + lever_height/2), (lever_x + lever_width, lever_y + lever_height/2), 2)
            # Label
            label_surface = self.font_small.render(f"L{i+1}", True, self.COLOR_UI_TEXT)
            self.screen.blit(label_surface, (lever_x + lever_width/2 - label_surface.get_width()/2, lever_y - 20))

    def close(self):
        pygame.font.quit()
        pygame.quit()

# Example of how to run the environment for human play
if __name__ == '__main__':
    # Important: Unset the dummy driver to see the display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # --- Pygame setup for human play ---
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Balance Scale Environment")
    clock = pygame.time.Clock()
    running = True

    while running:
        # --- Human Input ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Balances: {info['balances_achieved']}")
            obs, info = env.reset()

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        clock.tick(GameEnv.FPS)

    env.close()