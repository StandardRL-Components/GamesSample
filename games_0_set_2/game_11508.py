import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:05:35.409829
# Source Brief: brief_01508.md
# Brief Index: 1508
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball.
    The goal is to survive for 60 seconds, deflecting waves of projectiles
    and collecting power-ups. The ball's speed increases with each wall bounce.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a bouncing ball and survive for 60 seconds against waves of projectiles. "
        "Collect power-ups to grow larger and increase your score."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to change the direction of your bouncing ball. "
        "Avoid projectiles and collect the green power-ups to survive."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # Assumed for time-based calculations
        self.MAX_STEPS = self.FPS * 60  # 60 seconds

        # Colors (bright/high-contrast for gameplay, dark/desaturated for BG)
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_PLAYER = (255, 60, 60)
        self.COLOR_PLAYER_GLOW = (255, 60, 60)
        self.COLOR_PROJECTILE = (60, 180, 255)
        self.COLOR_POWERUP = (60, 255, 160)
        self.COLOR_WALL = (220, 220, 220)
        self.COLOR_TEXT = (240, 240, 240)

        # Game parameters
        self.BALL_INITIAL_RADIUS = 15
        self.BALL_INITIAL_SPEED = 3.0
        self.BALL_SPEED_INCREASE_FACTOR = 1.05
        self.PROJECTILE_SIZE = 8
        self.PROJECTILE_SPEED = 4.0
        self.PROJECTILE_TRAIL_LENGTH = 10
        self.POWERUP_SIZE = 12
        self.POWERUP_RADIUS_INCREASE = 1
        self.PROJECTILE_SPAWN_INTERVAL = int(1.5 * self.FPS)
        self.POWERUP_SPAWN_INTERVAL = int(5 * self.FPS)
        self.MAX_POWERUPS_ON_SCREEN = 3

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("monospace", 24, bold=True)
        except pygame.error:
            self.font = pygame.font.Font(None, 30)


        # --- State Variables (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_radius = None
        self.ball_speed = None
        self.projectiles = None
        self.powerups = None
        self.projectile_spawn_timer = None
        self.powerup_spawn_timer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player Ball
        self.ball_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        initial_angle = self.np_random.uniform(0, 2 * math.pi)
        self.ball_speed = self.BALL_INITIAL_SPEED
        self.ball_vel = pygame.math.Vector2(math.cos(initial_angle), math.sin(initial_angle)) * self.ball_speed
        self.ball_radius = self.BALL_INITIAL_RADIUS

        # Entities
        self.projectiles = []
        self.powerups = []
        self.projectile_spawn_timer = self.PROJECTILE_SPAWN_INTERVAL
        self.powerup_spawn_timer = self.POWERUP_SPAWN_INTERVAL

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1

        # 1. Handle Input
        self._handle_input(action)

        # 2. Update Game Logic
        self._update_ball()
        self._update_projectiles()  # Sets self.game_over on collision
        powerup_reward = self._update_powerups()  # Returns RL reward on collection

        # 3. Spawn new entities
        self._spawn_entities()

        # 4. Determine reward and termination status
        terminated = False
        reward = 0.0

        if self.game_over:
            reward = -100.0  # Terminal loss reward
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            reward = 100.0  # Terminal win reward
            self.score += 1000 # Bonus score for winning
            terminated = True
        else:
            # Non-terminal step reward
            reward = 0.1 + powerup_reward  # Survival + collection reward

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        # space_held (action[1]) and shift_held (action[2]) are unused

        if self.ball_vel.length() > 0:
            current_speed = self.ball_vel.length()
        else:
            current_speed = self.ball_speed

        if movement == 1:  # Up
            self.ball_vel = pygame.math.Vector2(0, -1) * current_speed
        elif movement == 2:  # Down
            self.ball_vel = pygame.math.Vector2(0, 1) * current_speed
        elif movement == 3:  # Left
            self.ball_vel = pygame.math.Vector2(-1, 0) * current_speed
        elif movement == 4:  # Right
            self.ball_vel = pygame.math.Vector2(1, 0) * current_speed
        # movement == 0 is a no-op, velocity remains unchanged

    def _update_ball(self):
        self.ball_pos += self.ball_vel
        bounced = False
        
        # Wall collisions
        if self.ball_pos.x - self.ball_radius < 0:
            self.ball_pos.x = self.ball_radius
            self.ball_vel.x *= -1
            bounced = True
        elif self.ball_pos.x + self.ball_radius > self.WIDTH:
            self.ball_pos.x = self.WIDTH - self.ball_radius
            self.ball_vel.x *= -1
            bounced = True

        if self.ball_pos.y - self.ball_radius < 0:
            self.ball_pos.y = self.ball_radius
            self.ball_vel.y *= -1
            bounced = True
        elif self.ball_pos.y + self.ball_radius > self.HEIGHT:
            self.ball_pos.y = self.HEIGHT - self.ball_radius
            self.ball_vel.y *= -1
            bounced = True

        if bounced:
            # sfx: wall_bounce
            self.ball_speed *= self.BALL_SPEED_INCREASE_FACTOR
            if self.ball_vel.length() > 0:
                self.ball_vel.scale_to_length(self.ball_speed)

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj['trail'].append(proj['pos'].copy())
            if len(proj['trail']) > self.PROJECTILE_TRAIL_LENGTH:
                proj['trail'].pop(0)
            proj['pos'] += proj['vel']

            # Check for collision with player ball
            distance = self.ball_pos.distance_to(proj['pos'])
            if distance < self.ball_radius + self.PROJECTILE_SIZE / 2:
                self.game_over = True
                # sfx: player_hit
                return

            # Remove projectiles that are off-screen
            if not (0 < proj['pos'].x < self.WIDTH and 0 < proj['pos'].y < self.HEIGHT):
                self.projectiles.remove(proj)

    def _update_powerups(self):
        rl_reward = 0.0
        for p_up in self.powerups[:]:
            distance = self.ball_pos.distance_to(p_up['pos'])
            if distance < self.ball_radius + self.POWERUP_SIZE:
                self.powerups.remove(p_up)
                self.ball_radius = min(self.ball_radius + self.POWERUP_RADIUS_INCREASE, self.HEIGHT / 4)
                self.score += 50  # Add to game score for UI
                rl_reward += 1.0  # Add to RL agent's reward
                # sfx: powerup_collect
        return rl_reward

    def _spawn_entities(self):
        # Spawn projectiles in waves
        self.projectile_spawn_timer -= 1
        if self.projectile_spawn_timer <= 0:
            self.projectile_spawn_timer = self.PROJECTILE_SPAWN_INTERVAL
            cluster_size = 1 + int(self.steps / (self.FPS * 10))
            cluster_size = min(cluster_size, 40)

            for _ in range(cluster_size):
                edge = self.np_random.integers(4)
                if edge == 0: pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), -self.PROJECTILE_SIZE)
                elif edge == 1: pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.PROJECTILE_SIZE)
                elif edge == 2: pos = pygame.math.Vector2(-self.PROJECTILE_SIZE, self.np_random.uniform(0, self.HEIGHT))
                else: pos = pygame.math.Vector2(self.WIDTH + self.PROJECTILE_SIZE, self.np_random.uniform(0, self.HEIGHT))

                target_pos = self.ball_pos + pygame.math.Vector2(self.np_random.uniform(-50, 50), self.np_random.uniform(-50, 50))
                vel = (target_pos - pos).normalize() * self.PROJECTILE_SPEED
                self.projectiles.append({'pos': pos, 'vel': vel, 'trail': []})
            # sfx: projectile_spawn_wave

        # Spawn power-ups
        self.powerup_spawn_timer -= 1
        if self.powerup_spawn_timer <= 0:
            self.powerup_spawn_timer = self.POWERUP_SPAWN_INTERVAL
            if len(self.powerups) < self.MAX_POWERUPS_ON_SCREEN:
                pos = pygame.math.Vector2(
                    self.np_random.uniform(50, self.WIDTH - 50),
                    self.np_random.uniform(50, self.HEIGHT - 50)
                )
                self.powerups.append({'pos': pos})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), 2)

        for proj in self.projectiles:
            if len(proj['trail']) > 1:
                for i in range(len(proj['trail']) - 1):
                    fade_ratio = (i + 1) / len(proj['trail'])
                    fade_color = (self.COLOR_PROJECTILE[0] * fade_ratio, self.COLOR_PROJECTILE[1] * fade_ratio, self.COLOR_PROJECTILE[2] * fade_ratio)
                    pygame.draw.line(self.screen, fade_color, proj['trail'][i], proj['trail'][i+1], max(1, int(self.PROJECTILE_SIZE * 0.5 * fade_ratio)))

        for proj in self.projectiles:
            rect = pygame.Rect(proj['pos'].x - self.PROJECTILE_SIZE / 2, proj['pos'].y - self.PROJECTILE_SIZE / 2, self.PROJECTILE_SIZE, self.PROJECTILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, rect, border_radius=2)

        for p_up in self.powerups:
            s = self.POWERUP_SIZE
            points = [(p_up['pos'].x, p_up['pos'].y - s), (p_up['pos'].x - s, p_up['pos'].y + s*0.7), (p_up['pos'].x + s, p_up['pos'].y + s*0.7)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_POWERUP)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_POWERUP)

        glow_radius = int(self.ball_radius * 1.5)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        alpha = 60 + 20 * math.sin(self.steps * 0.2)
        pygame.draw.circle(glow_surface, (*self.COLOR_PLAYER_GLOW, int(alpha)), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (int(self.ball_pos.x - glow_radius), int(self.ball_pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), int(self.ball_radius), self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), int(self.ball_radius), self.COLOR_PLAYER)

    def _render_ui(self):
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        timer_surface = self.font.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surface, (10, 10))

        current_score = self.score + self.steps
        score_text = f"SCORE: {current_score}"
        score_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surface.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_surface, score_rect)

        if self.game_over or self.steps >= self.MAX_STEPS:
            win = self.steps >= self.MAX_STEPS and not self.game_over
            end_text = "SURVIVED!" if win else "GAME OVER"
            end_color = (100, 255, 100) if win else (255, 100, 100)
            end_surface = self.font.render(end_text, True, end_color)
            end_rect = end_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_surface, end_rect)

    def _get_info(self):
        return {
            "score": self.score + self.steps,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # Un-comment the following line to run in a window
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    
    env = GameEnv()
    obs, info = env.reset()

    # The following is a human-playable version of the environment
    # It is not used by the evaluation system, but is useful for testing
    if os.environ.get("SDL_VIDEODRIVER", "dummy") != "dummy":
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Bounce Royale - Human Player")
        clock = pygame.time.Clock()

        running = True
        total_reward = 0
        while running:
            movement = 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            action = [movement, space_held, shift_held]

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            clock.tick(env.FPS)

        env.close()
    else:
        print("Running in headless mode. No visual output will be provided.")
        # Example of running a few steps headlessly
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print("Episode finished.")
                env.reset()
        env.close()
        print("Headless execution example finished.")