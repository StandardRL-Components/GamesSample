
# Generated: 2025-08-28T04:11:55.593763
# Source Brief: brief_02247.md
# Brief Index: 2247

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: ←→ to move the paddle."

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade brick breaker with modern particle effects. Clear all three levels to win."
    )

    # Frames auto-advance at 30fps.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG_TOP = (10, 0, 30)
    COLOR_BG_BOTTOM = (40, 0, 60)
    COLOR_PADDLE = (220, 220, 220)
    COLOR_BALL = (255, 255, 0)
    COLOR_PARTICLE = (255, 100, 0)
    COLOR_TEXT = (255, 255, 255)
    BRICK_COLORS = [
        (255, 50, 50), (255, 150, 50), (50, 255, 50),
        (50, 150, 255), (150, 50, 255)
    ]

    # Game parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 15
    BALL_RADIUS = 8
    INITIAL_LIVES = 5
    MAX_LEVELS = 3
    MAX_STEPS = 10000

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
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 24)

        self._create_background()
        
        # All state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.level = 0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.bricks = []
        self.particles = []
        self.np_random = None

        # self.validate_implementation() # Commented out for submission as per standard practice

    def _create_background(self):
        """Creates a pre-rendered gradient background surface for performance."""
        self.background = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.background, color, (0, y), (self.SCREEN_WIDTH, y))

    def _generate_bricks(self):
        self.bricks = []
        brick_width = 50
        brick_height = 20
        rows = 5 + self.level
        cols = self.SCREEN_WIDTH // (brick_width + 4)
        x_offset = (self.SCREEN_WIDTH - cols * (brick_width + 4)) // 2
        y_offset = 50

        base_density = 0.5
        density = min(1.0, base_density + (self.level - 1) * 0.2)

        for r in range(rows):
            for c in range(cols):
                if self.np_random.random() < density:
                    brick_x = x_offset + c * (brick_width + 4)
                    brick_y = y_offset + r * (brick_height + 4)
                    color = self.np_random.choice(self.BRICK_COLORS)
                    self.bricks.append(
                        (pygame.Rect(brick_x, brick_y, brick_width, brick_height), color)
                    )

    def _reset_ball_and_paddle(self):
        self.paddle = pygame.Rect(
            self.SCREEN_WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.SCREEN_HEIGHT - 30,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        
        ball_speed = 2.0 + (self.level - 1) * 0.2
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25) # Upwards cone
        self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * ball_speed
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES
        self.level = 1
        self.particles = []

        self._reset_ball_and_paddle()
        self._generate_bricks()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement = action[0]

        # 1. Handle player input
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        # 2. Update ball position
        self.ball_pos += self.ball_vel

        # 3. Handle collisions
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.SCREEN_WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
            # sfx: wall_bounce
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.SCREEN_HEIGHT - self.BALL_RADIUS)
            # sfx: wall_bounce

        # Paddle collision
        if self.paddle.colliderect(ball_rect) and self.ball_vel.y > 0:
            # sfx: paddle_hit
            self.ball_vel.y *= -1
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS # Prevent sticking

            hit_pos = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            hit_pos = np.clip(hit_pos, -1, 1)
            
            if abs(hit_pos) <= 0.5:
                reward += 0.1 # Center hit
            else:
                reward -= 0.02 # Edge hit

            self.ball_vel.x += hit_pos * 2.0
            self.ball_vel.normalize_ip()
            ball_speed = 2.0 + (self.level - 1) * 0.2
            self.ball_vel *= ball_speed

        # Brick collisions
        hit_index = ball_rect.collidelist([b[0] for b in self.bricks])
        if hit_index != -1:
            # sfx: brick_break
            brick_rect, _ = self.bricks.pop(hit_index)
            reward += 1
            self.score += 10

            for _ in range(15):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                pos = pygame.Vector2(brick_rect.center)
                lifespan = self.np_random.integers(15, 30)
                self.particles.append({'pos': pos, 'vel': vel, 'lifespan': lifespan})

            overlap = ball_rect.clip(brick_rect)
            if overlap.width < overlap.height:
                self.ball_vel.x *= -1
            else:
                self.ball_vel.y *= -1
        
        if 0 < abs(self.ball_vel.y) < 0.2:
            self.ball_vel.y = 0.2 * np.sign(self.ball_vel.y)

        if self.ball_pos.y > self.paddle.bottom:
            reward -= 0.01

        if self.ball_pos.y - self.BALL_RADIUS > self.SCREEN_HEIGHT:
            # sfx: lose_life
            self.lives -= 1
            reward -= 10
            if self.lives > 0:
                self._reset_ball_and_paddle()
            else:
                self.game_over = True

        if not self.bricks:
            # sfx: level_complete
            reward += 100
            self.level += 1
            if self.level > self.MAX_LEVELS:
                self.game_over = True
                reward += 100
            else:
                self._reset_ball_and_paddle()
                self._generate_bricks()

        # 4. Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['lifespan'] -= 1

        # 5. Update game state
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        # Draw bricks
        for brick_rect, color in self.bricks:
            pygame.draw.rect(self.screen, color, brick_rect)
            pygame.draw.rect(self.screen, tuple(min(255, c + 30) for c in color), brick_rect, 2)

        # Draw paddle
        paddle_color_dark = tuple(max(0, c - 50) for c in self.COLOR_PADDLE)
        pygame.draw.rect(self.screen, paddle_color_dark, self.paddle.move(0, 3))
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle)

        # Draw ball
        x, y = int(self.ball_pos.x), int(self.ball_pos.y)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS + 2, (255, 255, 150, 80))

        # Draw particles
        for p in self.particles:
            alpha = p['lifespan'] / 30.0
            color = (
                int(self.COLOR_PARTICLE[0]),
                int(self.COLOR_PARTICLE[1] * alpha),
                int(self.COLOR_PARTICLE[2] * alpha)
            )
            size = int(self.BALL_RADIUS / 2 * alpha)
            if size > 0:
                pygame.draw.circle(self.screen, color, p['pos'], size)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        level_text = self.font_small.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        level_rect = level_text.get_rect(centerx=self.SCREEN_WIDTH // 2, y=10)
        self.screen.blit(level_text, level_rect)

        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        lives_rect = lives_text.get_rect(right=self.SCREEN_WIDTH - 10, y=10)
        self.screen.blit(lives_text, lives_rect)

        if self.game_over:
            msg = "GAME OVER" if self.lives <= 0 else "YOU WIN!"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _get_observation(self):
        self.screen.blit(self.background, (0, 0))
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "level": self.level,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
        
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Brick Breaker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(GameEnv.FPS)

    env.close()