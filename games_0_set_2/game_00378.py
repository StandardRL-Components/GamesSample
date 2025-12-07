import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ and ↓ to move the paddle. Try to hit the ball with the top or bottom of your paddle for bonus points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pixel Pong: A minimalist Pong game. Score 7 points to win, but miss 3 balls and you lose. Risky hits give bonus rewards."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 8
        self.CELL_WIDTH = self.WIDTH // self.GRID_COLS
        self.CELL_HEIGHT = self.HEIGHT // self.GRID_ROWS
        self.PADDLE_HEIGHT_CELLS = 3
        self.PADDLE_MAX_Y = self.GRID_ROWS - self.PADDLE_HEIGHT_CELLS
        self.WIN_SCORE = 7
        self.MAX_LIVES = 3
        self.MAX_STEPS = 1800 # 60 seconds at 30fps

        # Colors
        self.COLOR_BG = (15, 15, 15)
        self.COLOR_GRID = (40, 40, 40)
        self.COLOR_PADDLE = (50, 205, 50) # LimeGreen
        self.COLOR_BALL = (230, 230, 230)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEART = (220, 20, 60) # Crimson
        self.COLOR_HEART_EMPTY = (70, 70, 70)
        self.COLOR_PARTICLE = (255, 255, 255)
        self.COLOR_SCORE_PARTICLE = (255, 215, 0) # Gold

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('monospace', 24, bold=True)
        self.big_font = pygame.font.SysFont('monospace', 48, bold=True)
        
        # Game state variables (will be initialized in reset)
        self.paddle_y = 0
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        
        # Initialize state variables
        # self.reset() is called by the wrapper/test harness, not needed here

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize paddle
        self.paddle_y = self.PADDLE_MAX_Y // 2

        # Initialize ball
        self._reset_ball()

        # Initialize game state
        self.score = 0
        self.lives = self.MAX_LIVES
        self.steps = 0
        self.game_over = False
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _reset_ball(self, serve_left=True):
        """Resets the ball to the center with a new random velocity."""
        self.ball_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        if not serve_left:
            angle += math.pi
        
        speed_multiplier = 1.0 + 0.1 * self.score
        base_speed = self.WIDTH / 10.0
        speed = base_speed * speed_multiplier
        
        self.ball_vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0
        terminated = False
        truncated = False

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]

            if movement == 1:  # Up
                self.paddle_y = max(0, self.paddle_y - 1)
            elif movement == 2:  # Down
                self.paddle_y = min(self.PADDLE_MAX_Y, self.paddle_y + 1)
            
            # Update game logic
            reward += self._update_ball()
            self._update_particles()
            self.steps += 1

            # Check for termination (win/loss)
            if self.score >= self.WIN_SCORE:
                reward += 10.0
                terminated = True
            elif self.lives <= 0:
                reward -= 10.0
                terminated = True
            
            # Check for truncation (time limit)
            if self.steps >= self.MAX_STEPS:
                truncated = True
            
            if terminated or truncated:
                self.game_over = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_ball(self):
        """Updates ball position and handles all collisions. Returns event-based reward."""
        reward = 0
        # Frame-rate independent speed, assuming 30fps
        time_delta_factor = self.clock.get_time() / (1000.0 / 30.0) if self.clock.get_time() > 0 else 1
        self.ball_pos += self.ball_vel * (time_delta_factor / 30.0)

        # Wall collisions (top/bottom/left)
        if self.ball_pos.y <= 0 or self.ball_pos.y >= self.HEIGHT:
            self.ball_vel.y *= -1
            self.ball_pos.y = max(0, min(self.ball_pos.y, self.HEIGHT))
            self._create_particles(self.ball_pos)

        if self.ball_pos.x <= 0:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(0, self.ball_pos.x)
            self._create_particles(self.ball_pos)

        # Paddle collision check
        paddle_x = self.WIDTH - self.CELL_WIDTH
        paddle_rect = pygame.Rect(
            paddle_x,
            self.paddle_y * self.CELL_HEIGHT,
            self.CELL_WIDTH,
            self.PADDLE_HEIGHT_CELLS * self.CELL_HEIGHT
        )
        ball_size = self.CELL_WIDTH / 2
        ball_rect = pygame.Rect(self.ball_pos.x - ball_size/2, self.ball_pos.y - ball_size/2, ball_size, ball_size)

        if self.ball_vel.x > 0 and ball_rect.colliderect(paddle_rect):
            hit_pos_normalized = (self.ball_pos.y - paddle_rect.top) / paddle_rect.height
            hit_pos_normalized = max(0, min(1, hit_pos_normalized))
            
            self.ball_vel.x *= -1
            
            deflection = (hit_pos_normalized - 0.5) * 1.5
            self.ball_vel.y += deflection * abs(self.ball_vel.x) * 0.1
            
            self.ball_pos.x = paddle_x - ball_size/2 - 1

            is_risky_hit = hit_pos_normalized < 0.2 or hit_pos_normalized > 0.8
            if is_risky_hit:
                reward += 1.0
                self.score += 1
                self._create_particles(self.ball_pos, 20, self.COLOR_SCORE_PARTICLE)
                if self.score < self.WIN_SCORE:
                    self._reset_ball(serve_left=False)
            else:
                reward += 0.1
                self._create_particles(self.ball_pos, 10, self.COLOR_PADDLE)
        
        # Ball goes out of bounds (player misses)
        if self.ball_pos.x > self.WIDTH:
            self.lives -= 1
            reward -= 1.0
            if self.lives > 0:
                self._reset_ball(serve_left=True)

        return reward

    def _create_particles(self, pos, count=5, color=None):
        if color is None:
            color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            lifetime = self.np_random.integers(10, 20)
            self.particles.append([pos.copy(), vel, lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1]
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(1, self.GRID_COLS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i * self.CELL_WIDTH, 0), (i * self.CELL_WIDTH, self.HEIGHT), 1)
        for i in range(1, self.GRID_ROWS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i * self.CELL_HEIGHT), (self.WIDTH, i * self.CELL_HEIGHT), 1)

        # Draw paddle
        paddle_rect = pygame.Rect(
            self.WIDTH - self.CELL_WIDTH,
            self.paddle_y * self.CELL_HEIGHT,
            self.CELL_WIDTH,
            self.PADDLE_HEIGHT_CELLS * self.CELL_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect)

        # Draw ball
        ball_size = self.CELL_WIDTH / 2
        ball_rect = pygame.Rect(
            int(self.ball_pos.x - ball_size / 2),
            int(self.ball_pos.y - ball_size / 2),
            int(ball_size),
            int(ball_size)
        )
        pygame.draw.rect(self.screen, self.COLOR_BALL, ball_rect)

        # Draw particles
        for p in self.particles:
            pos, _, lifetime, color = p
            size = max(1, int(3 * (lifetime / 20.0)))
            pygame.draw.rect(self.screen, color, (int(pos.x - size/2), int(pos.y - size/2), size, size))

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        for i in range(self.MAX_LIVES):
            heart_pos_x = self.WIDTH - 20 - (i * 30)
            if i < self.lives:
                self._draw_heart(heart_pos_x, 20, self.COLOR_HEART)
            else:
                self._draw_heart(heart_pos_x, 20, self.COLOR_HEART_EMPTY, filled=False)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.big_font.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _draw_heart(self, x, y, color, filled=True):
        points = [
            (x, y - 5), (x + 5, y - 10), (x + 10, y - 5),
            (x, y + 5),
            (x - 10, y - 5), (x - 5, y - 10)
        ]
        pygame.draw.polygon(self.screen, color, points, 0 if filled else 2)

    def _get_info(self):
        return {
            "score": self.score,
            "lives": self.lives,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

# This block allows the game to be run directly for human play testing
if __name__ == "__main__":
    # Allow display for human play
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Pong")
    
    running = True
    # Use a mutable list for actions, as numpy arrays are immutable
    action = [0, 0, 0]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        action[0] = 0 # No-op by default
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Lives: {info['lives']}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            action = [0, 0, 0]

    env.close()