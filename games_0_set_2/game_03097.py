
# Generated: 2025-08-28T06:57:38.497526
# Source Brief: brief_03097.md
# Brief Index: 3097

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    user_guide = (
        "Controls: Use Left and Right arrow keys to move the paddle. "
        "Aim your shots by hitting the ball with different parts of the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker where strategic paddle positioning and "
        "risk-taking are rewarded."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.BRICK_COLORS = [
            (217, 87, 99), (217, 142, 87), (175, 217, 87),
            (87, 217, 142), (87, 175, 217), (142, 87, 217)
        ]

        # Paddle
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 8

        # Ball
        self.BALL_RADIUS = 7
        self.BALL_INITIAL_SPEED = 4
        self.BALL_MAX_SPEED = 7
        self.BALL_BOUNCE_GAIN = 1.8 # Multiplier for x-velocity on paddle edge hits

        # Bricks
        self.BRICK_ROWS = 5
        self.BRICK_COLS = 15
        self.BRICK_COUNT = self.BRICK_ROWS * self.BRICK_COLS
        self.BRICK_WIDTH = 38
        self.BRICK_HEIGHT = 18
        self.BRICK_GAP = 4
        self.BRICKS_START_Y = 50

        # --- Gymnasium Setup ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        self.heart_icon = self._create_heart_icon()

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.time_remaining = 0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.bricks = []
        self.particles = []
        self.last_paddle_hit_offset = 0.0
        self.current_step_reward = 0.0
        
        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        self.time_remaining = self.MAX_STEPS
        self.last_paddle_hit_offset = 0.0

        # Paddle
        paddle_y = self.HEIGHT - 40
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Ball
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        self.ball_vel = [
            self.BALL_INITIAL_SPEED * math.cos(angle),
            self.BALL_INITIAL_SPEED * math.sin(angle),
        ]
        
        # Bricks
        self.bricks = []
        self.brick_colors = []
        total_brick_width = self.BRICK_COLS * (self.BRICK_WIDTH + self.BRICK_GAP) - self.BRICK_GAP
        start_x = (self.WIDTH - total_brick_width) / 2
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                x = start_x + j * (self.BRICK_WIDTH + self.BRICK_GAP)
                y = self.BRICKS_START_Y + i * (self.BRICK_HEIGHT + self.BRICK_GAP)
                self.bricks.append(pygame.Rect(x, y, self.BRICK_WIDTH, self.BRICK_HEIGHT))
                self.brick_colors.append(self.BRICK_COLORS[i % len(self.BRICK_COLORS)])

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.current_step_reward = 0
        self.steps += 1
        self.time_remaining -= 1

        self._handle_input(action)
        self._update_ball()
        self._update_particles()
        
        terminated = self._check_termination()
        
        # Apply terminal rewards
        if terminated:
            if not self.bricks: # Win condition
                self.current_step_reward += 50
            else: # Loss condition
                self.current_step_reward -= 50

        reward = self.current_step_reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED

        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

    def _update_ball(self):
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left < 0 or ball_rect.right > self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos[0]))
            # sfx: wall_bounce
        if ball_rect.top < 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = max(self.BALL_RADIUS, self.ball_pos[1])
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: paddle_hit
            offset = self.ball_pos[0] - self.paddle.centerx
            normalized_offset = offset / (self.PADDLE_WIDTH / 2)
            self.last_paddle_hit_offset = normalized_offset
            
            # Change horizontal velocity based on where it hit
            self.ball_vel[0] = normalized_offset * self.BALL_INITIAL_SPEED * self.BALL_BOUNCE_GAIN
            self.ball_vel[1] *= -1

            # Normalize speed to prevent it from getting too fast
            speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            if speed > self.BALL_MAX_SPEED:
                scale = self.BALL_MAX_SPEED / speed
                self.ball_vel[0] *= scale
                self.ball_vel[1] *= scale
            
            # Ensure minimum vertical speed
            self.ball_vel[1] = -abs(self.ball_vel[1]) if self.ball_vel[1] > -1 else -1

            # Prevent ball from getting stuck in paddle
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS

        # Brick collisions
        collided_idx = ball_rect.collidelist(self.bricks)
        if collided_idx != -1:
            # sfx: brick_break
            brick_rect = self.bricks.pop(collided_idx)
            brick_color = self.brick_colors.pop(collided_idx)
            
            self._create_explosion(brick_rect.center, brick_color)
            
            # Reward logic
            self.score += 1
            self.current_step_reward += 1.0 # Base reward for breaking a brick
            if abs(self.last_paddle_hit_offset) > 0.7:
                self.current_step_reward += 0.1 # Bonus for risky edge hit
            else:
                self.current_step_reward -= 0.02 # Small penalty for safe center hit

            # Reflection logic
            prev_ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_vel[0] - self.BALL_RADIUS, self.ball_pos[1] - self.ball_vel[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
            
            if prev_ball_rect.bottom <= brick_rect.top or prev_ball_rect.top >= brick_rect.bottom:
                self.ball_vel[1] *= -1
            if prev_ball_rect.right <= brick_rect.left or prev_ball_rect.left >= brick_rect.right:
                self.ball_vel[0] *= -1

        # Lose life
        if ball_rect.top > self.HEIGHT:
            # sfx: lose_life
            self.lives -= 1
            self.current_step_reward -= 5.0
            if self.lives > 0:
                # Reset ball
                self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
                angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
                self.ball_vel = [self.BALL_INITIAL_SPEED * math.cos(angle), self.BALL_INITIAL_SPEED * math.sin(angle)]
            else:
                self.game_over = True
    
    def _create_explosion(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [speed * math.cos(angle), speed * math.sin(angle)]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'lifespan': self.np_random.integers(20, 40),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['lifespan'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0]

    def _check_termination(self):
        if self.game_over:
            return True
        if self.lives <= 0:
            self.game_over = True
            return True
        if not self.bricks:
            self.game_over = True
            return True
        if self.time_remaining <= 0:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background Grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Bricks
        for i, brick in enumerate(self.bricks):
            pygame.draw.rect(self.screen, self.brick_colors[i], brick, border_radius=3)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 40))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color
            )
            
        # Ball
        x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:04d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Timer
        time_str = f"{self.time_remaining // self.FPS:02d}"
        time_text = self.font_large.render(time_str, True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(centerx=self.WIDTH / 2, y=5)
        self.screen.blit(time_text, time_rect)

        # Lives
        for i in range(self.lives):
            self.screen.blit(self.heart_icon, (self.WIDTH - 30 - i * 25, 10))
            
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if not self.bricks else "GAME OVER"
            end_text = self.font_large.render(message, True, self.COLOR_BALL if not self.bricks else self.BRICK_COLORS[0])
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_remaining": len(self.bricks),
            "time_remaining": self.time_remaining,
        }

    def _create_heart_icon(self):
        size = 20
        heart = pygame.Surface((size, size), pygame.SRCALPHA)
        color = (255, 80, 80)
        points = [
            (size // 2, size),
            (0, size // 3),
            (size // 4, 0),
            (size // 2, size // 4),
            (size * 3 // 4, 0),
            (size, size // 3)
        ]
        pygame.gfxdraw.aapolygon(heart, points, color)
        pygame.gfxdraw.filled_polygon(heart, points, color)
        return heart

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test assertions from brief
        assert self.score == info['score']
        assert self.lives == info['lives']
        assert self.steps == info['steps']
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Interactive Play ---
    # This part is for human testing and is not part of the Gymnasium environment
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)
    
    while not done:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']}")
    env.close()