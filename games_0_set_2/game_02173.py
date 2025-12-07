import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Clear all bricks before they reach the bottom or time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Paddle Panic: A fast-paced arcade game where you deflect a ball to destroy descending bricks. Clear the screen to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 3600  # 120 seconds * 30 FPS
        self.WALL_THICKNESS = 10

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_WALL = (180, 180, 180)
        self.COLOR_TEXT = (255, 255, 255)
        self.BRICK_COLORS = {
            1: (200, 70, 70),   # Red
            2: (70, 200, 70),   # Green
            3: (70, 70, 200)    # Blue
        }

        # Game parameters
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.BALL_SPEED = 8
        self.BRICK_ROWS = 5
        self.BRICK_COLS = 10
        self.BRICK_WIDTH = (self.SCREEN_WIDTH - 2 * self.WALL_THICKNESS) / self.BRICK_COLS
        self.BRICK_HEIGHT = 20
        self.BRICK_DROP_SPEED = 0.05

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)

        # --- State Variables ---
        # Initialize to default/empty states to prevent crashes before the first reset.
        self.paddle = pygame.Rect(0, 0, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.bricks = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.time_remaining = 0
        self.ball_y_history = deque(maxlen=60)

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize paddle
        paddle_y = self.SCREEN_HEIGHT - 40
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Initialize ball
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25) # Upwards
        self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_SPEED
        
        # Initialize bricks
        self.bricks = []
        brick_points = [1, 1, 2, 2, 3]
        for r in range(self.BRICK_ROWS):
            for c in range(self.BRICK_COLS):
                points = brick_points[r]
                color = self.BRICK_COLORS[points]
                brick_rect = pygame.Rect(
                    self.WALL_THICKNESS + c * self.BRICK_WIDTH,
                    self.WALL_THICKNESS + 50 + r * (self.BRICK_HEIGHT + 5),
                    self.BRICK_WIDTH - 2,
                    self.BRICK_HEIGHT - 2,
                )
                self.bricks.append({"rect": brick_rect, "color": color, "points": points, "y_float": float(brick_rect.y)})

        # Reset other state
        self.particles = []
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.time_remaining = self.MAX_STEPS
        self.ball_y_history = deque(maxlen=60)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0

        # --- Update Game Logic ---
        self._update_paddle(movement)
        self._update_ball()
        collision_reward = self._handle_collisions()
        reward += collision_reward
        self._update_bricks()
        self._update_particles()
        
        # --- Update Timers and Counters ---
        self.steps += 1
        self.time_remaining -= 1

        # --- Check Termination Conditions ---
        terminated = False
        
        # Win Condition
        if not self.bricks:
            self.game_won = True
            terminated = True
            reward += 50.0

        # Lose Conditions
        brick_loss = any(b["rect"].bottom >= self.SCREEN_HEIGHT for b in self.bricks)
        time_out = self.time_remaining <= 0
        step_limit_reached = self.steps >= self.MAX_STEPS

        if brick_loss:
            reward -= 50.0
            terminated = True
        elif time_out or step_limit_reached:
            terminated = True
        
        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_paddle(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.left = max(self.WALL_THICKNESS, self.paddle.left)
        self.paddle.right = min(self.SCREEN_WIDTH - self.WALL_THICKNESS, self.paddle.right)

    def _update_ball(self):
        self.ball_pos += self.ball_vel
        
        # Anti-softlock mechanism
        self.ball_y_history.append(self.ball_pos.y)
        if len(self.ball_y_history) == self.ball_y_history.maxlen:
            if np.std(self.ball_y_history) < 0.1: # Ball is stuck horizontally
                self.ball_vel.y += self.np_random.choice([-0.5, 0.5])
                self.ball_vel = self.ball_vel.normalize() * self.BALL_SPEED

    def _handle_collisions(self):
        reward = 0.0
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Walls
        if ball_rect.left <= self.WALL_THICKNESS:
            self.ball_vel.x *= -1
            ball_rect.left = self.WALL_THICKNESS
        if ball_rect.right >= self.SCREEN_WIDTH - self.WALL_THICKNESS:
            self.ball_vel.x *= -1
            ball_rect.right = self.SCREEN_WIDTH - self.WALL_THICKNESS
        if ball_rect.top <= self.WALL_THICKNESS:
            self.ball_vel.y *= -1
            ball_rect.top = self.WALL_THICKNESS
        
        self.ball_pos.x, self.ball_pos.y = ball_rect.centerx, ball_rect.centery
        
        # Paddle
        if self.paddle.colliderect(ball_rect) and self.ball_vel.y > 0:
            reward += 0.1
            self.ball_vel.y *= -1
            
            # Add horizontal influence based on hit location
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += offset * 2.5
            self.ball_vel = self.ball_vel.normalize() * self.BALL_SPEED
            
            ball_rect.bottom = self.paddle.top
            self.ball_pos.y = ball_rect.centery
            # Sound: paddle_hit.wav

        # Bricks
        for i, brick in reversed(list(enumerate(self.bricks))):
            if brick["rect"].colliderect(ball_rect):
                reward += brick["points"]
                self.score += brick["points"]
                self._create_particles(brick["rect"].center, brick["color"])
                
                # Simple collision response: reverse vertical velocity
                self.ball_vel.y *= -1
                
                self.bricks.pop(i)
                # Sound: brick_destroy.wav
                break
        
        return reward

    def _update_bricks(self):
        for brick in self.bricks:
            brick["y_float"] += self.BRICK_DROP_SPEED
            brick["rect"].y = int(brick["y_float"])

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "radius": self.np_random.uniform(2, 5),
                "life": 20,
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Damping
            p["life"] -= 1
            p["radius"] -= 0.1
        self.particles = [p for p in self.particles if p["life"] > 0 and p["radius"] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))

        # Particles
        for p in self.particles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            radius = int(p["radius"])
            if radius > 0:
                alpha = int(255 * (p["life"] / 20))
                color = (*p["color"], alpha)
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))
                
        # Bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick["color"], brick["rect"], border_radius=3)
            # Add a slight inner bevel for depth
            highlight = tuple(min(255, c+40) for c in brick["color"])
            shadow = tuple(max(0, c-40) for c in brick["color"])
            pygame.draw.line(self.screen, highlight, brick["rect"].topleft, brick["rect"].topright, 1)
            pygame.draw.line(self.screen, highlight, brick["rect"].topleft, brick["rect"].bottomleft, 1)
            pygame.draw.line(self.screen, shadow, brick["rect"].bottomleft, brick["rect"].bottomright, 1)
            pygame.draw.line(self.screen, shadow, brick["rect"].topright, brick["rect"].bottomright, 1)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball
        ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.SCREEN_HEIGHT - 30))

        # Time
        time_seconds = self.time_remaining // self.FPS
        time_text = self.font_small.render(f"TIME: {time_seconds}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - self.WALL_THICKNESS - 10, self.SCREEN_HEIGHT - 30))

        # Game Over / Win Message
        if self.game_over:
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (0, 255, 0) if self.game_won else (255, 0, 0)
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "bricks_left": len(self.bricks)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        # print("✓ Implementation validated successfully")

# Example usage for testing
if __name__ == '__main__':
    # The environment is created in headless mode, which is correct.
    # To actually see the game, we need to unset the dummy video driver
    # and create a display.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Paddle Panic")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # Space and Shift are not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Display the observation ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(env.FPS)
        
    env.close()