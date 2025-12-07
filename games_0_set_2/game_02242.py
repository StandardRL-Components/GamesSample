
# Generated: 2025-08-27T19:44:18.900099
# Source Brief: brief_02242.md
# Brief Index: 2242

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Try to clear all the bricks!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A vibrant, modern take on the classic brick breaker. Use your paddle to deflect the ball, destroy bricks, and achieve a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.BALL_RADIUS = 8
        self.BRICK_WIDTH, self.BRICK_HEIGHT = 58, 20
        self.BRICK_ROWS, self.BRICK_COLS = 4, 10
        self.MAX_STEPS = 1000
        self.INITIAL_LIVES = 5

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (0, 150, 255)
        self.COLOR_PADDLE_ACCENT = (100, 200, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_UI = (220, 220, 220)
        self.BRICK_COLORS = [
            (255, 80, 80), (255, 160, 80), (255, 240, 80), (80, 255, 80)
        ]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.bricks = None
        self.particles = None
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.total_bricks = 0
        self.game_over_message = ""
        
        self.reset()
        
        # Self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over_message = ""
        
        # Paddle
        paddle_y = self.HEIGHT - 40
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        # Ball
        self.ball = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        initial_angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        ball_speed = 7.0
        self.ball_vel = pygame.Vector2(math.cos(initial_angle) * ball_speed, math.sin(initial_angle) * ball_speed)

        # Bricks
        self.bricks = []
        self.total_bricks = self.BRICK_ROWS * self.BRICK_COLS
        top_offset = 50
        x_gap, y_gap = 6, 6
        total_brick_width = self.BRICK_COLS * self.BRICK_WIDTH + (self.BRICK_COLS - 1) * x_gap
        start_x = (self.WIDTH - total_brick_width) / 2

        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                brick_x = start_x + j * (self.BRICK_WIDTH + x_gap)
                brick_y = top_offset + i * (self.BRICK_HEIGHT + y_gap)
                brick_rect = pygame.Rect(brick_x, brick_y, self.BRICK_WIDTH, self.BRICK_HEIGHT)
                self.bricks.append({"rect": brick_rect, "color": self.BRICK_COLORS[i % len(self.BRICK_COLORS)]})
        
        # Particles
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        # --- Handle Action ---
        movement = action[0]
        paddle_speed = 12.0
        if movement == 3:  # Left
            self.paddle.x -= paddle_speed
            reward -= 0.01 # Small penalty for moving
        elif movement == 4:  # Right
            self.paddle.x += paddle_speed
            reward -= 0.01 # Small penalty for moving

        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.WIDTH, self.paddle.right)

        # --- Update Game State ---
        if not self.game_over_message:
            self.ball += self.ball_vel

            # --- Collisions ---
            # Walls
            if self.ball.x - self.BALL_RADIUS <= 0 or self.ball.x + self.BALL_RADIUS >= self.WIDTH:
                self.ball.x = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball.x))
                self.ball_vel.x *= -1
                # sfx: wall_bounce
            if self.ball.y - self.BALL_RADIUS <= 0:
                self.ball.y = self.BALL_RADIUS
                self.ball_vel.y *= -1
                # sfx: wall_bounce

            # Paddle
            paddle_collision = self.paddle.collidepoint(self.ball.x, self.ball.y + self.BALL_RADIUS)
            if paddle_collision and self.ball_vel.y > 0:
                self.ball.bottom = self.paddle.top
                
                # Influence bounce angle
                offset = (self.ball.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                bounce_angle = math.radians(-90 + offset * 60) # -30 to -150 degrees
                
                speed = self.ball_vel.length()
                self.ball_vel.x = speed * math.cos(bounce_angle)
                self.ball_vel.y = speed * math.sin(bounce_angle)
                
                reward += 0.1 # Reward for hitting the ball
                # sfx: paddle_hit

            # Bricks
            hit_brick_idx = -1
            for i, brick in enumerate(self.bricks):
                if brick["rect"].collidepoint(self.ball):
                    hit_brick_idx = i
                    break
            
            if hit_brick_idx != -1:
                brick_hit = self.bricks.pop(hit_brick_idx)
                reward += 1.0 # Reward for destroying a brick
                # sfx: brick_destroy
                
                # Create particles
                for _ in range(15):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 4)
                    vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
                    self.particles.append({
                        "pos": self.ball.copy(),
                        "vel": vel,
                        "lifespan": self.np_random.uniform(15, 30),
                        "color": brick_hit["color"]
                    })

                # Bounce logic
                # Simple vertical bounce is robust and feels good
                self.ball_vel.y *= -1

            # --- Ball Lost ---
            if self.ball.y - self.BALL_RADIUS > self.HEIGHT:
                self.lives -= 1
                # sfx: lose_life
                if self.lives > 0:
                    # Reset ball
                    self.ball = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
                    initial_angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                    self.ball_vel = pygame.Vector2(math.cos(initial_angle) * 7.0, math.sin(initial_angle) * 7.0)
                else:
                    terminated = True
                    reward = -100.0
                    self.game_over_message = "YOU LOSE"

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
        
        # --- Check Win/Termination ---
        if not self.bricks and not self.game_over_message:
            terminated = True
            reward = 100.0
            self.game_over_message = "YOU WIN!"

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            if not self.game_over_message:
                 self.game_over_message = "TIME UP"


        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _render_game(self):
        # Background gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG[0] * (1 - interp) + 5 * interp),
                int(self.COLOR_BG[1] * (1 - interp) + 5 * interp),
                int(self.COLOR_BG[2] * (1 - interp) + 35 * interp),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick["color"], brick["rect"], border_radius=3)
            # Add a subtle highlight for 3D effect
            highlight = tuple(min(255, c + 30) for c in brick["color"])
            pygame.draw.rect(self.screen, highlight, (brick["rect"].x, brick["rect"].y, brick["rect"].width, 4), border_top_left_radius=3, border_top_right_radius=3)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_ACCENT, (self.paddle.x, self.paddle.y, self.paddle.width, 5), border_top_left_radius=4, border_top_right_radius=4)
        
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 30))))
            color = p["color"] + (alpha,)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p["pos"].x - 2), int(p["pos"].y - 2)))
            
        # Ball
        if self.lives > 0:
            # Ball glow effect
            glow_radius = self.BALL_RADIUS + 5
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (255, 255, 255, 50))
            self.screen.blit(glow_surf, (int(self.ball.x - glow_radius), int(self.ball.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Main ball
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball.x), int(self.ball.y), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, int(self.ball.x), int(self.ball.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        life_radius = 6
        life_color = self.COLOR_PADDLE
        for i in range(self.lives):
            x = self.WIDTH - 20 - i * (life_radius * 2 + 5)
            pygame.gfxdraw.filled_circle(self.screen, x, 20, life_radius, life_color)
            pygame.gfxdraw.aacircle(self.screen, x, 20, life_radius, life_color)
            
        # Game Over Message
        if self.game_over_message:
            msg_surf = self.font_game_over.render(self.game_over_message, True, self.COLOR_UI)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_remaining": len(self.bricks)
        }

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Requires a display. Will not work in a purely headless environment.
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy" # Uncomment for headless execution test

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Brick Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # Movement, space, shift
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()
                total_reward = 0

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Control the frame rate of the playable demo

    env.close()