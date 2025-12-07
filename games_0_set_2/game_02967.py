
# Generated: 2025-08-28T06:34:42.082570
# Source Brief: brief_02967.md
# Brief Index: 2967

        
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


class Particle:
    """A simple class for a single particle."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-4, 0)
        self.lifespan = random.randint(15, 30)  # Frames
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            # Fade out effect
            alpha = max(0, min(255, int(255 * (self.lifespan / 30))))
            size = max(1, int(4 * (self.lifespan / 30)))
            rect = pygame.Rect(int(self.x - size / 2), int(self.y - size / 2), size, size)
            
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, (*self.color, alpha), (0, 0, size, size))
            surface.blit(temp_surf, rect.topleft)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ← and → to move the paddle. Bounce the ball to destroy all the bricks."
    )

    game_description = (
        "A fast-paced, retro arcade game. Destroy all the bricks with the ball while keeping it in play with your paddle."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Spaces
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
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (20, 40, 80)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_PADDLE_GLOW = (200, 200, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 150)
        self.COLOR_WALL = (100, 120, 150)
        self.COLOR_TEXT = (220, 220, 240)
        self.BRICK_COLORS = [
            (255, 50, 50), (255, 150, 50), (255, 255, 50),
            (50, 255, 50), (50, 150, 255), (150, 50, 255)
        ]

        # Fonts
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_msg = pygame.font.Font(pygame.font.get_default_font(), 48)
        except IOError:
            self.font_ui = pygame.font.SysFont("Arial", 18)
            self.font_msg = pygame.font.SysFont("Arial", 48)

        # Game constants
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_MAX_SPEED = 8
        self.WALL_THICKNESS = 10
        self.BRICK_WIDTH, self.BRICK_HEIGHT = 38, 15
        self.MAX_STEPS = 2500
        self.INITIAL_LIVES = 2
        self.NUM_BRICKS = 75

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.bricks = []
        self.particles = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.win = False

        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )
        
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        initial_speed = 5
        self.ball_vel = [initial_speed * math.sin(angle), -initial_speed * math.cos(angle)]

        self.bricks = []
        brick_cols = 15
        brick_rows = self.NUM_BRICKS // brick_cols
        total_brick_width = brick_cols * (self.BRICK_WIDTH + 2)
        start_x = (self.WIDTH - total_brick_width) / 2
        for i in range(brick_rows):
            for j in range(brick_cols):
                color = self.BRICK_COLORS[i % len(self.BRICK_COLORS)]
                brick_rect = pygame.Rect(
                    start_x + j * (self.BRICK_WIDTH + 2),
                    50 + i * (self.BRICK_HEIGHT + 2),
                    self.BRICK_WIDTH, self.BRICK_HEIGHT
                )
                self.bricks.append({"rect": brick_rect, "color": color})

        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement = action[0]
        
        # --- Handle Input ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = max(self.WALL_THICKNESS, min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH - self.WALL_THICKNESS))

        # --- Update Game State ---
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifespan > 0]

        # --- Collision Detection and Reward Calculation ---
        reward = -0.01  # Small penalty for time passing
        brick_hit_this_step = False
        life_lost_this_step = False

        # Wall collisions
        if ball_rect.left <= self.WALL_THICKNESS:
            ball_rect.left = self.WALL_THICKNESS
            self.ball_vel[0] *= -1
        if ball_rect.right >= self.WIDTH - self.WALL_THICKNESS:
            ball_rect.right = self.WIDTH - self.WALL_THICKNESS
            self.ball_vel[0] *= -1
        if ball_rect.top <= 0:
            ball_rect.top = 0
            self.ball_vel[1] *= -1

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # Prevent ball from getting stuck in paddle
            ball_rect.bottom = self.paddle.top
            
            # Change angle based on hit location
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = self.BALL_MAX_SPEED * offset
            self.ball_vel[1] *= -1
            
            # Normalize speed
            speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            if speed > self.BALL_MAX_SPEED:
                scale = self.BALL_MAX_SPEED / speed
                self.ball_vel[0] *= scale
                self.ball_vel[1] *= scale

        # Brick collisions
        hit_brick = ball_rect.collidelist([b["rect"] for b in self.bricks])
        if hit_brick != -1:
            brick_data = self.bricks.pop(hit_brick)
            brick_rect = brick_data["rect"]
            
            # Particle effect
            for _ in range(15):
                self.particles.append(Particle(brick_rect.centerx, brick_rect.centery, brick_data["color"]))

            # Determine bounce direction
            prev_ball_center = (ball_rect.centerx - self.ball_vel[0], ball_rect.centery - self.ball_vel[1])
            
            if (prev_ball_center[1] <= brick_rect.top or prev_ball_center[1] >= brick_rect.bottom):
                 self.ball_vel[1] *= -1
            else: # Hit from side
                 self.ball_vel[0] *= -1

            reward += 1
            self.score += 10
            brick_hit_this_step = True
        
        # Update ball position from rect after collision adjustments
        self.ball_pos = [ball_rect.centerx, ball_rect.centery]

        # Ball out of bounds
        if ball_rect.top > self.HEIGHT:
            self.lives -= 1
            life_lost_this_step = True
            if self.lives <= 0:
                self.game_over = True
                reward = -100
            else:
                # Reset ball and paddle
                self.paddle.centerx = self.WIDTH / 2
                self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
                angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
                self.ball_vel = [5 * math.sin(angle), -5 * math.cos(angle)]

        if not life_lost_this_step:
            reward += 0.1 # Reward for keeping ball in play

        # --- Termination Check ---
        self.steps += 1
        terminated = False
        if len(self.bricks) == 0:
            self.game_over = True
            self.win = True
            terminated = True
            reward = 100
        elif self.lives <= 0:
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_gradient_background(self):
        """Draws a vertical gradient."""
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _get_observation(self):
        self._render_gradient_background()
        
        # Render walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.HEIGHT))
        
        # Render bricks
        for brick_data in self.bricks:
            # Draw a slightly darker outline
            outline_rect = brick_data["rect"].inflate(2, 2)
            outline_color = tuple(c * 0.7 for c in brick_data["color"])
            pygame.draw.rect(self.screen, outline_color, outline_rect, border_radius=3)
            pygame.draw.rect(self.screen, brick_data["color"], brick_data["rect"], border_radius=3)

        # Render particles
        for p in self.particles:
            p.draw(self.screen)

        # Render paddle with glow
        glow_rect = self.paddle.inflate(8, 8)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PADDLE_GLOW, 50), glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Render ball with antialiasing and glow
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        glow_radius = int(self.BALL_RADIUS * 1.8)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, glow_radius, (*self.COLOR_BALL_GLOW, 80))
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

        # Render UI
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, 10))
        
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - self.WALL_THICKNESS - 10, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            msg_surf = self.font_msg.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_remaining": len(self.bricks),
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Override the dummy screen with a real one for display
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Breakout")
    
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                total_reward = 0
        
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Draw the observation to the display screen
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(60) # Control the frame rate

    env.close()