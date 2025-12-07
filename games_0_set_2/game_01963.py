
# Generated: 2025-08-28T03:14:34.824832
# Source Brief: brief_01963.md
# Brief Index: 1963

        
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
    """
    A top-down arcade Breakout game where the player controls a paddle to bounce a ball and destroy bricks.
    The goal is to destroy all bricks without losing all three lives.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Other keys have no effect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade Breakout clone. Move the paddle to bounce the ball and destroy all the bricks. You have 3 lives."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 10
    BALL_RADIUS = 6
    BALL_MAX_SPEED = 8
    BRICK_WIDTH, BRICK_HEIGHT = 58, 20
    BRICK_COLS, BRICK_ROWS = 10, 5
    MAX_STEPS = 30 * 60 # 60 seconds at 30fps

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_PADDLE = (0, 150, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_WALL = (100, 100, 120)
    BRICK_COLORS = [
        (255, 50, 50), (255, 150, 50), (255, 255, 50), 
        (50, 255, 50), (50, 150, 255)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
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
        
        # State variables are initialized in reset()
        self.paddle = None
        self.ball = None
        self.ball_velocity = None
        self.bricks = None
        self.particles = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.np_random = None

        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize paddle
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        # Initialize ball
        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS,
            self.paddle.top - self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )
        # Random initial direction
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        initial_speed = 5
        self.ball_velocity = [math.cos(angle) * initial_speed, math.sin(angle) * initial_speed]

        # Initialize bricks
        self.bricks = []
        brick_total_width = self.BRICK_COLS * (self.BRICK_WIDTH + 2)
        start_x = (self.WIDTH - brick_total_width) / 2
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                brick = pygame.Rect(
                    start_x + j * (self.BRICK_WIDTH + 2),
                    50 + i * (self.BRICK_HEIGHT + 2),
                    self.BRICK_WIDTH,
                    self.BRICK_HEIGHT
                )
                self.bricks.append(brick)
        
        self.total_bricks = len(self.bricks)

        # Initialize game state
        self.particles = []
        self.lives = 3
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.02  # Small penalty for each step

        # --- Handle Input ---
        movement = action[0]  # 0-4: none/up/down/left/right
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Keep paddle within bounds
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        # --- Update Game Logic ---
        self._update_ball()
        bricks_destroyed = self._handle_collisions()
        
        if bricks_destroyed > 0:
            reward += bricks_destroyed  # +1 per brick
            if bricks_destroyed > 1:
                reward += 5  # Bonus for multi-hit

        self._update_particles()
        
        # --- Check for Termination ---
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.win:
                reward += 100
            elif self.lives <= 0:
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        self.ball.x += self.ball_velocity[0]
        self.ball.y += self.ball_velocity[1]

    def _handle_collisions(self):
        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_velocity[0] *= -1
            self.ball.left = max(0, self.ball.left)
            self.ball.right = min(self.WIDTH, self.ball.right)
            # sfx: wall_bounce
        if self.ball.top <= 0:
            self.ball_velocity[1] *= -1
            self.ball.top = max(0, self.ball.top)
            # sfx: wall_bounce

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_velocity[1] > 0:
            self.ball_velocity[1] *= -1
            
            # Influence horizontal velocity based on hit location
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_velocity[0] += offset * 2.5
            
            # Cap ball speed
            speed = math.hypot(*self.ball_velocity)
            if speed > self.BALL_MAX_SPEED:
                scale = self.BALL_MAX_SPEED / speed
                self.ball_velocity[0] *= scale
                self.ball_velocity[1] *= scale
            
            self.ball.bottom = self.paddle.top
            # sfx: paddle_hit

        # Brick collisions
        bricks_destroyed = 0
        collided_index = self.ball.collidelist(self.bricks)
        if collided_index != -1:
            brick = self.bricks[collided_index]
            
            # Determine bounce direction (simple approach)
            # Check overlap to decide horizontal or vertical bounce
            prev_ball_center = (self.ball.centerx - self.ball_velocity[0], self.ball.centery - self.ball_velocity[1])
            
            if prev_ball_center[1] < brick.top or prev_ball_center[1] > brick.bottom:
                 self.ball_velocity[1] *= -1
            else:
                 self.ball_velocity[0] *= -1

            # Remove bricks and create particles
            # Check for multiple collisions (e.g., corner hit)
            indices_to_remove = sorted(self.ball.collidelistall(self.bricks), reverse=True)
            for index in indices_to_remove:
                removed_brick = self.bricks.pop(index)
                self.score += 1
                bricks_destroyed += 1
                color_index = (removed_brick.y - 50) // (self.BRICK_HEIGHT + 2)
                self._create_particles(removed_brick.center, self.BRICK_COLORS[color_index % len(self.BRICK_COLORS)])
                # sfx: brick_destroy
            
        # Missed ball
        if self.ball.top > self.HEIGHT:
            self.lives -= 1
            self.game_over = self.lives <= 0
            if not self.game_over:
                self._reset_ball()
            # sfx: life_lost
        
        return bricks_destroyed

    def _check_termination(self):
        if self.game_over:
            return True
        if not self.bricks:
            self.win = True
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _reset_ball(self):
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top - 5
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        initial_speed = 5
        self.ball_velocity = [math.cos(angle) * initial_speed, math.sin(angle) * initial_speed]

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                "pos": list(pos),
                "vel": velocity,
                "life": self.np_random.integers(15, 25),
                "color": color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw bricks
        for i, brick in enumerate(self.bricks):
            color_index = (brick.y - 50) // (self.BRICK_HEIGHT + 2)
            color = self.BRICK_COLORS[color_index % len(self.BRICK_COLORS)]
            pygame.draw.rect(self.screen, color, brick, border_radius=3)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 25))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, temp_surf.get_rect())
            self.screen.blit(temp_surf, (int(p['pos'][0]), int(p['pos'][1])))

        # Draw paddle with glow
        glow_rect = self.paddle.inflate(6, 6)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PADDLE, 50), glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Draw ball with glow
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS + 3, (*self.COLOR_BALL, 50))
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS + 3, (*self.COLOR_BALL, 50))
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_ui.render("LIVES:", True, (255, 255, 255))
        self.screen.blit(lives_text, (self.WIDTH - 150, 10))
        for i in range(self.lives):
            life_paddle = pygame.Rect(self.WIDTH - 80 + i * 25, 12, 20, 5)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_paddle, border_radius=2)
            
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (50, 255, 50) if self.win else (255, 50, 50)
            
            end_text = self.font_game_over.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
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
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

# Example of how to run the environment
if __name__ == "__main__":
    import time

    env = GameEnv()
    obs, info = env.reset()
    
    # Use Pygame for human interaction
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Breakout")
    
    terminated = False
    total_reward = 0
    
    # Map keys to actions
    key_map = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while not terminated:
        # --- Human Input ---
        movement = 0 # No-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        # Action is always a list of 3 integers for MultiDiscrete
        action = [movement, 0, 0] # space and shift are not used
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render ---
        # The observation is already the rendered image
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            time.sleep(3) # Pause to see final screen
            obs, info = env.reset()
            terminated = False
            total_reward = 0
            
        # Control frame rate
        env.clock.tick(30)
        
    env.close()