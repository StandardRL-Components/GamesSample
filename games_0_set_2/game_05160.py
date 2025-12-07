
# Generated: 2025-08-28T04:08:54.888220
# Source Brief: brief_05160.md
# Brief Index: 5160

        
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
    user_guide = (
        "Controls: ←→ to move the paddle. Hit the ball to score points. "
        "Edge hits are riskier but give more reward. "
        "Score 7 points to win, miss 3 balls and you lose."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced retro arcade game. Control a paddle to hit a "
        "bouncing ball. Aim for the paddle's edges for bonus points, "
        "but be careful not to miss!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (20, 20, 35)
    COLOR_PADDLE = (0, 255, 255)  # Cyan
    COLOR_BALL = (255, 255, 0)   # Yellow
    COLOR_WALL = (220, 220, 220)
    COLOR_TEXT = (255, 255, 255)
    COLOR_DANGER = (255, 50, 50)
    COLOR_PARTICLE = (255, 100, 0) # Orange/Red

    # Game parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 8
    BALL_SIZE = 10
    INITIAL_BALL_SPEED = 5.0
    MAX_STEPS = 1500 # Increased for longer rallies
    WIN_SCORE = 7
    MAX_MISSES = 3
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.score = 0
        self.misses = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.particles = []

        self.paddle = pygame.Rect(
            self.SCREEN_WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        self.ball = pygame.Rect(
            self.SCREEN_WIDTH // 2 - self.BALL_SIZE // 2,
            self.SCREEN_HEIGHT // 2 - self.BALL_SIZE // 2,
            self.BALL_SIZE,
            self.BALL_SIZE,
        )

        # Randomize initial ball direction
        angle = self.np_random.uniform(math.pi * 0.25, math.pi * 0.75)
        if self.np_random.choice([True, False]):
             angle += math.pi
        
        self.ball_vel = pygame.Vector2(
            math.cos(angle) * self.INITIAL_BALL_SPEED,
            math.sin(angle) * self.INITIAL_BALL_SPEED
        )
        # Ensure it doesn't start going straight up/down
        if abs(self.ball_vel.y) < 2:
            self.ball_vel.y = 2 if self.ball_vel.y > 0 else -2

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            # --- Action Handling ---
            movement = action[0]
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED

            # Clamp paddle to screen
            self.paddle.x = max(0, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH))

            # --- Game Logic Update ---
            self._update_ball()
            self._update_particles()
            
            # --- Collision Detection ---
            # Ball with Paddle
            if self.ball.colliderect(self.paddle) and self.ball_vel.y > 0:
                # Placeholder for hit sound
                # sfx: paddle_hit
                
                self.score += 1
                reward += 1.0  # Base reward for scoring a point

                # Calculate hit position on paddle (-1 left, 0 center, 1 right)
                hit_pos = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                
                if abs(hit_pos) > 0.5: # Risky hit (outer 25% on each side)
                    reward += 0.5
                    self._create_particles(self.ball.center, 20, self.COLOR_PARTICLE)
                else: # Safe hit (central 50%)
                    reward += 0.1

                # Reverse ball's vertical velocity
                self.ball_vel.y *= -1
                
                # Add horizontal "spin" based on hit position
                self.ball_vel.x += hit_pos * 2.0
                
                # Nudge ball out of paddle to prevent sticking
                self.ball.bottom = self.paddle.top
                
                # Increase ball speed every 2 points
                if self.score > 0 and self.score % 2 == 0:
                    current_speed = self.ball_vel.length()
                    self.ball_vel.scale_to_length(current_speed + 0.5)

            # Ball with Walls
            if self.ball.left <= 0 or self.ball.right >= self.SCREEN_WIDTH:
                self.ball_vel.x *= -1
                self.ball.left = max(0, self.ball.left)
                self.ball.right = min(self.SCREEN_WIDTH, self.ball.right)
                # sfx: wall_bounce
            if self.ball.top <= 0:
                self.ball_vel.y *= -1
                self.ball.top = max(0, self.ball.top)
                # sfx: wall_bounce

            # Ball miss
            if self.ball.top >= self.SCREEN_HEIGHT:
                self.misses += 1
                reward = -1.0
                # sfx: miss_ball
                if self.misses < self.MAX_MISSES:
                    self._reset_ball()
                
        # --- Termination Check ---
        if self.score >= self.WIN_SCORE:
            terminated = True
            self.game_over = True
            reward = 100.0
            # sfx: win_game
        elif self.misses >= self.MAX_MISSES:
            terminated = True
            self.game_over = True
            reward = -100.0
            # sfx: lose_game
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        self.ball.x += self.ball_vel.x
        self.ball.y += self.ball_vel.y

    def _reset_ball(self):
        self.ball.center = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        current_speed = self.ball_vel.length()
        self.ball_vel = pygame.Vector2(math.cos(angle) * current_speed, math.sin(angle) * current_speed)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] - 0.1)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })
    
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
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), int(p['size']))

        # Draw walls (simple lines for retro feel)
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, 0), (self.SCREEN_WIDTH, 0), 2)
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, 0), (0, self.SCREEN_HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - 1, 0), (self.SCREEN_WIDTH - 1, self.SCREEN_HEIGHT), 2)
        
        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with a glow effect
        ball_center = (int(self.ball.centerx), int(self.ball.centery))
        glow_color = (*self.COLOR_BALL, 50) # RGBA for alpha
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_SIZE, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_SIZE // 2, self.COLOR_BALL)
        
    def _render_ui(self):
        # Render Score
        score_text = self.font_medium.render(f"{self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH // 2, 30))
        self.screen.blit(score_text, score_rect)

        # Render Misses
        miss_icon_radius = 8
        miss_spacing = 25
        start_x = self.SCREEN_WIDTH // 2 + 60
        for i in range(self.MAX_MISSES - self.misses):
            pygame.draw.circle(self.screen, self.COLOR_DANGER, (start_x + i * miss_spacing, 30), miss_icon_radius)

        # Render Game Over/Win message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = self.COLOR_PADDLE
            else:
                msg = "GAME OVER"
                color = self.COLOR_DANGER
            
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            # Simple shadow effect
            shadow_text = self.font_large.render(msg, True, (0,0,0))
            shadow_rect = shadow_text.get_rect(center=(self.SCREEN_WIDTH // 2 + 3, self.SCREEN_HEIGHT // 2 + 3))
            self.screen.blit(shadow_text, shadow_rect)
            self.screen.blit(end_text, end_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
            "ball_speed": self.ball_vel.length() if self.ball_vel else 0
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    
    # --- Game Loop ---
    while not terminated:
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()
                terminated = False # Continue playing after reset
        
        env.clock.tick(30) # Run at 30 FPS

    print(f"Game Over. Final Info: {info}")
    env.close()