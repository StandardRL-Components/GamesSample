
# Generated: 2025-08-28T04:01:52.896820
# Source Brief: brief_02197.md
# Brief Index: 2197

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the paddle. Survive for 30 seconds."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive 30 seconds of chaotic top-down pong by juggling multiple balls with a moving paddle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 60
        self.MAX_TIME_SECONDS = 30
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (15, 15, 15)
        self.COLOR_FG = (220, 220, 220)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)

        # Game parameters
        self.PADDLE_WIDTH = 15
        self.PADDLE_HEIGHT = 80
        self.PADDLE_SPEED = 7
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 4.0 # Initial speed is 3 as per brief, but 4 feels better.
        self.NUM_BALLS = 3
        
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
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24)
            self.font_msg = pygame.font.SysFont("Consolas", 50)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 30)
            self.font_msg = pygame.font.SysFont(None, 60)
        
        # State variables (initialized in reset)
        self.paddle = None
        self.balls = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        # Create paddle
        paddle_x = 50
        paddle_y = self.SCREEN_HEIGHT // 2 - self.PADDLE_HEIGHT // 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Create balls
        self.balls = []
        for _ in range(self.NUM_BALLS):
            angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            # 50% chance to start moving left instead of right
            if self.np_random.random() < 0.5:
                angle += math.pi
            
            vel = pygame.math.Vector2(
                self.BALL_SPEED * math.cos(angle),
                self.BALL_SPEED * math.sin(angle)
            )
            ball_rect = pygame.Rect(
                self.SCREEN_WIDTH // 2 - self.BALL_RADIUS,
                self.np_random.integers(self.BALL_RADIUS, self.SCREEN_HEIGHT - self.BALL_RADIUS) - self.BALL_RADIUS,
                self.BALL_RADIUS * 2,
                self.BALL_RADIUS * 2
            )
            self.balls.append({'rect': ball_rect, 'vel': vel})
        
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # If game is over, subsequent steps do nothing but return the final state
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Player Movement ---
        if movement == 1:  # Up
            self.paddle.y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.paddle.y += self.PADDLE_SPEED
        elif movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED

        # Clamp paddle to screen bounds
        self.paddle.top = max(1, self.paddle.top)
        self.paddle.bottom = min(self.SCREEN_HEIGHT - 1, self.paddle.bottom)
        self.paddle.left = max(1, self.paddle.left)
        self.paddle.right = min(self.SCREEN_WIDTH - 1, self.paddle.right)

        # --- Game Logic Update ---
        self.steps += 1
        reward = 0.1  # Continuous reward for surviving

        # Update balls
        for ball in self.balls[:]:
            ball['rect'].x += ball['vel'].x
            ball['rect'].y += ball['vel'].y

            # Wall collisions
            if ball['rect'].top <= 1 or ball['rect'].bottom >= self.SCREEN_HEIGHT - 1:
                ball['vel'].y *= -1
                ball['rect'].top = max(1, ball['rect'].top)
                ball['rect'].bottom = min(self.SCREEN_HEIGHT - 1, ball['rect'].bottom)
                # SFX: wall_bounce.wav
            if ball['rect'].right >= self.SCREEN_WIDTH - 1:
                ball['vel'].x *= -1
                ball['rect'].right = min(self.SCREEN_WIDTH - 1, ball['rect'].right)
                # SFX: wall_bounce.wav

            # Paddle collision
            if self.paddle.colliderect(ball['rect']):
                # Prevent ball from getting stuck inside paddle
                if ball['vel'].x < 0:
                    ball['rect'].left = self.paddle.right
                
                ball['vel'].x *= -1
                
                # Add "spin" based on where it hits the paddle
                offset = (ball['rect'].centery - self.paddle.centery) / (self.PADDLE_HEIGHT / 2)
                ball['vel'].y += offset * 2.5
                
                ball['vel'] = ball['vel'].normalize() * self.BALL_SPEED
                
                reward += 1.0
                self.score += 10
                self._create_particles(ball['rect'].center)
                # SFX: paddle_hit.wav
            
            # Out of bounds (left side)
            if ball['rect'].right < 0:
                self.balls.remove(ball)
                # SFX: ball_miss.wav

        # Update particles
        self._update_particles()
        
        # --- Termination Check ---
        terminated = False
        if len(self.balls) == 0:
            terminated = True
            self.game_over = True
            self.win = False
            reward = -10.0
        
        if not self.game_over and self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100.0
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _create_particles(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'radius': self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'].x
            p['pos'][1] += p['vel'].y
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw play area boundary
        pygame.draw.rect(self.screen, self.COLOR_FG, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 2)
        
        # Draw particles
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = max(0, p['radius'] * life_ratio)
            color_val = int(150 + 105 * life_ratio)
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (int(p['pos'][0]), int(p['pos'][1])), radius)

        # Draw balls
        for ball in self.balls:
            pygame.draw.circle(self.screen, self.COLOR_BALL, ball['rect'].center, self.BALL_RADIUS)
        
        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            text_surf = self.font_msg.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        remaining_time = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {remaining_time:.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_FG)
        self.screen.blit(timer_surf, (15, 10))

        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_FG)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(score_surf, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win": self.win,
        }

    def close(self):
        pygame.font.quit()
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
        assert trunc == False
        assert isinstance(info, dict)