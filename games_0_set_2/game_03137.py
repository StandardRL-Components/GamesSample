
# Generated: 2025-08-28T07:06:08.463864
# Source Brief: brief_03137.md
# Brief Index: 3137

        
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
    game_description = "A retro-inspired Breakout clone where you clear bricks for points and try to achieve a high score."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 12
    PADDLE_SPEED = 8
    BALL_RADIUS = 6
    INITIAL_BALL_SPEED = 4.5
    MAX_BALL_SPEED = 10.0
    BRICK_ROWS = 6
    BRICK_COLS = 14
    BRICK_WIDTH = 40
    BRICK_HEIGHT = 15
    BRICK_GAP = 4
    BRICK_AREA_Y_START = 50

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (30, 30, 45)
    COLOR_PADDLE = (255, 180, 0)
    COLOR_PADDLE_ACCENT = (255, 220, 100)
    COLOR_BALL = (255, 255, 255)
    COLOR_BALL_GLOW = (200, 200, 255)
    COLOR_TEXT = (220, 220, 220)
    BRICK_COLORS = {
        'red': (217, 87, 99),
        'blue': (69, 173, 224),
        'green': (90, 200, 150)
    }
    BRICK_DATA = [
        {'color': 'red', 'score': 50, 'reward': 5.0},
        {'color': 'red', 'score': 50, 'reward': 5.0},
        {'color': 'blue', 'score': 20, 'reward': 2.0},
        {'color': 'blue', 'score': 20, 'reward': 2.0},
        {'color': 'green', 'score': 10, 'reward': 1.0},
        {'color': 'green', 'score': 10, 'reward': 1.0},
    ]

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Etc...
        self.paddle = None
        self.ball = None
        self.bricks = []
        self.particles = []
        
        # Initialize state variables
        self.reset()
        
        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.ball_speed = self.INITIAL_BALL_SPEED
        self.last_score_milestone = 0
        
        # Paddle
        paddle_y = self.SCREEN_HEIGHT - 40
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) // 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Ball
        self._reset_ball()

        # Bricks
        self.bricks = []
        for r in range(self.BRICK_ROWS):
            for c in range(self.BRICK_COLS):
                brick_data = self.BRICK_DATA[r]
                x = c * (self.BRICK_WIDTH + self.BRICK_GAP) + self.BRICK_GAP * 4
                y = r * (self.BRICK_HEIGHT + self.BRICK_GAP) + self.BRICK_AREA_Y_START
                rect = pygame.Rect(x, y, self.BRICK_WIDTH, self.BRICK_HEIGHT)
                self.bricks.append({
                    'rect': rect,
                    'color': self.BRICK_COLORS[brick_data['color']],
                    'score': brick_data['score'],
                    'reward': brick_data['reward']
                })
        
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _reset_ball(self):
        """Resets the ball's position and velocity after losing a life."""
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upward angle
        self.ball = {
            'pos': [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1],
            'vel': [self.ball_speed * math.cos(angle), self.ball_speed * math.sin(angle)],
            'trail': []
        }

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Action Handling ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = max(0, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH))

        # --- Game Logic ---
        self.steps += 1
        reward = -0.02 # Time penalty for each step
        
        # Ball trail
        self.ball['trail'].append(tuple(self.ball['pos']))
        if len(self.ball['trail']) > 5:
            self.ball['trail'].pop(0)

        # Update ball position
        self.ball['pos'][0] += self.ball['vel'][0]
        self.ball['pos'][1] += self.ball['vel'][1]
        ball_rect = pygame.Rect(self.ball['pos'][0] - self.BALL_RADIUS, self.ball['pos'][1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)

        # --- Collisions ---
        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball['vel'][0] *= -1
            ball_rect.left = max(1, ball_rect.left) # Prevent getting stuck
            ball_rect.right = min(self.SCREEN_WIDTH - 1, ball_rect.right)
            # sfx: wall_bounce

        if ball_rect.top <= 0:
            self.ball['vel'][1] *= -1
            ball_rect.top = 1 # Prevent getting stuck
            # sfx: wall_bounce
        
        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball['vel'][1] > 0:
            self.ball['vel'][1] *= -1
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball['vel'][0] += offset * 2.5 # Influence angle
            self.ball['vel'][0] = max(-self.ball_speed, min(self.ball_speed, self.ball['vel'][0]))
            self.ball['vel'][1] = -abs(self.ball['vel'][1]) # Ensure it goes up
            # sfx: paddle_hit

        # Brick collisions
        hit_brick_idx = ball_rect.collidelist([b['rect'] for b in self.bricks])
        if hit_brick_idx != -1:
            brick_data = self.bricks.pop(hit_brick_idx)
            # sfx: brick_break
            
            self.score += brick_data['score']
            reward += brick_data['reward'] + 0.1 # Base reward + per-hit reward
            
            self._create_particles(brick_data['rect'].center, brick_data['color'])
            
            # Simple but effective collision response
            self.ball['vel'][1] *= -1
        
        # --- State Updates ---
        if ball_rect.top > self.SCREEN_HEIGHT:
            self.lives -= 1
            # sfx: lose_life
            if self.lives > 0:
                self._reset_ball()
            else:
                self.game_over = True
                reward -= 100

        # Difficulty scaling
        if self.score // 500 > self.last_score_milestone:
            self.last_score_milestone = self.score // 500
            new_speed = self.INITIAL_BALL_SPEED + self.last_score_milestone * 0.5
            self.ball_speed = min(self.MAX_BALL_SPEED, new_speed)
            
            current_mag = math.sqrt(self.ball['vel'][0]**2 + self.ball['vel'][1]**2)
            if current_mag > 0:
                self.ball['vel'][0] = (self.ball['vel'][0] / current_mag) * self.ball_speed
                self.ball['vel'][1] = (self.ball['vel'][1] / current_mag) * self.ball_speed

        self._update_particles()
        
        # --- Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.score >= 5000:
                reward += 100
            elif not self.bricks:
                reward += 50
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _check_termination(self):
        return self.score >= 5000 or self.lives <= 0 or self.steps >= 10000 or not self.bricks

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'lifespan': self.np_random.integers(20, 40),
                'color': color, 'radius': self.np_random.uniform(1, 3)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        # Clear screen with background
        self._render_game()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def _render_game(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        
        # --- Bricks ---
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick['color'], brick['rect'], border_radius=2)
            pygame.draw.rect(self.screen, self.COLOR_BG, brick['rect'], 1)

        # --- Particles ---
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 40))
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p['color'], alpha), (p['radius'], p['radius']), p['radius'])
            self.screen.blit(s, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

        # --- Paddle ---
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        accent_rect = self.paddle.copy()
        accent_rect.height = self.PADDLE_HEIGHT // 3
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_ACCENT, accent_rect, border_top_left_radius=3, border_top_right_radius=3)

        # --- Ball ---
        if self.ball:
            # Trail
            for i, pos in enumerate(self.ball['trail']):
                alpha = int(100 * (i / len(self.ball['trail'])))
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.BALL_RADIUS, (*self.COLOR_BALL_GLOW, alpha))

            # Glow
            glow_radius = int(self.BALL_RADIUS * 1.8)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_BALL_GLOW, 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (int(self.ball['pos'][0] - glow_radius), int(self.ball['pos'][1] - glow_radius)))
            
            # Ball
            pygame.gfxdraw.aacircle(self.screen, int(self.ball['pos'][0]), int(self.ball['pos'][1]), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball['pos'][0]), int(self.ball['pos'][1]), self.BALL_RADIUS, self.COLOR_BALL)
        
        # --- UI ---
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        lives_text = self.font_large.render(f"BALLS: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "YOU WIN!" if self.score >= 5000 or not self.bricks else "GAME OVER"
            end_text_surf = self.font_large.render(win_text, True, self.COLOR_PADDLE)
            end_text_rect = end_text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(end_text_surf, end_text_rect)

            final_score_surf = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20))
            self.screen.blit(final_score_surf, final_score_rect)
            
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