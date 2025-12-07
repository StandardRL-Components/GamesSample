
# Generated: 2025-08-27T22:27:55.367499
# Source Brief: brief_03134.md
# Brief Index: 3134

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, retro-style brick breaker. Clear all bricks before time runs out or you lose all your lives."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # As per brief's interpolation suggestion
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.INITIAL_BALL_SPEED = 6
        self.MAX_BALL_SPEED = 12
        
        self.BRICK_ROWS, self.BRICK_COLS = 6, 14
        self.BRICK_WIDTH = self.WIDTH // self.BRICK_COLS
        self.BRICK_HEIGHT = 20
        self.BRICK_AREA_TOP_MARGIN = 50

        # Colors
        self.COLOR_BG = (20, 20, 40)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_PADDLE_GLOW = (180, 180, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 150)
        self.COLOR_TEXT = (220, 220, 255)
        self.BRICK_COLORS = [
            (255, 50, 50), (255, 150, 50), (255, 255, 50),
            (50, 255, 50), (50, 150, 255), (150, 50, 255)
        ]
        self.COLOR_GRID = (30, 30, 60)

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
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)

        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_on_paddle = None
        self.bricks = None
        self.initial_brick_count = None
        self.score = None
        self.lives = None
        self.steps = None
        self.time_remaining = None
        self.game_over = None
        self.particles = []
        self.ball_trail = []
        self.steps_since_brick_hit = None
        
        # This will be called once to set up the initial state
        self.reset()
        
        # Self-check
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball_on_paddle = True
        self._reset_ball_on_paddle()

        self.bricks = []
        for r in range(self.BRICK_ROWS):
            for c in range(self.BRICK_COLS):
                brick = pygame.Rect(
                    c * self.BRICK_WIDTH,
                    self.BRICK_AREA_TOP_MARGIN + r * self.BRICK_HEIGHT,
                    self.BRICK_WIDTH,
                    self.BRICK_HEIGHT,
                )
                self.bricks.append({"rect": brick, "color": self.BRICK_COLORS[r % len(self.BRICK_COLORS)]})
        self.initial_brick_count = len(self.bricks)

        self.score = 0
        self.lives = 3
        self.steps = 0
        self.time_remaining = self.MAX_STEPS
        self.game_over = False
        self.particles = []
        self.ball_trail = []
        self.steps_since_brick_hit = 0

        return self._get_observation(), self._get_info()

    def _reset_ball_on_paddle(self):
        self.ball_on_paddle = True
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top
        self.ball_vel = [0, 0]

    def step(self, action):
        reward = 0
        
        if self.game_over:
            terminated = self._check_termination()
            return self._get_observation(), 0, terminated, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1
        
        # Update counters
        self.steps += 1
        self.time_remaining -= 1

        # --- Game Logic ---
        
        # 1. Handle player input
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            reward -= 0.02 # As per brief, penalty for moving
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            reward -= 0.02

        # Keep paddle on screen
        self.paddle.x = max(0, min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH))

        # 2. Launch ball
        if self.ball_on_paddle:
            self.ball.centerx = self.paddle.centerx
            if space_pressed:
                # sfx: launch_ball.wav
                self.ball_on_paddle = False
                initial_angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
                self.ball_vel = [
                    self.INITIAL_BALL_SPEED * math.cos(initial_angle),
                    self.INITIAL_BALL_SPEED * math.sin(initial_angle)
                ]
                self.steps_since_brick_hit = 0
        
        # 3. Update ball position and handle collisions
        if not self.ball_on_paddle:
            self.ball_trail.append(self.ball.center)
            if len(self.ball_trail) > 5:
                self.ball_trail.pop(0)

            self.ball.x += self.ball_vel[0]
            self.ball.y += self.ball_vel[1]
            self.steps_since_brick_hit += 1

            # Wall collisions
            if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
                self.ball_vel[0] *= -1
                self.ball.left = max(0, self.ball.left)
                self.ball.right = min(self.WIDTH, self.ball.right)
                # sfx: wall_bounce.wav
            if self.ball.top <= 0:
                self.ball_vel[1] *= -1
                self.ball.top = max(0, self.ball.top)
                # sfx: wall_bounce.wav

            # Paddle collision
            if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
                # sfx: paddle_hit.wav
                self.ball_vel[1] *= -1
                self.ball.bottom = self.paddle.top # Prevent sticking
                
                offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] += offset * 2.5
                
                speed = math.hypot(*self.ball_vel)
                current_max_speed = self.INITIAL_BALL_SPEED + (self.MAX_BALL_SPEED - self.INITIAL_BALL_SPEED) * (1 - len(self.bricks) / self.initial_brick_count)
                if speed > 0:
                    self.ball_vel[0] = (self.ball_vel[0] / speed) * current_max_speed
                    self.ball_vel[1] = (self.ball_vel[1] / speed) * current_max_speed

            # Brick collisions
            hit_brick = None
            for brick_data in self.bricks:
                if self.ball.colliderect(brick_data["rect"]):
                    hit_brick = brick_data
                    break
            
            if hit_brick:
                # sfx: brick_destroy.wav
                self.bricks.remove(hit_brick)
                self.score += 10
                reward += 0.1
                self.ball_vel[1] *= -1
                self._create_particles(hit_brick["rect"].center, hit_brick["color"])
                self.steps_since_brick_hit = 0

            # Anti-softlock mechanism (as per brief, 100 steps)
            if self.steps_since_brick_hit > 100:
                self.ball_vel[0] += self.np_random.uniform(-0.5, 0.5)
                self.ball_vel[1] += self.np_random.uniform(-0.5, 0.5)
                self.steps_since_brick_hit = 0

            # Miss (ball goes past paddle)
            if self.ball.top >= self.HEIGHT:
                # sfx: lose_life.wav
                self.lives -= 1
                reward -= 1.0
                self.ball_trail.clear()
                if self.lives > 0:
                    self._reset_ball_on_paddle()
                else:
                    self.game_over = True
        
        # 4. Update particles
        self._update_particles()
        
        # 5. Check for termination conditions
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if not self.bricks: # Win condition
                self.score += 1000 # Bonus points not part of reward
                reward += 100
            elif self.time_remaining <= 0: # Time out
                reward -= 10

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _check_termination(self):
        return self.lives <= 0 or not self.bricks or self.time_remaining <= 0 or self.steps >= self.MAX_STEPS

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
        # Background Grid
        for x in range(0, self.WIDTH, self.BRICK_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.BRICK_AREA_TOP_MARGIN), (x, self.HEIGHT))
        for y in range(self.BRICK_AREA_TOP_MARGIN, self.HEIGHT, self.BRICK_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Bricks
        for brick_data in self.bricks:
            r = brick_data["rect"]
            c = brick_data["color"]
            pygame.draw.rect(self.screen, c, r)
            pygame.draw.rect(self.screen, self.COLOR_BG, r, 2) # Border

        # Paddle
        pygame.gfxdraw.box(self.screen, self.paddle, self.COLOR_PADDLE_GLOW)
        inner_paddle = self.paddle.inflate(-6, -6)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, inner_paddle)
        
        # Ball Trail
        for i, pos in enumerate(self.ball_trail):
            alpha = int(255 * (i / len(self.ball_trail)) * 0.5)
            pygame.gfxdraw.filled_circle(
                self.screen, int(pos[0]), int(pos[1]), self.BALL_RADIUS, (*self.COLOR_BALL_GLOW, alpha)
            )

        # Ball
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        
        # Particles
        for p in self.particles:
            pygame.draw.rect(self.screen, p['color'], (p['x'], p['y'], p['size'], p['size']))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time
        time_seconds = max(0, self.time_remaining // self.FPS)
        time_text = self.font_small.render(f"TIME: {time_seconds}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH // 2 - time_text.get_width() // 2, 10))

        # Lives
        lives_text = self.font_small.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 150, 10))
        for i in range(self.lives):
            life_icon_rect = pygame.Rect(self.WIDTH - 80 + (i * 25), 12, 20, 5)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_icon_rect)

        # Game Over Message
        if self.game_over:
            msg = ""
            if not self.bricks:
                msg = "YOU WIN!"
            elif self.lives <= 0:
                msg = "GAME OVER"
            elif self.time_remaining <= 0:
                msg = "TIME UP"
            
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_remaining": len(self.bricks),
            "time_remaining": self.time_remaining,
        }
        
    def _create_particles(self, pos, color):
        for _ in range(15): # Number of particles
            self.particles.append({
                'x': pos[0],
                'y': pos[1],
                'vx': self.np_random.uniform(-3, 3),
                'vy': self.np_random.uniform(-4, 2),
                'size': self.np_random.integers(2, 5),
                'life': 20, # Frames to live
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.2 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

    def close(self):
        pygame.quit()