import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Survive for 60 seconds."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive the onslaught of multiple bouncing balls in a retro arcade arena. "
        "Deflect balls with your paddle to score points and stay in the game. Last 60 seconds to win."
    )

    # Frames auto-advance at a fixed rate for real-time gameplay.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 12
        self.BALL_RADIUS = 8
        self.PADDLE_SPEED = 15
        self.BALL_SPEED = 6
        self.GRID_SPACING = 40
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS

        # --- Colors ---
        self.COLOR_BG = (20, 40, 30)
        self.COLOR_GRID = (30, 60, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_PADDLE_HIT = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_OVERLAY = (0, 0, 0, 180) # RGBA

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_large = pygame.font.Font(None, 72)
        self.font_small = pygame.font.Font(None, 36)
        
        # --- State Variables ---
        self.paddle = None
        self.balls = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.paddle_hit_timer = 0
        self.np_random = None

        # This validation function is called to ensure the environment conforms to the API.
        # It needs a fully initialized state to run, so we call reset() inside it.
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 20,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.balls = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.paddle_hit_timer = 0

        num_balls = 3
        for _ in range(num_balls):
            self._spawn_ball()
        
        return self._get_observation(), self._get_info()

    def _spawn_ball(self):
        ball_pos = pygame.Vector2(
            self.np_random.uniform(self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS),
            self.np_random.uniform(self.BALL_RADIUS, self.HEIGHT / 2)
        )
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_SPEED
        self.balls.append({'pos': ball_pos, 'vel': ball_vel})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Unpack Action & Update Player ---
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = max(0, min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH))
        
        # --- 2. Update Game Logic ---
        self.steps += 1
        reward = 0.01  # Reward for surviving one step

        if self.paddle_hit_timer > 0:
            self.paddle_hit_timer -= 1

        for ball in self.balls:
            ball['pos'] += ball['vel']
            
            # Wall collisions
            if ball['pos'].x <= self.BALL_RADIUS or ball['pos'].x >= self.WIDTH - self.BALL_RADIUS:
                ball['vel'].x *= -1
                ball['pos'].x = max(self.BALL_RADIUS, min(ball['pos'].x, self.WIDTH - self.BALL_RADIUS))
                self._create_particles(ball['pos'], 5)
            
            if ball['pos'].y <= self.BALL_RADIUS:
                ball['vel'].y *= -1
                ball['pos'].y = self.BALL_RADIUS
                self._create_particles(ball['pos'], 5)

            # Paddle collision
            if self.paddle.colliderect(pygame.Rect(ball['pos'].x - self.BALL_RADIUS, ball['pos'].y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)) and ball['vel'].y > 0:
                ball['vel'].y *= -1
                offset = (ball['pos'].x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                ball['vel'].x += offset * 2.5
                ball['vel'] = ball['vel'].normalize() * self.BALL_SPEED
                
                ball['pos'].y = self.paddle.top - self.BALL_RADIUS
                reward += 1
                self.score += 1
                self.paddle_hit_timer = 5
                self._create_particles(ball['pos'], 15, self.COLOR_PADDLE_HIT)

            # Bottom wall collision (Game Over)
            if ball['pos'].y >= self.HEIGHT:
                self.game_over = True
                self.win = False

        # Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1

        # --- 3. Check Termination ---
        terminated = False
        truncated = False # Gymnasium API: truncated is for time limits, not game over
        if self.game_over:
            terminated = True
            if not self.win:
                reward = -100  # Loss penalty
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = True
            terminated = True # Can be True or False, but often True on win
            truncated = True # Truncated is the correct signal for a time limit
            reward = 100  # Win bonus
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _create_particles(self, pos, count, color=None):
        if color is None:
            color = self.COLOR_BALL
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _get_observation(self):
        # 1. Draw background and grid
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.WIDTH, self.GRID_SPACING):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SPACING):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # 2. Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 20))
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'].x), int(p['pos'].y), 2, (*p['color'], alpha)
            )

        # 3. Draw balls
        for ball in self.balls:
            pos_x, pos_y = int(ball['pos'].x), int(ball['pos'].y)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, self.BALL_RADIUS + 3, (*self.COLOR_BALL, 50))
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, self.BALL_RADIUS, self.COLOR_BALL)

        # 4. Draw paddle (ensure it's not None)
        if self.paddle:
            paddle_color = self.COLOR_PADDLE_HIT if self.paddle_hit_timer > 0 else self.COLOR_PADDLE
            pygame.draw.rect(self.screen, paddle_color, self.paddle, border_radius=3)
            glow_rect = self.paddle.inflate(6, 6)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*paddle_color, 60), (0, 0, *glow_rect.size), border_radius=6)
            self.screen.blit(glow_surface, glow_rect.topleft)

        # 5. Draw UI
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # 6. Draw Game Over/Win screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.win else "GAME OVER"
            end_color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_large.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

        # 7. Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset (must be called before _get_observation to initialize state)
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test observation space (now that state is initialized)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)