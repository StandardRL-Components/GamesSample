
# Generated: 2025-08-27T21:17:11.037883
# Source Brief: brief_02736.md
# Brief Index: 2736

        
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
    """A simple particle for effects like explosions."""
    def __init__(self, pos, np_random):
        self.np_random = np_random
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        self.pos = list(pos)
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.lifespan = self.np_random.integers(15, 30)
        self.color = (255, 255, 255)
        self.radius = self.np_random.uniform(1, 3)

    def update(self):
        """Update particle position and lifespan."""
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[0] *= 0.95  # Friction
        self.vel[1] *= 0.95
        self.lifespan -= 1
        return self.lifespan > 0

    def draw(self, surface):
        """Draw the particle on the screen."""
        if self.lifespan > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / 20))))
            color = (*self.color, alpha)
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (int(self.pos[0] - self.radius), int(self.pos[1] - self.radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: ←→ to move the paddle. Press space to launch the ball."
    game_description = "A fast-paced, neon-themed arcade game. Bounce the ball to break all the blocks and clear the stages."
    auto_advance = True

    # Constants
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 8
    INITIAL_BALL_SPEED = 6.0
    FPS = 30
    STAGE_TIME_SECONDS = 60
    MAX_STAGES = 3

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_PADDLE = (0, 200, 200)
    COLOR_PADDLE_HIGHLIGHT = (150, 255, 255)
    COLOR_BALL = (255, 255, 0)
    BLOCK_COLORS = [(255, 50, 50), (50, 255, 50), (50, 100, 255), (255, 50, 255)]
    COLOR_TEXT = (220, 220, 240)
    COLOR_GAMEOVER = (255, 80, 80)
    COLOR_WIN = (80, 255, 80)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_on_paddle = None
        self.ball_speed = None
        self.blocks = []
        self.particles = []
        self.score = 0
        self.stage = 0
        self.lives = 0
        self.stage_timer = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.score = 0
        self.stage = 1
        self.lives = 3
        self.game_over = False
        self.win = False
        self.steps = 0
        self.particles.clear()

        self._setup_stage()
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.stage_timer = self.STAGE_TIME_SECONDS * self.FPS
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball_speed = self.INITIAL_BALL_SPEED + (self.stage - 1) * 0.7
        self._reset_ball()
        self._create_block_layout()
        self.particles.clear()

    def _reset_ball(self):
        self.ball_on_paddle = True
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = [math.sin(angle) * self.ball_speed, -math.cos(angle) * self.ball_speed]

    def _create_block_layout(self):
        self.blocks.clear()
        block_width, block_height = 58, 20
        gap = 4
        
        layout_map = {1: (4, 10, 'full'), 2: (6, 10, 'checker'), 3: (8, 10, 'pyramid')}
        rows, cols, pattern = layout_map.get(self.stage, (4, 10, 'full'))

        start_x = (self.WIDTH - (cols * (block_width + gap))) // 2
        start_y = 60

        for r in range(rows):
            for c in range(cols):
                if pattern == 'checker' and (r + c) % 2 == 0: continue
                if pattern == 'pyramid' and (c < r or c >= cols - r): continue
                
                color = self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)]
                block = pygame.Rect(
                    start_x + c * (block_width + gap),
                    start_y + r * (block_height + gap),
                    block_width,
                    block_height,
                )
                self.blocks.append({"rect": block, "color": color})

    def step(self, action):
        reward = 0
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle Input & Update Paddle
        movement = action[0]
        space_held = action[1] == 1

        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            reward -= 0.01
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            reward -= 0.01

        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.WIDTH, self.paddle.right)

        if self.ball_on_paddle and space_held:
            self.ball_on_paddle = False
            # sfx: launch_ball

        # 2. Update Game State
        self.stage_timer -= 1
        self.steps += 1

        if self.ball_on_paddle:
            self.ball_pos[0] = self.paddle.centerx
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
        else:
            reward -= 0.002  # Small penalty to encourage faster clears
            self._update_ball_position()
            reward += self._handle_collisions()

        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        # 4. Check for Stage/Game End Conditions
        terminated = False
        if not self.blocks:
            # sfx: stage_clear
            reward += 5.0
            self.score += 100
            self.stage += 1
            if self.stage > self.MAX_STAGES:
                self.game_over = True
                self.win = True
                reward += 100.0
            else:
                self._setup_stage()

        if self.stage_timer <= 0 and not self.game_over:
            self._lose_life()
            if not self.game_over:
                reward -= 10 # Penalty for timeout
                self._setup_stage()
            else:
                reward -= 100.0

        if self.game_over:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_ball_position(self):
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS,
            self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
        )

        # Wall collision
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos[0]))
            # sfx: wall_bounce
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = max(-self.ball_speed * 0.95, min(self.ball_speed * 0.95, self.ball_vel[0] + offset * 2.5))
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            # sfx: paddle_bounce

        # Block collision
        hit_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            hit_block = self.blocks.pop(hit_block_idx)
            reward += 1.0
            self.score += 10
            self._create_particles(hit_block['rect'].center)
            
            # Simple collision response
            self.ball_vel[1] *= -1
            # sfx: block_break

        # Bottom wall (lose life)
        if self.ball_pos[1] > self.HEIGHT:
            self._lose_life()
            if not self.game_over:
                self._reset_ball()
            else:
                reward -= 100.0

        return reward
    
    def _lose_life(self):
        self.lives -= 1
        # sfx: lose_life
        if self.lives <= 0:
            self.game_over = True
            self.win = False

    def _create_particles(self, pos):
        for _ in range(20):
            self.particles.append(Particle(pos, self.np_random))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for block_data in self.blocks:
            pygame.draw.rect(self.screen, block_data['color'], block_data['rect'], border_radius=2)
            highlight_color = tuple(min(255, c + 50) for c in block_data['color'])
            pygame.draw.rect(self.screen, highlight_color, block_data['rect'], 1, border_radius=2)

        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        highlight_rect = self.paddle.copy()
        highlight_rect.height = 3
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_HIGHLIGHT, highlight_rect, border_radius=3)

        center_x, center_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        glow_radius = self.BALL_RADIUS + 4
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_BALL, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (center_x - glow_radius, center_y - glow_radius))
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, self.BALL_RADIUS, self.COLOR_BALL)

        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        lives_text = self.font_small.render(f"BALLS: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        stage_text = self.font_small.render(f"STAGE: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH // 2 - stage_text.get_width() // 2, 10))

        timer_text = self.font_small.render(f"TIME: {self.stage_timer // self.FPS}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH // 2 - timer_text.get_width() // 2, 35))
        
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_GAMEOVER
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage, "lives": self.lives}

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")