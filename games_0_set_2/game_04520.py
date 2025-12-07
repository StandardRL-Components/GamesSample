
# Generated: 2025-08-28T02:38:43.556891
# Source Brief: brief_04520.md
# Brief Index: 4520

        
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

    user_guide = (
        "Controls: Use ← and → to move the paddle. Break all the bricks to advance to the next stage."
    )

    game_description = (
        "A retro-style block breaker. Deflect the ball to destroy bricks, clear all three stages to win. "
        "Aggressive edge-of-paddle hits are rewarded, but be careful not to lose the ball!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 3600 # 3 stages * 60s/stage * 20 steps/s (oops, FPS is 30) -> 5400
        self.MAX_STEPS = 5400 # 60s * 3 stages * 30fps
        self.TOTAL_STAGES = 3

        # Colors
        self.COLOR_BG = (15, 15, 35)
        self.COLOR_GRID = (30, 30, 60)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 255)
        self.BRICK_COLORS = [
            (255, 50, 50), (255, 150, 50), (50, 255, 50),
            (50, 150, 255), (150, 50, 255)
        ]

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
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)

        # Game State - initialized in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.stage = 0
        self.time_remaining = 0.0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_radius = 0
        self.base_ball_speed = 0
        self.bricks = []
        self.particles = []
        self.last_brick_hit_step = 0
        self.win_message = ""

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        self.stage = 0
        self.win_message = ""
        self.particles = []
        
        self._start_stage(new_game=True)
        
        return self._get_observation(), self._get_info()

    def _start_stage(self, new_game=False):
        if new_game:
            self.stage = 1
        else:
            self.stage += 1
        
        self.time_remaining = 60.0
        self.last_brick_hit_step = self.steps

        # Paddle
        paddle_width, paddle_height = 100, 15
        self.paddle = pygame.Rect(
            (self.WIDTH - paddle_width) / 2,
            self.HEIGHT - paddle_height - 10,
            paddle_width,
            paddle_height
        )

        # Ball
        self.ball_radius = 8
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.ball_radius - 1], dtype=float)
        
        # Ball speed increases with stage
        self.base_ball_speed = 6.0 + (self.stage - 1) * 0.4 # 0.2 units/sec * 30fps/step = 6
        
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upwards angle
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)]) * self.base_ball_speed
        
        self._generate_bricks()
        self.particles = []


    def _generate_bricks(self):
        self.bricks = []
        rows, cols = 5, 10
        brick_width, brick_height = 60, 20
        total_width = cols * (brick_width + 4)
        start_x = (self.WIDTH - total_width) / 2
        start_y = 50
        
        for r in range(rows):
            for c in range(cols):
                if self.np_random.random() > 0.1: # 90% chance of brick existing
                    color = self.BRICK_COLORS[(r + c) % len(self.BRICK_COLORS)]
                    brick = pygame.Rect(
                        start_x + c * (brick_width + 4),
                        start_y + r * (brick_height + 4),
                        brick_width,
                        brick_height
                    )
                    self.bricks.append({'rect': brick, 'color': color})


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        dt = self.clock.tick(self.FPS) / 1000.0 * 30 # Time delta scaled to 30fps
        reward = 0.0

        # 1. Handle Input
        movement = action[0]
        self._handle_paddle_movement(movement, dt)

        # 2. Update Game Logic
        ball_reward = self._update_ball(dt)
        reward += ball_reward
        self._update_particles(dt)

        self.time_remaining -= 1.0 / self.FPS
        self.steps += 1

        # 3. Check for state changes
        if not self.bricks: # Stage complete
            reward += 50.0
            if self.stage >= self.TOTAL_STAGES:
                reward += 150.0 # Game win bonus
                self.game_over = True
                self.win_message = "YOU WIN!"
            else:
                self._start_stage()
        
        # Anti-softlock
        if self.steps - self.last_brick_hit_step > 10 * self.FPS:
            self._reset_ball_velocity()
            self.last_brick_hit_step = self.steps

        # 4. Check Termination
        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.lives <= 0:
                reward -= 100.0
                self.win_message = "GAME OVER"
            elif self.time_remaining <= 0:
                 reward -= 100.0
                 self.win_message = "TIME UP"
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_paddle_movement(self, movement, dt):
        paddle_speed = 15.0 * dt
        if movement == 3:  # Left
            self.paddle.x -= paddle_speed
        elif movement == 4:  # Right
            self.paddle.x += paddle_speed
        
        self.paddle.x = max(0, min(self.paddle.x, self.WIDTH - self.paddle.width))

    def _update_ball(self, dt):
        reward = 0.0
        self.ball_pos += self.ball_vel * dt

        ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_radius, self.ball_pos[1] - self.ball_radius, self.ball_radius * 2, self.ball_radius * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(1, ball_rect.left)
            ball_rect.right = min(self.WIDTH - 1, ball_rect.right)
            self.ball_pos[0] = ball_rect.centerx
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(1, ball_rect.top)
            self.ball_pos[1] = ball_rect.centery
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            hit_pos_norm = (ball_rect.centerx - self.paddle.centerx) / (self.paddle.width / 2)
            hit_pos_norm = max(-0.95, min(0.95, hit_pos_norm)) # Clamp to avoid extreme angles

            # Reward for risky hits
            if abs(hit_pos_norm) > 0.7:
                reward += 0.1
            else:
                reward -= 0.02
            
            angle = math.acos(self.ball_vel[0] / np.linalg.norm(self.ball_vel))
            new_angle = math.pi * 1.5 + (math.pi / 2.5 * -hit_pos_norm)
            
            self.ball_vel[0] = math.cos(new_angle) * self.base_ball_speed
            self.ball_vel[1] = math.sin(new_angle) * self.base_ball_speed
            
            self.ball_pos[1] = self.paddle.top - self.ball_radius - 1
            # sfx: paddle_hit

        # Brick collisions
        hit_brick_idx = ball_rect.collidelist([b['rect'] for b in self.bricks])
        if hit_brick_idx != -1:
            brick_data = self.bricks[hit_brick_idx]
            brick_rect = brick_data['rect']
            
            # Determine collision side to correctly reverse velocity
            # A simple approach: check overlap amounts
            overlap_left = ball_rect.right - brick_rect.left
            overlap_right = brick_rect.right - ball_rect.left
            overlap_top = ball_rect.bottom - brick_rect.top
            overlap_bottom = brick_rect.bottom - ball_rect.top
            
            min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)

            if min_overlap == overlap_top or min_overlap == overlap_bottom:
                self.ball_vel[1] *= -1
            else:
                self.ball_vel[0] *= -1

            self._create_particles(brick_rect.center, brick_data['color'], 30)
            self.bricks.pop(hit_brick_idx)
            reward += 1.0
            self.score += 10
            self.last_brick_hit_step = self.steps
            # sfx: brick_destroy
        
        # Bottom wall (lose life)
        if ball_rect.top >= self.HEIGHT:
            self.lives -= 1
            reward -= 10.0 # Penalty for losing a life
            if self.lives > 0:
                self._reset_ball()
            # sfx: lose_life

        return reward

    def _reset_ball(self):
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.ball_radius - 1], dtype=float)
        self._reset_ball_velocity()

    def _reset_ball_velocity(self):
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)]) * self.base_ball_speed

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.uniform(0.3, 0.8)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifetime, 'color': color})

    def _update_particles(self, dt):
        for p in self.particles:
            p['pos'][0] += p['vel'][0] * dt
            p['pos'][1] += p['vel'][1] * dt
            p['vel'][1] += 2.0 * dt # Gravity
            p['life'] -= dt / 30.0
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        return self.lives <= 0 or self.time_remaining <= 0 or (self.stage > self.TOTAL_STAGES) or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        # Background grid
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Bricks
        for brick_data in self.bricks:
            pygame.draw.rect(self.screen, brick_data['color'], brick_data['rect'])
            pygame.draw.rect(self.screen, self.COLOR_BG, brick_data['rect'], 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Ball
        x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.ball_radius, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.ball_radius, self.COLOR_BALL)
        
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * 255)))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            temp_surf.fill(color)
            self.screen.blit(temp_surf, (int(p['pos'][0]), int(p['pos'][1])))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (10, 30))

        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.stage}/{self.TOTAL_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH - stage_text.get_width() - 10, 10))

        # Timer
        time_str = f"TIME: {max(0, int(self.time_remaining)):02d}"
        time_color = (255, 100, 100) if self.time_remaining < 10 else self.COLOR_TEXT
        time_text = self.font_small.render(time_str, True, time_color)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 30))

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_PADDLE)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
            "time_remaining": self.time_remaining,
        }
    
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()