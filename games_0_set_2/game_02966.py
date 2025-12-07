
# Generated: 2025-08-27T21:58:36.466629
# Source Brief: brief_02966.md
# Brief Index: 2966

        
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
        "Controls: ←→ to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Reflect a bouncing ball with a moving paddle to break bricks and achieve a target score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        self.screen_width = 640
        self.screen_height = 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.game_font = pygame.font.SysFont("monospace", 24, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 18)
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (220, 220, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_TEXT = (200, 200, 220)
        self.BRICK_COLORS = {
            1: (0, 180, 100),   # Green
            2: (0, 120, 220),   # Blue
            3: (200, 50, 50)    # Red
        }

        # Game parameters
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.BASE_BALL_SPEED = 5.0
        self.MAX_BALL_SPEED_MOD = 1.8
        self.MAX_LIVES = 3
        self.WIN_SCORE = 100
        self.MAX_STEPS = 10000

        # Game state variables (to be initialized in reset)
        self.steps = None
        self.score = None
        self.lives = None
        self.game_over = None
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.bricks = None
        self.particles = None
        self.combo_counter = None
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.combo_counter = 0
        self.particles = []

        paddle_y = self.screen_height - 40
        paddle_x = (self.screen_width - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        self._reset_ball()
        self._create_brick_layout()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 3=left, 4=right
        
        reward = -0.01  # Small time penalty to encourage speed

        self._update_paddle(movement)
        broken_brick = self._update_ball()
        self._update_particles()
        
        if broken_brick:
            points = broken_brick['points']
            self.score += points
            reward += 0.1 + points  # Reward for hitting + points value
            self.combo_counter += 1
            if self.combo_counter >= 3:
                reward += 5  # Combo bonus
                self.combo_counter = 0 # Reset combo after bonus
        else:
            self.combo_counter = 0 # Reset combo if no brick was hit

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE or not self.bricks:
                reward += 100  # Win bonus
            elif self.lives <= 0:
                reward -= 100  # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_paddle(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.screen_width, self.paddle.right)

    def _update_ball(self):
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS, 
            self.ball_pos[1] - self.BALL_RADIUS, 
            self.BALL_RADIUS * 2, self.BALL_RADIUS * 2
        )

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.screen_width:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = max(self.BALL_RADIUS, min(self.ball_pos[0], self.screen_width - self.BALL_RADIUS))
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = max(self.BALL_RADIUS, self.ball_pos[1])
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS - 1

            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2.5
            
            self._normalize_ball_speed()
            # sfx: paddle_hit

        # Brick collisions
        for brick in self.bricks[:]:
            if ball_rect.colliderect(brick['rect']):
                overlap = ball_rect.clip(brick['rect'])
                if overlap.width < overlap.height:
                    self.ball_vel[0] *= -1
                else:
                    self.ball_vel[1] *= -1
                
                self.bricks.remove(brick)
                self._create_particles(brick['rect'].center, brick['color'])
                # sfx: brick_break
                return brick # Return broken brick info

        # Ball out of bounds (lose life)
        if ball_rect.top >= self.screen_height:
            self.lives -= 1
            # sfx: lose_life
            if self.lives > 0:
                self._reset_ball()
            else:
                self.game_over = True
        
        return None

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
        }

    def _check_termination(self):
        return (
            self.lives <= 0 or
            self.score >= self.WIN_SCORE or
            self.steps >= self.MAX_STEPS or
            not self.bricks
        )

    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 5]
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        speed = self._get_current_ball_speed()
        self.ball_vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self._normalize_ball_speed()

    def _normalize_ball_speed(self):
        current_speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
        target_speed = self._get_current_ball_speed()
        if current_speed > 0:
            factor = target_speed / current_speed
            self.ball_vel[0] *= factor
            self.ball_vel[1] *= factor
        # Anti-stuck: ensure ball has some vertical velocity
        if abs(self.ball_vel[1]) < 0.2:
            self.ball_vel[1] = -0.2 if self.ball_vel[1] <= 0 else 0.2

    def _get_current_ball_speed(self):
        speed_multiplier = 1 + (self.score // 20) * 0.1
        return self.BASE_BALL_SPEED * min(speed_multiplier, self.MAX_BALL_SPEED_MOD)

    def _create_brick_layout(self):
        self.bricks = []
        brick_rows = 5
        brick_cols = 10
        brick_width = 60
        brick_height = 20
        gap = 4
        top_offset = 50
        side_offset = (self.screen_width - (brick_cols * (brick_width + gap) - gap)) / 2
        for r in range(brick_rows):
            for c in range(brick_cols):
                points = 1
                if r < 2: points = 3
                elif r < 4: points = 2
                x = side_offset + c * (brick_width + gap)
                y = top_offset + r * (brick_height + gap)
                self.bricks.append({
                    'rect': pygame.Rect(x, y, brick_width, brick_height),
                    'color': self.BRICK_COLORS[points],
                    'points': points
                })

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos), 'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color, 'radius': self.np_random.uniform(1, 4)
            })
            
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _render_game(self):
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick['color'], brick['rect'], border_radius=2)
        
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            px, py = int(p['pos'][0]), int(p['pos'][1])
            pr = int(p['radius'])
            r, g, b = p['color']
            temp_surf = pygame.Surface((pr * 2, pr * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (r,g,b,alpha), (pr, pr), pr)
            self.screen.blit(temp_surf, (px - pr, py - pr), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        for i in range(4, 0, -1):
            alpha = 80 - i * 20
            pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + i, (*self.COLOR_BALL, alpha))
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        score_text = self.game_font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        for i in range(self.lives):
            life_x = self.screen_width - 25 - (i * (self.BALL_RADIUS * 2 + 5))
            life_y = 10 + self.BALL_RADIUS
            pygame.gfxdraw.aacircle(self.screen, life_x, life_y, self.BALL_RADIUS, self.COLOR_PADDLE)
            pygame.gfxdraw.filled_circle(self.screen, life_x, life_y, self.BALL_RADIUS, self.COLOR_PADDLE)

        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = self.score >= self.WIN_SCORE or not self.bricks
            message = "YOU WIN!" if win_condition else "GAME OVER"
            end_text = self.game_font.render(message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.screen_width/2, self.screen_height/2 - 20))
            self.screen.blit(end_text, text_rect)
            
            final_score_text = self.small_font.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.screen_width/2, self.screen_height/2 + 20))
            self.screen.blit(final_score_text, final_score_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")