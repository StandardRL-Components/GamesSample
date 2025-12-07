
# Generated: 2025-08-28T04:41:25.924632
# Source Brief: brief_02393.md
# Brief Index: 2393

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A retro arcade block-breaker. Clear all 100 blocks to win. "
        "Bouncing the ball off the edges of the paddle sets up a risky, high-reward shot."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 30 * 90 # 90 seconds max episode length

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (30, 50, 80)
        self.COLOR_PADDLE = (220, 220, 220)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_UI = (200, 200, 255)
        self.BLOCK_COLORS = [
            (255, 70, 70), (255, 160, 70), (220, 220, 50),
            (70, 255, 70), (70, 160, 255), (160, 70, 255)
        ]

        # Game Element Properties
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.BASE_BALL_SPEED = 5.0
        self.BLOCK_COLS, self.BLOCK_ROWS = 10, 10
        self.BLOCK_WIDTH = (self.WIDTH - 20) // self.BLOCK_COLS
        self.BLOCK_HEIGHT = 20
        self.BLOCK_Y_OFFSET = 50

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_score = pygame.font.Font(None, 36)
        self.font_info = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- State Variables ---
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_on_paddle = None
        self.blocks = None
        self.score = None
        self.steps = None
        self.balls_left = None
        self.game_over = None
        self.ball_speed_multiplier = None
        self.last_hit_was_risky = None
        self.steps_since_last_block_hit = None
        self.particles = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )
        self.blocks = self._generate_blocks()
        self.score = 0
        self.steps = 0
        self.balls_left = 3
        self.game_over = False
        self.ball_speed_multiplier = 1.0
        self.last_hit_was_risky = False
        self.steps_since_last_block_hit = 0
        self.particles = []
        
        self._reset_ball()

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_on_paddle = True
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        self.ball_vel = [0, 0]

    def _generate_blocks(self):
        blocks = []
        for r in range(self.BLOCK_ROWS):
            for c in range(self.BLOCK_COLS):
                rect = pygame.Rect(
                    10 + c * self.BLOCK_WIDTH,
                    self.BLOCK_Y_OFFSET + r * self.BLOCK_HEIGHT,
                    self.BLOCK_WIDTH - 2, self.BLOCK_HEIGHT - 2
                )
                color = self.BLOCK_COLORS[(r + c) % len(self.BLOCK_COLORS)]
                blocks.append({"rect": rect, "color": color})
        return blocks

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Per-step penalty to encourage speed

        self._handle_input(action)
        ball_reward = self._update_ball()
        self._update_particles()
        reward += ball_reward
        
        self.steps += 1
        terminated = False
        
        if not self.blocks:  # Win
            reward += 100
            terminated = True
            self.game_over = True
        elif self.balls_left <= 0:  # Lose
            reward -= 50
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))
        
        if space_held and self.ball_on_paddle:
            self.ball_on_paddle = False
            launch_angle = self.np_random.uniform(-math.pi / 8, math.pi / 8)
            self.ball_vel = [
                math.sin(launch_angle) * self.BASE_BALL_SPEED,
                -math.cos(launch_angle) * self.BASE_BALL_SPEED
            ]
            # sound: launch_ball.wav

    def _update_ball(self):
        if self.ball_on_paddle:
            self._reset_ball()
            return 0
        
        reward = 0
        self.ball_pos[0] += self.ball_vel[0] * self.ball_speed_multiplier
        self.ball_pos[1] += self.ball_vel[1] * self.ball_speed_multiplier
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)

        # Wall collisions
        if ball_rect.left <= 0:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = self.BALL_RADIUS + 1
        if ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = self.WIDTH - self.BALL_RADIUS - 1
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS + 1
            # sound: wall_bounce.wav

        # Lose ball
        if ball_rect.top >= self.HEIGHT:
            self.balls_left -= 1
            if self.balls_left > 0: self._reset_ball()
            # sound: lose_ball.wav
            return 0

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            self.ball_vel[1] *= -1
            
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = self.BASE_BALL_SPEED * offset * 1.2
            
            speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            if speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / speed) * self.BASE_BALL_SPEED
                self.ball_vel[1] = (self.ball_vel[1] / speed) * self.BASE_BALL_SPEED
            
            self.last_hit_was_risky = abs(offset) > 0.9
            self.steps_since_last_block_hit = 0
            # sound: paddle_bounce.wav

        # Block collisions
        hit_block = None
        for block in self.blocks:
            if ball_rect.colliderect(block["rect"]):
                hit_block = block
                break

        if hit_block:
            # Reward logic
            if self.last_hit_was_risky:
                reward += 2.0
            else:
                reward -= 0.2
            self.last_hit_was_risky = False

            # Collision resolution
            prev_ball_rect = pygame.Rect(ball_rect.left - self.ball_vel[0], ball_rect.top - self.ball_vel[1], ball_rect.width, ball_rect.height)
            if prev_ball_rect.bottom <= hit_block["rect"].top or prev_ball_rect.top >= hit_block["rect"].bottom:
                self.ball_vel[1] *= -1
            else:
                self.ball_vel[0] *= -1

            self._create_particles(hit_block["rect"].center, hit_block["color"])
            self.blocks.remove(hit_block)
            self.score += 1
            self.steps_since_last_block_hit = 0
            
            blocks_destroyed = 100 - len(self.blocks)
            if blocks_destroyed in [25, 50, 75]:
                self.ball_speed_multiplier += 0.05
            # sound: block_break.wav
        else:
            self.steps_since_last_block_hit += 1

        # Anti-softlock mechanism
        if self.steps_since_last_block_hit > self.FPS * 4: # 4 seconds
            self.ball_vel[0] += self.np_random.uniform(-0.5, 0.5)
            self.ball_vel[1] = -abs(self.ball_vel[1]) # ensure it goes upwards
            self.steps_since_last_block_hit = 0

        # Speed assertion
        current_speed = math.hypot(self.ball_vel[0], self.ball_vel[1]) * self.ball_speed_multiplier
        assert current_speed < 20, f"Ball speed {current_speed} exceeded limit"
        
        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "lifetime": lifetime, "color": color})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            r = int(self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * y / self.HEIGHT)
            g = int(self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * y / self.HEIGHT)
            b = int(self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * y / self.HEIGHT)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=2)
            pygame.draw.rect(self.screen, tuple(min(255, c + 30) for c in block["color"]), block["rect"].inflate(-6, -6), border_radius=2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        pygame.draw.rect(self.screen, (255, 255, 255), self.paddle.inflate(-4, -4), border_radius=3)

        # Ball
        if not self.ball_on_paddle:
            glow_radius = int(self.BALL_RADIUS * 2.5)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_BALL, 20))
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, int(glow_radius * 0.7), (*self.COLOR_BALL, 30))
            self.screen.blit(glow_surf, (int(self.ball_pos[0] - glow_radius), int(self.ball_pos[1] - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)

        # Particles
        for p in self.particles:
            alpha = max(0, int(255 * (p["lifetime"] / 30.0)))
            color = (*p["color"], alpha)
            size = max(1, int(self.BALL_RADIUS * 0.4 * (p["lifetime"] / 30.0)))
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), size, color)

    def _render_ui(self):
        score_text = self.font_score.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        for i in range(self.balls_left):
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 25 - i * 25, 25, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 25 - i * 25, 25, self.BALL_RADIUS, self.COLOR_BALL)

        blocks_text = self.font_info.render(f"BLOCKS: {len(self.blocks)}", True, self.COLOR_UI)
        text_rect = blocks_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 20))
        self.screen.blit(blocks_text, text_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "YOU WIN!" if not self.blocks else "GAME OVER"
            text_surf = self.font_game_over.render(win_text, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        assert info['blocks_left'] == 100
        assert info['score'] == 0
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")