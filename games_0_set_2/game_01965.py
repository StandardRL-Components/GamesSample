
# Generated: 2025-08-28T03:14:52.541209
# Source Brief: brief_01965.md
# Brief Index: 1965

        
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
        "Controls: ←→ to move. Hold Shift for slower, precise movement. Press Space to launch the ball."
    )

    game_description = (
        "A retro-arcade block breaker with vibrant neon visuals. Clear all the blocks to win, but don't lose all your balls!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED_FAST = 12
        self.PADDLE_SPEED_SLOW = 5
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 5.0
        self.MAX_BALL_SPEED = 15.0
        self.BALL_SPEED_INCREMENT = 0.1
        self.BLOCK_ROWS, self.BLOCK_COLS = 10, 10
        self.TOTAL_BLOCKS = self.BLOCK_ROWS * self.BLOCK_COLS
        self.MAX_STEPS = 3000

        # --- Colors (Neon on Dark) ---
        self.COLOR_BG_START = (10, 0, 20)
        self.COLOR_BG_END = (30, 0, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_WALL = (100, 100, 255, 150)
        self.COLOR_TEXT = (220, 220, 255)
        self.BLOCK_COLORS = [
            (255, 0, 255), (0, 255, 255), (0, 255, 0),
            (255, 128, 0), (255, 0, 0)
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.big_font = pygame.font.Font(None, 72)
        
        # --- State Variables (initialized in reset) ---
        self.paddle_x = 0
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_attached = True
        self.current_ball_speed = 0.0
        self.blocks = []
        self.score = 0
        self.balls_remaining = 0
        self.steps = 0
        self.game_over = False
        self.blocks_destroyed_this_step = []
        self.blocks_destroyed_count = 0
        self.particles = []
        self.ball_trail = []
        
        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.paddle_x = self.WIDTH / 2
        
        self.blocks = []
        block_width = (self.WIDTH - 2) / self.BLOCK_COLS
        block_height = 20
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                color_index = (i // 2) % len(self.BLOCK_COLORS)
                color = self.BLOCK_COLORS[color_index]
                points = (self.BLOCK_ROWS // 2 - i // 2) * 2 + 1
                block_rect = pygame.Rect(
                    j * block_width + 1,
                    50 + i * block_height,
                    block_width - 2,
                    block_height - 2
                )
                self.blocks.append({'rect': block_rect, 'color': color, 'points': points})
        assert len(self.blocks) == self.TOTAL_BLOCKS

        self.score = 0
        self.balls_remaining = 3
        self.steps = 0
        self.game_over = False
        self.blocks_destroyed_this_step = []
        self.blocks_destroyed_count = 0
        self.particles = []
        self.ball_trail = []
        
        self._reset_ball()
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_attached = True
        self.current_ball_speed = self.INITIAL_BALL_SPEED
        self.ball_pos = pygame.Vector2(self.paddle_x, self.HEIGHT - self.PADDLE_HEIGHT - self.BALL_RADIUS - 5)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_trail.clear()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self.blocks_destroyed_this_step.clear()

        # 1. Handle Action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        old_paddle_x = self.paddle_x
        
        paddle_speed = self.PADDLE_SPEED_SLOW if shift_held else self.PADDLE_SPEED_FAST
        if movement == 3: self.paddle_x -= paddle_speed # Left
        elif movement == 4: self.paddle_x += paddle_speed # Right
            
        self.paddle_x = np.clip(self.paddle_x, self.PADDLE_WIDTH / 2, self.WIDTH - self.PADDLE_WIDTH / 2)
        
        if self.ball_attached and space_held:
            # sfx: launch_ball
            self.ball_attached = False
            angle = self.np_random.uniform(-math.pi / 6, math.pi / 6)
            self.ball_vel = pygame.Vector2(math.sin(angle), -math.cos(angle)) * self.current_ball_speed

        # 2. Update Physics & Get Event Rewards
        event_reward = self._update_game_physics()
        reward += event_reward

        # 3. Update Score & Calculate Block Rewards
        for block in self.blocks_destroyed_this_step:
            self.score += block['points']
            reward += block['points']
            if not self.ball_attached and self.ball_vel.magnitude() > 0 and abs(self.ball_vel.y) / self.ball_vel.magnitude() > 0.9:
                reward += 5 # Risky shot bonus

        # 4. Add Continuous Rewards/Penalties
        if not self.ball_attached:
            reward += 0.01 # Keep ball in play
            # Penalty for moving away from falling ball
            if self.ball_vel.y > 0:
                ball_dx = self.ball_pos.x - self.paddle_x
                paddle_move = self.paddle_x - old_paddle_x
                if paddle_move != 0 and np.sign(paddle_move) != np.sign(ball_dx) and abs(ball_dx) > self.PADDLE_WIDTH / 2:
                    reward -= 0.05
            # Penalty for cautious play (staying in center)
            if abs(self.paddle_x - self.WIDTH/2) < self.PADDLE_WIDTH/4:
                reward -= 0.01

        # 5. Update Visual Effects
        self._update_particles()
        self._update_ball_trail()

        # 6. Check Termination
        if len(self.blocks) == 0:
            self.game_over = True
            reward += 100 # Win bonus
            
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_game_physics(self):
        event_reward = 0
        if self.ball_attached:
            self.ball_pos.x = self.paddle_x
            return event_reward

        self.ball_pos += self.ball_vel
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        paddle_rect = self._get_paddle_rect()

        # Wall collisions
        if self.ball_pos.x <= self.BALL_RADIUS or self.ball_pos.x >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel.x *= -1; self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS) # sfx: wall_bounce
        if self.ball_pos.y <= self.BALL_RADIUS:
            self.ball_vel.y *= -1; self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.HEIGHT) # sfx: wall_bounce

        # Bottom edge (lose ball)
        if self.ball_pos.y >= self.HEIGHT + self.BALL_RADIUS:
            self.balls_remaining -= 1; event_reward -= 10 # sfx: lose_ball
            if self.balls_remaining <= 0: self.game_over = True
            else: self._reset_ball()
            return event_reward

        # Paddle collision
        if self.ball_vel.y > 0 and ball_rect.colliderect(paddle_rect):
            # sfx: paddle_hit
            hit_pos_norm = np.clip((self.ball_pos.x - self.paddle_x) / (self.PADDLE_WIDTH / 2), -1, 1)
            angle = hit_pos_norm * (math.pi / 2.5)
            self.ball_vel = pygame.Vector2(math.sin(angle), -math.cos(angle)).normalize() * self.current_ball_speed
            self.ball_pos.y = paddle_rect.top - self.BALL_RADIUS - 1

        # Block collisions
        for i in range(len(self.blocks) - 1, -1, -1):
            block = self.blocks[i]
            if ball_rect.colliderect(block['rect']):
                # sfx: block_break
                self._create_particles(block['rect'].center, block['color'])
                self.blocks_destroyed_this_step.append(block)
                self.blocks_destroyed_count += 1
                del self.blocks[i]
                
                # Collision response
                overlap = ball_rect.clip(block['rect'])
                if overlap.width < overlap.height: self.ball_vel.x *= -1
                else: self.ball_vel.y *= -1

                # Difficulty scaling
                if self.blocks_destroyed_count > 0 and self.blocks_destroyed_count % 20 == 0:
                    new_speed = self.current_ball_speed + self.BALL_SPEED_INCREMENT
                    self.current_ball_speed = min(new_speed, self.MAX_BALL_SPEED)
                    assert self.current_ball_speed <= self.MAX_BALL_SPEED
                    if self.ball_vel.magnitude_squared() > 0: self.ball_vel.scale_to_length(self.current_ball_speed)
                break
        return event_reward

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "balls": self.balls_remaining}

    def _render_game(self):
        self._draw_background()
        self._draw_walls()
        self._draw_blocks()
        self._draw_particles()
        self._draw_ball()
        self._draw_paddle()
        self._render_ui()

    def _draw_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = tuple(int(s * (1 - ratio) + e * ratio) for s, e in zip(self.COLOR_BG_START, self.COLOR_BG_END))
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _draw_walls(self):
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, 5))

    def _draw_blocks(self):
        for block in self.blocks:
            inner_color = tuple(c * 0.6 for c in block['color'])
            pygame.draw.rect(self.screen, inner_color, block['rect'])
            main_rect = block['rect'].inflate(-4, -4)
            pygame.draw.rect(self.screen, block['color'], main_rect)

    def _get_paddle_rect(self):
        return pygame.Rect(self.paddle_x - self.PADDLE_WIDTH / 2, self.HEIGHT - self.PADDLE_HEIGHT - 5, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

    def _draw_paddle(self):
        paddle_rect = self._get_paddle_rect()
        glow_rect = paddle_rect.inflate(8, 8)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PADDLE, 50), glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)

    def _draw_ball(self):
        for i, pos in enumerate(self.ball_trail):
            alpha = 60 * (i / len(self.ball_trail))
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.BALL_RADIUS, (*self.COLOR_BALL, int(alpha)))
        
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS + 4, (*self.COLOR_BALL, 60))
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS + 2, (*self.COLOR_BALL, 100))
        
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _create_particles(self, pos, color):
        for _ in range(15):
            vel = pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3))
            lifetime = self.np_random.integers(10, 20)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'lifetime': lifetime, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 1]
        for p in self.particles:
            p['pos'] += p['vel']; p['vel'] *= 0.95; p['lifetime'] -= 1

    def _draw_particles(self):
        for p in self.particles:
            size = max(0, int(p['lifetime'] / 4))
            alpha = max(0, int(255 * (p['lifetime'] / 20)))
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            surf.fill((*p['color'], alpha))
            self.screen.blit(surf, p['pos'] - pygame.Vector2(size/2, size/2))

    def _update_ball_trail(self):
        if not self.ball_attached:
            self.ball_trail.append(self.ball_pos.copy())
            if len(self.ball_trail) > 5: self.ball_trail.pop(0)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        for i in range(self.balls_remaining - 1):
            pos_x = self.WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            pygame.gfxdraw.filled_circle(self.screen, pos_x, 20, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos_x, 20, self.BALL_RADIUS, self.COLOR_BALL)

        if self.game_over:
            end_text = "YOU WIN!" if len(self.blocks) == 0 else "GAME OVER"
            text_surf = self.big_font.render(end_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")