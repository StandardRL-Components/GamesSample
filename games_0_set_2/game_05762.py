
# Generated: 2025-08-28T06:02:26.131535
# Source Brief: brief_05762.md
# Brief Index: 5762

        
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
        "Controls: Use ← and → to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-arcade block breaker. Clear the screen of all blocks using your paddle and ball. "
        "Hit the side walls to activate a score multiplier."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 10
        self.PADDLE_Y = self.HEIGHT - 30
        self.BALL_RADIUS = 7
        self.BALL_SPEED = 7.0
        self.MAX_STEPS = 1500

        # --- Colors ---
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_GRID = (20, 35, 60)
        self.COLOR_PADDLE = (0, 255, 255) # Cyan
        self.COLOR_BALL = (255, 255, 255) # White
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_BONUS = (255, 215, 0) # Gold
        self.BLOCK_COLORS = [
            (255, 0, 128),   # Magenta
            (0, 255, 0),     # Green
            (255, 128, 0),   # Orange
            (255, 255, 0),   # Yellow
            (128, 0, 255),   # Purple
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        
        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 0
        self.paddle_x = 0
        self.ball_pos = [0.0, 0.0]
        self.ball_vel = [0.0, 0.0]
        self.ball_attached = True
        self.blocks = []
        self.block_colors = []
        self.particles = []
        self.bonus_multiplier = 1.0
        self.bonus_timer = 0
        self.prev_space_held = False
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        self.bonus_multiplier = 1.0
        self.bonus_timer = 0
        self.particles.clear()
        
        self._create_blocks()
        self._reset_ball_and_paddle()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1

        # --- 1. Handle Input ---
        movement, space_held = action[0], action[1] == 1
        
        if movement == 3:  # Left
            self.paddle_x = max(0, self.paddle_x - self.PADDLE_SPEED)
            reward -= 0.01 # Discourage wiggling
        elif movement == 4:  # Right
            self.paddle_x = min(self.WIDTH - self.PADDLE_WIDTH, self.paddle_x + self.PADDLE_SPEED)
            reward -= 0.01
        
        if self.ball_attached and space_held and not self.prev_space_held:
            # sfx: launch_ball
            self.ball_attached = False
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            self.ball_vel = [math.cos(angle) * self.BALL_SPEED, math.sin(angle) * self.BALL_SPEED]
        self.prev_space_held = space_held

        # --- 2. Update Game Logic ---
        self._update_particles()
        
        if self.bonus_timer > 0:
            self.bonus_timer -= 1
            if self.bonus_timer == 0:
                self.bonus_multiplier = 1.0

        if self.ball_attached:
            self.ball_pos[0] = self.paddle_x + self.PADDLE_WIDTH / 2
        else:
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
            
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            # Wall collisions
            if ball_rect.left <= 0 and self.ball_vel[0] < 0:
                ball_rect.left = 0; self.ball_vel[0] *= -1; reward += self._activate_bonus() # sfx: bonus
            if ball_rect.right >= self.WIDTH and self.ball_vel[0] > 0:
                ball_rect.right = self.WIDTH; self.ball_vel[0] *= -1; reward += self._activate_bonus() # sfx: bonus
            if ball_rect.top <= 0 and self.ball_vel[1] < 0:
                ball_rect.top = 0; self.ball_vel[1] *= -1 # sfx: wall_bounce
            
            # Paddle collision
            paddle_rect = pygame.Rect(self.paddle_x, self.PADDLE_Y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
            if ball_rect.colliderect(paddle_rect) and self.ball_vel[1] > 0:
                # sfx: paddle_bounce
                self.ball_vel[1] *= -1
                offset = (ball_rect.centerx - paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] = max(-self.BALL_SPEED * 0.95, min(self.BALL_SPEED * 0.95, self.ball_vel[0] + offset * 2.5))
                self.ball_pos[1] = self.PADDLE_Y - self.BALL_RADIUS # Prevent sticking

            # Block collisions
            hit_block_idx = ball_rect.collidelist(self.blocks)
            if hit_block_idx != -1:
                # sfx: block_break
                block_rect = self.blocks.pop(hit_block_idx)
                block_color = self.block_colors.pop(hit_block_idx)
                self._create_particles_for_block(block_rect.center, block_color)
                
                reward += 1.0
                self.score += int(10 * self.bonus_multiplier)

                overlap = ball_rect.clip(block_rect)
                if overlap.width < overlap.height: self.ball_vel[0] *= -1
                else: self.ball_vel[1] *= -1
            
            self.ball_pos = [ball_rect.centerx, ball_rect.centery]

            # Floor collision (lose ball)
            if ball_rect.top >= self.HEIGHT:
                # sfx: lose_ball
                self.balls_left -= 1
                reward -= 10
                if self.balls_left > 0:
                    self._reset_ball_and_paddle()
                else:
                    self.game_over = True
        
        # --- 3. Check Termination ---
        terminated = self.game_over or len(self.blocks) == 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            if len(self.blocks) == 0:
                reward += 100 # Win bonus
                self.score += 500
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "balls": self.balls_left}

    def _reset_ball_and_paddle(self):
        self.paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        self.ball_attached = True
        self.ball_pos = [self.paddle_x + self.PADDLE_WIDTH / 2, self.PADDLE_Y - self.BALL_RADIUS]
        self.ball_vel = [0, 0]
        self.bonus_multiplier = 1.0
        self.bonus_timer = 0

    def _create_blocks(self):
        self.blocks.clear()
        self.block_colors.clear()
        
        num_cols = 10
        num_rows = 5
        block_width = self.WIDTH // num_cols
        block_height = 20
        top_offset = 50
        
        for r in range(num_rows):
            for c in range(num_cols):
                color = self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)]
                rect = pygame.Rect(c * block_width + 1, r * block_height + top_offset + 1, block_width - 2, block_height - 2)
                self.blocks.append(rect)
                self.block_colors.append(color)

    def _activate_bonus(self):
        if self.bonus_timer <= 0:
            self.bonus_multiplier = 2.0
            self.bonus_timer = 150 # 5 seconds at 30fps
            return 5
        else:
            self.bonus_timer = 150
            return 0

    def _render_game(self):
        self._draw_grid()
        self._draw_blocks()
        self._draw_particles()
        self._draw_paddle()
        self._draw_ball()

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        
        balls_text = self.font_large.render(f"BALLS: {self.balls_left}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.WIDTH - balls_text.get_width() - 10, 5))

        if self.bonus_timer > 0:
            flash = (self.bonus_timer % 30) > 15
            color = self.COLOR_BONUS if flash else self.COLOR_TEXT
            bonus_text = self.font_medium.render(f"{self.bonus_multiplier:.0f}x MULTIPLIER!", True, color)
            self.screen.blit(bonus_text, (self.WIDTH // 2 - bonus_text.get_width() // 2, 10))

    def _draw_grid(self):
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _draw_blocks(self):
        for i, block in enumerate(self.blocks):
            color = self.block_colors[i]
            border_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.rect(self.screen, border_color, block)
            inner_rect = block.inflate(-4, -4)
            pygame.draw.rect(self.screen, color, inner_rect)

    def _draw_paddle(self):
        rect = pygame.Rect(self.paddle_x, self.PADDLE_Y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self._draw_glow_rect(self.screen, self.COLOR_PADDLE, rect, 10, 4)

    def _draw_ball(self):
        self._draw_glow_circle(self.screen, self.COLOR_BALL, self.ball_pos, self.BALL_RADIUS, 15, 5)

    def _create_particles_for_block(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _draw_particles(self):
        temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for p in self.particles:
            alpha = int(200 * (p['life'] / p['max_life']))
            radius = int(4 * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.draw.circle(temp_surf, (*p['color'], alpha), p['pos'], radius)
        self.screen.blit(temp_surf, (0, 0))

    def _draw_glow_rect(self, surface, color, rect, glow_size, alpha_factor):
        temp_surf = pygame.Surface((rect.width + glow_size * 2, rect.height + glow_size * 2), pygame.SRCALPHA)
        for i in range(glow_size, 0, -1):
            alpha = int(alpha_factor * (1 - i / glow_size)**2)
            glow_rect = pygame.Rect(glow_size - i, glow_size - i, rect.width + i * 2, rect.height + i * 2)
            pygame.draw.rect(temp_surf, (*color, alpha), glow_rect, border_radius=4)
        
        pygame.draw.rect(temp_surf, color, (glow_size, glow_size, rect.width, rect.height), border_radius=3)
        surface.blit(temp_surf, (rect.x - glow_size, rect.y - glow_size))

    def _draw_glow_circle(self, surface, color, center, radius, glow_size, alpha_factor):
        temp_surf = pygame.Surface((radius * 2 + glow_size * 2, radius * 2 + glow_size * 2), pygame.SRCALPHA)
        center_in_surf = (temp_surf.get_width() // 2, temp_surf.get_height() // 2)
        for i in range(glow_size, 0, -1):
            alpha = int(alpha_factor * (1 - i / glow_size)**2)
            pygame.gfxdraw.filled_circle(temp_surf, center_in_surf[0], center_in_surf[1], radius + i, (*color, alpha))
        
        pygame.gfxdraw.aacircle(temp_surf, center_in_surf[0], center_in_surf[1], radius, color)
        pygame.gfxdraw.filled_circle(temp_surf, center_in_surf[0], center_in_surf[1], radius, color)
        surface.blit(temp_surf, (int(center[0]) - center_in_surf[0], int(center[1]) - center_in_surf[1]))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                env.reset()
                total_reward = 0

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            waiting_for_reset = True
            while waiting_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        print("Resetting game.")
                        env.reset()
                        total_reward = 0
                        waiting_for_reset = False
                clock.tick(30)

        clock.tick(30)

    env.close()