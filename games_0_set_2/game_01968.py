
# Generated: 2025-08-27T18:51:31.496951
# Source Brief: brief_01968.md
# Brief Index: 1968

        
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
        "A vibrant, grid-based Breakout game with neon visuals and a risk-reward scoring system."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 50)
    COLOR_BORDER = (0, 128, 255)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_PADDLE_GLOW = (100, 150, 255)
    COLOR_BALL = (0, 255, 255)
    COLOR_BALL_GLOW = (100, 200, 255)
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = [
        (255, 0, 128),  # Magenta
        (255, 128, 0),  # Orange
        (255, 255, 0),  # Yellow
        (0, 255, 0),    # Green
        (0, 128, 255),  # Blue
    ]
    
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 8.0
    BALL_RADIUS = 7
    BALL_MAX_SPEED = 10.0
    BALL_MIN_SPEED = 4.0
    BLOCK_ROWS = 5
    BLOCK_COLS = 11
    BLOCK_WIDTH = 50
    BLOCK_HEIGHT = 20
    BLOCK_SPACING = 4
    MAX_STEPS = 10000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_score = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_lives = pygame.font.SysFont("Arial", 30)
        self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)
        
        # Initialize state variables to be properly set in reset()
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.np_random = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.paddle_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self._reset_ball()

        self.blocks = []
        grid_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.SCREEN_WIDTH - grid_width) / 2
        start_y = 50
        for r in range(self.BLOCK_ROWS):
            row_blocks = []
            for c in range(self.BLOCK_COLS):
                x = start_x + c * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = start_y + r * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                color = self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)]
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                row_blocks.append({"rect": block_rect, "color": color, "active": True})
            self.blocks.append(row_blocks)

        self.particles = []
        self.lives = 3
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()
    
    def _reset_ball(self):
        self.ball_pos = np.array([self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS - 1], dtype=np.float64)
        angle = self.np_random.uniform(math.radians(225), math.radians(315))
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * self.BALL_MIN_SPEED
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0

        # 1. Update Paddle
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        self.paddle_rect.x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle_rect.x))

        # 2. Update Ball
        self.ball_pos += self.ball_vel
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2, self.BALL_RADIUS * 2
        )

        # 3. Collision Detection
        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(1, ball_rect.left)
            ball_rect.right = min(self.SCREEN_WIDTH - 1, ball_rect.right)
            self.ball_pos[0] = ball_rect.centerx
            # sfx_wall_hit
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(1, ball_rect.top)
            self.ball_pos[1] = ball_rect.centery
            # sfx_wall_hit

        # Bottom wall (lose life)
        if ball_rect.top >= self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 10.0
            # sfx_lose_life
            if self.lives <= 0:
                self.game_over = True
            else:
                self._reset_ball()
            terminated = self.game_over or self.steps >= self.MAX_STEPS
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Paddle collision
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel[1] > 0:
            # sfx_paddle_hit
            self.ball_vel[1] *= -1
            ball_rect.bottom = self.paddle_rect.top - 1
            self.ball_pos[1] = ball_rect.centery
            hit_pos_norm = (ball_rect.centerx - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += hit_pos_norm * 2.0
            reward += 0.1 * abs(hit_pos_norm) - 0.1 * (1 - abs(hit_pos_norm))
            speed = np.linalg.norm(self.ball_vel)
            if speed > self.BALL_MAX_SPEED: self.ball_vel = self.ball_vel / speed * self.BALL_MAX_SPEED
            elif speed < self.BALL_MIN_SPEED: self.ball_vel = self.ball_vel / speed * self.BALL_MIN_SPEED

        # Block collisions
        block_hit = False
        for row in self.blocks:
            for block_data in row:
                if block_data["active"]:
                    if not block_hit and ball_rect.colliderect(block_data["rect"]):
                        # sfx_block_break
                        block_data["active"] = False
                        block_hit = True
                        self.score += 1
                        reward += 1.0
                        self._create_particles(block_data["rect"].center, block_data["color"])
                        overlap_x = min(ball_rect.right, block_data["rect"].right) - max(ball_rect.left, block_data["rect"].left)
                        overlap_y = min(ball_rect.bottom, block_data["rect"].bottom) - max(ball_rect.top, block_data["rect"].top)
                        if overlap_x > overlap_y: self.ball_vel[1] *= -1
                        else: self.ball_vel[0] *= -1

        # 4. Update Particles
        self._update_particles()
        
        # 5. Check for win condition
        num_active_blocks = sum(b["active"] for row in self.blocks for b in row)
        if num_active_blocks == 0 and not self.game_over:
            self.game_over = True
            reward += 100.0 # Win bonus
            # sfx_win_game
        
        # 6. Update steps and check for max steps termination
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": np.array(pos, dtype=np.float64), "vel": vel,
                "lifetime": self.np_random.integers(15, 30), "color": color,
                "radius": self.np_random.uniform(1, 3)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifetime"] > 1]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["lifetime"] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 2)

    def _render_game_elements(self):
        for row in self.blocks:
            for block in row:
                if block["active"]:
                    pygame.draw.rect(self.screen, block["color"], block["rect"])
                    pygame.draw.rect(self.screen, self.COLOR_BG, block["rect"].inflate(-6, -6))
        
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), int(p["radius"]))
            
        glow_rect = self.paddle_rect.inflate(8, 8)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_PADDLE_GLOW, 50), glow_surface.get_rect(), border_radius=8)
        self.screen.blit(glow_surface, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=5)
        
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        current_radius = self.BALL_RADIUS + pulse * 2
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        glow_radius = int(current_radius * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_BALL_GLOW, 80), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (ball_pos_int[0] - glow_radius, ball_pos_int[1] - glow_radius))
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], int(current_radius), self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], int(current_radius), self.COLOR_BALL)

    def _render_ui(self):
        score_text = self.font_score.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        lives_text = " ".join(["❤"] * self.lives)
        lives_surface = self.font_lives.render(lives_text, True, (255, 50, 50))
        self.screen.blit(lives_surface, (self.SCREEN_WIDTH - lives_surface.get_width() - 10, 5))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            num_active_blocks = sum(b["active"] for row in self.blocks for b in row)
            win = num_active_blocks == 0
            end_text_str = "YOU WIN!" if win else "GAME OVER"
            end_color = (50, 255, 50) if win else (255, 50, 50)
            
            end_text = self.font_game_over.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.lives}

    def close(self):
        pygame.quit()