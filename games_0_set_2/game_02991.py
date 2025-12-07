
# Generated: 2025-08-28T06:38:51.470670
# Source Brief: brief_02991.md
# Brief Index: 2991

        
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
        "Controls: ↑ to move up, ↓ to move down. Hit the ball with the edge of your paddle for bonus points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based Pong game with a high-risk, high-reward scoring system. The ball gets faster with every point scored!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 10, 8
        self.CELL_W = self.WIDTH // self.GRID_W
        self.CELL_H = self.HEIGHT // self.GRID_H
        self.PADDLE_HEIGHT_CELLS = 2
        self.BALL_SIZE_CELLS = 1
        self.MAX_STEPS = 5000
        self.WIN_SCORE = 7
        self.MAX_MISSES = 3

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PADDLE = (50, 150, 255)
        self.COLOR_PADDLE_GLOW = (150, 200, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_ACCENT_GOOD = (50, 255, 150)
        self.COLOR_ACCENT_BAD = (255, 100, 100)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_flash = pygame.font.SysFont("Consolas", 20, bold=True)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.paddle_y = 0
        self.ball_pos = [0.0, 0.0]
        self.ball_vel = [0.0, 0.0]
        self.base_ball_speed = 0.0
        self.particles = []
        self.text_flashes = []
        self.ball_trail = []
        self.screen_shake = 0

        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.paddle_y = self.GRID_H // 2 - self.PADDLE_HEIGHT_CELLS // 2
        self.base_ball_speed = 0.2  # cells per frame
        self.particles = []
        self.text_flashes = []
        self.ball_trail = []
        self.screen_shake = 0
        
        self._reset_ball(player_scored=None)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = self._update_game_state(movement)
        
        self.steps += 1
        terminated = (self.score >= self.WIN_SCORE or 
                      self.misses >= self.MAX_MISSES or 
                      self.steps >= self.MAX_STEPS)
        
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                reward += 10
            elif self.misses >= self.MAX_MISSES:
                reward -= 10
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self, movement):
        # Handle paddle movement
        if movement == 1:  # Up
            self.paddle_y = max(0, self.paddle_y - 1)
        elif movement == 2:  # Down
            self.paddle_y = min(self.GRID_H - self.PADDLE_HEIGHT_CELLS, self.paddle_y + 1)
        
        # Update ball, particles, and flashes
        reward = self._update_ball()
        self._update_particles()
        self._update_text_flashes()
        self.screen_shake = max(0, self.screen_shake - 1)

        # Per-frame penalty for ball on player's side
        if self.ball_pos[0] > self.GRID_W / 2:
            reward -= 0.1

        return reward

    def _reset_ball(self, player_scored):
        self.ball_pos = [self.GRID_W / 2, self.GRID_H / 2]
        self.ball_trail = []
        
        # Launch ball
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        direction = -1 if self.np_random.random() < 0.5 else 1
        if player_scored is not None:
            direction = 1 if player_scored else -1

        self.ball_vel = [math.cos(angle) * direction, math.sin(angle)]
        
        # Normalize velocity vector
        norm = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
        if norm > 0:
            self.ball_vel = [v / norm for v in self.ball_vel]

    def _update_ball(self):
        reward = 0
        
        # Update trail
        self.ball_trail.append(list(self.ball_pos))
        if len(self.ball_trail) > 5:
            self.ball_trail.pop(0)

        # Move ball
        self.ball_pos[0] += self.ball_vel[0] * self.base_ball_speed
        self.ball_pos[1] += self.ball_vel[1] * self.base_ball_speed

        # Wall collisions
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.GRID_H - self.BALL_SIZE_CELLS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], 0, self.GRID_H - self.BALL_SIZE_CELLS)
            self._create_particles(self.ball_pos, self.COLOR_GRID, 5)
            # sfx: wall_bounce

        # Left wall (player scores)
        if self.ball_pos[0] < 0:
            self.score += 1
            reward += 1
            self.base_ball_speed += 0.02
            self._reset_ball(player_scored=True)
            self._create_text_flash("+1", (self.WIDTH // 4, self.HEIGHT // 2), self.COLOR_ACCENT_GOOD)
            # sfx: score_point
            return reward

        # Right wall (player misses)
        if self.ball_pos[0] >= self.GRID_W:
            self.misses += 1
            reward -= 1
            self._reset_ball(player_scored=False)
            self._create_text_flash("MISS", (self.WIDTH * 3 // 4, self.HEIGHT // 2), self.COLOR_ACCENT_BAD)
            # sfx: miss_point
            return reward

        # Paddle collision
        paddle_x = self.GRID_W - 1
        if self.ball_vel[0] > 0 and self.ball_pos[0] + self.BALL_SIZE_CELLS > paddle_x:
            ball_y_center = self.ball_pos[1] + self.BALL_SIZE_CELLS / 2
            if self.paddle_y <= ball_y_center <= self.paddle_y + self.PADDLE_HEIGHT_CELLS:
                self.ball_pos[0] = paddle_x - self.BALL_SIZE_CELLS
                self.ball_vel[0] *= -1

                hit_offset = (ball_y_center - (self.paddle_y + self.PADDLE_HEIGHT_CELLS / 2)) / (self.PADDLE_HEIGHT_CELLS / 2)
                self.ball_vel[1] = hit_offset * 0.8
                
                # Normalize velocity
                norm = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
                if norm > 0:
                    self.ball_vel = [v / norm for v in self.ball_vel]

                self.screen_shake = 8
                self._create_particles([paddle_x, self.ball_pos[1]], self.COLOR_PADDLE, 15)
                # sfx: paddle_hit

                reward += 0.1  # Base hit reward
                if abs(hit_offset) > 0.75: # Risky edge hit
                    reward += 2
                    self._create_text_flash("RISKY! +2", (self.WIDTH - 200, self.ball_pos[1] * self.CELL_H), self.COLOR_ACCENT_GOOD)
                else: # Safe center hit
                    reward -= 0.2
        return reward

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle_pos = [pos[0] * self.CELL_W, pos[1] * self.CELL_H]
            self.particles.append({'pos': particle_pos, 'vel': vel, 'life': 20, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _create_text_flash(self, text, pos, color):
        self.text_flashes.append({'text': text, 'pos': list(pos), 'life': 45, 'color': color})

    def _update_text_flashes(self):
        for f in self.text_flashes:
            f['pos'][1] -= 0.5
            f['life'] -= 1
        self.text_flashes = [f for f in self.text_flashes if f['life'] > 0]

    def _get_observation(self):
        render_surface = self.screen
        if self.screen_shake > 0:
            # Use a temporary surface for shaking to avoid permanent offsets
            temp_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
            self._render_all(temp_surface)
            shake_offset_x = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            shake_offset_y = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            self.screen.fill(self.COLOR_BG)
            self.screen.blit(temp_surface, (shake_offset_x, shake_offset_y))
        else:
            self._render_all(self.screen)
        
        arr = pygame.surfarray.array3d(render_surface)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_all(self, surface):
        surface.fill(self.COLOR_BG)
        self._render_game(surface)
        self._render_ui(surface)
    
    def _render_game(self, surface):
        # Draw grid
        for x in range(self.GRID_W + 1):
            px = x * self.CELL_W
            pygame.draw.line(surface, self.COLOR_GRID, (px, 0), (px, self.HEIGHT))
        for y in range(self.GRID_H + 1):
            py = y * self.CELL_H
            pygame.draw.line(surface, self.COLOR_GRID, (0, py), (self.WIDTH, py))

        # Draw paddle glow
        paddle_glow_rect = pygame.Rect(
            (self.GRID_W - 1) * self.CELL_W - 2,
            self.paddle_y * self.CELL_H - 2,
            self.CELL_W + 4,
            self.PADDLE_HEIGHT_CELLS * self.CELL_H + 4
        )
        pygame.draw.rect(surface, self.COLOR_PADDLE_GLOW, paddle_glow_rect, border_radius=6)

        # Draw paddle
        paddle_rect = pygame.Rect(
            (self.GRID_W - 1) * self.CELL_W,
            self.paddle_y * self.CELL_H,
            self.CELL_W,
            self.PADDLE_HEIGHT_CELLS * self.CELL_H
        )
        pygame.draw.rect(surface, self.COLOR_PADDLE, paddle_rect, border_radius=4)
        
        # Draw ball trail
        for i, pos in enumerate(self.ball_trail):
            alpha = (i + 1) / len(self.ball_trail)
            color = (self.COLOR_BALL[0], self.COLOR_BALL[1], self.COLOR_BALL[2], int(alpha * 100))
            radius = int((self.CELL_W / 2) * (i / len(self.ball_trail)))
            px = int(pos[0] * self.CELL_W + self.CELL_W / 2)
            py = int(pos[1] * self.CELL_H + self.CELL_H / 2)
            pygame.gfxdraw.filled_circle(surface, px, py, radius, color)

        # Draw ball
        ball_px = int(self.ball_pos[0] * self.CELL_W + self.CELL_W / 2)
        ball_py = int(self.ball_pos[1] * self.CELL_H + self.CELL_H / 2)
        ball_radius = int(self.CELL_W * self.BALL_SIZE_CELLS / 2)
        pygame.gfxdraw.filled_circle(surface, ball_px, ball_py, ball_radius, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(surface, ball_px, ball_py, ball_radius, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            alpha = p['life'] / 20.0
            radius = int(3 * alpha)
            color = (p['color'][0], p['color'][1], p['color'][2], int(alpha * 255))
            pygame.gfxdraw.filled_circle(surface, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def _render_ui(self, surface):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        surface.blit(score_text, (20, 10))

        # Misses
        miss_text = self.font_ui.render("MISSES:", True, self.COLOR_TEXT)
        surface.blit(miss_text, (self.WIDTH - 180, 10))
        for i in range(self.MAX_MISSES):
            color = self.COLOR_ACCENT_BAD if i < self.misses else self.COLOR_GRID
            x_surf = self.font_ui.render("X", True, color)
            surface.blit(x_surf, (self.WIDTH - 70 + i * 25, 10))
            
        # Text flashes
        for f in self.text_flashes:
            alpha = max(0, min(255, int(255 * (f['life'] / 45.0))))
            flash_surf = self.font_flash.render(f['text'], True, f['color'])
            flash_surf.set_alpha(alpha)
            text_rect = flash_surf.get_rect(center=f['pos'])
            surface.blit(flash_surf, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
            "ball_speed": self.base_ball_speed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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