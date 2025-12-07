
# Generated: 2025-08-27T22:15:05.284642
# Source Brief: brief_03060.md
# Brief Index: 3060

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker where risky plays are rewarded. "
        "Break all the blocks to win, but lose all your balls and it's game over."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.BALL_RADIUS = 8
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 60, 20
        self.UI_HEIGHT = 40
        self.GAME_AREA_Y_START = self.UI_HEIGHT
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PADDLE = (220, 220, 220)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_UI_TEXT = (200, 200, 220)
        self.BLOCK_COLORS = [
            (255, 87, 34),  # Deep Orange
            (33, 150, 243), # Blue
            (76, 175, 80),  # Green
            (255, 235, 59), # Yellow
            (156, 39, 176), # Purple
        ]

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        
        # Game state variables (initialized in reset)
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.score = None
        self.balls_left = None
        self.steps = None
        self.game_over = None
        
        self.reset()

        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset paddle
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        # Reset ball
        self._reset_ball()
        
        # Reset blocks (5 rows, 10 columns = 50 blocks)
        self.blocks = []
        num_cols = 10
        num_rows = 5
        block_total_width = num_cols * self.BLOCK_WIDTH
        start_x = (self.WIDTH - block_total_width) / 2
        for i in range(num_rows):
            for j in range(num_cols):
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                block_rect = pygame.Rect(
                    start_x + j * self.BLOCK_WIDTH,
                    self.GAME_AREA_Y_START + 20 + i * self.BLOCK_HEIGHT,
                    self.BLOCK_WIDTH,
                    self.BLOCK_HEIGHT
                )
                self.blocks.append({"rect": block_rect, "color": color})

        # Reset other state
        self.particles = []
        self.score = 0
        self.balls_left = 3
        self.steps = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1], dtype=float)
        self.ball_vel = np.array([0.0, 0.0], dtype=float)
        self.ball_launched = False
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.02  # Penalty for each step to encourage fast completion

        # 1. Handle player input
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        paddle_speed = 10
        if movement == 3:  # Left
            self.paddle.x -= paddle_speed
        elif movement == 4:  # Right
            self.paddle.x += paddle_speed
        
        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.WIDTH, self.paddle.right)
        
        if space_held and not self.ball_launched:
            self.ball_launched = True
            angle = self.np_random.uniform(math.pi / 4, 3 * math.pi / 4)
            speed = 6.0
            self.ball_vel = np.array([math.cos(angle) * speed, -math.sin(angle) * speed])
            # Sound: Ball Launch

        # 2. Update game state
        if self.ball_launched:
            self.ball_pos += self.ball_vel
        else:
            self.ball_pos[0] = self.paddle.centerx

        self._update_particles()
        
        # 3. Handle collisions
        ball_rect = pygame.Rect(
            int(self.ball_pos[0] - self.BALL_RADIUS), 
            int(self.ball_pos[1] - self.BALL_RADIUS), 
            self.BALL_RADIUS * 2, 
            self.BALL_RADIUS * 2
        )
        
        # Ball-wall collision
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            # Sound: Wall bounce
        if self.ball_pos[1] <= self.GAME_AREA_Y_START + self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.GAME_AREA_Y_START + self.BALL_RADIUS
            # Sound: Wall bounce
            
        # Ball-paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            reward += 0.1
            hit_pos_norm = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            angle = hit_pos_norm * (math.pi / 3)  # Max angle 60 degrees
            
            speed = max(6.0, np.linalg.norm(self.ball_vel)) # Anti-softlock
            self.ball_vel[0] = math.sin(angle) * speed
            self.ball_vel[1] = -math.cos(angle) * speed
            
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            # Sound: Paddle hit

        # Ball-block collision
        block_rects = [b['rect'] for b in self.blocks]
        hit_idx = ball_rect.collidelist(block_rects)
        if hit_idx != -1:
            reward += 1.0
            self.score += 10
            
            block_hit = self.blocks.pop(hit_idx)
            self._create_particles(block_hit['rect'].center, block_hit['color'])
            
            # Simple reflection logic
            dx = self.ball_pos[0] - block_hit['rect'].centerx
            dy = self.ball_pos[1] - block_hit['rect'].centery
            if abs(dx / self.BLOCK_WIDTH) > abs(dy / self.BLOCK_HEIGHT):
                self.ball_vel[0] *= -1
            else:
                self.ball_vel[1] *= -1
            # Sound: Block break
            
        # Ball lost
        if self.ball_pos[1] > self.HEIGHT + self.BALL_RADIUS:
            self.balls_left -= 1
            reward -= 1.0
            # Sound: Ball lost
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
                reward -= 100.0

        # 4. Check for termination
        terminated = self.game_over
        if not self.blocks:
            terminated = True
            self.game_over = True
            reward += 100.0
            self.score += 1000
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color
            })
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw blocks with 3D effect
        for block in self.blocks:
            r, c = block['rect'], block['color']
            pygame.draw.rect(self.screen, c, r)
            light_color = tuple(min(255, x + 40) for x in c)
            dark_color = tuple(max(0, x - 40) for x in c)
            pygame.draw.line(self.screen, light_color, r.topleft, r.topright, 2)
            pygame.draw.line(self.screen, light_color, r.topleft, r.bottomleft, 2)
            pygame.draw.line(self.screen, dark_color, r.bottomleft, r.bottomright, 2)
            pygame.draw.line(self.screen, dark_color, r.topright, r.bottomright, 2)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with glow
        center = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        glow_radius = self.BALL_RADIUS + 4
        glow_color = (120, 120, 0, 100)
        temp_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(temp_surf, (center[0] - glow_radius, center[1] - glow_radius))

        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * max(0, p['life'] / 30))
            color = p['color'] + (alpha,)
            size = int(max(1, p['life'] / 10))
            
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'][0]) - size, int(p['pos'][1]) - size))

    def _render_ui(self):
        # UI background bar and dividing line
        pygame.draw.rect(self.screen, (10, 15, 30), (0, 0, self.WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, self.UI_HEIGHT - 1), (self.WIDTH, self.UI_HEIGHT - 1), 2)
        
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, (self.UI_HEIGHT - score_text.get_height()) // 2))
        
        # Remaining balls
        for i in range(self.balls_left):
            x, y = self.WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 10)), self.UI_HEIGHT // 2
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if not self.blocks else "GAME OVER"
            end_text = self.font_large.render(msg, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
            
            if self.steps >= self.MAX_STEPS:
                time_text = self.font_small.render("TIME LIMIT REACHED", True, (200, 200, 200))
                time_rect = time_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 30))
                self.screen.blit(time_text, time_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks)
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