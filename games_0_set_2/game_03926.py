
# Generated: 2025-08-28T00:52:16.551851
# Source Brief: brief_03926.md
# Brief Index: 3926

        
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
        "Controls: ←→ to move the paddle. Try to break all the blocks with the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade game combining Breakout and Pong. Control the paddle to bounce a ball, "
        "break all the blocks, and achieve a high score before you run out of lives."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.BALL_RADIUS = 8
        self.PADDLE_SPEED = 12
        self.BALL_INITIAL_SPEED = 5
        self.MAX_BALL_HORIZONTAL_INFLUENCE = 7
        self.MAX_STEPS = 1000
        self.INITIAL_LIVES = 3

        # Colors
        self.COLOR_BG = (10, 10, 35)
        self.COLOR_GRID = (30, 30, 60)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BORDER = (128, 128, 128)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_HEART = (255, 80, 80)
        self.BLOCK_COLORS = [(66, 214, 245), (245, 66, 214), (66, 245, 129), (245, 150, 66)]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        
        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.blocks = None
        self.particles = None
        self.lives = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.consecutive_hits = 0

        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES
        self.consecutive_hits = 0
        self.particles = []

        # Paddle
        paddle_y = self.HEIGHT - 40
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Ball
        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS,
            self.paddle.top - self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
        )
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = [
            self.BALL_INITIAL_SPEED * math.cos(angle),
            -self.BALL_INITIAL_SPEED * math.sin(angle),
        ]

        # Blocks
        self._create_blocks()
        
        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        self.blocks = []
        block_width, block_height = 58, 20
        rows, cols = 4, 10
        for i in range(rows):
            for j in range(cols):
                block_x = j * (block_width + 2) + (self.WIDTH - cols * (block_width + 2)) / 2 + 1
                block_y = i * (block_height + 2) + 50
                rect = pygame.Rect(block_x, block_y, block_width, block_height)
                color = self.BLOCK_COLORS[(i + j) % len(self.BLOCK_COLORS)]
                self.blocks.append({"rect": rect, "color": color})
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0.1  # Continuous reward for playing
        
        # 1. Update Paddle
        paddle_moved = self._update_paddle(movement)
        if not paddle_moved:
            reward -= 0.2

        # 2. Update Ball
        block_hit, life_lost = self._update_ball()
        
        # 3. Update Particles
        self._update_particles()

        # 4. Calculate Rewards
        if block_hit:
            reward += 1.0
            self.score += 10
            self.consecutive_hits += 1
            if self.consecutive_hits > 1:
                reward += 2.0
                self.score += 20 # Bonus points
        
        if life_lost:
            reward -= 5.0
            self.consecutive_hits = 0
            # sfx: lose_life

        # 5. Check Termination
        terminated = False
        if not self.blocks: # Win condition
            self.game_over = True
            terminated = True
            reward += 50.0
            self.score += 1000 # Win bonus
        elif self.lives <= 0: # Lose condition
            self.game_over = True
            terminated = True
            reward -= 50.0
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_paddle(self, movement):
        moved = True
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        else:
            moved = False
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)
        return moved

    def _update_ball(self):
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]
        
        block_hit = False
        life_lost = False

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.x = np.clip(self.ball.x, 0, self.WIDTH - self.ball.width)
            # sfx: wall_bounce
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.y = np.clip(self.ball.y, 0, self.HEIGHT - self.ball.height)
            # sfx: wall_bounce

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball.bottom = self.paddle.top
            self.ball_vel[1] *= -1
            
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = offset * self.MAX_BALL_HORIZONTAL_INFLUENCE
            self.consecutive_hits = 0
            # sfx: paddle_hit
        
        # Block collisions
        for block_data in self.blocks[:]:
            if self.ball.colliderect(block_data["rect"]):
                self._create_particles(block_data["rect"].center, block_data["color"])
                
                # Simple bounce logic: reverse vertical velocity
                self.ball_vel[1] *= -1
                
                self.blocks.remove(block_data)
                block_hit = True
                # sfx: block_break
                break # Only break one block per frame

        # Missed ball
        if self.ball.top >= self.HEIGHT:
            self.lives -= 1
            life_lost = True
            if self.lives > 0:
                # Reset ball position
                self.ball.centerx = self.paddle.centerx
                self.ball.bottom = self.paddle.top
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                self.ball_vel = [self.BALL_INITIAL_SPEED * math.cos(angle), -self.BALL_INITIAL_SPEED * math.sin(angle)]

        return block_hit, life_lost

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                "pos": list(pos),
                "vel": vel,
                "life": self.np_random.uniform(20, 40),
                "color": color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95 # Drag
            p["vel"][1] *= 0.95
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render background grid
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Render border
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (0, 0, self.WIDTH, self.HEIGHT), 2)
        
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 40))
            if alpha > 0:
                size = int(p["life"] / 10)
                if size > 0:
                    surf = pygame.Surface((size, size), pygame.SRCALPHA)
                    surf.fill((*p["color"], alpha))
                    self.screen.blit(surf, (int(p["pos"][0] - size/2), int(p["pos"][1] - size/2)))

        # Render blocks
        for block_data in self.blocks:
            pygame.draw.rect(self.screen, block_data["color"], block_data["rect"])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block_data["color"]), block_data["rect"], 2)

        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Render ball with glow
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_alpha = 100
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_BALL, glow_alpha), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (int(self.ball.centerx - glow_radius), int(self.ball.centery - glow_radius)))
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)

        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            self._draw_heart(self.screen, self.WIDTH - 30 - i * 35, 25, 15, self.COLOR_HEART)

        # Game Over Message
        if self.game_over:
            msg = "YOU WIN!" if not self.blocks else "GAME OVER"
            color = (0, 255, 0) if not self.blocks else (255, 0, 0)
            game_over_text = self.font_game_over.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20,20))
            self.screen.blit(game_over_text, text_rect)

    def _draw_heart(self, surface, x, y, size, color):
        points = [
            (x, y - int(size * 0.4)),
            (x + int(size * 0.5), y - int(size * 0.8)),
            (x + size, y - int(size * 0.4)),
            (x + size, y),
            (x, y + size),
            (x - size, y),
            (x - size, y - int(size * 0.4)),
            (x - int(size * 0.5), y - int(size * 0.8)),
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Reset to initialize state for observation
        self.reset()
        
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