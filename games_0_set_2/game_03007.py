
# Generated: 2025-08-28T06:42:31.791118
# Source Brief: brief_03007.md
# Brief Index: 3007

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: ←→ to move the paddle. Press space to launch the ball."

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced, retro block-breaking game. Clear all the blocks without losing your balls to get the highest score."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        self.small_font = pygame.font.SysFont("Consolas", 18, bold=True)

        # Colors
        self.COLOR_BG1 = (10, 20, 40)
        self.COLOR_BG2 = (30, 40, 70)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 165, 0), # Orange
            (0, 255, 128),   # Green
            (255, 69, 0),  # Red-Orange
        ]

        # Game parameters
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 8
        self.MAX_STEPS = 1000
        self.PARTICLE_LIFESPAN = 30

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = False
        self.blocks = []
        self.particles = []
        self.balls_left = 0
        self.score = 0
        self.steps = 0
        
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.balls_left = 3
        self.particles = []
        self.ball_launched = False

        paddle_y = self.HEIGHT - 40
        paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)

        self.blocks = []
        block_width = 60
        block_height = 20
        gap = 5
        rows = 5
        cols = self.WIDTH // (block_width + gap)
        start_x = (self.WIDTH - cols * (block_width + gap) + gap) / 2
        start_y = 50
        for i in range(rows):
            for j in range(cols):
                if self.np_random.random() > 0.1:
                    color_index = self.np_random.integers(0, len(self.BLOCK_COLORS))
                    rect = pygame.Rect(
                        start_x + j * (block_width + gap),
                        start_y + i * (block_height + gap),
                        block_width,
                        block_height
                    )
                    self.blocks.append({"rect": rect, "color": self.BLOCK_COLORS[color_index]})

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False

        # --- Handle Actions ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            reward -= 0.02
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            reward -= 0.02
        
        self.paddle.x = max(0, min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH))

        if space_held and not self.ball_launched:
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi / 6, math.pi / 6)
            self.ball_vel = pygame.Vector2(math.sin(angle), -math.cos(angle)) * self.BALL_SPEED
            # sound: launch_ball.wav

        # --- Update Game State ---
        if self.ball_launched:
            self.ball_pos += self.ball_vel
        else:
            self.ball_pos.x = self.paddle.centerx
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS

        # --- Handle Collisions ---
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        if self.ball_pos.x <= self.BALL_RADIUS or self.ball_pos.x >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.ball_pos.x, self.WIDTH - self.BALL_RADIUS))
            # sound: wall_bounce.wav
        if self.ball_pos.y <= self.BALL_RADIUS:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            # sound: wall_bounce.wav

        if self.ball_pos.y >= self.HEIGHT - self.BALL_RADIUS:
            self.balls_left -= 1
            reward -= 10
            # sound: lose_ball.wav
            if self.balls_left > 0:
                self.ball_launched = False
            else:
                terminated = True

        if self.paddle.colliderect(ball_rect) and self.ball_vel.y > 0:
            self.ball_vel.y *= -1
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x = min(max(self.ball_vel.x + offset * 2, -self.BALL_SPEED), self.BALL_SPEED)
            self.ball_vel.scale_to_length(self.BALL_SPEED)
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
            # sound: paddle_hit.wav
        
        hit_block_index = ball_rect.collidelist([b["rect"] for b in self.blocks])
        if hit_block_index != -1:
            hit_block = self.blocks.pop(hit_block_index)
            # sound: block_break.wav
            reward += 1
            self.score += 10
            self._create_particles(hit_block["rect"].center, hit_block["color"])

            intersect = ball_rect.clip(hit_block["rect"])
            if intersect.width < intersect.height:
                self.ball_vel.x *= -1
            else:
                self.ball_vel.y *= -1
        
        self._update_particles()

        # --- Check Termination Conditions ---
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
        if not self.blocks:
            terminated = True
            reward += 100
            self.score += 500
            # sound: win_level.wav
        if self.balls_left <= 0:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(15, self.PARTICLE_LIFESPAN + 1)
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "color": color, "lifespan": lifespan})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["lifespan"] -= 1

    def _get_observation(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = tuple(int(c1 * (1 - interp) + c2 * interp) for c1, c2 in zip(self.COLOR_BG1, self.COLOR_BG2))
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Draw blocks with a 3D effect
        for block in self.blocks:
            brighter = tuple(min(255, c + 40) for c in block["color"])
            darker = tuple(max(0, c - 40) for c in block["color"])
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.polygon(self.screen, brighter, [block["rect"].topleft, block["rect"].topright, (block["rect"].right-2, block["rect"].top+2), (block["rect"].left+2, block["rect"].top+2)])
            pygame.draw.polygon(self.screen, darker, [block["rect"].bottomleft, block["rect"].bottomright, (block["rect"].right-2, block["rect"].bottom-2), (block["rect"].left+2, block["rect"].bottom-2)])

        # Draw particles with fade effect
        for p in self.particles:
            life_ratio = p["lifespan"] / self.PARTICLE_LIFESPAN
            size = int(life_ratio * 3)
            if size > 0:
                p_color = tuple(int(c * life_ratio + bgc * (1 - life_ratio)) for c, bgc in zip(p["color"], self.COLOR_BG2))
                pygame.draw.circle(self.screen, p_color, (int(p["pos"].x), int(p["pos"].y)), size)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with glow
        ball_center = (int(self.ball_pos.x), int(self.ball_pos.y))
        glow_color = (*self.COLOR_BALL, 100)
        pygame.gfxdraw.filled_circle(self.screen, *ball_center, self.BALL_RADIUS + 3, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, *ball_center, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, *ball_center, self.BALL_RADIUS, self.COLOR_BALL)

        # Draw UI
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        ball_icon_text = self.small_font.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(ball_icon_text, (self.WIDTH - 150, 15))
        for i in range(self.balls_left -1): # Don't show current ball in play
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 80 + i * 20, 25, 6, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 80 + i * 20, 25, 6, self.COLOR_BALL)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to get an initial observation
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test info from reset
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        # Reset again to ensure it works multiple times
        self.reset()
        # print("✓ Implementation validated successfully")