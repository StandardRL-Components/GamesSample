
# Generated: 2025-08-27T21:07:29.289948
# Source Brief: brief_02688.md
# Brief Index: 2688

        
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
        "A fast-paced, top-down block breaker. Destroy all blocks to win, but lose a life if the ball goes past your paddle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 8
    INITIAL_BALL_SPEED = 5
    MAX_BALL_SPEED = 10
    MAX_STEPS = 2000
    INITIAL_LIVES = 3

    # --- Colors ---
    COLOR_BG = (15, 15, 35) # Dark Blue
    COLOR_GRID = (30, 30, 60)
    COLOR_PADDLE = (240, 240, 240) # White
    COLOR_BALL = (255, 255, 0) # Yellow
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = [
        (255, 50, 50),   # Red
        (50, 255, 50),   # Green
        (50, 50, 255),   # Blue
        (255, 128, 0),   # Orange
        (128, 0, 255),   # Purple
        (0, 255, 255),   # Cyan
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 60)

        # Initialize state variables (will be properly set in reset)
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = True
        self.blocks = []
        self.particles = []
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.stuck_counter = 0

        # self.validate_implementation() # Uncomment for testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES

        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self._reset_ball()
        self.particles = []

        # Generate blocks
        self.blocks = []
        block_width = 58
        block_height = 20
        rows = 5
        cols = 10
        for r in range(rows):
            for c in range(cols):
                block_rect = pygame.Rect(
                    c * (block_width + 2) + 20,
                    r * (block_height + 2) + 40,
                    block_width,
                    block_height
                )
                color_index = self.np_random.integers(0, len(self.BLOCK_COLORS))
                self.blocks.append({"rect": block_rect, "color": self.BLOCK_COLORS[color_index]})

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        movement = action[0]
        space_held = action[1] == 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Handle Input & Paddle Movement ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        # --- 2. Handle Ball ---
        if self.ball_attached:
            self.ball_pos[0] = self.paddle.centerx
            if space_held:
                self.ball_attached = False
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                self.ball_vel = np.array([math.cos(angle), -math.sin(angle)]) * self.INITIAL_BALL_SPEED
                # sound: launch_ball.wav
        else:
            reward += self._update_ball_and_collisions()

        # --- 3. Update Particles ---
        self._update_particles()

        # --- 4. Update Game State & Termination ---
        self.steps += 1
        
        if not self.blocks: # Win
            reward += 100
            self.game_over = True
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_ball_and_collisions(self):
        reward = 0
        self.ball_pos += self.ball_vel

        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            # sound: wall_bounce.wav
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.HEIGHT)
            # sound: wall_bounce.wav

        # Bottom (lose life)
        if self.ball_pos[1] >= self.HEIGHT:
            self.lives -= 1
            # sound: lose_life.wav
            if self.lives <= 0:
                self.game_over = True
                reward -= 100
            else:
                self._reset_ball()
            return reward

        # Paddle collision
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        if self.paddle.colliderect(ball_rect) and self.ball_vel[1] > 0:
            # sound: paddle_hit.wav
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS

            offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2.5
            
            reward += 0.1 if abs(offset) > 0.7 else -0.02

            speed = np.linalg.norm(self.ball_vel)
            if speed < self.MAX_BALL_SPEED: self.ball_vel *= 1.02
            speed = np.linalg.norm(self.ball_vel)
            if speed > self.MAX_BALL_SPEED: self.ball_vel = (self.ball_vel / speed) * self.MAX_BALL_SPEED

        # Block collisions
        hit_block = None
        for block in self.blocks:
            if block["rect"].colliderect(ball_rect):
                hit_block = block
                break
        
        if hit_block:
            # sound: block_break.wav
            self.blocks.remove(hit_block)
            reward += 5
            self.score += 10
            self._create_particles(hit_block["rect"].center, hit_block["color"])

            prev_ball_pos = self.ball_pos - self.ball_vel
            dx = prev_ball_pos[0] - hit_block["rect"].centerx
            dy = prev_ball_pos[1] - hit_block["rect"].centery
            w, h = hit_block["rect"].width / 2, hit_block["rect"].height / 2
            
            if abs(dx / w) > abs(dy / h): self.ball_vel[0] *= -1
            else: self.ball_vel[1] *= -1

        # Anti-stuck logic
        if abs(self.ball_vel[1]) < 0.5: self.stuck_counter += 1
        else: self.stuck_counter = 0
        if self.stuck_counter > 60:
            self.ball_vel[1] += self.np_random.choice([-1, 1]) * 0.5
            self.stuck_counter = 0

        return reward

    def _reset_ball(self):
        self.ball_attached = True
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS], dtype=float)
        self.ball_vel = np.array([0.0, 0.0], dtype=float)
        self.stuck_counter = 0

    def _create_particles(self, pos, color):
        for _ in range(20):
            particle_vel = np.array([self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)])
            lifespan = self.np_random.uniform(15, 30)
            self.particles.append({
                "pos": np.array(pos, dtype=float), "vel": particle_vel,
                "lifespan": lifespan, "max_lifespan": lifespan, "color": color
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(0, self.WIDTH, 20): pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20): pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, tuple(c * 0.8 for c in block["color"]), block["rect"], 2)

        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Ball glow
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        for i in range(4, 0, -1):
            alpha = 100 - i * 25
            glow_radius = self.BALL_RADIUS + i * 2
            pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, glow_radius, (*self.COLOR_BALL, alpha))
        
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

        for p in self.particles:
            alpha = 255 * (p["lifespan"] / p["max_lifespan"])
            size = int(5 * (p["lifespan"] / p["max_lifespan"]))
            if size > 0:
                p_x, p_y = int(p["pos"][0] - size/2), int(p["pos"][1] - size/2)
                pygame.draw.rect(self.screen, (*p["color"], alpha), (p_x, p_y, size, size))

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        block_text = self.font_small.render(f"BLOCKS: {len(self.blocks)}", True, self.COLOR_TEXT)
        self.screen.blit(block_text, (self.WIDTH/2 - block_text.get_width()/2, self.HEIGHT - 30))

        if self.game_over:
            msg, color = ("YOU WIN!", (100, 255, 100)) if not self.blocks else ("GAME OVER", (255, 100, 100))
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))
        elif self.ball_attached:
            prompt_text = self.font_small.render("PRESS SPACE TO LAUNCH", True, self.COLOR_TEXT)
            self.screen.blit(prompt_text, prompt_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 50)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.lives, "blocks_left": len(self.blocks)}

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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