
# Generated: 2025-08-27T22:09:45.298126
# Source Brief: brief_03032.md
# Brief Index: 3032

        
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
    """
    A fast-paced, top-down block breaker where strategic angles and risky plays are rewarded.
    This environment prioritizes visual quality and satisfying game feel.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-arcade block breaker. Clear all the bricks without losing your three balls to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 8
        self.BALL_INITIAL_SPEED = 5.5
        self.MAX_STEPS = 2000

        # --- Colors (Vibrant Retro Arcade Theme) ---
        self.COLOR_BG = (20, 20, 40)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PADDLE = (240, 240, 240)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 0, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (255, 50, 50), (255, 150, 50), (255, 255, 50),
            (50, 255, 50), (50, 150, 255), (150, 50, 255)
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
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)

        # --- State variables (initialized in reset) ---
        self.np_random = None
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = None
        self.particles = None
        self.score = 0
        self.balls_left = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.ball_in_play = False
        self.last_collision_was_wall = False

        self.reset()
        # self.validate_implementation() # Uncomment to run self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # --- Game State ---
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.particles = []
        self.score = 0
        self.balls_left = 3
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.last_collision_was_wall = False

        self._create_blocks()
        self._reset_ball()

        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        self.blocks = []
        block_width = 50
        block_height = 20
        rows = 6
        cols = self.WIDTH // (block_width + 4)
        x_offset = (self.WIDTH - cols * (block_width + 4)) // 2
        y_offset = 50

        for r in range(rows):
            for c in range(cols):
                block_rect = pygame.Rect(
                    x_offset + c * (block_width + 4),
                    y_offset + r * (block_height + 4),
                    block_width,
                    block_height,
                )
                self.blocks.append(
                    {"rect": block_rect, "color": self.BLOCK_COLORS[r]}
                )

    def _reset_ball(self):
        self.ball_in_play = False
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            reward -= 0.02 # Small penalty for movement
        if movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            reward -= 0.02 # Small penalty for movement
        self.paddle.clamp_ip(self.screen.get_rect())

        if space_pressed and not self.ball_in_play:
            # Sound: Ball Launch
            self.ball_in_play = True
            angle = self.np_random.uniform(-math.pi / 6, math.pi / 6)
            self.ball_vel = [
                self.BALL_INITIAL_SPEED * math.sin(angle),
                -self.BALL_INITIAL_SPEED * math.cos(angle)
            ]

        # --- Update Game Logic ---
        if self.ball_in_play:
            reward += 0.01 # Continuous reward for keeping ball in play
            self._move_ball()
            step_reward = self._handle_collisions()
            reward += step_reward
        else: # Ball is on the paddle
            self.ball_pos[0] = self.paddle.centerx

        self._update_particles()

        # --- Check Termination ---
        self.steps += 1
        terminated = False
        if not self.blocks:
            self.game_won = True
            self.game_over = True
            terminated = True
            reward += 100 # Win bonus
        elif self.balls_left <= 0:
            self.game_over = True
            terminated = True
            reward -= 50 # Loss penalty
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _move_ball(self):
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS,
            self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
        )

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos[0]))
            self.last_collision_was_wall = True
            # Sound: Wall Bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = max(self.BALL_RADIUS, self.ball_pos[1])
            self.last_collision_was_wall = True
            # Sound: Wall Bounce

        # Ball lost
        if ball_rect.top >= self.HEIGHT:
            self.balls_left -= 1
            reward -= 1.0 # Penalty for losing a ball
            # Sound: Lose Ball
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
            return reward

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # Sound: Paddle Hit
            self.ball_vel[1] *= -1
            offset = (self.paddle.centerx - self.ball_pos[0]) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] -= offset * 2.5
            
            speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            if speed > 0:
                scale = self.BALL_INITIAL_SPEED / speed
                self.ball_vel[0] *= scale
                self.ball_vel[1] *= scale

            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            self.last_collision_was_wall = False

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                # Sound: Block Break
                self.score += 10
                reward += 1.0
                if self.last_collision_was_wall:
                    reward += 5.0
                    self.score += 25

                self._create_particles(block["rect"].center, block["color"])
                self.blocks.remove(block)

                # Simple but effective collision response
                prev_ball_center_y = self.ball_pos[1] - self.ball_vel[1]
                if prev_ball_center_y < block["rect"].top or prev_ball_center_y > block["rect"].bottom:
                    self.ball_vel[1] *= -1
                else:
                    self.ball_vel[0] *= -1

                self.last_collision_was_wall = False
                break

        return reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": color,
                "radius": self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=2)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block["color"]), block["rect"], 1)

        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 30.0))))
            s = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p["color"], alpha), (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(s, (p["pos"][0] - p["radius"], p["pos"][1] - p["radius"]))

        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        glow_surf = pygame.Surface((self.BALL_RADIUS*4, self.BALL_RADIUS*4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_BALL_GLOW, (self.BALL_RADIUS*2, self.BALL_RADIUS*2), self.BALL_RADIUS*1.5)
        self.screen.blit(glow_surf, (ball_x - self.BALL_RADIUS*2, ball_y - self.BALL_RADIUS*2))
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        ball_text = self.font_small.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(ball_text, (self.WIDTH - 150, 10))
        for i in range(self.balls_left - (1 if self.ball_in_play else 0)):
             pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 60 + i * 20, 22, 6, self.COLOR_BALL)
             pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 60 + i * 20, 22, 6, self.COLOR_BALL)

        if self.game_over:
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "balls_left": self.balls_left}

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")