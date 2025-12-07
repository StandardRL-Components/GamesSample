
# Generated: 2025-08-27T22:25:58.602396
# Source Brief: brief_03122.md
# Brief Index: 3122

        
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
        "Controls: Use ← and → to move the paddle. Press Space to launch the ball."
    )

    game_description = (
        "A retro arcade block-breaking game. Clear all blocks to advance, but don't lose your ball or run out of time!"
    )

    auto_advance = True

    # --- Constants ---
    # Game world
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    TIME_PER_STAGE = 60 * FPS  # 60 seconds
    MAX_STAGES = 3
    MAX_EPISODE_STEPS = (MAX_STAGES * TIME_PER_STAGE) + 100 # Failsafe

    # Colors
    COLOR_BG = (20, 20, 40)
    COLOR_GRID = (30, 30, 60)
    COLOR_PADDLE = (220, 220, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    BLOCK_COLORS = [
        (76, 175, 80),  # Green (1 HP)
        (255, 152, 0),  # Orange (2 HP)
        (244, 67, 54),  # Red (3 HP)
    ]

    # Paddle
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 12

    # Ball
    BALL_RADIUS = 8
    INITIAL_BALL_SPEED = 5.0
    BALL_SPEED_INCREMENT = 0.5

    # Blocks
    BLOCK_WIDTH, BLOCK_HEIGHT = 60, 20
    BLOCK_ROWS, BLOCK_COLS = 5, 10
    BLOCK_TOP_MARGIN = 50
    BLOCK_X_SPACING, BLOCK_Y_SPACING = 4, 4

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        self.np_random = None
        self.game_over_message = ""

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.balls_left = 3
        self.time_left = self.TIME_PER_STAGE
        self.ball_speed = self.INITIAL_BALL_SPEED

        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        self._reset_ball()
        self._generate_stage()

        self.particles = []

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_launched = False
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def _generate_stage(self):
        self.blocks = []
        grid_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_X_SPACING)
        start_x = (self.WIDTH - grid_width) // 2

        for r in range(self.BLOCK_ROWS):
            for c in range(self.BLOCK_COLS):
                health = min(self.stage, len(self.BLOCK_COLORS))
                block_rect = pygame.Rect(
                    start_x + c * (self.BLOCK_WIDTH + self.BLOCK_X_SPACING),
                    self.BLOCK_TOP_MARGIN + r * (self.BLOCK_HEIGHT + self.BLOCK_Y_SPACING),
                    self.BLOCK_WIDTH,
                    self.BLOCK_HEIGHT,
                )
                self.blocks.append({"rect": block_rect, "health": health})

    def _next_stage(self):
        self.stage += 1
        self.ball_speed += self.BALL_SPEED_INCREMENT
        self.time_left = self.TIME_PER_STAGE
        self._reset_ball()
        self._generate_stage()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Handle Input ---
        movement = action[0]
        space_pressed = action[1] == 1

        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.clamp_ip(self.screen.get_rect())

        if space_pressed and not self.ball_launched:
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi * 3 / 4, -math.pi / 4)
            self.ball_vel = [math.cos(angle) * self.ball_speed, math.sin(angle) * self.ball_speed]
            # sfx: launch_ball

        # --- 2. Update Game State ---
        self.steps += 1
        self.time_left -= 1

        if self.ball_launched:
            reward += 0.01  # Reward for keeping ball in play
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
            
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)

            # Wall collisions
            if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
                self.ball_vel[0] *= -1
                self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
                # sfx: wall_bounce
            if self.ball_pos[1] <= self.BALL_RADIUS:
                self.ball_vel[1] *= -1
                self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.HEIGHT)
                # sfx: wall_bounce

            # Paddle collision
            if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
                self.ball_vel[1] *= -1
                self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS

                hit_pos_norm = (self.ball_pos[0] - self.paddle.centerx) / (self.paddle.width / 2)
                self.ball_vel[0] += hit_pos_norm * 2.0
                
                # Normalize velocity to maintain constant speed
                current_speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
                if current_speed > 0:
                    self.ball_vel[0] = (self.ball_vel[0] / current_speed) * self.ball_speed
                    self.ball_vel[1] = (self.ball_vel[1] / current_speed) * self.ball_speed
                
                if abs(hit_pos_norm) > 0.75: # Risky hit on edge
                    reward += 0.5
                # sfx: paddle_bounce

            # Block collisions
            for block in self.blocks[:]:
                if ball_rect.colliderect(block["rect"]):
                    # sfx: block_hit
                    self._spawn_particles(block["rect"].center, self.BLOCK_COLORS[block["health"] - 1])
                    
                    # Simple but effective collision response
                    dx = self.ball_pos[0] - block["rect"].centerx
                    dy = self.ball_pos[1] - block["rect"].centery
                    w, h = block["rect"].width / 2, block["rect"].height / 2
                    
                    if abs(dx / w) > abs(dy / h): # Horizontal collision
                        self.ball_vel[0] *= -1
                    else: # Vertical collision
                        self.ball_vel[1] *= -1

                    block["health"] -= 1
                    if block["health"] <= 0:
                        self.blocks.remove(block)
                        self.score += 1
                        reward += 1.0
                    break

            # Ball loss
            if self.ball_pos[1] > self.HEIGHT:
                self.balls_left -= 1
                reward -= 10.0
                # sfx: lose_ball
                if self.balls_left > 0:
                    self._reset_ball()
                else:
                    terminated = True
                    self.game_over = True
                    self.game_over_message = "OUT OF BALLS"
                    reward -= 100.0
        else:
            # Ball follows paddle before launch
            self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]

        # Update particles
        for p in self.particles[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1
            if p[4] <= 0:
                self.particles.remove(p)

        # --- 3. Check for State Transitions ---
        if not self.blocks and not self.game_over:
            if self.stage < self.MAX_STAGES:
                reward += 10.0
                self._next_stage()
                # sfx: stage_clear
            else:
                self.score += 100
                reward += 100.0
                terminated = True
                self.game_over = True
                self.game_over_message = "YOU WIN!"
                # sfx: game_win

        if self.time_left <= 0 and not self.game_over:
            terminated = True
            self.game_over = True
            self.game_over_message = "TIME'S UP"
            reward -= 50.0
            # sfx: game_over
        
        if self.steps >= self.MAX_EPISODE_STEPS and not self.game_over:
            terminated = True
            self.game_over = True
            self.game_over_message = "MAX STEPS REACHED"


        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.np_random.integers(10, 20)
            self.particles.append([pos[0], pos[1], vx, vy, life, color])

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Blocks
        for block in self.blocks:
            color = self.BLOCK_COLORS[min(block["health"] - 1, len(self.BLOCK_COLORS) - 1)]
            pygame.draw.rect(self.screen, color, block["rect"], border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), block["rect"], width=2, border_radius=3)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)

        # Particles
        for p in self.particles:
            size = max(0, int(p[4] / 4))
            pygame.draw.rect(self.screen, p[5], (int(p[0]), int(p[1]), size, size))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Stage
        stage_text = self.font_ui.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Balls
        for i in range(self.balls_left):
            pygame.gfxdraw.filled_circle(self.screen, 20 + i * 25, self.HEIGHT - 20, 8, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, 20 + i * 25, self.HEIGHT - 20, 8, self.COLOR_BALL)

        # Timer
        time_sec = self.time_left // self.FPS
        timer_text = self.font_ui.render(f"TIME: {time_sec}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, self.HEIGHT - timer_text.get_height() - 10))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        game_over_text = self.font_game_over.render(self.game_over_message, True, self.COLOR_TEXT)
        text_rect = game_over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(game_over_text, text_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "balls_left": self.balls_left,
            "time_left_seconds": self.time_left // self.FPS,
        }

    def close(self):
        pygame.quit()
        super().close()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    
    print(env.user_guide)

    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000)
    env.close()