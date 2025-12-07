
# Generated: 2025-08-27T13:41:15.512448
# Source Brief: brief_00452.md
# Brief Index: 452

        
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
        "Controls: ←→ to move the paddle. Break all the blocks to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker where risk-taking is rewarded. Clear all blocks to win, but you only have 3 balls."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 7
        self.BALL_MIN_Y_VEL = 3 # Prevents horizontal loops
        self.MAX_STEPS = 2000

        # --- Color Palette (Retro Arcade) ---
        self.COLOR_BG = (15, 15, 25) # Dark blue-black
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0) # Bright Yellow
        self.COLOR_WALL = (50, 50, 70)
        self.COLOR_TEXT = (220, 220, 240)
        self.BLOCK_COLORS = [
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 0),    # Green
            (255, 165, 0),  # Orange
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 28)
        self.font_tiny = pygame.font.Font(None, 20)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.combo_counter = 0

        # Initialize state
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        self.combo_counter = 0

        # Paddle
        paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT - 10
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1)
        
        # Set initial ball velocity
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4) # Random initial angle
        self.ball_vel = pygame.Vector2(math.sin(angle), -math.cos(angle)) * self.BALL_SPEED
        if abs(self.ball_vel.y) < self.BALL_MIN_Y_VEL:
             self.ball_vel.y = -self.BALL_MIN_Y_VEL if self.ball_vel.y < 0 else self.BALL_MIN_Y_VEL


        # Blocks
        self.blocks = []
        block_width, block_height = 58, 20
        num_cols, num_rows = 10, 5
        x_gap, y_gap = 6, 6
        start_x = (self.WIDTH - (num_cols * (block_width + x_gap) - x_gap)) / 2
        start_y = 50
        for i in range(num_rows):
            for j in range(num_cols):
                color = self.BLOCK_COLORS[(i + j) % len(self.BLOCK_COLORS)]
                block = pygame.Rect(
                    start_x + j * (block_width + x_gap),
                    start_y + i * (block_height + y_gap),
                    block_width,
                    block_height
                )
                self.blocks.append({"rect": block, "color": color})

        # Particles
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Unpack Action ---
        movement = action[0]  # 3=left, 4=right

        # --- Calculate Reward ---
        reward = -0.02  # Time penalty

        # --- Update Paddle ---
        if movement == 3:  # Move left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Move right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))
        
        # "Safe play" penalty
        is_ball_approaching = self.ball_vel.y > 0 and self.ball_pos.y > self.HEIGHT * 0.7
        safe_zone_start = self.WIDTH * 0.4
        safe_zone_end = self.WIDTH * 0.6
        is_in_safe_zone = safe_zone_start < self.paddle.centerx < safe_zone_end
        if is_ball_approaching and is_in_safe_zone:
            reward -= 2.0


        # --- Update Ball ---
        self.ball_pos += self.ball_vel

        # --- Collision Detection ---
        # Walls
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos.x))
            # sfx: wall_bounce
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            # sfx: wall_bounce

        # Paddle
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        if self.paddle.colliderect(ball_rect) and self.ball_vel.y > 0:
            # sfx: paddle_hit
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x = offset * self.BALL_SPEED * 0.8 # Max horizontal speed change
            self.ball_vel.y *= -1
            
            # Ensure minimum vertical speed and normalize total speed
            if abs(self.ball_vel.y) < self.BALL_MIN_Y_VEL:
                self.ball_vel.y = -self.BALL_MIN_Y_VEL
            self.ball_vel.normalize_ip()
            self.ball_vel *= self.BALL_SPEED

            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS # Prevent sticking
            self.combo_counter = 0 # Reset combo on paddle hit

        # Blocks
        hit_block_idx = -1
        for i, block_data in enumerate(self.blocks):
            if block_data["rect"].colliderect(ball_rect):
                hit_block_idx = i
                break
        
        if hit_block_idx != -1:
            # sfx: block_break
            block_data = self.blocks.pop(hit_block_idx)
            self.score += 1
            reward += 1.0

            if self.combo_counter > 0:
                reward += 5.0 # Combo bonus
                self.score += 5

            self.combo_counter += 1

            # Create particle explosion
            for _ in range(20):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                p_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                p_life = self.np_random.integers(15, 30)
                self.particles.append([block_data["rect"].center, p_vel, p_life, block_data["color"]])

            # Bounce logic
            self.ball_vel.y *= -1

        # Miss (bottom wall)
        if self.ball_pos.y + self.BALL_RADIUS >= self.HEIGHT:
            # sfx: lose_ball
            self.balls_left -= 1
            self.combo_counter = 0
            if self.balls_left > 0:
                # Reset ball position
                self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1)
                angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
                self.ball_vel = pygame.Vector2(math.sin(angle), -math.cos(angle)) * self.BALL_SPEED
                if abs(self.ball_vel.y) < self.BALL_MIN_Y_VEL:
                     self.ball_vel.y = -self.BALL_MIN_Y_VEL
            else:
                self.game_over = True


        # --- Update Particles ---
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0] += p[1] # Update position
            p[2] -= 1 # Decrement lifespan

        # --- Update Steps and Check Termination ---
        self.steps += 1
        terminated = False
        if self.game_over or len(self.blocks) == 0 or self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        if terminated:
            if len(self.blocks) == 0:
                reward += 100 # Win bonus
                self.score += 100
            elif self.balls_left <= 0:
                reward -= 50 # Lose penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        # --- Clear Screen ---
        self.screen.fill(self.COLOR_BG)

        # --- Render Game Elements ---
        # Blocks
        for block_data in self.blocks:
            pygame.draw.rect(self.screen, block_data["color"], block_data["rect"])
            pygame.draw.rect(self.screen, self.COLOR_BG, block_data["rect"], 1) # Outline

        # Particles
        for pos, vel, life, color in self.particles:
            alpha = int(255 * (life / 30))
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            s.fill((color[0], color[1], color[2], alpha))
            self.screen.blit(s, (int(pos[0]), int(pos[1])))

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball (with glow)
        ball_center = (int(self.ball_pos.x), int(self.ball_pos.y))
        glow_color = (*self.COLOR_BALL, 50)
        pygame.gfxdraw.filled_circle(self.screen, *ball_center, self.BALL_RADIUS + 3, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, *ball_center, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, *ball_center, self.BALL_RADIUS, self.COLOR_BALL)


        # --- Render UI ---
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls left display
        for i in range(self.balls_left):
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 20 - (i * 20), 22, 6, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 20 - (i * 20), 22, 6, self.COLOR_BALL)

        # Combo display
        if self.combo_counter > 1:
            combo_text = self.font_small.render(f"COMBO x{self.combo_counter}!", True, self.COLOR_BALL)
            text_rect = combo_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 80))
            self.screen.blit(combo_text, text_rect)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if len(self.blocks) == 0:
                end_text = self.font_large.render("YOU WIN!", True, (0, 255, 0))
            else:
                end_text = self.font_large.render("GAME OVER", True, (255, 0, 0))
            
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(end_text, text_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 30))
            self.screen.blit(final_score_text, score_rect)

        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_remaining": len(self.blocks),
        }

    def render(self):
        # This method is not strictly required by the new API but is good practice
        return self._get_observation()

    def close(self):
        pygame.quit()
        
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

if __name__ == "__main__":
    # --- Manual Play ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Create a window to display the game
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    running = True
    total_reward = 0

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if not terminated:
            # --- Action Mapping for Human ---
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4
            
            action = [movement, 0, 0] # space and shift are unused

            # --- Step the Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # --- Rendering ---
        # Pygame uses a different coordinate system, so we need to transpose
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control FPS
        env.clock.tick(60)

    env.close()