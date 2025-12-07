
# Generated: 2025-08-27T12:27:10.558246
# Source Brief: brief_00046.md
# Brief Index: 46

        
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
        "A retro arcade block breaker. Clear all the blocks without losing your three lives. "
        "Score more points by playing riskily near the edges of the screen."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For auto_advance=True

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PADDLE = (240, 240, 240)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_DANGER = (100, 20, 20)
        self.COLOR_GRID = (30, 35, 50)
        self.BLOCK_COLORS = [
            (255, 87, 34), (255, 193, 7), (139, 195, 74),
            (0, 188, 212), (33, 150, 243), (103, 58, 183)
        ]
        self.COLOR_TEXT = (220, 220, 220)

        # Entity properties
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 15
        self.BALL_RADIUS = 8
        self.BALL_SPEED_INITIAL = 8
        self.BALL_MAX_VEL_X_INFLUENCE = 5

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
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- State Variables (initialized in reset) ---
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.blocks = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.ball_launched = None
        self.particles = None
        self.last_space_press = False # To detect rising edge of space press
        self._max_blocks = 0

        # Initialize state
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_launched = False
        self._reset_ball()

        self.lives = 3
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.last_space_press = False

        # Create block layout
        self.blocks = []
        block_width, block_height = 58, 20
        gap = 6
        num_cols = self.WIDTH // (block_width + gap)
        start_x = (self.WIDTH - num_cols * (block_width + gap) + gap) // 2
        for row in range(6):
            for col in range(num_cols):
                block_rect = pygame.Rect(
                    start_x + col * (block_width + gap),
                    50 + row * (block_height + gap),
                    block_width,
                    block_height
                )
                self.blocks.append({"rect": block_rect, "color": self.BLOCK_COLORS[row % len(self.BLOCK_COLORS)]})
        self._max_blocks = len(self.blocks)

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        """Resets the ball to be attached to the paddle."""
        self.ball_launched = False
        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS,
            self.paddle.top - self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )
        self.ball_vel = [0, 0]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.02 # Small penalty per step to encourage speed
        
        # --- Unpack Actions ---
        movement = action[0]  # 0=none, 1=up, 2=down, 3=left, 4=right
        space_held = action[1] == 1
        # shift_held is action[2], but unused in this game

        # --- Handle Input ---
        # Paddle movement
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.clamp_ip(self.screen.get_rect())

        # Ball launch (on rising edge of space press)
        if space_held and not self.last_space_press and not self.ball_launched:
            self.ball_launched = True
            # Launch with a slight random horizontal component
            initial_vx = self.np_random.uniform(-1, 1)
            self.ball_vel = [initial_vx, -self.BALL_SPEED_INITIAL]
            # sound: ball_launch.wav
        self.last_space_press = space_held

        # --- Update Game Logic ---
        if not self.ball_launched:
            # Ball follows paddle before launch
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top
        else:
            self._move_ball()
            reward += self._handle_collisions()

        self._update_particles()

        # --- Check for Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100 # Win bonus
            elif self.lives <= 0:
                reward -= 100 # Loss penalty
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        self.steps += 1
        if not self.blocks:
            self.win = True
            self.game_over = True
            return True
        if self.lives <= 0:
            self.game_over = True
            return True
        if self.steps >= 5000:
            self.game_over = True
            return True
        return False
        
    def _move_ball(self):
        """Updates ball position and handles wall bounces."""
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

        # Wall bounces
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.left = max(0, self.ball.left)
            self.ball.right = min(self.WIDTH, self.ball.right)
            # sound: wall_bounce.wav
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.top = max(0, self.ball.top)
            # sound: wall_bounce.wav

        # Out of bounds (lose life)
        if self.ball.top >= self.HEIGHT:
            self.lives -= 1
            # sound: lose_life.wav
            if self.lives > 0:
                self._reset_ball()
            else:
                self.game_over = True

    def _handle_collisions(self):
        """Handles ball collisions with paddle and blocks, returns reward."""
        collision_reward = 0

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            collision_reward += 0.1 # Reward for hitting the ball
            
            # Paddle risk/safety check
            center_zone_start = self.WIDTH * 0.25
            center_zone_end = self.WIDTH * 0.75
            if center_zone_start < self.paddle.centerx < center_zone_end:
                collision_reward -= 2.0 # Penalty for safe play
            
            # Bounce logic
            self.ball.bottom = self.paddle.top
            self.ball_vel[1] *= -1

            # Influence horizontal velocity based on hit location
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * self.BALL_MAX_VEL_X_INFLUENCE
            
            # Normalize ball speed to prevent it from getting too fast/slow
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if speed > 0:
                scale = self.BALL_SPEED_INITIAL / speed
                self.ball_vel[0] *= scale
                self.ball_vel[1] *= scale
            # sound: paddle_hit.wav

        # Block collision
        hit_block_idx = self.ball.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            block_hit = self.blocks[hit_block_idx]

            # Determine if it was a risky hit
            is_risky = self.paddle.left < 10 or self.paddle.right > self.WIDTH - 10
            collision_reward += 5.0 if is_risky else 1.0

            # Create particle explosion
            self._create_particles(block_hit['rect'].center, block_hit['color'])
            
            # Remove block
            self.blocks.pop(hit_block_idx)
            # sound: block_break.wav

            # Simple bounce logic
            self.ball_vel[1] *= -1
        
        return collision_reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            particle = {
                "pos": list(pos),
                "vel": [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)],
                "life": self.np_random.integers(10, 20),
                "radius": self.np_random.uniform(2, 5),
                "color": color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw danger zone
        pygame.draw.rect(self.screen, self.COLOR_DANGER, (0, self.HEIGHT - 5, self.WIDTH, 5))

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with a glow
        ball_center = (int(self.ball.centerx), int(self.ball.centery))
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], [int(c) for c in p['pos']], int(p['radius'] * (p['life'] / 20)))
            
    def _render_ui(self):
        # A simple score: 10 points per block broken
        self.score = (self._max_blocks - len(self.blocks)) * 10
        if self.win:
            self.score += 1000 # Big win bonus

        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        life_icon_surf = pygame.Surface((self.PADDLE_WIDTH / 5, self.PADDLE_HEIGHT / 2))
        life_icon_surf.fill(self.COLOR_PADDLE)
        for i in range(self.lives):
            self.screen.blit(life_icon_surf, (self.WIDTH - 30 - i * 25, 15))

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 128) if self.win else (255, 50, 50)
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Validating implementation...")
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

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    env = GameEnv()
    env.validate_implementation()

    # --- Interactive Play ---
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption(env.game_description)
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    keys_pressed = pygame.key.get_pressed()
    
    while not done:
        # --- Action mapping from keyboard to MultiDiscrete ---
        movement = 0 # No-op
        if keys_pressed[pygame.K_LEFT] or keys_pressed[pygame.K_a]:
            movement = 3
        elif keys_pressed[pygame.K_RIGHT] or keys_pressed[pygame.K_d]:
            movement = 4
        
        space_held = 1 if keys_pressed[pygame.K_SPACE] else 0
        shift_held = 1 if keys_pressed[pygame.K_LSHIFT] or keys_pressed[pygame.K_RSHIFT] else 0
        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render for human ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys_pressed = pygame.key.get_pressed()
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Info: {info}")
    env.close()