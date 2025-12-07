
# Generated: 2025-08-28T05:22:36.445179
# Source Brief: brief_05554.md
# Brief Index: 5554

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based block breaker where strategic paddle positioning and risk-taking are rewarded."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 45)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 80, 255), (80, 255, 255)]

    # Game settings
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 7
    INITIAL_BALL_SPEED = 5.0
    MAX_BALL_SPEED_X = 7.0
    INITIAL_LIVES = 3
    MAX_STEPS = 10000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # Randomness
        self.np_random = None

        # Initialize state variables
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
        self.lives = self.INITIAL_LIVES
        self.combo = 1
        self.blocks_destroyed = 0
        
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self._reset_ball()
        self._create_blocks()
        
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Handle Input ---
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        if self.ball_on_paddle and space_held:
            self.ball_on_paddle = False
            self.ball_vel_y = -self.current_ball_speed
            # sfx: launch_ball
        
        # --- Update Game Logic ---
        if not self.ball_on_paddle:
            reward += 0.01  # Reward for keeping ball in play
            
            self.ball_pos_x += self.ball_vel_x
            self.ball_pos_y += self.ball_vel_y
            ball_rect = self._get_ball_rect()

            # Wall collisions
            if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
                self.ball_vel_x *= -1
                ball_rect.left = np.clip(ball_rect.left, 0, self.SCREEN_WIDTH - ball_rect.width)
                self.ball_pos_x = ball_rect.centerx
                # sfx: wall_bounce
            if ball_rect.top <= 0:
                self.ball_vel_y *= -1
                ball_rect.top = 0
                self.ball_pos_y = ball_rect.centery
                # sfx: wall_bounce

            # Paddle collision
            if ball_rect.colliderect(self.paddle) and self.ball_vel_y > 0:
                self.ball_vel_y *= -1
                
                # Calculate rebound angle
                hit_pos = (ball_rect.centerx - self.paddle.centerx) / (self.paddle.width / 2)
                self.ball_vel_x = self.MAX_BALL_SPEED_X * hit_pos
                
                # Normalize speed
                self._normalize_ball_velocity()

                # Edge catch reward
                if abs(hit_pos) > 0.75:
                    reward += 0.5 * self.combo

                self.combo += 1
                ball_rect.bottom = self.paddle.top
                self.ball_pos_y = ball_rect.centery
                # sfx: paddle_bounce
            
            # Block collisions
            hit_block_idx = ball_rect.collidelist(self.blocks)
            if hit_block_idx != -1:
                block = self.blocks.pop(hit_block_idx)
                block_color = self.block_colors.pop(hit_block_idx)
                
                # sfx: block_break
                self._create_particles(block.center, block_color)
                
                # Simple collision response
                self.ball_vel_y *= -1
                
                self.score += self.combo
                reward += 1.0
                self.blocks_destroyed += 1
                
                # Increase ball speed every 10 blocks
                if self.blocks_destroyed % 10 == 0:
                    self.current_ball_speed = min(self.INITIAL_BALL_SPEED + (self.blocks_destroyed // 10) * 1.0, 10.0)
                    self._normalize_ball_velocity()

            # Lost life
            if ball_rect.top >= self.SCREEN_HEIGHT:
                self.lives -= 1
                self.combo = 1
                reward -= 1.0
                # sfx: lose_life
                if self.lives > 0:
                    self._reset_ball()
                else:
                    self.game_over = True
        else:
            self._reset_ball_position()

        # --- Update Particles ---
        self._update_particles()
        
        # --- Check Termination ---
        self.steps += 1
        if self.game_over or not self.blocks or self.steps >= self.MAX_STEPS:
            terminated = True
            if not self.blocks: # Win
                reward += 100
                self.score += 1000 # Bonus for winning
            elif self.game_over: # Loss
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "combo": self.combo,
            "blocks_remaining": len(self.blocks),
        }
    
    def _render_all(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Blocks
        for i, block in enumerate(self.blocks):
            pygame.draw.rect(self.screen, self.block_colors[i], block)
            pygame.draw.rect(self.screen, self.COLOR_BG, block, 1)

        # Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball
        ball_rect = self._get_ball_rect()
        pygame.gfxdraw.filled_circle(self.screen, ball_rect.centerx, ball_rect.centery, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_rect.centerx, ball_rect.centery, self.BALL_RADIUS, self.COLOR_BALL)

        # UI
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        
        combo_text = self.font_small.render(f"COMBO: x{self.combo}", True, self.COLOR_TEXT)
        self.screen.blit(combo_text, (10, 35))

        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH - 20 - i * 25, 20, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH - 20 - i * 25, 20, self.BALL_RADIUS, self.COLOR_BALL)

    def _create_blocks(self):
        self.blocks = []
        self.block_colors = []
        block_width = 60
        block_height = 20
        gap = 4
        rows = 5
        cols = 10
        
        for r in range(rows):
            for c in range(cols):
                x = c * (block_width + gap) + gap
                y = r * (block_height + gap) + 60
                self.blocks.append(pygame.Rect(x, y, block_width, block_height))
                self.block_colors.append(self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)])

    def _reset_ball(self):
        self.ball_on_paddle = True
        self.current_ball_speed = self.INITIAL_BALL_SPEED
        self.ball_vel_x = 0
        self.ball_vel_y = 0
        self._reset_ball_position()

    def _reset_ball_position(self):
        self.ball_pos_x = self.paddle.centerx
        self.ball_pos_y = self.paddle.top - self.BALL_RADIUS

    def _get_ball_rect(self):
        return pygame.Rect(
            self.ball_pos_x - self.BALL_RADIUS,
            self.ball_pos_y - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )

    def _normalize_ball_velocity(self):
        speed = math.sqrt(self.ball_vel_x**2 + self.ball_vel_y**2)
        if speed > 0:
            self.ball_vel_x = (self.ball_vel_x / speed) * self.current_ball_speed
            self.ball_vel_y = (self.ball_vel_y / speed) * self.current_ball_speed

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.uniform(2, 5),
                'color': color,
                'life': self.np_random.integers(20, 40)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Set up a window to display the game
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    while running:
        # --- Human Input ---
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
        total_reward += reward

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}")
            obs, info = env.reset()
            total_reward = 0
            
        # Control the frame rate
        env.clock.tick(60)

    env.close()