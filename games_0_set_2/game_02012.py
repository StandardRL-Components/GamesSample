
# Generated: 2025-08-27T18:58:02.071967
# Source Brief: brief_02012.md
# Brief Index: 2012

        
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
        "Controls: ← to move left, → to move right."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block-breaker. Use the paddle to bounce the ball and destroy all the blocks. Don't let the ball hit the floor!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 2000
        
        # Colors (retro neon theme)
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 200, 0)
        self.COLOR_BALL_GLOW = (255, 100, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Yellow
            (0, 255, 0),    # Green
            (255, 80, 80),  # Red
        ]

        # Paddle settings
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 10

        # Ball settings
        self.BALL_RADIUS = 7
        self.BALL_SPEED_INITIAL = 5
        self.BALL_SPEED_MAX = 8
        
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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        
        # Call reset to populate initial state
        self.reset()

        # Run self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        
        # Paddle
        paddle_y = self.SCREEN_HEIGHT - 40
        paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball
        self._reset_ball()

        # Blocks
        self.blocks = []
        n_rows = 5
        n_cols = 10
        block_width = 58
        block_height = 20
        total_block_width = n_cols * (block_width + 4)
        start_x = (self.SCREEN_WIDTH - total_block_width) / 2
        start_y = 50
        for i in range(n_rows):
            for j in range(n_cols):
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                block_x = start_x + j * (block_width + 4)
                block_y = start_y + i * (block_height + 4)
                self.blocks.append({"rect": pygame.Rect(block_x, block_y, block_width, block_height), "color": color})

        # Particles
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        """Resets the ball's position and velocity."""
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1)
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Downward angle
        self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_SPEED_INITIAL
    
    def step(self, action):
        # Base reward for surviving a step
        reward = -0.01

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update paddle position
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)
        
        # Update ball position and handle collisions
        collision_reward = self._move_ball_and_collide()
        reward += collision_reward

        # Update particles
        self._update_particles()
        
        # Update step counter
        self.steps += 1

        # Check for termination conditions
        terminated = False
        if len(self.blocks) == 0: # Victory
            reward += 100
            terminated = True
        elif self.lives <= 0: # Loss
            # The -50 reward for losing a life is handled in _move_ball_and_collide
            terminated = True
        elif self.steps >= self.MAX_STEPS: # Timeout
            terminated = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _move_ball_and_collide(self):
        """Updates ball position and handles all collisions."""
        reward = 0
        
        self.ball_pos += self.ball_vel
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel.x *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.SCREEN_WIDTH, ball_rect.right)
            # Sound: wall_hit.wav
        if ball_rect.top <= 0:
            self.ball_vel.y *= -1
            ball_rect.top = max(0, ball_rect.top)
            # Sound: wall_hit.wav
        
        # Bottom wall (lose life)
        if ball_rect.bottom >= self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 50
            # Sound: lose_life.wav
            if self.lives > 0:
                self._reset_ball()
            return reward

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel.y > 0:
            # Sound: paddle_hit.wav
            self.ball_vel.y *= -1
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS # Prevent sticking

            # Calculate hit position for reward and angle change
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            
            # Reward for risky vs safe play
            if abs(offset) > 0.4: # Edge hit
                reward += 0.1
            else: # Center hit
                reward -= 0.2

            # Change ball's horizontal velocity based on hit location
            self.ball_vel.x += offset * 2.5
            self.ball_vel.normalize_ip()
            current_speed = math.hypot(self.ball_vel.x, self.ball_vel.y) * self.BALL_SPEED_INITIAL
            self.ball_vel *= min(self.BALL_SPEED_MAX, current_speed * 1.02) # Slightly speed up on hit
            
        # Block collisions
        collided_block_index = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if collided_block_index != -1:
            # Sound: block_break.wav
            block_data = self.blocks.pop(collided_block_index)
            block_rect = block_data['rect']
            reward += 1
            self.score += 1

            # Create particles
            self._create_particles(block_rect.center, block_data['color'])

            # Determine collision side to reflect ball correctly
            prev_ball_rect = pygame.Rect(ball_rect.x - self.ball_vel.x, ball_rect.y - self.ball_vel.y, ball_rect.width, ball_rect.height)
            
            # Check for horizontal collision
            if prev_ball_rect.right <= block_rect.left or prev_ball_rect.left >= block_rect.right:
                self.ball_vel.x *= -1
            # Check for vertical collision
            if prev_ball_rect.bottom <= block_rect.top or prev_ball_rect.top >= block_rect.bottom:
                self.ball_vel.y *= -1

        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(10, 25)
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "lifetime": lifetime, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # --- Render all game elements ---
        # Particles (rendered first, behind other elements)
        for p in self.particles:
            size = max(0, int(p['lifetime'] / 4))
            pygame.draw.rect(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y), size, size))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, self.COLOR_BG, block['rect'], 1) # Outline

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball (with glow effect)
        ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # --- Render UI overlay ---
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        lives_text = self.font_large.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To display the game in a window ---
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Get user input for manual play
        action = np.array([0, 0, 0]) # Default no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Control FPS

    env.close()