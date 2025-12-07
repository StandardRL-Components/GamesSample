
# Generated: 2025-08-27T17:58:28.134696
# Source Brief: brief_01696.md
# Brief Index: 1696

        
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
    A fast-paced, grid-based block breaker where strategic paddle positioning
    and risk-taking are rewarded. The player controls a paddle to bounce a ball,
    breaking a grid of blocks.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based block breaker. Position your paddle to break all the blocks and achieve a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000 * self.FPS // 30 # Scale max steps if FPS changes

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 200, 0)
        self.COLOR_BALL_GLOW = (255, 200, 0, 64)
        self.COLOR_BLOCK = (100, 120, 200)
        self.COLOR_BLOCK_OUTLINE = (150, 170, 250)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_PARTICLE = (255, 220, 50)
        
        # Game element properties
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_Y = self.HEIGHT - 40
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 6
        self.MAX_BALL_SPEED = 12
        self.BLOCK_ROWS = 5
        self.BLOCK_COLS = 10
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 60, 20
        self.BLOCK_SPACING = 4
        self.BLOCK_START_Y = 50
        
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
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # --- Game State ---
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = True
        self.blocks = []
        self.particles = []
        self.score = 0
        self.lives = 0
        self.steps = 0
        
        # Initialize state variables
        self.reset()
        
        # --- Validation ---
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        
        # Paddle
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.PADDLE_Y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Blocks
        self.blocks = []
        grid_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.WIDTH - grid_width) / 2
        for row in range(self.BLOCK_ROWS):
            for col in range(self.BLOCK_COLS):
                x = start_x + col * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.BLOCK_START_Y + row * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                self.blocks.append(pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT))
        
        # Particles
        self.particles = []

        # Ball
        self._reset_ball()
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_attached = True
        self.ball_pos = pygame.math.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.math.Vector2(0, 0)
        
    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed

        # 1. Handle player input
        self._handle_input(action)
        
        # 2. Update game state
        self._update_ball()
        self._update_particles()
        
        # 3. Handle collisions and calculate rewards
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # 4. Check for termination
        terminated = self._check_termination()
        if terminated:
            if len(self.blocks) == 0:
                reward += 100  # Win bonus
            elif self.lives <= 0:
                reward += -100 # Loss penalty

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        # Paddle movement
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        # Launch ball
        if space_held and self.ball_attached:
            self.ball_attached = False
            # Sound: Ball Launch
            self.ball_vel = pygame.math.Vector2(
                (random.random() - 0.5) * 2, -1
            ).normalize() * self.INITIAL_BALL_SPEED

    def _update_ball(self):
        if self.ball_attached:
            self.ball_pos.x = self.paddle.centerx
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
        else:
            self.ball_pos += self.ball_vel
            # Speed cap
            speed = self.ball_vel.length()
            if speed > self.MAX_BALL_SPEED:
                self.ball_vel.scale_to_length(self.MAX_BALL_SPEED)

    def _update_particles(self):
        # Update and remove old particles
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(
            self.ball_pos.x - self.BALL_RADIUS,
            self.ball_pos.y - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
        )

        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos.x))
            # Sound: Wall Bounce
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            # Sound: Wall Bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel.y > 0:
            # Sound: Paddle Hit
            self.ball_vel.y *= -1
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS # Prevent sticking
            
            # Risky hit reward
            hit_offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += hit_offset * 4 # Add spin
            
            if abs(hit_offset) > 0.7: # Edge hit
                reward += 5
            else: # Center hit
                reward += -2

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block):
                # Sound: Block Break
                reward += 1
                self.score += 10
                self.blocks.remove(block)
                self._create_particles(block.center)
                
                # Determine bounce direction
                # A simple but effective way is to check which axis has less overlap
                overlap_x = min(ball_rect.right, block.right) - max(ball_rect.left, block.left)
                overlap_y = min(ball_rect.bottom, block.bottom) - max(ball_rect.top, block.top)

                if overlap_x < overlap_y:
                    self.ball_vel.x *= -1
                else:
                    self.ball_vel.y *= -1
                break

        # Lose a life
        if self.ball_pos.y > self.HEIGHT:
            # Sound: Lose Life
            self.lives -= 1
            if self.lives > 0:
                self._reset_ball()
        
        return reward
    
    def _create_particles(self, pos):
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'lifetime': random.randint(10, 20)
            })

    def _check_termination(self):
        return (
            self.lives <= 0 or 
            len(self.blocks) == 0 or 
            self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background_grid()
        self._render_blocks()
        self._render_particles()
        self._render_paddle()
        self._render_ball()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_grid(self):
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_blocks(self):
        for block in self.blocks:
            pygame.draw.rect(self.screen, self.COLOR_BLOCK, block)
            pygame.draw.rect(self.screen, self.COLOR_BLOCK_OUTLINE, block, 2)
    
    def _render_particles(self):
        for p in self.particles:
            size = max(0, int(p['lifetime'] / 4))
            rect = pygame.Rect(int(p['pos'].x - size/2), int(p['pos'].y - size/2), size, size)
            pygame.draw.rect(self.screen, self.COLOR_PARTICLE, rect)

    def _render_paddle(self):
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
    
    def _render_ball(self):
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_BALL_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (int(self.ball_pos.x - glow_radius), int(self.ball_pos.y - glow_radius)))
        
        # Main ball
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
    
    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Lives
        lives_text = self.font_main.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 180, 10))
        for i in range(self.lives):
            life_paddle = pygame.Rect(self.WIDTH - 80 + (i * 25), 16, 20, 5)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_paddle, border_radius=2)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks)
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

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you need a different setup that uses pygame.display
    # This example demonstrates the Gym API usage with a random agent.
    
    print("--- Environment Info ---")
    print(f"Description: {env.game_description}")
    print(f"Controls: {env.user_guide}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    
    # Create a window to display the game
    pygame.display.set_caption("Block Breaker Gym Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Random Agent ---
        # action = env.action_space.sample()

        # --- Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Not used in this game

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        # Pygame uses (width, height) but numpy uses (height, width, channels)
        # So we need to transpose the observation from (H, W, C) back to (W, H, C)
        # for pygame.surfarray.make_surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
            
        env.clock.tick(env.FPS)

    env.close()