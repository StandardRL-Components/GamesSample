
# Generated: 2025-08-27T17:15:35.950880
# Source Brief: brief_01476.md
# Brief Index: 1476

        
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
        "Controls: Use ← and → to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade game. Control the paddle to bounce the ball and break all the blocks. "
        "Score points for hitting and destroying blocks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PADDLE = (0, 150, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BLOCK_HEALTHY = (0, 200, 100)
        self.COLOR_BLOCK_DAMAGED = (220, 50, 50)
        self.COLOR_PARTICLE = (255, 200, 0)
        self.COLOR_UI = (240, 240, 240)

        # Game constants
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 5.0
        self.MAX_BALL_SPEED = 10.0
        self.MAX_STEPS = 5000
        
        # Game state variables are initialized in reset()
        self.paddle = None
        self.ball = None
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.blocks_destroyed = 0
        self.total_blocks = 0
        
        # Initialize state variables for the first time
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.blocks_destroyed = 0
        
        # Reset paddle
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        # Reset ball
        start_angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball = {
            "pos": [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1],
            "vel": [self.INITIAL_BALL_SPEED * math.cos(start_angle), self.INITIAL_BALL_SPEED * math.sin(start_angle)],
            "speed": self.INITIAL_BALL_SPEED
        }
        
        # Reset blocks
        self.blocks = []
        block_rows = 4
        block_cols = 10
        block_width = 58
        block_height = 20
        block_spacing = 6
        start_x = (self.WIDTH - (block_cols * (block_width + block_spacing) - block_spacing)) / 2
        start_y = 40
        for r in range(block_rows):
            for c in range(block_cols):
                health = max(1, block_rows - r)
                self.blocks.append({
                    "rect": pygame.Rect(
                        start_x + c * (block_width + block_spacing),
                        start_y + r * (block_height + block_spacing),
                        block_width,
                        block_height
                    ),
                    "health": health,
                    "max_health": health,
                    "active": True
                })
        self.total_blocks = len(self.blocks)
        
        # Reset particles
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0=none, 3=left, 4=right
        
        # Initialize reward for this step
        reward = -0.01  # Small penalty to encourage faster completion

        # 1. Update Paddle
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)
        
        # 2. Update Ball
        self.ball["pos"][0] += self.ball["vel"][0]
        self.ball["pos"][1] += self.ball["vel"][1]
        
        ball_rect = pygame.Rect(
            self.ball["pos"][0] - self.BALL_RADIUS, 
            self.ball["pos"][1] - self.BALL_RADIUS, 
            self.BALL_RADIUS * 2, 
            self.BALL_RADIUS * 2
        )
        
        # 3. Ball Collision Logic
        # Wall collision
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball["vel"][0] *= -1
            ball_rect.left = np.clip(ball_rect.left, 0, self.WIDTH - ball_rect.width)
            # Sound effect: wall bounce
        if ball_rect.top <= 0:
            self.ball["vel"][1] *= -1
            ball_rect.top = np.clip(ball_rect.top, 0, self.HEIGHT - ball_rect.height)
            # Sound effect: wall bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball["vel"][1] > 0:
            self.ball["vel"][1] *= -1
            
            # Change horizontal velocity based on where it hit the paddle
            hit_offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball["vel"][0] = self.ball["speed"] * hit_offset * 1.2
            
            # Normalize velocity to maintain constant speed
            current_vel_mag = math.sqrt(self.ball["vel"][0]**2 + self.ball["vel"][1]**2)
            if current_vel_mag > 0:
                scale = self.ball["speed"] / current_vel_mag
                self.ball["vel"][0] *= scale
                self.ball["vel"][1] *= scale
            # Sound effect: paddle hit
            
        # Block collision
        for block in self.blocks:
            if block["active"] and ball_rect.colliderect(block["rect"]):
                block["health"] -= 1
                reward += 1.0
                
                # Determine bounce direction
                if abs(ball_rect.centerx - block["rect"].centerx) > abs(ball_rect.centery - block["rect"].centery):
                    self.ball["vel"][0] *= -1
                else:
                    self.ball["vel"][1] *= -1

                if block["health"] <= 0:
                    block["active"] = False
                    reward += 2.0
                    self.blocks_destroyed += 1
                    self.score += 10
                    self._create_particles(block["rect"].center)
                    # Sound effect: block destroy
                    
                    # Increase difficulty
                    if self.blocks_destroyed > 0 and self.blocks_destroyed % 5 == 0:
                        new_speed = self.ball["speed"] + 0.5
                        self.ball["speed"] = min(new_speed, self.MAX_BALL_SPEED)

                else:
                    self.score += 1
                    # Sound effect: block hit
                
                break # Only one block collision per frame

        # 4. Update Particles
        self._update_particles()
        
        # 5. Check Termination Conditions
        self.steps += 1
        terminated = False
        
        # Win condition
        if self.blocks_destroyed == self.total_blocks:
            reward = 100.0
            terminated = True
            self.game_over = True
            # Sound effect: win
        
        # Lose condition
        if ball_rect.top >= self.HEIGHT:
            reward = -100.0
            terminated = True
            self.game_over = True
            # Sound effect: lose
            
        # Max steps condition
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _create_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render blocks
        for block in self.blocks:
            if block["active"]:
                health_ratio = block["health"] / block["max_health"]
                color = self._lerp_color(self.COLOR_BLOCK_DAMAGED, self.COLOR_BLOCK_HEALTHY, health_ratio)
                pygame.draw.rect(self.screen, color, block["rect"], border_radius=3)
                
        # Render particles
        for p in self.particles:
            size = max(0, int(p["life"] / 6))
            pygame.draw.rect(self.screen, self.COLOR_PARTICLE, (int(p["pos"][0]), int(p["pos"][1]), size, size))

        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)
        
        # Render ball
        pygame.gfxdraw.aacircle(
            self.screen, 
            int(self.ball["pos"][0]), 
            int(self.ball["pos"][1]), 
            self.BALL_RADIUS, 
            self.COLOR_BALL
        )
        pygame.gfxdraw.filled_circle(
            self.screen, 
            int(self.ball["pos"][0]), 
            int(self.ball["pos"][1]), 
            self.BALL_RADIUS, 
            self.COLOR_BALL
        )
        
    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

    def _lerp_color(self, c1, c2, t):
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t)
        )
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_destroyed": self.blocks_destroyed,
            "ball_speed": self.ball["speed"]
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game
    env = GameEnv(render_mode="rgb_array")
    
    # Use a separate display for human interaction
    human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Breakout")
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        # Get player input from Pygame events
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        action = [movement, 0, 0] # space and shift are unused

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the human-facing screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle game over
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Handle quitting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Control the frame rate
        env.clock.tick(60)

    env.close()