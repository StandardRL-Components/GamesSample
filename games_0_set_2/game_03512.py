
# Generated: 2025-08-27T23:34:14.384847
# Source Brief: brief_03512.md
# Brief Index: 3512

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# --- Helper Classes for Game Objects ---

class Paddle:
    def __init__(self, x, y, width, height, speed, screen_width):
        self.rect = pygame.Rect(x - width // 2, y, width, height)
        self.color = (255, 255, 255)
        self.speed = speed
        self.screen_width = screen_width

    def move(self, direction):
        self.rect.x += direction * self.speed
        self.rect.left = max(0, self.rect.left)
        self.rect.right = min(self.screen_width, self.rect.right)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect, border_radius=3)

class Ball:
    def __init__(self, x, y, radius, screen_width, screen_height):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = (0, 255, 128)
        self.glow_color = (0, 255, 128, 50)
        self.base_speed = 4.0
        self.vx = 0
        self.vy = 0
        self.screen_width = screen_width
        self.screen_height = screen_height

    def launch(self, np_random):
        angle = np_random.uniform(math.pi * 0.25, math.pi * 0.75)
        self.vx = self.base_speed * math.cos(angle)
        self.vy = -self.base_speed * math.sin(angle)

    def update(self):
        self.x += self.vx
        self.y += self.vy

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)

    def update_speed(self, blocks_destroyed_count):
        # As per brief: speed increases by 0.2 for every 10 blocks destroyed
        speed_increase_factor = 1.0 + (blocks_destroyed_count // 10) * (0.2 / self.base_speed)
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        if current_speed > 0:
            self.vx = (self.vx / current_speed) * self.base_speed * speed_increase_factor
            self.vy = (self.vy / current_speed) * self.base_speed * speed_increase_factor

    def draw(self, surface):
        # Glow effect
        glow_radius = int(self.radius * 2.5)
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), glow_radius, self.glow_color)
        # Main ball
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.radius, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), self.radius, self.color)

class Block:
    def __init__(self, x, y, width, height, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.border_color = tuple(min(255, c + 50) for c in color)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        pygame.draw.rect(surface, self.border_color, self.rect, 1)

class Particle:
    def __init__(self, x, y, np_random):
        self.x = x
        self.y = y
        self.vx = np_random.uniform(-1.5, 1.5)
        self.vy = np_random.uniform(-1.5, 1.5)
        self.radius = np_random.uniform(1, 3)
        self.lifespan = 30  # 1 second at 30fps
        self.life = self.lifespan
        self.color = (255, 255, 255)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.05 # a little gravity
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.lifespan))
            color = self.color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), color)


# --- Gymnasium Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ← and → to move the paddle. Break all the blocks to win."
    )

    game_description = (
        "A minimalist block breaker. Clear the screen of blocks by bouncing the ball. "
        "Get bonus points for risky hits on the edge of the paddle."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.width, self.height = 640, 400
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Initialize Pygame and fonts
        pygame.init()
        pygame.font.init()
        try:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_game_over = pygame.font.Font(None, 64)
        except IOError:
            self.font_ui = pygame.font.SysFont("sans", 24)
            self.font_game_over = pygame.font.SysFont("sans", 64)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Colors
        self.COLOR_BG = (15, 15, 25)

        # Game state variables are initialized in reset()
        self.paddle = None
        self.ball = None
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.balls_left = 0
        self.game_over = False
        self.blocks_destroyed_count = 0
        self.total_blocks = 50

        # Initialize state
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        self.blocks_destroyed_count = 0

        # Create paddle
        self.paddle = Paddle(self.width // 2, self.height - 30, 100, 10, 8, self.width)

        # Create ball
        self.ball = Ball(self.width // 2, self.height // 2 + 50, 7, self.width, self.height)
        self.ball.launch(self.np_random)

        # Create blocks
        self.blocks = []
        block_rows = 5
        block_cols = 10
        block_width = (self.width - 40) // block_cols
        block_height = 20
        start_y = 50
        for i in range(block_rows):
            for j in range(block_cols):
                x = 20 + j * block_width
                y = start_y + i * block_height
                # Color varies with height
                blue_val = 100 + (i * 25)
                color = (0, 70, blue_val)
                self.blocks.append(Block(x, y, block_width - 2, block_height - 2, color))
        self.total_blocks = len(self.blocks)

        # Clear particles
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.02  # Time penalty

        # --- Handle Input ---
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.move(-1)
        elif movement == 4:  # Right
            self.paddle.move(1)

        # --- Update Game State ---
        self.ball.update()
        
        # Update particles
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

        # --- Collision Detection ---
        # Ball with walls
        if self.ball.x - self.ball.radius < 0 or self.ball.x + self.ball.radius > self.width:
            self.ball.vx *= -1
            self.ball.x = max(self.ball.radius, min(self.ball.x, self.width - self.ball.radius))
            # sfx: wall_bounce

        if self.ball.y - self.ball.radius < 0:
            self.ball.vy *= -1
            self.ball.y = self.ball.radius
            # sfx: wall_bounce

        # Ball with bottom (lose life)
        if self.ball.y + self.ball.radius > self.height:
            self.balls_left -= 1
            if self.balls_left <= 0:
                self.game_over = True
                reward -= 100  # Loss penalty
                # sfx: game_over_loss
            else:
                # Reset ball position
                self.ball.x = self.paddle.rect.centerx
                self.ball.y = self.height // 2 + 50
                self.ball.launch(self.np_random)
                # sfx: lose_life

        # Ball with paddle
        ball_rect = self.ball.get_rect()
        if ball_rect.colliderect(self.paddle.rect) and self.ball.vy > 0:
            # sfx: paddle_hit
            self.ball.vy *= -1
            self.ball.y = self.paddle.rect.top - self.ball.radius

            # Influence horizontal velocity based on hit location
            offset = (self.ball.x - self.paddle.rect.centerx) / (self.paddle.rect.width / 2)
            self.ball.vx = self.ball.base_speed * offset * 1.5 # *1.5 for more control
            
            # Clamp horizontal velocity to prevent extreme angles
            self.ball.vx = max(-self.ball.base_speed, min(self.ball.vx, self.ball.base_speed))

            # Reward for risky play
            if abs(offset) > 0.8: # Outer 20% of the paddle (10% on each side)
                reward += 2.0
            else:
                reward += 0.1

        # Ball with blocks
        for block in self.blocks[:]:
            if ball_rect.colliderect(block.rect):
                # sfx: block_break
                reward += 1.0

                # Determine collision side to correctly reflect velocity
                prev_ball_rect = pygame.Rect(self.ball.x - self.ball.vx - self.ball.radius, self.ball.y - self.ball.vy - self.ball.radius, self.ball.radius*2, self.ball.radius*2)
                
                # Check for horizontal collision
                if prev_ball_rect.centery >= block.rect.bottom or prev_ball_rect.centery <= block.rect.top:
                    self.ball.vy *= -1
                else: # Vertical collision
                    self.ball.vx *= -1

                self.blocks.remove(block)
                self.blocks_destroyed_count += 1
                
                # Spawn particles
                for _ in range(10):
                    self.particles.append(Particle(block.rect.centerx, block.rect.centery, self.np_random))
                
                # Update ball speed if a multiple of 10 blocks is destroyed
                if self.blocks_destroyed_count > 0 and self.blocks_destroyed_count % 10 == 0:
                    self.ball.update_speed(self.blocks_destroyed_count)
                
                break # Only handle one block collision per frame

        # --- Update Score and Steps ---
        self.score += reward
        self.steps += 1

        # --- Check Termination Conditions ---
        if not self.blocks:
            self.game_over = True
            reward += 100  # Win bonus
            self.score += 100
            # sfx: game_over_win
            
        if self.steps >= 1000:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw blocks
        for block in self.blocks:
            block.draw(self.screen)
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
            
        # Draw paddle
        self.paddle.draw(self.screen)
        
        # Draw ball
        self.ball.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Balls left
        ball_icon_radius = 5
        for i in range(self.balls_left):
            x = self.width - 20 - (i * (ball_icon_radius * 2 + 5))
            pygame.draw.circle(self.screen, (255, 255, 255), (x, 18), ball_icon_radius)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            if not self.blocks:
                msg = "YOU WIN!"
                color = (0, 255, 128)
            else:
                msg = "GAME OVER"
                color = (255, 50, 50)
                
            game_over_text = self.font_game_over.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
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
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Test a few random steps
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"Episode finished. Final Info: {info}")
            obs, info = env.reset()
    
    print("Environment ran for 100 steps successfully.")
    env.close()

    # Example of how to visualize the game with Pygame
    print("\nStarting interactive visualization...")
    
    # We need to unset the dummy driver to show a window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    display_screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Block Breaker")
    
    running = True
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        # The env auto-advances, so we just step once per frame
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            pygame.time.wait(2000) # Pause for 2 seconds on game over
            obs, info = env.reset()
            
        env.clock.tick(30) # Run at 30 FPS
        
    env.close()