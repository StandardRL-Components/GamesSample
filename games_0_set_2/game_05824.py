
# Generated: 2025-08-28T06:12:18.797395
# Source Brief: brief_05824.md
# Brief Index: 5824

        
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


class Particle:
    """A simple particle for effects."""
    def __init__(self, x, y, color, lifetime_range=(10, 20), speed_range=(1, 3), radius_range=(1, 3)):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(*speed_range)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = random.randint(*lifetime_range)
        self.initial_lifetime = self.lifetime
        self.color = color
        self.radius = random.randint(*radius_range)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0:
            # Fade out effect
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            # Use gfxdraw for anti-aliased circles
            pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), self.radius, (*self.color, alpha))
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.radius, (*self.color, alpha))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Use ← and → to move the paddle."
    game_description = "A retro arcade game. Control the paddle to deflect a powerful energy ball and shatter the block field above. Don't let the ball pass you!"
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (220, 220, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BLOCK_HEALTHY = (0, 200, 100)
        self.COLOR_BLOCK_DAMAGED = (255, 200, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_BOUNDARY = (80, 80, 100)
        
        # Game constants
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 7
        self.BALL_SPEED = 5
        self.MAX_BALL_SPIN = 2.5
        self.BLOCK_COLS, self.BLOCK_ROWS = 10, 4
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 58, 20
        self.BLOCK_SPACING = 6
        self.MAX_STEPS = 30 * 30 # 30 seconds at 30fps
        
        # State variables are initialized in reset()
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = None
        self.particles = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        
        # Paddle
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT - 10
        self.paddle = pygame.Rect((self.WIDTH - self.PADDLE_WIDTH) // 2, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        # Ball
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        self.ball_vel = [random.choice([-1, 1]) * self.BALL_SPEED * 0.5, -self.BALL_SPEED]
        
        # Blocks
        self.blocks = []
        total_block_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.WIDTH - total_block_width) // 2
        start_y = 50
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                x = start_x + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = start_y + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({'rect': block_rect, 'health': 2})
        
        # Particles
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage efficiency
        
        if not self.game_over:
            # Unpack action
            movement = action[0]
            
            # --- 1. Update Paddle ---
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED
            self.paddle.clamp_ip(self.screen.get_rect())

            # --- 2. Update Ball ---
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            # --- 3. Collision Detection ---
            # Walls
            if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
                self.ball_vel[0] *= -1
                ball_rect.left = max(0, ball_rect.left)
                ball_rect.right = min(self.WIDTH, ball_rect.right)
                self.ball_pos[0] = ball_rect.centerx
                reward -= 0.1 # Penalty for hitting side walls
            if ball_rect.top <= 0:
                self.ball_vel[1] *= -1
                ball_rect.top = max(0, ball_rect.top)
                self.ball_pos[1] = ball_rect.centery

            # Paddle
            if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
                self.ball_vel[1] *= -1
                ball_rect.bottom = self.paddle.top
                self.ball_pos[1] = ball_rect.centery
                
                # Add spin based on hit location
                offset = (ball_rect.centerx - self.paddle.centerx) / (self.paddle.width / 2)
                spin = offset * self.MAX_BALL_SPIN
                self.ball_vel[0] = np.clip(self.ball_vel[0] + spin, -self.BALL_SPEED, self.BALL_SPEED)
                
                reward += 0.2 # Reward for hitting the ball
                # # PADDLE_HIT_SOUND

            # Blocks
            hit_block = None
            for block in self.blocks:
                if ball_rect.colliderect(block['rect']):
                    hit_block = block
                    break
            
            if hit_block:
                block['health'] -= 1
                
                # Determine collision side to correctly reflect the ball
                overlap = ball_rect.clip(block['rect'])
                if overlap.width < overlap.height:
                    self.ball_vel[0] *= -1 # Side collision
                else:
                    self.ball_vel[1] *= -1 # Top/bottom collision

                if block['health'] <= 0:
                    self.score += 10
                    reward += 10
                    self.blocks.remove(block)
                    # Create particles
                    for _ in range(20):
                        self.particles.append(Particle(block['rect'].centerx, block['rect'].centery, self.COLOR_BLOCK_DAMAGED))
                    # # BLOCK_DESTROY_SOUND
                else:
                    reward += 1
                    for _ in range(5):
                        self.particles.append(Particle(ball_rect.centerx, ball_rect.centery, self.COLOR_BLOCK_HEALTHY, lifetime_range=(5,10), radius_range=(1,2)))
                    # # BLOCK_HIT_SOUND
            
            # Miss (bottom wall)
            if ball_rect.top >= self.HEIGHT:
                self.lives -= 1
                reward -= 5
                if self.lives > 0:
                    # Reset ball position
                    self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
                    self.ball_vel = [random.choice([-1, 1]) * self.BALL_SPEED * 0.5, -self.BALL_SPEED]
                    # # LIFE_LOST_SOUND
                
        # --- 4. Update Particles ---
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifetime > 0]
        
        # --- 5. Check Termination ---
        self.steps += 1
        terminated = False
        if self.lives <= 0:
            terminated = True
            reward -= 100
            self.game_over = True
            # # GAME_OVER_SOUND
        elif not self.blocks:
            terminated = True
            reward += 100
            self.game_over = True
            # # VICTORY_SOUND
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw blocks
        for block in self.blocks:
            color = self.COLOR_BLOCK_HEALTHY if block['health'] == 2 else self.COLOR_BLOCK_DAMAGED
            pygame.draw.rect(self.screen, color, block['rect'], border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_BG, block['rect'], width=2, border_radius=3)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Draw ball with a trail effect
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 20, 10))

        # Game Over / Victory message
        if self.game_over:
            if not self.blocks:
                msg = "VICTORY!"
                color = self.COLOR_BLOCK_HEALTHY
            else:
                msg = "GAME OVER"
                color = self.COLOR_BLOCK_DAMAGED
            
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(over_text, text_rect)
            
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "human" to display the game window
    render_mode = "human" # "rgb_array" for training, "human" for playing

    if render_mode == "human":
        # Add 'human' to metadata for compatibility with gym.make
        GameEnv.metadata["render_modes"].append("human")
        
        class HumanGameEnv(GameEnv):
            def __init__(self, render_mode="human"):
                super().__init__(render_mode)
                self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
                pygame.display.set_caption("Breakout")

            def _get_observation(self):
                # In human mode, we render to the display screen
                self.screen.fill(self.COLOR_BG)
                self._render_game()
                self._render_ui()
                pygame.display.flip()
                
                # We still need to return the array for the agent
                arr = pygame.surfarray.array3d(self.screen)
                return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

        env = HumanGameEnv(render_mode="human")
    else:
        env = GameEnv()

    obs, info = env.reset()
    done = False
    
    # --- Manual Play Loop ---
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        if render_mode == "human":
            # Map keyboard keys to actions for human play
            action.fill(0)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            if keys[pygame.K_RIGHT]:
                action[0] = 4
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset() # Reset on 'R' key
        else:
            # Agent would choose an action here
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if render_mode == "human":
            env.clock.tick(30) # Control FPS for human play
        
        if done:
            print(f"Game Over. Final Info: {info}")
            if render_mode == "human":
                pygame.time.wait(2000) # Pause before closing
            obs, info = env.reset()

    env.close()