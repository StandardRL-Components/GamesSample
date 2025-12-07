
# Generated: 2025-08-27T17:39:26.065158
# Source Brief: brief_01600.md
# Brief Index: 1600

        
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


# Helper class for particle effects
class Particle:
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = np_random.integers(20, 40)
        self.color = color
        self.radius = np_random.uniform(2, 6)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.radius *= 0.95  # Shrink over time

    def draw(self, surface):
        if self.lifespan > 0 and self.radius > 1:
            pos = (int(self.x), int(self.y))
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(self.radius), self.color)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(self.radius), self.color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move the paddle. Break all the blocks to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker. Strategic paddle positioning is key to maximizing score and survival."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 8
        self.BALL_SPEED_INITIAL = 5
        self.BALL_SPEED_MAX = 10
        self.MAX_STEPS = 5000
        self.INITIAL_LIVES = 3

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_GRID = (25, 25, 45)
        self.COLOR_PADDLE = (220, 220, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 150, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = [
            (217, 87, 99), (217, 146, 87), (175, 217, 87),
            (87, 217, 146), (87, 146, 217)
        ]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_score = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # Initialize state variables
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = None
        self.particles = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.particles = []

        # Paddle state
        paddle_y = self.SCREEN_HEIGHT - self.PADDLE_HEIGHT * 2
        paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle_rect = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Blocks state
        self.blocks = []
        block_width = 58
        block_height = 20
        for i in range(5):  # Rows
            for j in range(10):  # Columns
                block_x = j * (block_width + 6) + 2
                block_y = i * (block_height + 6) + 40
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                self.blocks.append({'rect': pygame.Rect(block_x, block_y, block_width, block_height), 'color': color})
        
        self._launch_ball()

        return self._get_observation(), self._get_info()

    def _launch_ball(self):
        self.ball_pos = [self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS - 1]
        angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4) # Upwards
        self.ball_vel = [
            self.BALL_SPEED_INITIAL * math.cos(angle),
            self.BALL_SPEED_INITIAL * math.sin(angle)
        ]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(30) # Ensure consistent frame rate for physics
        self.steps += 1
        reward = 0

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Reward Calculation ---
        # 1. Survival reward
        reward += 0.01 
        
        # 2. Penalty for moving paddle away from ball's x-position when ball is descending
        if self.ball_vel[1] > 0:
            ball_x = self.ball_pos[0]
            paddle_center_x = self.paddle_rect.centerx
            is_moving_left = movement == 3
            is_moving_right = movement == 4
            
            if (is_moving_left and paddle_center_x > ball_x) or \
               (is_moving_right and paddle_center_x < ball_x):
                pass # Moving towards the ball, no penalty
            elif is_moving_left or is_moving_right:
                reward -= 0.02

        # --- Game Logic ---
        # 1. Move paddle
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        self.paddle_rect.clamp_ip(self.screen.get_rect())

        # 2. Move ball
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # 3. Collision detection
        # Walls
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.clamp_ip(self.screen.get_rect()) # Prevent getting stuck
            self.ball_pos[0] = ball_rect.centerx
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            
        # Paddle
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel[1] > 0:
            # Sound: paddle_hit.wav
            offset = (ball_rect.centerx - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2.5
            self.ball_vel[1] *= -1
            
            # Clamp ball speed
            speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            if speed > self.BALL_SPEED_MAX:
                scale = self.BALL_SPEED_MAX / speed
                self.ball_vel = [self.ball_vel[0] * scale, self.ball_vel[1] * scale]
            
            # Ensure ball is above paddle after bounce
            self.ball_pos[1] = self.paddle_rect.top - self.BALL_RADIUS - 1

        # Blocks
        hit_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            # Sound: block_break.wav
            block_hit = self.blocks.pop(hit_block_idx)
            reward += 1.0  # Reward for breaking a block
            self.score += 10
            
            # Create particles
            for _ in range(15):
                self.particles.append(Particle(ball_rect.centerx, ball_rect.centery, block_hit['color'], self.np_random))

            # Determine bounce direction
            # A simple vertical bounce is most common and stable for this genre
            self.ball_vel[1] *= -1
            
        # 4. Update particles
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0:
                self.particles.remove(p)

        # 5. Check for life loss
        if ball_rect.top > self.SCREEN_HEIGHT:
            # Sound: lose_life.wav
            self.lives -= 1
            reward -= 10.0 # Penalty for losing a life
            if self.lives > 0:
                self._launch_ball()
            else:
                self.game_over = True

        # --- Termination Check ---
        terminated = False
        if self.game_over: # Lost all lives
            terminated = True
            reward -= 50.0 # Penalty for losing the game
        elif not self.blocks: # Won the game
            terminated = True
            reward += 100.0 # Bonus for winning
            self.game_over = True # To show win message
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw blocks
        for block in self.blocks:
            r = block['rect']
            c = block['color']
            # Draw a slight 3D effect
            pygame.draw.rect(self.screen, tuple(max(0, val-30) for val in c), r.move(2, 2))
            pygame.draw.rect(self.screen, c, r)
            
        # Draw paddle
        pygame.draw.rect(self.screen, tuple(max(0, val-50) for val in self.COLOR_PADDLE), self.paddle_rect.move(3,3))
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=3)
        
        # Draw ball with glow
        if self.ball_pos:
            pos = (int(self.ball_pos[0]), int(self.ball_pos[1]))
            # Glow effect
            glow_surf = pygame.Surface((self.BALL_RADIUS * 4, self.BALL_RADIUS * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_BALL_GLOW, (self.BALL_RADIUS * 2, self.BALL_RADIUS * 2), self.BALL_RADIUS * 1.5)
            self.screen.blit(glow_surf, (pos[0] - self.BALL_RADIUS * 2, pos[1] - self.BALL_RADIUS * 2))
            # Main ball
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_score.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        life_radius = 8
        for i in range(self.lives):
            x = self.SCREEN_WIDTH - 20 - i * (life_radius * 2 + 5)
            y = 10 + life_radius
            pygame.gfxdraw.filled_circle(self.screen, x, y, life_radius, self.COLOR_PADDLE)
            pygame.gfxdraw.aacircle(self.screen, x, y, life_radius, self.COLOR_PADDLE)

        # Game Over / Win Message
        if self.game_over:
            if not self.blocks: # Win condition
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else: # Lose condition
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
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

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Display the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        # If game is over, wait for a key press to reset
        if terminated:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    obs, info = env.reset()
                    terminated = False
                if event.type == pygame.QUIT:
                    running = False

    env.close()