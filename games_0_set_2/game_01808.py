
# Generated: 2025-08-27T18:21:50.161897
# Source Brief: brief_01808.md
# Brief Index: 1808

        
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
        "A retro block-breaking game. Use the paddle to bounce the ball and destroy all the blocks before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_WALL = (150, 150, 160)
        self.COLOR_PADDLE = (50, 255, 50)
        self.COLOR_PADDLE_GLOW = (50, 255, 50, 50)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BALL_GLOW = (255, 255, 255, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.BLOCK_COLORS = {
            1: (255, 50, 50),    # Red
            2: (255, 150, 50),   # Orange
            3: (255, 255, 50),   # Yellow
            4: (50, 255, 50),    # Green
            5: (50, 150, 255)    # Blue
        }

        # Game parameters
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 9
        self.WALL_THICKNESS = 10
        self.MAX_EPISODE_STEPS = self.FPS * 60 # 60 seconds

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
        self.font = pygame.font.Font(None, 36)
        
        # State variables (will be initialized in reset)
        self.paddle = None
        self.ball = None
        self.ball_velocity = None
        self.ball_attached = None
        self.blocks = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.balls_remaining = 0
        self.time_remaining = 0
        self.game_over = False
        
        # Initialize state
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_remaining = 3
        self.time_remaining = self.MAX_EPISODE_STEPS
        
        self.particles = []
        self._generate_blocks()
        self._reset_ball_and_paddle()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1
        step_reward = -0.01  # Time penalty

        # --- Action Handling ---
        movement = action[0]
        space_pressed = action[1] == 1

        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, self.WALL_THICKNESS, self.WIDTH - self.WALL_THICKNESS - self.PADDLE_WIDTH)

        if self.ball_attached and space_pressed:
            self.ball_attached = False
            self.ball_velocity = [self.np_random.uniform(-0.5, 0.5), -1]
            # Sound: ball_launch
            
        # --- Game Logic ---
        step_reward += self._update_ball()
        self._update_particles()
        
        # --- Termination Check ---
        terminated = False
        if not any(b['active'] for b in self.blocks):
            step_reward += 50 # Win bonus
            terminated = True
            self.game_over = True
        elif self.balls_remaining <= 0:
            terminated = True # Loss
            self.game_over = True
        elif self.time_remaining <= 0:
            step_reward -= 10 # Timeout penalty
            terminated = True
            self.game_over = True
            
        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _reset_ball_and_paddle(self):
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )
        self.ball_attached = True
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball_velocity = [0, 0]

    def _generate_blocks(self):
        self.blocks = []
        block_width, block_height = 58, 20
        rows, cols = 5, 10
        points = [5, 4, 3, 2, 1]
        for r in range(rows):
            for c in range(cols):
                block_x = self.WALL_THICKNESS + c * (block_width + 2) + 1
                block_y = self.WALL_THICKNESS + 50 + r * (block_height + 2)
                point_val = points[r]
                self.blocks.append({
                    'rect': pygame.Rect(block_x, block_y, block_width, block_height),
                    'color': self.BLOCK_COLORS[point_val],
                    'points': point_val,
                    'active': True
                })

    def _update_ball(self):
        reward = 0
        if self.ball_attached:
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top
            return reward

        # Normalize velocity
        vel_mag = math.sqrt(self.ball_velocity[0]**2 + self.ball_velocity[1]**2)
        if vel_mag > 0:
            self.ball_velocity[0] = (self.ball_velocity[0] / vel_mag) * self.BALL_SPEED
            self.ball_velocity[1] = (self.ball_velocity[1] / vel_mag) * self.BALL_SPEED
        
        self.ball.x += self.ball_velocity[0]
        self.ball.y += self.ball_velocity[1]

        # Wall collisions
        if self.ball.left <= self.WALL_THICKNESS or self.ball.right >= self.WIDTH - self.WALL_THICKNESS:
            self.ball_velocity[0] *= -1
            self.ball.left = max(self.ball.left, self.WALL_THICKNESS)
            self.ball.right = min(self.ball.right, self.WIDTH - self.WALL_THICKNESS)
            # Sound: wall_bounce
        if self.ball.top <= self.WALL_THICKNESS:
            self.ball_velocity[1] *= -1
            self.ball.top = max(self.ball.top, self.WALL_THICKNESS)
            # Sound: wall_bounce

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_velocity[1] > 0:
            reward += 0.1
            self.ball.bottom = self.paddle.top
            
            # Change horizontal velocity based on hit location
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_velocity[0] = offset * 1.5 # Multiplier for more control
            self.ball_velocity[1] *= -1
            # Sound: paddle_hit

        # Block collisions
        for block in self.blocks:
            if block['active'] and self.ball.colliderect(block['rect']):
                block['active'] = False
                reward += block['points']
                self.score += block['points']
                
                self._create_particles(block['rect'].center, block['color'])
                
                # Reflection logic
                prev_ball_rect = self.ball.copy()
                prev_ball_rect.x -= self.ball_velocity[0]
                prev_ball_rect.y -= self.ball_velocity[1]

                if prev_ball_rect.bottom <= block['rect'].top or prev_ball_rect.top >= block['rect'].bottom:
                    self.ball_velocity[1] *= -1
                else:
                    self.ball_velocity[0] *= -1
                # Sound: block_break
                break

        # Bottom wall (lose ball)
        if self.ball.top >= self.HEIGHT:
            self.balls_remaining -= 1
            reward -= 50
            if self.balls_remaining > 0:
                self._reset_ball_and_paddle()
            # Sound: lose_ball

        return reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': self.np_random.integers(10, 20),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.HEIGHT))
        
        # Render blocks
        for block in self.blocks:
            if block['active']:
                pygame.draw.rect(self.screen, block['color'], block['rect'])
                pygame.draw.rect(self.screen, tuple(min(c+30, 255) for c in block['color']), block['rect'], 2)

        # Render paddle with glow
        glow_rect = self.paddle.inflate(10, 10)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, self.COLOR_PADDLE_GLOW, glow_surface.get_rect(), border_radius=5)
        self.screen.blit(glow_surface, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Render ball with glow
        if not self.game_over:
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW)
            pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)

        # Render particles
        for p in self.particles:
            alpha = max(0, int(255 * (p['lifespan'] / 20)))
            color = p['color'] + (alpha,)
            size = max(1, int(3 * (p['lifespan'] / 20)))
            pygame.draw.circle(self.screen, color, [int(c) for c in p['pos']], size)

        # Render UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.HEIGHT - 40))
        
        # Balls
        ball_text = self.font.render(f"BALLS: {max(0, self.balls_remaining - 1 if self.ball_attached else self.balls_remaining)}", True, self.COLOR_TEXT)
        self.screen.blit(ball_text, (self.WIDTH - ball_text.get_width() - self.WALL_THICKNESS - 10, self.HEIGHT - 40))

        # Time
        time_seconds = self.time_remaining // self.FPS
        time_text = self.font.render(f"TIME: {time_seconds}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH // 2 - time_text.get_width() // 2, self.WALL_THICKNESS + 5))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_remaining": self.balls_remaining,
            "time_remaining": self.time_remaining,
            "blocks_cleared": sum(1 for b in self.blocks if not b['active']),
            "total_blocks": len(self.blocks)
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

# Example usage to run and display the game
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker RL Environment")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        if done:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds before closing
    
    env.close()