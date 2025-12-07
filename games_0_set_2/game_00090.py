
# Generated: 2025-08-27T12:34:02.083750
# Source Brief: brief_00090.md
# Brief Index: 90

        
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
        "A fast-paced, top-down block breaker where risky plays are rewarded and cautious play is penalized."
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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Colors
        self.COLOR_BG = (16, 16, 48) # Dark Blue
        self.COLOR_GRID = (32, 32, 80)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0) # Bright Yellow
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = [
            (255, 80, 80),  # Red
            (80, 255, 80),  # Green
            (80, 80, 255),  # Blue
            (255, 165, 0), # Orange
            (128, 0, 128), # Purple
        ]

        # Game constants
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 12
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 7
        self.BASE_BALL_SPEED = 6.0
        self.MAX_STEPS = 5000
        self.INITIAL_BALLS = 3
        
        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.balls_left = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.balls_left = self.INITIAL_BALLS
        self.particles = []

        self._create_blocks()
        self._reset_paddle_and_ball()
        
        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        self.blocks = []
        block_width = 58
        block_height = 20
        gap = 6
        num_cols = 10
        num_rows = 5
        start_x = (self.WIDTH - (num_cols * (block_width + gap) - gap)) / 2
        start_y = 50
        
        for i in range(num_rows):
            for j in range(num_cols):
                x = start_x + j * (block_width + gap)
                y = start_y + i * (block_height + gap)
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                self.blocks.append({"rect": pygame.Rect(x, y, block_width, block_height), "color": color})

    def _reset_paddle_and_ball(self):
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2, 
            self.HEIGHT - self.PADDLE_HEIGHT - 10, 
            self.PADDLE_WIDTH, 
            self.PADDLE_HEIGHT
        )
        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS, 
            self.paddle.top - self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2, 
            self.BALL_RADIUS * 2
        )
        self.ball_vel = [0, 0]
        self.ball_launched = False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.02  # Small penalty per step to encourage speed
        
        movement = action[0]
        space_pressed = action[1] == 1
        
        # 1. Handle player input
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        if not self.ball_launched:
            self.ball.centerx = self.paddle.centerx
            if space_pressed:
                self.ball_launched = True
                # Sound: Ball Launch
                angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
                self.ball_vel = [self.BASE_BALL_SPEED * math.sin(angle), -self.BASE_BALL_SPEED * math.cos(angle)]

        # 2. Update game logic
        if self.ball_launched:
            reward += self._update_ball()
        
        self._update_particles()
        
        # 3. Check for termination
        self.steps += 1
        terminated = False
        if not self.blocks:
            self.game_won = True
            self.game_over = True
            terminated = True
            reward += 100
        elif self.balls_left <= 0:
            self.game_over = True
            terminated = True
            reward -= 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        reward = 0
        
        # Speed scaling
        blocks_destroyed = 50 - len(self.blocks)
        speed_multiplier = 1.0 + blocks_destroyed * 0.01 # 0.1% is too slow, let's do 1%
        speed_multiplier = min(speed_multiplier, 1.5) # Cap at 50% increase
        current_speed = self.BASE_BALL_SPEED * speed_multiplier
        
        # Normalize velocity and apply current speed
        vel_magnitude = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
        if vel_magnitude > 0:
            self.ball_vel[0] = (self.ball_vel[0] / vel_magnitude) * current_speed
            self.ball_vel[1] = (self.ball_vel[1] / vel_magnitude) * current_speed
        
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.left = np.clip(self.ball.left, 0, self.WIDTH - self.ball.width)
            # Sound: Wall Bounce
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.top = np.clip(self.ball.top, 0, self.HEIGHT - self.ball.height)
            # Sound: Wall Bounce

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # Sound: Paddle Hit
            reward += 0.1
            
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            
            # Risky play reward
            if abs(offset) > 0.9: # Outer 10%
                reward += 2.0
            # Safe play penalty
            elif abs(offset) < 0.25: # Inner 50% (half of half)
                reward -= 0.2
            
            bounce_angle = offset * (math.pi / 2.5) # Max 72 degrees
            self.ball_vel[0] = current_speed * math.sin(bounce_angle)
            self.ball_vel[1] = -current_speed * math.cos(bounce_angle)
            self.ball.bottom = self.paddle.top

        # Block collision
        hit_block_idx = self.ball.collidelistall([b['rect'] for b in self.blocks])
        if hit_block_idx:
            for idx in sorted(hit_block_idx, reverse=True):
                block_dict = self.blocks[idx]
                block_rect = block_dict['rect']
                
                # Sound: Block Break
                self._spawn_particles(block_rect.center, block_dict['color'])
                reward += 1.0
                self.score += 10
                
                # Determine bounce direction
                prev_ball_center = (self.ball.centerx - self.ball_vel[0], self.ball.centery - self.ball_vel[1])
                
                # Check if the ball was primarily moving horizontally or vertically into the block
                if prev_ball_center[1] < block_rect.top or prev_ball_center[1] > block_rect.bottom:
                    self.ball_vel[1] *= -1
                else:
                    self.ball_vel[0] *= -1
                    
                del self.blocks[idx]
            # Avoid multiple block breaks in one frame from causing issues
            if not self.blocks:
                return reward


        # Ball lost
        if self.ball.top >= self.HEIGHT:
            # Sound: Ball Lost
            self.balls_left -= 1
            if self.balls_left > 0:
                self._reset_paddle_and_ball()
            else:
                self.game_over = True
        
        return reward

    def _spawn_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "life": life, "color": color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
            pygame.draw.rect(self.screen, tuple(min(255, c + 50) for c in block['color']), block['rect'], 2, border_radius=3)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball
        if self.balls_left > 0 or self.ball_launched:
            center = (int(self.ball.centerx), int(self.ball.centery))
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.BALL_RADIUS, self.COLOR_BALL)
            # Add a "glow"
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.BALL_RADIUS + 2, (255, 255, 0, 50))

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * 15)))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 4, 4))
            self.screen.blit(temp_surf, (int(p['pos'][0]), int(p['pos'][1])))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Ball icons
        for i in range(self.balls_left - (1 if not self.ball_launched else 0)):
            pos = (self.WIDTH - 20 - i * 20, 20)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (0, 255, 0) if self.game_won else (255, 0, 0)
            end_text = self.font_game_over.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks)
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # Get user input
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Match the intended FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()