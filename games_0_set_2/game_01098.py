
# Generated: 2025-08-27T16:02:27.749624
# Source Brief: brief_01098.md
# Brief Index: 1098

        
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
        "Controls: ←→ to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A visually stunning, procedurally generated Block Breaker game. "
        "Destroy all blocks to win."
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
        
        # Colors (Neon on dark theme)
        self.COLOR_BG_TOP = (10, 0, 20)
        self.COLOR_BG_BOTTOM = (30, 0, 40)
        self.COLOR_PADDLE = (0, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 255)
        self.BLOCK_COLORS = [
            (255, 0, 128), (128, 0, 255), (0, 128, 255), 
            (0, 255, 128), (128, 255, 0)
        ]

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Game constants
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 4.0
        self.MAX_BALL_SPEED = 10.0
        self.MAX_BALLS = 3
        self.MAX_STEPS = 3000 # Increased for longer episodes
        self.BLOCK_ROWS = 5
        self.BLOCK_COLS = 10
        self.BLOCK_WIDTH = self.WIDTH // self.BLOCK_COLS
        self.BLOCK_HEIGHT = 20
        self.NUM_BLOCKS = self.BLOCK_ROWS * self.BLOCK_COLS

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.balls_left = 0
        self.ball_speed_multiplier = 1.0
        self.steps_since_last_hit = 0
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.balls_left = self.MAX_BALLS
        self.ball_speed_multiplier = 1.0
        self.steps_since_last_hit = 0
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self._generate_blocks()
        self._reset_ball()
        self.particles.clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = -0.01  # Small penalty per step to encourage efficiency
        
        self._handle_input(movement)
        
        # Update ball and check for loss
        ball_lost = self._update_ball()
        if ball_lost:
            reward -= 5
            self.balls_left -= 1
            if self.balls_left > 0:
                self._reset_ball()
                # sfx: lose_life
            else:
                reward -= 100 # Large penalty for game over
        else:
            # Handle collisions only if ball is in play
            collision_reward = self._handle_collisions()
            reward += collision_reward

        # Update particles
        self._update_particles()
        
        # Update difficulty
        if self.steps > 0 and self.steps % 50 == 0:
            current_speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            if current_speed < self.MAX_BALL_SPEED:
                self.ball_speed_multiplier += 0.05
                self._set_ball_speed(self.INITIAL_BALL_SPEED * self.ball_speed_multiplier)

        self.steps += 1
        terminated = self._check_termination()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        self._render_background()
        self._render_blocks()
        self._render_paddle()
        self._render_ball()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks)
        }
        
    # --- Helper Methods ---

    def _generate_blocks(self):
        self.blocks = []
        y_offset = 40
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                block_rect = pygame.Rect(
                    j * self.BLOCK_WIDTH,
                    y_offset + i * self.BLOCK_HEIGHT,
                    self.BLOCK_WIDTH,
                    self.BLOCK_HEIGHT
                )
                color_index = (i + j) % len(self.BLOCK_COLORS)
                self.blocks.append((block_rect, self.BLOCK_COLORS[color_index]))

    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4) # Upward angle
        speed = self.INITIAL_BALL_SPEED * self.ball_speed_multiplier
        self.ball_vel = [speed * math.cos(angle), speed * math.sin(angle)]
        self.steps_since_last_hit = 0

    def _set_ball_speed(self, speed):
        current_speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
        if current_speed > 0:
            factor = speed / current_speed
            self.ball_vel[0] *= factor
            self.ball_vel[1] *= factor

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

    def _update_ball(self):
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos[0]))
            # sfx: wall_bounce
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS
            # sfx: wall_bounce

        # Ball lost
        if self.ball_pos[1] >= self.HEIGHT:
            return True
        return False

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS # Prevent sticking
            
            # Change horizontal velocity based on where it hits the paddle
            offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = self.INITIAL_BALL_SPEED * self.ball_speed_multiplier * offset * 1.5
            self._set_ball_speed(self.INITIAL_BALL_SPEED * self.ball_speed_multiplier)
            # sfx: paddle_bounce
            self._spawn_particles(self.paddle.clipline(ball_rect.centerx, ball_rect.centery, self.ball_pos[0] - self.ball_vel[0], self.ball_pos[1] - self.ball_vel[1])[0], self.COLOR_PADDLE, 5)

        # Block collisions
        hit_block = False
        for i, (block, color) in enumerate(self.blocks):
            if ball_rect.colliderect(block):
                # Determine collision side to correctly reflect the ball
                # A simple approximation: check overlap amounts
                overlap_x = (ball_rect.width / 2 + block.width / 2) - abs(ball_rect.centerx - block.centerx)
                overlap_y = (ball_rect.height / 2 + block.height / 2) - abs(ball_rect.centery - block.centery)
                
                if overlap_x < overlap_y:
                    self.ball_vel[0] *= -1
                else:
                    self.ball_vel[1] *= -1

                self._spawn_particles(block.center, color, 20)
                self.blocks.pop(i)
                reward += 1
                self.score += 1
                hit_block = True
                # sfx: block_break
                break
        
        if hit_block:
            self.steps_since_last_hit = 0
        else:
            self.steps_since_last_hit += 1
            
        # Anti-softlock mechanism
        if self.steps_since_last_hit > 100:
             self._reset_ball()

        return reward

    def _check_termination(self):
        if self.balls_left <= 0:
            return True
        if not self.blocks: # All blocks destroyed
            self.score += 50 # Victory bonus
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _spawn_particles(self, position, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [speed * math.cos(angle), speed * math.sin(angle)]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([list(position), vel, lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]
        
    # --- Rendering Methods ---

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_blocks(self):
        for block, color in self.blocks:
            inner_rect = block.inflate(-4, -4)
            pygame.draw.rect(self.screen, color, block, border_radius=3)
            pygame.draw.rect(self.screen, (0,0,0), inner_rect, border_radius=3)
            
            # Glow effect
            glow_color = tuple(min(255, c + 50) for c in color)
            pygame.draw.rect(self.screen, glow_color, block.inflate(2,2), width=1, border_radius=4)


    def _render_paddle(self):
        # Glow effect
        for i in range(4, 0, -1):
            alpha = 80 - i * 20
            color = (*self.COLOR_PADDLE, alpha)
            s = pygame.Surface((self.PADDLE_WIDTH + i*2, self.PADDLE_HEIGHT + i*2), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect(), border_radius=5)
            self.screen.blit(s, (self.paddle.x - i, self.paddle.y - i))
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

    def _render_ball(self):
        x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
        # Glow effect
        for i in range(8, 0, -1):
            alpha = 100 - i * 12
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS + i, (*self.COLOR_BALL, alpha))
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_particles(self):
        for pos, vel, lifetime, color in self.particles:
            radius = int(max(0, (lifetime / 30) * 4))
            alpha = int(max(0, (lifetime / 30) * 150))
            p_color = (*color, alpha)
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, p_color, (radius, radius), radius)
            self.screen.blit(s, (int(pos[0] - radius), int(pos[1] - radius)))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Balls left
        ball_text = self.font_small.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(ball_text, (self.WIDTH - 150, 10))
        for i in range(self.balls_left - 1):
             pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 60 + i * 20, 25, 6, self.COLOR_BALL)
             pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 60 + i * 20, 25, 6, self.COLOR_BALL)

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
    import time
    
    # Set this to True to see the game window
    human_mode = True 
    
    if human_mode:
        env = GameEnv(render_mode="human")
        pygame.display.set_caption("Block Breaker")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    else:
        env = GameEnv()

    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # --- Manual Control Mapping ---
    # 0=none, 1=up, 2=down, 3=left, 4=right
    # [movement, space, shift]
    action = [0, 0, 0] 

    while not done:
        if human_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            action[0] = 0 # Default to no-op
            if keys[pygame.K_LEFT]:
                action[0] = 3
            if keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Render to the display window
            frame = env._get_observation()
            frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()
            
            # The environment clock is managed internally, but we add a small delay for human playability
            time.sleep(1/30) 

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if done:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            # A small pause before restarting
            if human_mode:
                time.sleep(2)

    env.close()