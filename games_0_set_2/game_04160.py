
# Generated: 2025-08-28T01:34:44.287399
# Source Brief: brief_04160.md
# Brief Index: 4160

        
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
        "A fast-paced arcade game. Bounce the ball with your paddle to break all the blocks before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS_PER_STAGE = 60 * self.FPS # 60 seconds per stage

        # Colors
        self.COLOR_BG = (15, 15, 35)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 0, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 165, 0), # Orange
            (0, 255, 0),   # Green
        ]

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
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.balls_left = 0
        self.time_left = 0
        self.paddle = None
        self.ball = None
        self.ball_velocity = [0, 0]
        self.ball_launched = False
        self.blocks = []
        self.particles = []
        
        self.reset()
        
        # Run self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.balls_left = 3
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Sets up the game state for the current stage."""
        # Reset paddle
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        # Reset ball
        self.BALL_RADIUS = 8
        self.ball_launched = False
        base_speed = 4 + (self.stage - 1) * 0.5 # Ball speed increases with stage
        self.ball_speed = base_speed
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top
        self.ball_velocity = [0, 0]

        # Reset timer
        self.time_left = self.MAX_STEPS_PER_STAGE

        # Generate blocks
        self.blocks = []
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 58, 20
        padding = 6
        
        if self.stage == 1: # Simple grid
            for row in range(4):
                for col in range(10):
                    x = col * (self.BLOCK_WIDTH + padding) + padding
                    y = row * (self.BLOCK_HEIGHT + padding) + 50
                    block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                    color = self.BLOCK_COLORS[row % len(self.BLOCK_COLORS)]
                    self.blocks.append((block_rect, color))
        elif self.stage == 2: # Pyramid
            for row in range(6):
                num_cols = 10 - row * 2
                start_col = row
                for col in range(num_cols):
                    x = (start_col + col) * (self.BLOCK_WIDTH + padding) + padding
                    y = row * (self.BLOCK_HEIGHT + padding) + 50
                    block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                    color = self.BLOCK_COLORS[row % len(self.BLOCK_COLORS)]
                    self.blocks.append((block_rect, color))
        elif self.stage == 3: # Checkerboard
             for row in range(5):
                for col in range(10):
                    if (row + col) % 2 == 0:
                        x = col * (self.BLOCK_WIDTH + padding) + padding
                        y = row * (self.BLOCK_HEIGHT + padding) + 50
                        block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                        color = self.BLOCK_COLORS[row % len(self.BLOCK_COLORS)]
                        self.blocks.append((block_rect, color))

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_left -= 1
        
        reward = -0.01  # Small penalty for time passing
        
        # --- Handle Input ---
        movement = action[0]
        space_pressed = action[1] == 1
        
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        if space_pressed and not self.ball_launched:
            self.ball_launched = True
            angle = (self.np_random.uniform(-math.pi/4, math.pi/4))
            self.ball_velocity = [self.ball_speed * math.sin(angle), -self.ball_speed * math.cos(angle)]
            # sfx: ball_launch

        # --- Update Game Logic ---
        if self.ball_launched:
            self.ball.x += self.ball_velocity[0]
            self.ball.y += self.ball_velocity[1]
        else: # Ball is attached to paddle
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top

        # Ball collisions
        if self.ball_launched:
            # Wall collision
            if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
                self.ball_velocity[0] *= -1
                self.ball.left = max(0, self.ball.left)
                self.ball.right = min(self.WIDTH, self.ball.right)
                # sfx: wall_bounce
            if self.ball.top <= 0:
                self.ball_velocity[1] *= -1
                self.ball.top = max(0, self.ball.top)
                # sfx: wall_bounce

            # Paddle collision
            if self.ball.colliderect(self.paddle) and self.ball_velocity[1] > 0:
                self.ball.bottom = self.paddle.top
                self.ball_velocity[1] *= -1
                
                # Influence horizontal velocity based on hit location
                offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_velocity[0] = self.ball_speed * offset
                
                # Normalize velocity to maintain constant speed
                speed = math.sqrt(self.ball_velocity[0]**2 + self.ball_velocity[1]**2)
                if speed > 0:
                    self.ball_velocity[0] = (self.ball_velocity[0] / speed) * self.ball_speed
                    self.ball_velocity[1] = (self.ball_velocity[1] / speed) * self.ball_speed
                # sfx: paddle_bounce

            # Block collision
            hit_block = None
            for i in range(len(self.blocks) - 1, -1, -1):
                block_rect, color = self.blocks[i]
                if self.ball.colliderect(block_rect):
                    hit_block = self.blocks.pop(i)
                    reward += 1.0 # Reward for breaking a block
                    self.score += 10

                    # Create particles
                    for _ in range(10):
                        particle_vel = [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)]
                        particle_pos = list(hit_block[0].center)
                        particle_life = self.np_random.integers(10, 20)
                        self.particles.append([particle_pos, particle_vel, particle_life, hit_block[1]])
                    
                    # Naive bounce: just reverse vertical velocity
                    self.ball_velocity[1] *= -1
                    # sfx: block_break
                    break # Only break one block per frame
            
            # Floor collision (lose a ball)
            if self.ball.top >= self.HEIGHT:
                self.balls_left -= 1
                self.ball_launched = False
                # sfx: lose_ball
                if self.balls_left < 0:
                    self.game_over = True
                    reward -= 100 # Penalty for losing
                else:
                    reward -= 10 # Penalty for losing a ball

        # --- Update Particles ---
        for p in self.particles:
            p[0][0] += p[1][0] # pos x
            p[0][1] += p[1][1] # pos y
            p[2] -= 1 # lifetime
        self.particles = [p for p in self.particles if p[2] > 0]

        # --- Check Termination Conditions ---
        terminated = self.game_over
        
        # Time runs out
        if self.time_left <= 0 and not terminated:
            terminated = True
            reward -= 100 # Penalty for running out of time

        # Stage clear
        if not self.blocks and not terminated:
            reward += 5 # Reward for clearing a stage
            self.score += 50
            if self.stage < 3:
                self.stage += 1
                self._setup_stage()
                # sfx: stage_clear
            else: # Game won
                terminated = True
                reward += 100 # Big reward for winning
                self.score += 1000
                # sfx: win_game
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw blocks
        for block_rect, color in self.blocks:
            pygame.draw.rect(self.screen, color, block_rect, border_radius=3)
            
        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Draw particles
        for pos, vel, life, color in self.particles:
            size = max(0, life / 4)
            pygame.draw.rect(self.screen, color, (int(pos[0]), int(pos[1]), size, size))

        # Draw ball with glow
        # The glow is a larger, semi-transparent circle
        pygame.gfxdraw.filled_circle(
            self.screen,
            int(self.ball.centerx),
            int(self.ball.centery),
            int(self.BALL_RADIUS * 1.5),
            self.COLOR_BALL_GLOW
        )
        pygame.gfxdraw.aacircle(
            self.screen,
            int(self.ball.centerx),
            int(self.ball.centery),
            int(self.BALL_RADIUS * 1.5),
            self.COLOR_BALL_GLOW
        )
        # The main ball is a solid circle on top
        pygame.gfxdraw.filled_circle(
            self.screen,
            int(self.ball.centerx),
            int(self.ball.centery),
            self.BALL_RADIUS,
            self.COLOR_BALL
        )
        pygame.gfxdraw.aacircle(
            self.screen,
            int(self.ball.centerx),
            int(self.ball.centery),
            self.BALL_RADIUS,
            self.COLOR_BALL
        )

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls
        balls_text = self.font_ui.render(f"BALLS: {max(0, self.balls_left)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(balls_text, (self.WIDTH - balls_text.get_width() - 10, 10))
        
        # Timer
        time_str = f"{self.time_left // self.FPS:02d}"
        time_text = self.font_ui.render(f"TIME: {time_str}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH/2 - time_text.get_width()/2, 10))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if not self.blocks and self.stage == 3:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_game_over.render(msg, True, self.COLOR_PADDLE)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "balls_left": self.balls_left,
            "time_left": self.time_left,
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Breakout Arcade")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(env.FPS)
        
    env.close()