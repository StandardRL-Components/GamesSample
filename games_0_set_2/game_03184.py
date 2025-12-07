
# Generated: 2025-08-27T22:36:46.572723
# Source Brief: brief_03184.md
# Brief Index: 3184

        
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
        "Controls: ←→ to move the paddle. Try to destroy all the blocks with the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block-breaker. Destroy all blocks to win, but don't let the ball fall past your paddle or you'll lose a life!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WALL_THICKNESS = 10

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
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_WALL = (100, 110, 130)
        self.COLOR_PADDLE = (240, 240, 240)
        self.COLOR_BALL = (255, 220, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = {
            1: (0, 200, 100),  # Green
            3: (0, 150, 255),  # Blue
            5: (255, 80, 80),   # Red
        }

        # Game constants
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 7
        self.INITIAL_BALL_SPEED = 5
        self.MAX_STEPS = 5000
        self.INITIAL_LIVES = 3

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.bounce_counter = 0

        self.np_random = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.particles = []
        self.bounce_counter = 0

        # Setup paddle
        paddle_y = self.SCREEN_HEIGHT - 40
        paddle_x = self.SCREEN_WIDTH / 2 - self.PADDLE_WIDTH / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Setup ball
        self._reset_ball()

        # Setup blocks
        self.blocks = []
        block_width = 38
        block_height = 18
        gap = 4
        num_cols = 15
        num_rows = 5
        start_x = (self.SCREEN_WIDTH - (num_cols * (block_width + gap))) / 2
        start_y = 50
        
        for i in range(num_rows):
            for j in range(num_cols):
                points = 1 if i >= 3 else (3 if i >= 1 else 5)
                color = self.BLOCK_COLORS[points]
                block_rect = pygame.Rect(
                    start_x + j * (block_width + gap),
                    start_y + i * (block_height + gap),
                    block_width,
                    block_height
                )
                self.blocks.append({"rect": block_rect, "points": points, "color": color})
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upwards
        self.ball_vel = [
            self.INITIAL_BALL_SPEED * math.cos(angle),
            self.INITIAL_BALL_SPEED * math.sin(angle)
        ]
    
    def step(self, action):
        reward = -0.02  # Small penalty for each step to encourage efficiency
        
        # Unpack factorized action
        movement = action[0]
        
        # --- Update Game Logic ---
        
        # 1. Move Paddle
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen bounds
        self.paddle.x = max(self.WALL_THICKNESS, min(self.paddle.x, self.SCREEN_WIDTH - self.WALL_THICKNESS - self.PADDLE_WIDTH))

        # 2. Update Ball
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Ball collisions
        # Wall collisions
        if ball_rect.left <= self.WALL_THICKNESS:
            ball_rect.left = self.WALL_THICKNESS
            self.ball_vel[0] *= -1
            self.bounce_counter += 1
            # sfx: wall_bounce
        if ball_rect.right >= self.SCREEN_WIDTH - self.WALL_THICKNESS:
            ball_rect.right = self.SCREEN_WIDTH - self.WALL_THICKNESS
            self.ball_vel[0] *= -1
            self.bounce_counter += 1
            # sfx: wall_bounce
        if ball_rect.top <= self.WALL_THICKNESS:
            ball_rect.top = self.WALL_THICKNESS
            self.ball_vel[1] *= -1
            self.bounce_counter += 1
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            reward += 0.1
            self.bounce_counter = 0
            # sfx: paddle_hit
            
            # Change angle based on hit position
            hit_offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            new_angle = math.pi * 1.5 - hit_offset * (math.pi / 3) # +/- 60 degrees from vertical
            speed = math.hypot(*self.ball_vel)
            self.ball_vel[0] = speed * math.cos(new_angle)
            self.ball_vel[1] = speed * math.sin(new_angle)
            
            # Ensure ball is above paddle to prevent getting stuck
            ball_rect.bottom = self.paddle.top
        
        self.ball_pos = [ball_rect.centerx, ball_rect.centery]

        # Block collisions
        hit_block_idx = ball_rect.collidelist([b["rect"] for b in self.blocks])
        if hit_block_idx != -1:
            # sfx: block_hit
            self.bounce_counter = 0
            block = self.blocks.pop(hit_block_idx)
            reward += block["points"]
            self.score += block["points"]

            # Create particles
            for _ in range(15):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                p_vel = [speed * math.cos(angle), speed * math.sin(angle)]
                p_life = self.np_random.integers(10, 20)
                self.particles.append({"pos": list(block["rect"].center), "vel": p_vel, "life": p_life, "color": self.COLOR_BALL})
            
            # Simple bounce logic
            # Determine if collision was more horizontal or vertical
            # This is a simplification; more complex logic could be used
            dbx = self.ball_pos[0] - block["rect"].centerx
            dby = self.ball_pos[1] - block["rect"].centery
            if abs(dbx) > abs(dby):
                self.ball_vel[0] *= -1
            else:
                self.ball_vel[1] *= -1

        # Anti-softlock
        if self.bounce_counter > 20:
            self.ball_vel[1] += self.np_random.uniform(-0.5, 0.5)
            self.bounce_counter = 0

        # 3. Update Particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95 # Drag
            p["vel"][1] *= 0.95
            p["life"] -= 1

        # 4. Check for Termination
        self.steps += 1
        terminated = False
        
        # Lost life
        if ball_rect.top > self.SCREEN_HEIGHT:
            # sfx: lose_life
            self.lives -= 1
            self.bounce_counter = 0
            reward -= 10
            if self.lives > 0:
                self._reset_ball()
            else:
                self.game_over = True

        if len(self.blocks) == 0: # Win condition
            reward += 100
            self.game_over = True
            # sfx: win_game
        elif self.game_over: # Loss condition
            reward -= 100
            # sfx: lose_game

        if self.game_over or self.steps >= self.MAX_STEPS:
            terminated = True
        
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
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, self.COLOR_BG, block["rect"], 2) # Outline

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Draw ball with antialiasing
        x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 20))))
            color_with_alpha = p["color"] + (alpha,)
            temp_surf = pygame.Surface((2, 2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color_with_alpha, (0, 0, 2, 2))
            self.screen.blit(temp_surf, (int(p["pos"][0]), int(p["pos"][1])))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, self.SCREEN_HEIGHT - 35))

        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 20, self.SCREEN_HEIGHT - 35))

        if self.game_over:
            end_font = pygame.font.SysFont("monospace", 48, bold=True)
            msg = "YOU WIN!" if len(self.blocks) == 0 else "GAME OVER"
            end_text = end_font.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
        }

    def close(self):
        pygame.font.quit()
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