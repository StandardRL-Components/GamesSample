
# Generated: 2025-08-27T21:12:32.161978
# Source Brief: brief_02710.md
# Brief Index: 2710

        
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
    A grid-based Breakout game where strategic paddle positioning and risk-taking are rewarded.
    The player controls a paddle to bounce a ball, breaking blocks to score points and clear stages.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ← and → to move the paddle. Your goal is to break all the blocks with the ball."
    )

    game_description = (
        "A retro arcade game. Bounce the ball to destroy all the blocks on the screen. "
        "Clear three stages to win. Lose all your balls and the game is over."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 80, 10
        self.PADDLE_SPEED = 8
        self.BALL_SIZE = 8
        self.GRID_COLS, self.GRID_ROWS = 20, 25
        self.BLOCK_WIDTH = self.WIDTH // self.GRID_COLS
        self.BLOCK_HEIGHT = 12
        self.INITIAL_BALLS = 3
        self.MAX_STAGES = 3
        self.MAX_STEPS = 5000

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.BLOCK_COLORS = {
            1: (0, 200, 100), # Green
            2: (100, 150, 255), # Blue
            3: (255, 100, 100)  # Red
        }

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        
        # Initialize state variables
        self.paddle_x = 0
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = 0
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.stage = 0
        self.balls_left = 0
        self.game_over = False
        self.win = False
        self.step_reward = 0.0
        self.blocks_hit_this_step = 0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.balls_left = self.INITIAL_BALLS
        self.game_over = False
        self.win = False
        self.particles.clear()
        
        self._setup_stage()
        self._launch_ball()
        
        return self._get_observation(), self._get_info()
    
    def _setup_stage(self):
        self.blocks = []
        # Procedural but deterministic block layout based on stage
        stage_seed = self.stage
        for r in range(5 + self.stage):  # More rows in later stages
            for c in range(self.GRID_COLS):
                # Use a simple pattern based on stage and position
                val = (r + c + stage_seed * 3) % 11
                if val < 4:
                    block_type = 1 # Green
                elif val < 7:
                    block_type = 2 # Blue
                elif val < 9:
                    block_type = 3 # Red
                else:
                    continue # Empty space
                
                x = c * self.BLOCK_WIDTH
                y = 50 + r * self.BLOCK_HEIGHT
                self.blocks.append({
                    "rect": pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
                    "type": block_type,
                    "color": self.BLOCK_COLORS[block_type]
                })

    def _launch_ball(self):
        self.paddle_x = self.WIDTH / 2 - self.PADDLE_WIDTH / 2
        self.ball_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        
        # Base speed increases with stage
        self.ball_speed = 6.0 + (self.stage - 1) * 0.5
        
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = [math.cos(angle) * self.ball_speed, math.sin(angle) * self.ball_speed]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.step_reward = 0.0
        self.blocks_hit_this_step = 0
        
        # --- Action Handling ---
        movement = action[0]
        prev_paddle_x = self.paddle_x
        
        if movement == 3:  # Left
            self.paddle_x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle_x += self.PADDLE_SPEED
            
        self.paddle_x = np.clip(self.paddle_x, 0, self.WIDTH - self.PADDLE_WIDTH)
        
        # --- Update Game Logic ---
        self._update_ball()
        self._update_particles()
        
        # --- Reward Calculation ---
        # +0.1 for surviving a frame
        self.step_reward += 0.01 
        
        # -0.2 for moving away from the ball
        paddle_moved_left = self.paddle_x < prev_paddle_x
        paddle_moved_right = self.paddle_x > prev_paddle_x
        if (self.ball_vel[0] > 0 and paddle_moved_left) or \
           (self.ball_vel[0] < 0 and paddle_moved_right):
            self.step_reward -= 0.02
        
        # +5 bonus for multi-block hits
        if self.blocks_hit_this_step > 1:
            self.step_reward += 5.0
            
        # --- Termination Check ---
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_SIZE/2, self.ball_pos[1] - self.BALL_SIZE/2, self.BALL_SIZE, self.BALL_SIZE)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = np.clip(ball_rect.left, 0, self.WIDTH - self.BALL_SIZE)
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = 0
            # sfx: wall_bounce

        # Paddle collision
        paddle_rect = pygame.Rect(self.paddle_x, self.HEIGHT - self.PADDLE_HEIGHT - 10, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        if ball_rect.colliderect(paddle_rect) and self.ball_vel[1] > 0:
            # sfx: paddle_bounce
            offset = (ball_rect.centerx - paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            bounce_angle = offset * (math.pi / 2.5) # Max 72 degrees
            
            self.ball_vel[0] = self.ball_speed * math.sin(bounce_angle)
            self.ball_vel[1] = -self.ball_speed * math.cos(bounce_angle)

            # Ensure ball is above paddle to prevent sticking
            ball_rect.bottom = paddle_rect.top
            
        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                # sfx: block_break
                self.blocks.remove(block)
                self.blocks_hit_this_step += 1
                
                # Reward based on block type
                reward_val = block["type"]
                self.step_reward += reward_val
                self.score += reward_val * 10
                
                # Create particles
                self._create_particles(block["rect"].center, block["color"])
                
                # Bounce logic
                prev_ball_rect = pygame.Rect(ball_rect.x - self.ball_vel[0], ball_rect.y - self.ball_vel[1], self.BALL_SIZE, self.BALL_SIZE)
                if prev_ball_rect.bottom <= block["rect"].top or prev_ball_rect.top >= block["rect"].bottom:
                    self.ball_vel[1] *= -1
                else:
                    self.ball_vel[0] *= -1
                break

        # Ball miss
        if ball_rect.top > self.HEIGHT:
            # sfx: lose_ball
            self.balls_left -= 1
            self.step_reward -= 1.0
            if self.balls_left <= 0:
                self.game_over = True
            else:
                self._launch_ball()

        # Stage clear
        if not self.blocks:
            # sfx: stage_clear
            self.stage += 1
            self.step_reward += 50.0
            self.score += 1000
            if self.stage > self.MAX_STAGES:
                self.game_over = True
                self.win = True
                self.step_reward += 100.0
            else:
                self._setup_stage()
                self._launch_ball()
        
        self.ball_pos = [ball_rect.centerx, ball_rect.centery]

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for c in range(1, self.GRID_COLS):
            x = c * self.BLOCK_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for r in range(1, self.GRID_ROWS):
            y = r * (self.BLOCK_HEIGHT * 1.5) # Visual spacing
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, self.COLOR_BG, block["rect"], 1)

        # Draw paddle
        paddle_rect = pygame.Rect(self.paddle_x, self.HEIGHT - self.PADDLE_HEIGHT - 10, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)
        
        # Draw ball with a glow
        ball_center = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        glow_color = (*self.COLOR_BALL, 100)
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_SIZE, (*self.COLOR_BALL, 50))
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], int(self.BALL_SIZE * 0.75), self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            size = max(1, int(3 * (p["lifespan"] / 30)))
            pygame.draw.rect(self.screen, color, (int(p["pos"][0]), int(p["pos"][1]), size, size))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH // 2 - stage_text.get_width() // 2, 10))
        
        # Balls
        ball_text = self.font_small.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(ball_text, (self.WIDTH - 120, 10))
        for i in range(self.balls_left):
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 60 + i * 15, 16, 5, self.COLOR_BALL)
            
        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "balls_left": self.balls_left,
            "game_over": self.game_over,
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a window to display the game
    pygame.display.set_caption("Breakout Arcade")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    clock = pygame.time.Clock()
    
    # Action state
    movement = 0 # 0=none, 3=left, 4=right
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT and movement == 3:
                    movement = 0
                elif event.key == pygame.K_RIGHT and movement == 4:
                    movement = 0
    
        action = [movement, 0, 0] # space and shift are unused
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if info['game_over']:
            # Pause for a moment on game over before closing
            pygame.time.wait(2000)
            terminated = True
            
        clock.tick(60) # Run at 60 FPS for smooth human gameplay
        
    env.close()