
# Generated: 2025-08-27T16:30:08.490637
# Source Brief: brief_01244.md
# Brief Index: 1244

        
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
        "Controls: Use ← and → to move the paddle. Clear all blocks to advance."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker. Clear all blocks on the screen by bouncing the ball with your paddle. Clear 3 stages to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
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
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_WALL = (200, 200, 220)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BLOCK_GREEN = (50, 255, 150)
        self.COLOR_BLOCK_BLUE = (50, 150, 255)
        self.COLOR_BLOCK_RED = (255, 50, 100)
        self.COLOR_TEXT = (240, 240, 240)

        # Fonts
        self.font_main = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # Game constants
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 10
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 6
        self.BALL_INITIAL_SPEED = 4.0
        self.BALL_MAX_SPEED = 8.0
        self.WALL_THICKNESS = 10
        self.MAX_STEPS = 3600 # 2 minutes at 30fps
        self.MAX_STAGES = 3
        self.STAGE_CLEAR_DELAY = 90 # 3 seconds at 30fps

        # Initialize state variables
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.current_ball_speed = None
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.balls_left = 0
        self.current_stage = 0
        self.stage_clear_timer = 0
        self.blocks_destroyed_this_stage = 0
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.balls_left = 3
        self.current_stage = 1
        self.stage_clear_timer = 0
        self.particles = []

        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        # Reset paddle
        self.paddle_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 20,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        # Reset ball
        self.ball_pos = [self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS]
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        self.current_ball_speed = self.BALL_INITIAL_SPEED
        self.ball_vel = [self.current_ball_speed * math.cos(angle), self.current_ball_speed * math.sin(angle)]
        
        # Reset stage-specific state
        self.blocks = []
        self.blocks_destroyed_this_stage = 0
        
        # Generate blocks based on stage
        block_width, block_height = 40, 15
        gap = 4
        
        if self.current_stage == 1: # Simple grid
            rows, cols = 5, 12
            for r in range(rows):
                for c in range(cols):
                    x = self.WALL_THICKNESS + 10 + c * (block_width + gap)
                    y = self.WALL_THICKNESS + 40 + r * (block_height + gap)
                    color, points = self._get_block_properties(r)
                    self.blocks.append({'rect': pygame.Rect(x, y, block_width, block_height), 'color': color, 'points': points})

        elif self.current_stage == 2: # Pyramid
            rows = 7
            for r in range(rows):
                cols = r + 1
                start_x = self.SCREEN_WIDTH/2 - (cols/2 * (block_width + gap))
                for c in range(cols):
                    x = start_x + c * (block_width + gap)
                    y = self.WALL_THICKNESS + 40 + r * (block_height + gap)
                    color, points = self._get_block_properties(rows - 1 - r)
                    self.blocks.append({'rect': pygame.Rect(x, y, block_width, block_height), 'color': color, 'points': points})

        elif self.current_stage == 3: # Alternating rows
            rows, cols = 8, 12
            for r in range(rows):
                current_cols = cols if r % 2 == 0 else cols - 1
                offset = 0 if r % 2 == 0 else (block_width + gap) / 2
                for c in range(current_cols):
                    x = self.WALL_THICKNESS + 10 + offset + c * (block_width + gap)
                    y = self.WALL_THICKNESS + 40 + r * (block_height + gap)
                    color, points = self._get_block_properties(r)
                    self.blocks.append({'rect': pygame.Rect(x, y, block_width, block_height), 'color': color, 'points': points})

    def _get_block_properties(self, row):
        if row < 2: return self.COLOR_BLOCK_RED, 5
        if row < 5: return self.COLOR_BLOCK_BLUE, 3
        return self.COLOR_BLOCK_GREEN, 1

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # Handle stage clear transition
        if self.stage_clear_timer > 0:
            self.stage_clear_timer -= 1
            if self.stage_clear_timer == 0:
                self._setup_stage()
            return self._get_observation(), 0, False, False, self._get_info()

        # Unpack action
        movement = action[0]
        
        # 1. Update paddle position
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
            reward -= 0.01 # Small penalty for movement
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
            reward -= 0.01 # Small penalty for movement
        
        self.paddle_rect.left = max(self.WALL_THICKNESS, self.paddle_rect.left)
        self.paddle_rect.right = min(self.SCREEN_WIDTH - self.WALL_THICKNESS, self.paddle_rect.right)

        # 2. Update ball position and check collisions
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= self.WALL_THICKNESS:
            self.ball_vel[0] *= -1
            ball_rect.left = self.WALL_THICKNESS
        if ball_rect.right >= self.SCREEN_WIDTH - self.WALL_THICKNESS:
            self.ball_vel[0] *= -1
            ball_rect.right = self.SCREEN_WIDTH - self.WALL_THICKNESS
        if ball_rect.top <= self.WALL_THICKNESS:
            self.ball_vel[1] *= -1
            ball_rect.top = self.WALL_THICKNESS
        
        # Paddle collision
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel[1] > 0:
            # Sound: paddle_hit.wav
            ball_rect.bottom = self.paddle_rect.top
            self.ball_vel[1] *= -1
            
            # Change horizontal velocity based on hit location
            offset = (ball_rect.centerx - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = self.current_ball_speed * offset * 1.2
            
            # Normalize to maintain speed
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            self.ball_vel[0] = (self.ball_vel[0] / speed) * self.current_ball_speed
            self.ball_vel[1] = (self.ball_vel[1] / speed) * self.current_ball_speed

        # Block collisions
        hit_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            # Sound: block_break.wav
            block_hit = self.blocks[hit_block_idx]
            
            # Determine bounce direction
            dx = ball_rect.centerx - block_hit['rect'].centerx
            dy = ball_rect.centery - block_hit['rect'].centery
            
            if abs(dx / block_hit['rect'].width) > abs(dy / block_hit['rect'].height):
                self.ball_vel[0] *= -1
            else:
                self.ball_vel[1] *= -1
                
            self.score += block_hit['points']
            reward += block_hit['points'] + 0.1 # Base reward + hit reward
            
            self._create_particles(block_hit['rect'].center, block_hit['color'])
            self.blocks.pop(hit_block_idx)
            self.blocks_destroyed_this_stage += 1
            
            # Increase ball speed every 25 blocks
            if self.blocks_destroyed_this_stage > 0 and self.blocks_destroyed_this_stage % 25 == 0:
                self.current_ball_speed = min(self.BALL_MAX_SPEED, self.current_ball_speed + 0.5)

        # Update ball position from rect
        self.ball_pos = [ball_rect.centerx, ball_rect.centery]

        # 3. Check for game state changes
        # Ball lost
        if ball_rect.top > self.SCREEN_HEIGHT:
            # Sound: lose_ball.wav
            self.balls_left -= 1
            reward -= 50
            if self.balls_left <= 0:
                self.game_over = True
            else:
                # Reset ball and paddle for the next turn
                self._setup_stage() # This also resets ball/paddle

        # Stage clear
        if not self.blocks:
            if self.current_stage < self.MAX_STAGES:
                # Sound: stage_clear.wav
                self.current_stage += 1
                self.stage_clear_timer = self.STAGE_CLEAR_DELAY
                reward += 50
            else:
                # Sound: game_win.wav
                self.game_won = True
                self.game_over = True
                reward += 100

        # 4. Update particles
        self._update_particles()
        
        # 5. Check termination
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [speed * math.cos(angle), speed * math.sin(angle)]
            lifespan = self.np_random.integers(10, 25)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, self.COLOR_BG, block['rect'], 1)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=3)
        
        # Ball
        if self.balls_left > 0:
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 25))))
            color = (*p['color'], alpha)
            s = pygame.Surface((2, 2), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])))
    
    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.SCREEN_HEIGHT - 28))
        
        # Balls
        balls_text = self.font_main.render(f"BALLS: {self.balls_left}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.SCREEN_WIDTH - balls_text.get_width() - self.WALL_THICKNESS - 10, self.SCREEN_HEIGHT - 28))

        # Stage
        stage_text = self.font_main.render(f"STAGE: {self.current_stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH/2 - stage_text.get_width()/2, self.SCREEN_HEIGHT - 28))
        
        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            msg_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

        # Stage Clear message
        elif self.stage_clear_timer > 0:
            msg = f"STAGE {self.current_stage - 1} CLEARED"
            msg_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "balls_left": self.balls_left,
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print(env.user_guide)
    
    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_SHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Pygame rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Match the intended FPS
        
    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()