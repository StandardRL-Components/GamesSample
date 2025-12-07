
# Generated: 2025-08-28T06:22:39.665732
# Source Brief: brief_02915.md
# Brief Index: 2915

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-arcade block breaker. Clear all blocks to win, but watch the timer and your lives!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    
    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (20, 40, 80)
    COLOR_PADDLE = (220, 220, 220)
    COLOR_BALL = (255, 255, 0)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_SPECIAL_BLOCK = (255, 215, 0)
    BLOCK_COLORS = [
        (255, 87, 34), (255, 193, 7), (139, 195, 74),
        (0, 188, 212), (33, 150, 243), (103, 58, 183)
    ]

    # Game Parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 7
    INITIAL_BALL_SPEED = 6
    MAX_BALL_SPEED_X = 8
    
    INITIAL_LIVES = 3
    MAX_STAGES = 3
    STAGE_TIME_SECONDS = 60
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.stage = 0
        self.time_left = 0
        self.ball_speed_multiplier = 1.0
        
        self.reset()
        
        # This is a dummy check to make sure the implementation is valid.
        # It's not part of the standard Gym API but is required by the prompt.
        if hasattr(self, 'validate_implementation'):
            self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.stage = 1
        self.game_over = False
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Initializes the game state for the current stage."""
        self.paddle = pygame.Rect(
            self.SCREEN_WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.SCREEN_HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self._reset_ball()
        
        self.time_left = self.STAGE_TIME_SECONDS * self.FPS
        self.ball_speed_multiplier = 1.0 + 0.1 * (self.stage - 1)
        
        self.blocks = []
        base_block_count = 40
        num_blocks = int(base_block_count * (1 + 0.2 * (self.stage - 1)))
        
        block_width = 50
        block_height = 20
        cols = self.SCREEN_WIDTH // (block_width + 5)
        
        for i in range(num_blocks):
            row = i // cols
            col = i % cols
            
            x = col * (block_width + 5) + 30
            y = row * (block_height + 5) + 50
            
            is_special = self.np_random.random() < 0.1
            color = self.COLOR_SPECIAL_BLOCK if is_special else self.np_random.choice(self.BLOCK_COLORS)
            
            block_rect = pygame.Rect(x, y, block_width, block_height)
            self.blocks.append({'rect': block_rect, 'color': color, 'special': is_special})
            
    def _reset_ball(self):
        """Resets the ball to the starting position on the paddle."""
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4) # Upwards angle
        speed = self.INITIAL_BALL_SPEED * self.ball_speed_multiplier
        self.ball_vel = [speed * math.sin(angle), -speed * math.cos(angle)]

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0.01  # Small reward for surviving
        terminated = self.game_over

        if not terminated:
            # 1. Handle Action
            movement = action[0]
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED
            
            self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

            # 2. Update Game Logic
            self._move_ball()
            
            # 3. Handle Collisions and Rewards
            collision_reward = self._handle_collisions()
            reward += collision_reward

            # 4. Update Timers and Particles
            self.time_left -= 1
            self._update_particles()
            
            # 5. Check Termination Conditions
            if not self.blocks: # Stage clear
                self.score += 50
                reward += 50
                if self.stage < self.MAX_STAGES:
                    self.stage += 1
                    self._setup_stage()
                    # Sound: Stage Clear
                else: # Game won
                    self.score += 100
                    reward += 100
                    terminated = True
                    self.game_over = True
                    # Sound: Game Win
            
            if self.time_left <= 0:
                reward -= 100
                terminated = True
                self.game_over = True
                # Sound: Game Over
                
            if self.lives <= 0:
                reward -= 100
                terminated = True
                self.game_over = True
                # Sound: Game Over
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_ball(self):
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.SCREEN_WIDTH, ball_rect.right)
            # Sound: Wall Bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(0, ball_rect.top)
            # Sound: Wall Bounce

        # Bottom wall (lose life)
        if ball_rect.top >= self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 5
            if self.lives > 0:
                self._reset_ball()
                # Sound: Lose Life
            return reward

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            
            # Influence horizontal velocity based on hit location
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = self.MAX_BALL_SPEED_X * offset * self.ball_speed_multiplier
            
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS # Prevent sticking
            # Sound: Paddle Hit
            
        # Block collisions
        hit_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            block_data = self.blocks[hit_block_idx]
            block_rect = block_data['rect']

            # Determine collision side to correctly reflect the ball
            prev_ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_vel[0] - self.BALL_RADIUS, self.ball_pos[1] - self.ball_vel[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
            
            if prev_ball_rect.bottom <= block_rect.top or prev_ball_rect.top >= block_rect.bottom:
                 self.ball_vel[1] *= -1
            else:
                 self.ball_vel[0] *= -1
            
            # Reward and effects
            if block_data['special']:
                reward += 2.0
                self.score += 20
            else:
                reward += 1.0
                self.score += 10
            
            self._create_particles(block_rect.center, block_data['color'])
            self.blocks.pop(hit_block_idx)
            # Sound: Block Break
            
        # Anti-softlock: if ball is in a horizontal loop
        if abs(self.ball_vel[1]) < 0.5:
            self.ball_vel[1] += self.np_random.choice([-0.5, 0.5])

        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, 20)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self._render_background()
        self._render_game_objects()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
        }

    def _render_background(self):
        """Draws a vertical gradient for the background."""
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game_objects(self):
        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block['color']), block['rect'], 2)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with a glow effect
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 3, (100, 100, 0, 100))
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_particles(self):
        for p in self.particles:
            alpha = p['life'] / 20
            color = (p['color'][0], p['color'][1], p['color'][2], int(alpha * 255))
            size = int(p['life'] / 5)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            if size > 0:
                pygame.draw.rect(self.screen, p['color'], (pos[0], pos[1], size, size))

    def _render_ui(self):
        # Score
        score_surf = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Lives
        lives_text = "LIVES: " + "♥ " * self.lives
        lives_surf = self.font_large.render(lives_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_surf, (self.SCREEN_WIDTH - lives_surf.get_width() - 10, 10))

        # Timer
        time_seconds = max(0, self.time_left // self.FPS)
        timer_surf = self.font_large.render(f"TIME: {time_seconds}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH // 2 - timer_surf.get_width() // 2, 10))
        
        # Stage
        stage_surf = self.font_small.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_surf, (self.SCREEN_WIDTH // 2 - stage_surf.get_width() // 2, 35))

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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    
    # Test reset
    obs, info = env.reset()
    print("Reset successful. Initial info:", info)
    
    # Test a few steps with random actions
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if (i + 1) % 20 == 0:
            print(f"Step {i+1}: Reward={reward:.2f}, Info={info}, Terminated={terminated}")
        if terminated:
            print("Episode terminated.")
            break
            
    env.close()
    print("Environment closed.")