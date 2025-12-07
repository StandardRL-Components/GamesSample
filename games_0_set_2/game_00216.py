
# Generated: 2025-08-27T12:57:34.352988
# Source Brief: brief_00216.md
# Brief Index: 216

        
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
        "A fast-paced, top-down block breaker where risk-taking is rewarded. Clear all the blocks to win, but lose all your lives and it's game over."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # Colors
        self.COLOR_BG = (13, 13, 38) # Dark Blue
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (0, 255, 255) # Bright Cyan
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = [
            (255, 0, 128),   # Hot Pink
            (0, 255, 0),     # Lime Green
            (255, 128, 0),   # Bright Orange
            (255, 255, 0),   # Yellow
            (0, 128, 255),   # Bright Blue
        ]
        
        # Game constants
        self.MAX_STEPS = 1000
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 7
        self.MAX_BALL_REFLECT_X = 5
        self.INITIAL_LIVES = 3

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.lives = None
        self.combo = None
        
        # Initialize state
        self.reset()

        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_launched = False
        self._reset_ball()

        # Create block grid
        self.blocks = []
        block_w, block_h = 58, 15
        rows, cols = 10, 10
        start_x = (self.SCREEN_WIDTH - (cols * (block_w + 2))) / 2
        start_y = 50
        for i in range(rows):
            for j in range(cols):
                color_index = (i // 2) % len(self.BLOCK_COLORS)
                block_rect = pygame.Rect(
                    start_x + j * (block_w + 2),
                    start_y + i * (block_h + 2),
                    block_w, block_h
                )
                self.blocks.append({"rect": block_rect, "color": self.BLOCK_COLORS[color_index]})
        
        self.total_blocks = len(self.blocks)
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES
        self.combo = 1
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        """Resets the ball to the paddle's center."""
        self.ball_launched = False
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS], dtype=float)
        self.ball_vel = np.array([0.0, 0.0], dtype=float)

    def step(self, action):
        reward = -0.2 # Small penalty per step to encourage action
        self.steps += 1
        
        if self.game_over:
            return (
                self._get_observation(),
                0, # No reward if game is over
                True,
                False,
                self._get_info()
            )

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # 1. Handle player input
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        # Launch ball
        if space_held and not self.ball_launched:
            # sfx: launch_ball.wav
            self.ball_launched = True
            initial_vx = (self.np_random.random() - 0.5) * 2
            self.ball_vel = np.array([initial_vx, -self.BALL_SPEED])

        # 2. Update game state
        if not self.ball_launched:
            self.ball_pos[0] = self.paddle.centerx
        else:
            self.ball_pos += self.ball_vel

        # Ball collision physics
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        
        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.SCREEN_WIDTH, ball_rect.right)
            self.ball_pos[0] = ball_rect.centerx
            # sfx: bounce_wall.wav

        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = 0
            self.ball_pos[1] = ball_rect.centery
            # sfx: bounce_wall.wav

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: bounce_paddle.wav
            self.ball_vel[1] *= -1
            
            # Change horizontal velocity based on where it hit the paddle
            dist_from_center = ball_rect.centerx - self.paddle.centerx
            normalized_dist = dist_from_center / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = normalized_dist * self.MAX_BALL_REFLECT_X
            
            # Normalize to constant speed
            speed = np.linalg.norm(self.ball_vel)
            if speed > 0:
                self.ball_vel = (self.ball_vel / speed) * self.BALL_SPEED
            
            ball_rect.bottom = self.paddle.top
            self.ball_pos[1] = ball_rect.centery

        # Block collisions
        hit_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            # sfx: break_block.wav
            block_hit = self.blocks.pop(hit_block_idx)
            
            # Create particles
            for _ in range(15):
                self.particles.append(Particle(block_hit['rect'].center, block_hit['color'], self.np_random))

            reward += self.combo
            self.score += self.combo
            self.combo += 1
            
            # Simple bounce logic
            self.ball_vel[1] *= -1

        # Ball out of bounds
        if ball_rect.top > self.SCREEN_HEIGHT:
            # sfx: lose_life.wav
            self.lives -= 1
            self.combo = 1
            self._reset_ball()
            if self.lives <= 0:
                self.game_over = True
                reward -= 100 # Loss penalty

        # Update particles
        self.particles = [p for p in self.particles if p.update()]
        
        # 3. Check for termination
        terminated = False
        if self.lives <= 0:
            self.game_over = True
            terminated = True
        
        if not self.blocks:
            self.game_over = True
            terminated = True
            reward += 100 # Win bonus
            self.score += 100
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
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
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, self.COLOR_BG, block['rect'], 1) # Border

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with glow
        ball_center = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        for i in range(self.BALL_RADIUS, 0, -1):
            alpha = 150 * (1 - i / self.BALL_RADIUS)
            glow_color = (*self.COLOR_BALL, alpha)
            pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], i + 3, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

        # Combo
        if self.combo > 1:
            combo_text = self.font_ui.render(f"COMBO: x{self.combo}", True, self.COLOR_TEXT)
            self.screen.blit(combo_text, ((self.SCREEN_WIDTH - combo_text.get_width())/2, 10))
            
        # Game Over / Win message
        if self.game_over:
            if not self.blocks:
                msg = "YOU WIN!"
                color = (0, 255, 128)
            else:
                msg = "GAME OVER"
                color = (255, 0, 0)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
            "combo": self.combo
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

class Particle:
    def __init__(self, pos, color, np_random):
        self.pos = list(pos)
        self.np_random = np_random
        angle = self.np_random.random() * 2 * math.pi
        speed = self.np_random.random() * 2 + 1
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.lifespan = self.np_random.integers(20, 40)
        self.color = color
        self.radius = self.np_random.integers(2, 5)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1
        return self.lifespan > 0

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / 30))))
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (int(self.pos[0] - self.radius), int(self.pos[1] - self.radius)))

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Block Breaker")
    screen_display = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        # --- Human input mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Not used
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to screen ---
        # Pygame uses (W, H) but our obs is (H, W, C), so we need to handle it.
        # The env._get_observation() already created the frame, we can just grab its internal surface.
        surf = pygame.transform.rotate(env.screen, -90)
        surf = pygame.transform.flip(surf, True, False)
        screen_display.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Tick the clock ---
        clock.tick(30) # Match the auto_advance rate
        
    env.close()
    pygame.quit()