
# Generated: 2025-08-28T05:32:02.570146
# Source Brief: brief_05606.md
# Brief Index: 5606

        
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
    A top-down arcade block-breaker game.

    The player controls a paddle at the bottom of the screen to bounce a ball
    upwards, destroying a descending wall of blocks. The goal is to clear
    all blocks before they reach the bottom or before running out of lives.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Break all the descending blocks in this top-down arcade-style "
        "block breaker before they reach the bottom."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For auto-advance logic
        
        # Colors
        self.COLOR_BG = (15, 15, 35)
        self.COLOR_PADDLE = (0, 150, 255)
        self.COLOR_PADDLE_GLOW = (0, 75, 128)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BALL_GLOW = (128, 128, 128)
        self.COLOR_WALL = (50, 50, 90)
        self.COLOR_TEXT = (200, 200, 255)
        self.BLOCK_COLORS = [
            (0, 255, 128), (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

        # Game parameters
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 80, 12
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 6
        self.BALL_MAX_SPEED_Y = 8
        self.BALL_MAX_SPEED_X = 10
        self.BLOCK_ROWS, self.BLOCK_COLS = 5, 10
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 58, 18
        self.BLOCK_SPACING = 6
        self.INITIAL_LIVES = 3
        self.MAX_STEPS = 3600 # 2 minutes at 30fps

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # --- Game State Initialization ---
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_on_paddle = None
        self.blocks = None
        self.particles = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.game_won = None
        self.block_descent_speed = None
        self.initial_block_count = 0
        self.rng = None

        # Call reset to initialize the state for the first time
        self.reset()

        # Validate implementation after initialization
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # Fallback if seed is not provided in older gym versions
            self.rng = np.random.default_rng()

        # Initialize all game state
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2, 
            self.HEIGHT - 40, 
            self.PADDLE_WIDTH, 
            self.PADDLE_HEIGHT
        )
        self.lives = self.INITIAL_LIVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.particles = []
        self.block_descent_speed = 0.03 # pixels per step

        self._setup_blocks()
        self._reset_ball()
        
        return self._get_observation(), self._get_info()

    def _setup_blocks(self):
        self.blocks = []
        total_block_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.WIDTH - total_block_width) // 2
        start_y = 60
        
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                x = start_x + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = start_y + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({'rect': block_rect, 'color': color, 'orig_y': y})
        self.initial_block_count = len(self.blocks)

    def _reset_ball(self):
        self.ball_on_paddle = True
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        space_held = action[1] == 1
        
        paddle_moved = self._handle_input(movement, space_held)

        # --- Game Logic Update ---
        reward = 0
        blocks_destroyed_this_step = 0
        row_cleared = False
        life_lost = False

        if paddle_moved:
            reward -= 0.02 # Small penalty for movement

        self._update_game_state()
        
        # Ball logic
        if not self.ball_on_paddle:
            reward += 0.01 # Reward for keeping ball in play
            
            # Ball collisions
            blocks_destroyed, row_cleared = self._handle_ball_collisions()
            blocks_destroyed_this_step += blocks_destroyed
            
            # Check for losing a life
            if self.ball_pos[1] > self.HEIGHT + self.BALL_RADIUS:
                # sfx: life_lost_sound
                self.lives -= 1
                life_lost = True
                if self.lives > 0:
                    self._reset_ball()
                else:
                    self.game_over = True

        # --- Reward Calculation ---
        reward += blocks_destroyed_this_step * 1.0
        if row_cleared:
            reward += 5.0
        
        # --- Termination Check ---
        if self.lives <= 0:
            self.game_over = True
            reward -= 100.0
        
        if not self.blocks:
            self.game_over = True
            self.game_won = True
            reward += 100.0

        if any(block['rect'].bottom >= self.paddle.top for block in self.blocks):
            self.game_over = True
            reward -= 100.0
            
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        self.score += reward
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        paddle_moved = False
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
            paddle_moved = True
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
            paddle_moved = True
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)
        
        if self.ball_on_paddle and space_held:
            # sfx: launch_sound
            self.ball_on_paddle = False
            initial_vx = self.rng.uniform(-2, 2)
            self.ball_vel = [initial_vx, -self.BALL_MAX_SPEED_Y * 0.75]
        
        return paddle_moved

    def _update_game_state(self):
        # Update ball position
        if not self.ball_on_paddle:
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
        else:
            self.ball_pos[0] = self.paddle.centerx
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS

        # Update block positions
        if self.steps > 0 and self.steps % 500 == 0:
             self.block_descent_speed += 0.005 # Increase difficulty
        for block in self.blocks:
            block['rect'].y += self.block_descent_speed
        
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _handle_ball_collisions(self):
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        blocks_destroyed = 0
        row_cleared = False
        
        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            # sfx: wall_bounce_sound
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS
            # sfx: wall_bounce_sound

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: paddle_bounce_sound
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS # Prevent sticking

            # Add "spin" based on hit location
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = offset * self.BALL_MAX_SPEED_X
            self.ball_vel[0] = np.clip(self.ball_vel[0], -self.BALL_MAX_SPEED_X, self.BALL_MAX_SPEED_X)
        
        # Block collisions
        hit_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            # sfx: block_break_sound
            hit_block = self.blocks[hit_block_idx]
            
            # Determine if it's a horizontal or vertical collision
            prev_ball_pos = [self.ball_pos[0] - self.ball_vel[0], self.ball_pos[1] - self.ball_vel[1]]
            
            # Simple check: if ball center was above/below block in prev frame, it's a vertical hit
            if prev_ball_pos[1] < hit_block['rect'].top or prev_ball_pos[1] > hit_block['rect'].bottom:
                 self.ball_vel[1] *= -1
            else: # Otherwise, horizontal hit
                 self.ball_vel[0] *= -1

            self._create_particles(hit_block['rect'].center, hit_block['color'])
            
            # Check if row is cleared
            row_y = hit_block['orig_y']
            self.blocks.pop(hit_block_idx)
            blocks_destroyed += 1
            
            is_row_empty = not any(b['orig_y'] == row_y for b in self.blocks)
            if is_row_empty:
                # sfx: row_clear_sound
                row_cleared = True
        
        # Clamp ball speed
        self.ball_vel[1] = np.clip(self.ball_vel[1], -self.BALL_MAX_SPEED_Y, self.BALL_MAX_SPEED_Y)
        if abs(self.ball_vel[1]) < 2: # Prevent ball from getting too slow vertically
            self.ball_vel[1] = 2 * np.sign(self.ball_vel[1])

        return blocks_destroyed, row_cleared

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.rng.integers(10, 25)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})
    
    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game()
        
        # Render UI
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 25.0))
            p_color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, p_color)

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_BG, block['rect'].inflate(-4, -4), border_radius=2)
            
        # Draw paddle with glow
        glow_rect = self.paddle.inflate(6, 6)
        glow_rect.center = self.paddle.center
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_GLOW, glow_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=6)
        
        # Draw ball with glow
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 2, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Blocks remaining
        blocks_text = self.font_small.render(f"BLOCKS: {len(self.blocks)}", True, self.COLOR_TEXT)
        self.screen.blit(blocks_text, (self.WIDTH // 2 - blocks_text.get_width() // 2, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_BALL)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)
        elif self.ball_on_paddle:
            launch_text = self.font_small.render("PRESS SPACE TO LAUNCH", True, self.COLOR_TEXT)
            text_rect = launch_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 50))
            self.screen.blit(launch_text, text_rect)
    
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
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
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
    # Set `auto_advance` to True for human play
    env = GameEnv(render_mode="rgb_array")
    env.auto_advance = True
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    # --- Game Loop ---
    while not terminated:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Action Generation (from keyboard) ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # Pygame uses a different coordinate system, so we need to transpose
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate Control ---
        if env.auto_advance:
            clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000)
    env.close()