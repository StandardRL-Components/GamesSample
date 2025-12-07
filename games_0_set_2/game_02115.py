
# Generated: 2025-08-28T03:46:16.844348
# Source Brief: brief_02115.md
# Brief Index: 2115

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "A fast-paced, retro block-breaker. Clear the screen of all blocks to win, but don't lose all your balls!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 6.0
        self.MAX_STEPS = 1500 # Increased for more play time
        self.INITIAL_LIVES = 3
        
        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (220, 220, 220)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (200, 200, 200)
        self.BLOCK_COLORS = [
            (255, 50, 50), (255, 150, 50), (50, 255, 50), 
            (50, 150, 255), (150, 50, 255)
        ]

        # --- Gym Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Game State (initialized in reset) ---
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.ball_on_paddle = True
        self.game_over = False
        
        self.reset()

        # self.validate_implementation() # Optional: Call to self-check
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset paddle
        paddle_y = self.HEIGHT - 40
        paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Reset ball
        self._reset_ball()

        # Reset blocks
        self._create_block_grid()

        # Reset particles
        self.particles = []

        # Reset game state
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.steps = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_on_paddle = True
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def _create_block_grid(self):
        self.blocks = []
        block_width, block_height = 40, 20
        gap = 4
        num_cols = 15
        num_rows = 5
        grid_width = num_cols * (block_width + gap) - gap
        start_x = (self.WIDTH - grid_width) / 2
        start_y = 50

        for i in range(num_rows):
            for j in range(num_cols):
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                x = start_x + j * (block_width + gap)
                y = start_y + i * (block_height + gap)
                rect = pygame.Rect(x, y, block_width, block_height)
                self.blocks.append({'rect': rect, 'color': color})
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.02  # Time penalty

        # --- Handle Input ---
        movement = action[0]
        space_pressed = action[1] == 1
        
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        if movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = max(0, min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH))

        if self.ball_on_paddle and space_pressed:
            # Sound: Launch
            self.ball_on_paddle = False
            launch_angle = self.np_random.uniform(-0.6, 0.6)
            self.ball_vel = [self.BALL_SPEED * launch_angle, -self.BALL_SPEED]
            # Ensure vertical speed is significant
            self.ball_vel[1] = -abs(self.BALL_SPEED * math.cos(launch_angle))


        # --- Update Game Logic ---
        event_reward = self._update_ball()
        reward += event_reward
        self._update_particles()
        
        # --- Check Termination ---
        self.steps += 1
        win_condition = not self.blocks
        lose_condition = self.lives <= 0
        timeout_condition = self.steps >= self.MAX_STEPS
        
        terminated = win_condition or lose_condition or timeout_condition

        if win_condition and not self.game_over:
            reward += 100  # Win bonus
        if lose_condition and not self.game_over:
            reward = -100 # Loss penalty overrides other rewards
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        if self.ball_on_paddle:
            self.ball_pos[0] = self.paddle.centerx
            return 0

        # Move ball
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0:
            ball_rect.left = 0
            self.ball_vel[0] *= -1
        if ball_rect.right >= self.WIDTH:
            ball_rect.right = self.WIDTH
            self.ball_vel[0] *= -1
        if ball_rect.top <= 0:
            ball_rect.top = 0
            self.ball_vel[1] *= -1

        # Floor collision (lose life)
        if ball_rect.top >= self.HEIGHT:
            self.lives -= 1
            # Sound: Lose life
            if self.lives > 0:
                self._reset_ball()
                return -5
            else:
                return 0 # The -100 is applied in step()

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # Sound: Paddle hit
            ball_rect.bottom = self.paddle.top
            self.ball_pos[1] = ball_rect.centery
            self.ball_vel[1] *= -1

            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2.5
            
            # Normalize to constant speed
            speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            if speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / speed) * self.BALL_SPEED
                self.ball_vel[1] = (self.ball_vel[1] / speed) * self.BALL_SPEED
            
            return 0.1

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block['rect']):
                # Sound: Block break
                self._create_particles(block['rect'].center, block['color'])
                self.blocks.remove(block)
                self.score += 1

                # Bounce logic
                prev_ball_rect = ball_rect.copy()
                prev_ball_rect.x -= self.ball_vel[0]
                prev_ball_rect.y -= self.ball_vel[1]
                
                if prev_ball_rect.bottom <= block['rect'].top or prev_ball_rect.top >= block['rect'].bottom:
                    self.ball_vel[1] *= -1
                else:
                    self.ball_vel[0] *= -1
                
                return 1.0 # Reward for breaking a block
        
        return 0

    def _create_particles(self, pos, color):
        for _ in range(15):
            vel = [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Drag
            p['vel'][1] *= 0.95
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
        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            r, g, b = p['color']
            # Using simple rects for particles for performance and style
            pygame.draw.rect(self.screen, (r, g, b, alpha), (*p['pos'], 2, 2))

        # Draw ball with anti-aliasing
        x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Draw score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Draw lives
        for i in range(self.lives):
            x = self.WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 10))
            y = 10 + self.BALL_RADIUS
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_PADDLE)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_PADDLE)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
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
    # pip install gymnasium[classic-control]
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Block Breaker Gym Env")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to Screen ---
        # Pygame uses (width, height), but our obs is (height, width, 3)
        # So we need to transpose it back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & FPS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Info: {info}")
    env.close()