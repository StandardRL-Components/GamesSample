
# Generated: 2025-08-28T01:33:32.448408
# Source Brief: brief_04144.md
# Brief Index: 4144

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set a dummy video driver to run Pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based block breaker. Clear all blocks to advance. "
        "Running out of time or losing all your balls ends the game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STAGES = 3
        self.MAX_LIVES = 3
        self.STAGE_TIME_SECONDS = 60
        self.MAX_STEPS = self.STAGE_TIME_SECONDS * self.FPS * self.MAX_STAGES

        # Colors
        self.COLOR_BG = (15, 23, 42)  # Dark Slate Blue
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (250, 204, 21)  # Yellow
        self.COLOR_UI_TEXT = (226, 232, 240)
        self.COLOR_GRID = (30, 41, 59)
        self.BLOCK_COLORS = [
            (239, 68, 68),   # Red
            (59, 130, 246),  # Blue
            (34, 197, 94),   # Green
            (249, 115, 22),  # Orange
            (168, 85, 247),  # Purple
        ]

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
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stage = 1
        self.lives = self.MAX_LIVES
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.02  # Time penalty to encourage efficiency
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            
            self._handle_input(movement, space_held)
            reward += self._update_game_state()

        self.steps += 1
        
        # Check termination conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Paddle Movement
        if movement == 3:  # Left
            self.paddle.x -= self.paddle_speed
        elif movement == 4:  # Right
            self.paddle.x += self.paddle_speed
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.paddle.width)

        # Launch Ball
        if space_held and self.ball_attached:
            self.ball_attached = False
            # Sound: Ball Launch
            ball_speed_multiplier = 1.0 + (self.stage - 1) * 0.2
            self.ball_vel = [
                self.np_random.uniform(-0.5, 0.5),
                -self.base_ball_speed * ball_speed_multiplier
            ]

    def _update_game_state(self):
        reward = 0
        
        # Update time
        self.time_remaining -= 1
        if self.time_remaining <= 0:
            reward += self._lose_life()

        # Update ball position
        if self.ball_attached:
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top
        else:
            self.ball.x += self.ball_vel[0]
            self.ball.y += self.ball_vel[1]

        # Update particles
        self._update_particles()
        
        # Handle collisions
        reward += self._handle_collisions()

        # Check for stage clear
        if not self.blocks:
            # Sound: Stage Clear
            reward += 50
            self.stage += 1
            if self.stage > self.MAX_STAGES:
                reward += 100
                self.win = True
                self.game_over = True
            else:
                self._setup_stage()
        
        return reward

    def _handle_collisions(self):
        reward = 0
        
        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.x = np.clip(self.ball.x, self.ball_radius, self.WIDTH - self.ball_radius)
            # Sound: Wall Bounce
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.y = np.clip(self.ball.y, self.ball_radius, self.HEIGHT - self.ball_radius)
            # Sound: Wall Bounce

        # Bottom wall (lose life)
        if self.ball.top >= self.HEIGHT:
            reward += self._lose_life()
        
        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            reward += 0.1
            self.ball.bottom = self.paddle.top
            self.ball_vel[1] *= -1
            
            # Influence horizontal velocity based on hit location
            offset = (self.ball.centerx - self.paddle.centerx) / (self.paddle.width / 2)
            self.ball_vel[0] += offset * 2.0
            
            # Normalize ball speed to prevent it from getting too fast/slow
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            ball_speed_multiplier = 1.0 + (self.stage - 1) * 0.2
            target_speed = self.base_ball_speed * ball_speed_multiplier
            self.ball_vel[0] = (self.ball_vel[0] / speed) * target_speed
            self.ball_vel[1] = (self.ball_vel[1] / speed) * target_speed
            # Sound: Paddle Hit

        # Block collisions
        hit_block_idx = self.ball.collidelist(self.blocks)
        if hit_block_idx != -1:
            reward += 1
            block = self.blocks.pop(hit_block_idx)
            # Sound: Block Break
            
            # Create particle explosion
            self._create_particles(block.center, self.BLOCK_COLORS[hit_block_idx % len(self.BLOCK_COLORS)])

            # Simple bounce logic
            prev_ball_center = (self.ball.centerx - self.ball_vel[0], self.ball.centery - self.ball_vel[1])
            
            # Determine if it was a horizontal or vertical collision
            if (prev_ball_center[1] < block.top or prev_ball_center[1] > block.bottom):
                self.ball_vel[1] *= -1
            else:
                self.ball_vel[0] *= -1
        
        return reward

    def _lose_life(self):
        # Sound: Life Lost
        self.lives -= 1
        if self.lives <= 0:
            self.game_over = True
            return -5
        else:
            self._reset_ball_and_timer()
            return -5

    def _reset_ball_and_timer(self):
        self.ball_attached = True
        self.time_remaining = self.STAGE_TIME_SECONDS * self.FPS

    def _setup_stage(self):
        # Paddle
        self.paddle_width, self.paddle_height = 100, 15
        self.paddle = pygame.Rect(
            (self.WIDTH - self.paddle_width) / 2,
            self.HEIGHT - 40,
            self.paddle_width,
            self.paddle_height
        )
        self.paddle_speed = 8

        # Ball
        self.ball_radius = 8
        self.base_ball_speed = 5.0
        self.ball = pygame.Rect(0, 0, self.ball_radius * 2, self.ball_radius * 2)
        
        # Blocks
        self._create_blocks()
        
        # Particles
        self.particles = []

        # Reset ball position and timer
        self._reset_ball_and_timer()

    def _create_blocks(self):
        self.blocks = []
        block_width, block_height = 58, 20
        gap = 6
        rows, cols = 0, 0
        
        if self.stage == 1:
            rows, cols = 5, 10
            for r in range(rows):
                for c in range(cols):
                    self.blocks.append(pygame.Rect(
                        c * (block_width + gap) + gap,
                        r * (block_height + gap) + 60,
                        block_width,
                        block_height
                    ))
        elif self.stage == 2:
            rows, cols = 7, 10
            for r in range(rows):
                for c in range(cols):
                    if (c < 2 or c > 7) or (r > 1 and r < 5):
                         self.blocks.append(pygame.Rect(
                            c * (block_width + gap) + gap,
                            r * (block_height + gap) + 60,
                            block_width,
                            block_height
                        ))
        elif self.stage == 3:
            rows, cols = 8, 10
            for r in range(rows):
                for c in range(cols):
                    if (r+c) % 2 == 0:
                        self.blocks.append(pygame.Rect(
                            c * (block_width + gap) + gap,
                            r * (block_height + gap) + 50,
                            block_width,
                            block_height
                        ))

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'lifetime': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['lifetime'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['lifetime'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_all(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_grid()
        self._render_blocks()
        self._render_paddle()
        self._render_ball()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()

    def _render_grid(self):
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_blocks(self):
        for i, block in enumerate(self.blocks):
            color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
            pygame.draw.rect(self.screen, color, block, border_radius=3)

    def _render_paddle(self):
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

    def _render_ball(self):
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.ball_radius, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.ball_radius, self.COLOR_BALL)
    
    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_medium.render(f"BALLS: {self.lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (self.WIDTH // 2 - stage_text.get_width() // 2, 10))
        
        # Time
        time_str = f"TIME: {self.time_remaining // self.FPS:02d}"
        time_text = self.font_small.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH // 2 - time_text.get_width() // 2, 30))

        # Game Over / Win Message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        self.score = max(0, int(self.score + self._update_game_state())) # Update score in info
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "lives": self.lives,
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
    # Re-enable the normal video driver
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "windows" on Windows, "x11" or "wayland" on Linux

    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    print(env.user_guide)

    while not terminated:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()

        clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']}")
    env.close()