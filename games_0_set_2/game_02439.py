
# Generated: 2025-08-28T04:49:07.803799
# Source Brief: brief_02439.md
# Brief Index: 2439

        
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
        "A fast-paced, neon-drenched block breaker. Clear all blocks across 3 stages to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_large = pygame.font.SysFont("Consolas", 48)
            self.font_medium = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 60)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PADDLE = (0, 150, 255)
        self.COLOR_PADDLE_GLOW = (0, 100, 200)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BALL_GLOW = (200, 200, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_ACCENT = (255, 255, 0)
        self.BLOCK_COLORS = [
            (255, 0, 128), (0, 255, 255), (128, 255, 0),
            (255, 128, 0), (128, 0, 255)
        ]

        # Game constants
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 7
        self.INITIAL_BALL_SPEED = 5
        self.MAX_BOUNCE_ANGLE = math.pi / 3  # 60 degrees
        self.MAX_EPISODE_STEPS = 5000
        
        # State variables (initialized in reset)
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = None
        self.ball_speed = None
        self.blocks = None
        self.particles = None
        self.steps = None
        self.score = None
        self.lives = None
        self.stage = None
        self.game_over = None
        self.game_won = None
        self.ball_y_history = None
        self.ball_x_history = None

        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_speed = self.INITIAL_BALL_SPEED
        self._reset_ball()
        
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.stage = 1
        self.game_over = False
        self.game_won = False
        
        self.particles = []
        self.ball_y_history = []
        self.ball_x_history = []

        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_attached = True
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def _setup_stage(self):
        self.blocks = []
        block_width, block_height = 50, 20
        gap = 5
        rows, cols = 0, 0
        
        if self.stage == 1: # Pyramid
            rows, cols = 5, 11
            for r in range(rows):
                for c in range(r, cols - r):
                    if c >= 0 and c < cols:
                        x = c * (block_width + gap) + (self.WIDTH - cols * (block_width + gap)) / 2
                        y = r * (block_height + gap) + 50
                        color = self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)]
                        self.blocks.append({"rect": pygame.Rect(x, y, block_width, block_height), "color": color})

        elif self.stage == 2: # Grid
            rows, cols = 6, 10
            for r in range(rows):
                for c in range(cols):
                    x = c * (block_width + gap) + (self.WIDTH - cols * (block_width + gap)) / 2 + gap/2
                    y = r * (block_height + gap) + 50
                    color = self.BLOCK_COLORS[(r+c) % len(self.BLOCK_COLORS)]
                    self.blocks.append({"rect": pygame.Rect(x, y, block_width, block_height), "color": color})
                    
        elif self.stage == 3: # Alternating
            rows, cols = 5, 11
            for r in range(rows):
                for c in range(cols):
                    if (r + c) % 2 == 0:
                        x = c * (block_width + gap) + (self.WIDTH - cols * (block_width + gap)) / 2
                        y = r * (block_height + gap) + 60
                        color = self.BLOCK_COLORS[c % len(self.BLOCK_COLORS)]
                        self.blocks.append({"rect": pygame.Rect(x, y, block_width, block_height), "color": color})
        
    def step(self, action):
        if self.game_over or self.game_won:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        step_reward = 0

        # Handle Input
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            step_reward -= 0.02
        if movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            step_reward -= 0.02
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH))

        if self.ball_attached:
            self.ball_pos[0] = self.paddle.centerx
            if space_held:
                # Sound: Ball Launch
                self.ball_attached = False
                angle = (self.np_random.random() * 0.6 - 0.3) * math.pi
                self.ball_vel = [math.sin(angle) * self.ball_speed, -math.cos(angle) * self.ball_speed]
        else:
            # Update ball position
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]

            # Anti-softlock mechanism
            self._apply_anti_stuck()

            # Collisions
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            # Walls
            if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
                self.ball_vel[0] *= -1
                ball_rect.left = max(0, ball_rect.left)
                ball_rect.right = min(self.WIDTH, ball_rect.right)
                self.ball_pos[0] = ball_rect.centerx
                # Sound: Wall Bounce
            if ball_rect.top <= 0:
                self.ball_vel[1] *= -1
                ball_rect.top = max(0, ball_rect.top)
                self.ball_pos[1] = ball_rect.centery
                # Sound: Wall Bounce

            # Paddle
            if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
                # Sound: Paddle Bounce
                offset = (self.paddle.centerx - self.ball_pos[0]) / (self.PADDLE_WIDTH / 2)
                angle = offset * self.MAX_BOUNCE_ANGLE
                
                self.ball_vel[0] = -math.sin(angle) * self.ball_speed
                self.ball_vel[1] = -math.cos(angle) * self.ball_speed
                
                self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS

            # Blocks
            hit_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
            if hit_block_idx != -1:
                # Sound: Block Break
                block = self.blocks.pop(hit_block_idx)
                self._create_particles(block['rect'].center, block['color'])
                self.score += 10
                step_reward += 1.1 # +1 for breaking, +0.1 for hitting

                # Simple bounce logic
                self.ball_vel[1] *= -1

            # Miss
            if ball_rect.top >= self.HEIGHT:
                # Sound: Life Lost
                self.lives -= 1
                self._reset_ball()
                if self.lives <= 0:
                    self.game_over = True

        # Check for stage clear
        if not self.blocks and not self.game_won:
            self.stage += 1
            self.score += 100
            step_reward += 5
            if self.stage > 3:
                self.game_won = True
            else:
                # Sound: Stage Clear
                self.ball_speed += 0.5
                self._setup_stage()
                self._reset_ball()

        # Update particles
        self._update_particles()
        
        self.steps += 1
        
        terminated = self.game_over or self.game_won or self.steps >= self.MAX_EPISODE_STEPS
        
        if terminated:
            if self.game_won:
                step_reward += 100
            if self.game_over:
                step_reward -= 100
        
        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _apply_anti_stuck(self):
        self.ball_y_history.append(self.ball_pos[1])
        self.ball_x_history.append(self.ball_pos[0])
        if len(self.ball_y_history) > 30:
            self.ball_y_history.pop(0)
            self.ball_x_history.pop(0)
            if max(self.ball_y_history) - min(self.ball_y_history) < 1.0:
                self.ball_vel[1] += (self.np_random.random() - 0.5) * 0.2
            if max(self.ball_x_history) - min(self.ball_x_history) < 1.0:
                self.ball_vel[0] += (self.np_random.random() - 0.5) * 0.2
            # Renormalize speed
            current_speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if current_speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / current_speed) * self.ball_speed
                self.ball_vel[1] = (self.ball_vel[1] / current_speed) * self.ball_speed


    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_game(self):
        # Blocks
        for block in self.blocks:
            r = block['rect']
            c = block['color']
            darker_c = (max(0, c[0]-50), max(0, c[1]-50), max(0, c[2]-50))
            pygame.draw.rect(self.screen, darker_c, r.move(3, 3))
            pygame.draw.rect(self.screen, c, r)

        # Paddle
        pygame.gfxdraw.box(self.screen, self.paddle.move(2,2), (*self.COLOR_PADDLE_GLOW, 100))
        pygame.gfxdraw.box(self.screen, self.paddle, self.COLOR_PADDLE)

        # Ball
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 3, (*self.COLOR_BALL_GLOW, 100))
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            color = (*p['color'], alpha)
            size = int(3 * (p['lifespan'] / 30))
            if size > 0:
                rect = pygame.Rect(p['pos'][0] - size/2, p['pos'][1] - size/2, size, size)
                pygame.gfxdraw.box(self.screen, rect, color)
    
    def _render_ui(self):
        # Score
        score_surf = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH // 2 - score_surf.get_width() // 2, 10))

        # Lives
        lives_surf = self.font_medium.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_surf, (self.WIDTH - lives_surf.get_width() - 20, 10))

        # Stage
        stage_surf = self.font_medium.render(f"STAGE: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_surf, (20, 10))

        # Game Over / Win message
        if self.game_over:
            msg_surf = self.font_large.render("GAME OVER", True, self.COLOR_TEXT_ACCENT)
            self.screen.blit(msg_surf, (self.WIDTH // 2 - msg_surf.get_width() // 2, self.HEIGHT // 2 - msg_surf.get_height() // 2))
        elif self.game_won:
            msg_surf = self.font_large.render("YOU WIN!", True, self.COLOR_TEXT_ACCENT)
            self.screen.blit(msg_surf, (self.WIDTH // 2 - msg_surf.get_width() // 2, self.HEIGHT // 2 - msg_surf.get_height() // 2))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Create a display window
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    
    while running:
        # Get user input
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Unused
        
        action = [movement, space_held, shift_held]
        
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Control frame rate
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
            terminated = False

    env.close()