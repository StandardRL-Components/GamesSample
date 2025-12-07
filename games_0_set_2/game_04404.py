
# Generated: 2025-08-28T02:17:30.815520
# Source Brief: brief_04404.md
# Brief Index: 4404

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro block breaker. Clear all bricks across 3 stages before the time runs out or you lose all your balls. Risky paddle hits are rewarded with more control."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    
    # Colors
    COLOR_BG = (15, 15, 35)
    COLOR_GRID = (30, 30, 50)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_PADDLE_GLOW = (200, 200, 255, 100)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (255, 255, 150, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    BRICK_COLORS = [
        (255, 50, 50), (255, 150, 50), (255, 255, 50),
        (50, 255, 50), (50, 150, 255), (150, 50, 255)
    ]

    # Game element properties
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 16
    PADDLE_SPEED = 12
    BALL_RADIUS = 8
    BALL_SPEED = 8
    BRICK_WIDTH = 48
    BRICK_HEIGHT = 20
    
    MAX_STAGE = 3
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

        # Fonts
        self.font_ui = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 60)
        self.font_subtitle = pygame.font.Font(None, 30)

        # State variables (initialized in reset)
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.bricks = None
        self.particles = None
        self.score = None
        self.balls_left = None
        self.current_stage = None
        self.game_over = None
        self.game_won = None
        self.stage_timer = None
        self.steps = None
        self.ball_y_history = None
        
        self.reset()
        
        # This check is for development and can be removed in production
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.balls_left = 3
        self.current_stage = 1
        self.game_over = False
        self.game_won = False
        self.steps = 0
        self.particles = []
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_launched = False
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS], dtype=float)
        self.ball_vel = np.array([0.0, 0.0], dtype=float)
        self.stage_timer = self.STAGE_TIME_SECONDS * self.FPS
        self.ball_y_history = deque(maxlen=self.FPS) # 1 second history
        self._generate_bricks()

    def _generate_bricks(self):
        self.bricks = []
        brick_y_offset = 60
        num_cols = self.SCREEN_WIDTH // (self.BRICK_WIDTH + 4)
        col_width = self.BRICK_WIDTH + 4
        x_start_offset = (self.SCREEN_WIDTH - num_cols * col_width) / 2 + 2

        if self.current_stage == 1: # Standard wall
            for row in range(6):
                for col in range(num_cols):
                    brick_rect = pygame.Rect(
                        x_start_offset + col * col_width,
                        brick_y_offset + row * (self.BRICK_HEIGHT + 4),
                        self.BRICK_WIDTH,
                        self.BRICK_HEIGHT
                    )
                    self.bricks.append({'rect': brick_rect, 'color': self.BRICK_COLORS[row % len(self.BRICK_COLORS)]})
        elif self.current_stage == 2: # Pyramid
            for row in range(7):
                row_cols = num_cols - 2 * row
                if row_cols <= 0: break
                row_x_offset = x_start_offset + row * col_width
                for col in range(row_cols):
                    brick_rect = pygame.Rect(
                        row_x_offset + col * col_width,
                        brick_y_offset + row * (self.BRICK_HEIGHT + 4),
                        self.BRICK_WIDTH,
                        self.BRICK_HEIGHT
                    )
                    self.bricks.append({'rect': brick_rect, 'color': self.BRICK_COLORS[row % len(self.BRICK_COLORS)]})
        elif self.current_stage == 3: # Checkered
             for row in range(8):
                for col in range(num_cols):
                    if (row + col) % 2 == 0:
                        brick_rect = pygame.Rect(
                            x_start_offset + col * col_width,
                            brick_y_offset + row * (self.BRICK_HEIGHT + 4),
                            self.BRICK_WIDTH,
                            self.BRICK_HEIGHT
                        )
                        self.bricks.append({'rect': brick_rect, 'color': self.BRICK_COLORS[row % len(self.BRICK_COLORS)]})


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_pressed = action[1] == 1
        
        reward = -0.01  # Time penalty
        terminated = False
        
        # --- Handle Input ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        if space_pressed and not self.ball_launched:
            self.ball_launched = True
            angle = (self.np_random.random() - 0.5) * (math.pi / 4) # -22.5 to +22.5 degrees
            self.ball_vel = np.array([math.sin(angle) * self.BALL_SPEED, -math.cos(angle) * self.BALL_SPEED])
            # sfx: launch_ball

        # --- Update Game State ---
        self.steps += 1
        self.stage_timer -= 1

        # Ball movement
        if self.ball_launched:
            self.ball_pos += self.ball_vel
            self.ball_y_history.append(self.ball_pos[1])
        else:
            self.ball_pos[0] = self.paddle.centerx
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS

        # Particle update
        self._update_particles()

        # --- Collision Detection ---
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
            # sfx: wall_bounce
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS
            # sfx: wall_bounce

        # Paddle collision
        if self.ball_vel[1] > 0 and ball_rect.colliderect(self.paddle):
            reward += 0.1
            dist = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = dist * self.BALL_SPEED * 1.1 # More horizontal control
            self.ball_vel[1] *= -1
            self.ball_vel /= np.linalg.norm(self.ball_vel)
            self.ball_vel *= self.BALL_SPEED
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS # Prevent sticking
            self._create_particles(self.ball_pos, self.COLOR_PADDLE, 5, is_paddle_hit=True)
            # sfx: paddle_hit

        # Brick collisions
        for brick in self.bricks[:]:
            if ball_rect.colliderect(brick['rect']):
                reward += 1
                self.score += 1
                
                # Determine collision side to correctly reflect
                # A simple but effective method
                prev_ball_pos = self.ball_pos - self.ball_vel
                if (prev_ball_pos[0] < brick['rect'].left or prev_ball_pos[0] > brick['rect'].right):
                    self.ball_vel[0] *= -1
                else:
                    self.ball_vel[1] *= -1

                self._create_particles(brick['rect'].center, brick['color'], 15)
                self.bricks.remove(brick)
                # sfx: brick_break
                break # Process one brick collision per frame

        # --- Check Game Conditions ---
        # Ball lost
        if self.ball_pos[1] > self.SCREEN_HEIGHT:
            self.balls_left -= 1
            reward -= 2
            # sfx: lose_ball
            if self.balls_left <= 0:
                self.game_over = True
            else:
                self._setup_stage() # Only resets ball/paddle, not score/stage

        # Time up
        if self.stage_timer <= 0:
            self.game_over = True
            terminated = True
        
        # Stage clear
        if not self.bricks:
            reward += 10
            # sfx: stage_clear
            self.current_stage += 1
            if self.current_stage > self.MAX_STAGE:
                self.game_won = True
                reward += 100
                # sfx: game_win
            else:
                self._setup_stage()

        # Anti-softlock
        if self.ball_launched and len(self.ball_y_history) == self.ball_y_history.maxlen:
            if max(self.ball_y_history) - min(self.ball_y_history) < 1.0:
                self.ball_vel[1] += (self.np_random.random() - 0.5) * 2
                self.ball_vel /= np.linalg.norm(self.ball_vel)
                self.ball_vel *= self.BALL_SPEED

        if self.game_over or self.game_won:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _create_particles(self, pos, color, count, is_paddle_hit=False):
        for _ in range(count):
            if is_paddle_hit:
                angle = math.pi * self.np_random.random()
                speed = 1 + self.np_random.random() * 2
            else:
                angle = 2 * math.pi * self.np_random.random()
                speed = 2 + self.np_random.random() * 4
            
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.FPS // 2 + self.np_random.integers(0, self.FPS // 2)
            radius = 2 + self.np_random.random() * 2
            self.particles.append({'pos': list(pos), 'vel': vel, 'radius': radius, 'color': color, 'lifetime': lifetime})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifetime'] -= 1
            p['radius'] -= 0.05
            if p['lifetime'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Background grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick['color'], brick['rect'])
            pygame.draw.rect(self.screen, self.COLOR_BG, brick['rect'], 1) # Outline

        # Paddle
        glow_surf = pygame.Surface((self.PADDLE_WIDTH + 20, self.PADDLE_HEIGHT + 20), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PADDLE_GLOW, glow_surf.get_rect(), border_radius=12)
        self.screen.blit(glow_surf, (self.paddle.x - 10, self.paddle.y - 10), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Ball
        ball_center = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        glow_surf = pygame.Surface((self.BALL_RADIUS * 4, self.BALL_RADIUS * 4), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, self.BALL_RADIUS*2, self.BALL_RADIUS*2, self.BALL_RADIUS*2, self.COLOR_BALL_GLOW)
        self.screen.blit(glow_surf, (ball_center[0] - self.BALL_RADIUS*2, ball_center[1] - self.BALL_RADIUS*2), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p['color'])

    def _render_text(self, text, font, position, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW):
        shadow = font.render(text, True, shadow_color)
        self.screen.blit(shadow, (position[0] + 2, position[1] + 2))
        rendered_text = font.render(text, True, color)
        self.screen.blit(rendered_text, position)

    def _render_ui(self):
        # Score
        self._render_text(f"SCORE: {self.score}", self.font_ui, (10, 10))
        
        # Balls left
        ball_text = self.font_ui.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(ball_text, (self.SCREEN_WIDTH - 160, 10))
        for i in range(self.balls_left):
            pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH - 80 + i * 25, 18, 8, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH - 80 + i * 25, 18, 8, self.COLOR_BALL)

        # Stage
        self._render_text(f"STAGE {self.current_stage}", self.font_ui, (self.SCREEN_WIDTH / 2 - 40, self.SCREEN_HEIGHT - 30))

        # Timer
        secs = self.stage_timer // self.FPS
        self._render_text(f"TIME: {secs}", self.font_ui, (self.SCREEN_WIDTH / 2 - 40, 10))

        # Game Over / Win Message
        if self.game_over:
            self._render_text("GAME OVER", self.font_title, (self.SCREEN_WIDTH/2 - 150, self.SCREEN_HEIGHT/2 - 50))
            self._render_text(f"FINAL SCORE: {self.score}", self.font_subtitle, (self.SCREEN_WIDTH/2 - 100, self.SCREEN_HEIGHT/2 + 10))
        elif self.game_won:
            self._render_text("YOU WIN!", self.font_title, (self.SCREEN_WIDTH/2 - 120, self.SCREEN_HEIGHT/2 - 50))
            self._render_text(f"FINAL SCORE: {self.score}", self.font_subtitle, (self.SCREEN_WIDTH/2 - 100, self.SCREEN_HEIGHT/2 + 10))


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

# Example usage for interactive play
if __name__ == "__main__":
    import sys
    
    # Set SDL_VIDEODRIVER to "dummy" if you want to run headless
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Create a real window for interactive play
    pygame.display.set_caption("Block Breaker")
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    print(GameEnv.user_guide)
    print(GameEnv.game_description)

    while not terminated:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render to the real screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling and clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        env.clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()
    pygame.quit()
    sys.exit()