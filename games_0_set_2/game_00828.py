
# Generated: 2025-08-27T14:54:45.031968
# Source Brief: brief_00828.md
# Brief Index: 828

        
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
        "Controls: Use ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade classic. Control the paddle to deflect the ball and destroy all the bricks to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 7
        self.BASE_BALL_SPEED = 4.0
        self.MAX_STEPS = 10000
        self.INITIAL_LIVES = 3
        self.BRICK_ROWS = 6
        self.BRICK_COLS = 10
        self.BRICK_WIDTH = self.WIDTH // self.BRICK_COLS
        self.BRICK_HEIGHT = 20
        self.BRICK_TOP_OFFSET = 50

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (220, 220, 255)
        self.COLOR_BALL = (100, 255, 100)
        self.COLOR_BALL_GLOW = (150, 255, 150)
        self.COLOR_UI = (200, 200, 220)
        self.BRICK_COLORS = [
            (217, 87, 99), (217, 144, 87), (195, 217, 87),
            (87, 217, 134), (87, 168, 217), (139, 87, 217)
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)

        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_pos_float = None
        self.ball_vel = None
        self.ball_launched = None
        self.bricks = None
        self.initial_brick_count = self.BRICK_ROWS * self.BRICK_COLS
        self.current_ball_speed = self.BASE_BALL_SPEED
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.consecutive_hits = 0

        self.reset()
        
        # Self-validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES
        self.consecutive_hits = 0
        self.current_ball_speed = self.BASE_BALL_SPEED

        # Paddle
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT - 10
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2, paddle_y,
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )

        # Ball
        self.ball_launched = False
        self._reset_ball()

        # Bricks
        self.bricks = []
        for row in range(self.BRICK_ROWS):
            for col in range(self.BRICK_COLS):
                brick = pygame.Rect(
                    col * self.BRICK_WIDTH,
                    self.BRICK_TOP_OFFSET + row * self.BRICK_HEIGHT,
                    self.BRICK_WIDTH, self.BRICK_HEIGHT
                )
                self.bricks.append(brick)

        self.particles.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for each step
        
        if not self.game_over:
            self._handle_input(action)
            reward += self._update_game_state()

        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            if self.lives <= 0:
                reward -= 100
            elif len(self.bricks) == 0:
                reward += 100
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_pressed = action[1] == 1

        # Paddle Movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        # Launch Ball
        if space_pressed and not self.ball_launched:
            self.ball_launched = True
            # sfx: launch_ball.wav
            initial_angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            self.ball_vel = [
                math.cos(initial_angle) * self.current_ball_speed,
                math.sin(initial_angle) * self.current_ball_speed
            ]

    def _update_game_state(self):
        step_reward = 0

        if not self.ball_launched:
            self._reset_ball_position()
            return step_reward

        # Move Ball
        self.ball_pos_float[0] += self.ball_vel[0]
        self.ball_pos_float[1] += self.ball_vel[1]
        self.ball.center = (int(self.ball_pos_float[0]), int(self.ball_pos_float[1]))

        # Wall Collision
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.left = max(0, self.ball.left)
            self.ball.right = min(self.WIDTH, self.ball.right)
            # sfx: wall_bounce.wav
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.top = max(0, self.ball.top)
            # sfx: wall_bounce.wav

        # Paddle Collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            self.ball.bottom = self.paddle.top

            # Angle change based on hit location
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = offset * self.current_ball_speed * 0.8
            self._normalize_ball_velocity()
            # sfx: paddle_bounce.wav

        # Brick Collision
        hit_brick_index = self.ball.collidelist(self.bricks)
        if hit_brick_index != -1:
            brick = self.bricks.pop(hit_brick_index)
            # sfx: brick_destroy.wav

            # Reward calculation
            step_reward += 1.0  # Base reward for destroying a brick
            step_reward += 0.1  # Continuous feedback reward
            step_reward += 0.5 * self.consecutive_hits # Combo bonus
            self.score += 10 + 5 * self.consecutive_hits
            self.consecutive_hits += 1

            # Collision response
            self._handle_brick_collision(brick)
            self._create_particles(brick.center, self._get_brick_color(brick))
            self._update_ball_speed()

        # Life Lost
        if self.ball.top >= self.HEIGHT:
            self.lives -= 1
            self.consecutive_hits = 0
            self.ball_launched = False
            self._reset_ball()
            # sfx: life_lost.wav
            if self.lives > 0:
                step_reward -= 10 # Penalty for losing a life (non-terminal)


        # Update particles
        self._update_particles()
        
        return step_reward

    def _handle_brick_collision(self, brick):
        overlap = self.ball.clip(brick)
        if overlap.width > overlap.height:
            self.ball_vel[1] *= -1
            # Move ball out of collision
            if self.ball.centery < brick.centery:
                self.ball.bottom = brick.top
            else:
                self.ball.top = brick.bottom
        else:
            self.ball_vel[0] *= -1
            # Move ball out of collision
            if self.ball.centerx < brick.centerx:
                self.ball.right = brick.left
            else:
                self.ball.left = brick.right
        
        self._normalize_ball_velocity()

    def _update_ball_speed(self):
        bricks_destroyed = self.initial_brick_count - len(self.bricks)
        speed_increase_tiers = bricks_destroyed // 10
        self.current_ball_speed = self.BASE_BALL_SPEED + speed_increase_tiers * 0.5
        self._normalize_ball_velocity()

    def _normalize_ball_velocity(self):
        norm = math.hypot(*self.ball_vel)
        if norm > 0:
            self.ball_vel[0] = (self.ball_vel[0] / norm) * self.current_ball_speed
            self.ball_vel[1] = (self.ball_vel[1] / norm) * self.current_ball_speed
        else: # If ball somehow stops, give it a kick
            self.ball_vel[1] = -self.current_ball_speed

    def _reset_ball(self):
        self.ball_vel = [0, 0]
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball_pos_float = [0.0, 0.0]
        self._reset_ball_position()

    def _reset_ball_position(self):
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top
        self.ball_pos_float = [float(self.ball.centerx), float(self.ball.centery)]

    def _check_termination(self):
        return self.lives <= 0 or len(self.bricks) == 0 or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw Bricks
        for brick in self.bricks:
            color = self._get_brick_color(brick)
            pygame.draw.rect(self.screen, color, brick)
            pygame.draw.rect(self.screen, self.COLOR_BG, brick, 1)

        # Draw Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw Ball with glow
        center = (int(self.ball.centerx), int(self.ball.centery))
        glow_radius = int(self.BALL_RADIUS * 1.5)
        for i in range(glow_radius, self.BALL_RADIUS, -1):
            alpha = 50 * (1 - (i - self.BALL_RADIUS) / (glow_radius - self.BALL_RADIUS))
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], i, (*self.COLOR_BALL_GLOW, int(alpha)))
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Draw Particles
        for p in self.particles:
            pygame.draw.rect(self.screen, p['color'], p['rect'])

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            life_paddle_rect = pygame.Rect(
                self.WIDTH - (i + 1) * 40, 10, 30, 8
            )
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_paddle_rect, border_radius=2)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_left": len(self.bricks),
        }

    def _get_brick_color(self, brick):
        row_index = (brick.y - self.BRICK_TOP_OFFSET) // self.BRICK_HEIGHT
        return self.BRICK_COLORS[row_index % len(self.BRICK_COLORS)]

    def _create_particles(self, pos, color):
        for _ in range(15):
            particle = {
                'rect': pygame.Rect(pos[0], pos[1], 3, 3),
                'vel': [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)],
                'life': self.np_random.integers(15, 30),
                'color': color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p['rect'].x += p['vel'][0]
            p['rect'].y += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

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
    import sys
    
    env = GameEnv(render_mode="rgb_array")
    
    # Use a dummy screen for playing, as the env renders to a surface
    play_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Breakout")
    
    obs, info = env.reset()
    done = False
    
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
        shift_held = 0 # Shift is not used in this game
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        play_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame rate ---
        env.clock.tick(60)

    print(f"Game Over! Final Score: {info['score']}")
    env.close()
    pygame.quit()
    sys.exit()