
# Generated: 2025-08-28T06:37:26.841720
# Source Brief: brief_02989.md
# Brief Index: 2989

        
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
    """
    A fast-paced, arcade-style brick breaker game.

    The player controls a paddle at the bottom of the screen to bounce a ball
    upwards, breaking a grid of bricks. The goal is to clear all bricks without
    losing all lives. The game features a combo multiplier, increasing ball speed,
    and particle effects for a visually engaging experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Break all the bricks with a paddle in this fast-paced, grid-based arcade game."
    )

    # Frames auto-advance at 30fps.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 7
        self.INITIAL_LIVES = 5
        self.MAX_STEPS = 10000
        self.INITIAL_BALL_SPEED = 5
        self.BRICK_ROWS, self.BRICK_COLS = 5, 10
        
        # --- Colors ---
        self.COLOR_BG = (20, 20, 40)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PADDLE = (220, 220, 220)
        self.COLOR_PADDLE_HL = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (200, 200, 0)
        self.BRICK_COLORS = [(255, 50, 50), (255, 150, 50), (255, 255, 50), (50, 255, 50), (50, 150, 255)]
        self.COLOR_TEXT = (255, 255, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_multiplier = pygame.font.Font(None, 48)

        # --- Game State Variables ---
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_launched = None
        self.ball_base_speed = None
        self.lives = None
        self.score = None
        self.bricks = None
        self.brick_colors = None
        self.particles = None
        self.steps = None
        self.game_over = None
        self.bricks_destroyed_this_episode = None
        self.consecutive_hits = None
        self.multiplier = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Paddle ---
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT - 10
        self.paddle = pygame.Rect((self.WIDTH - self.PADDLE_WIDTH) / 2, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # --- Initialize Ball ---
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball_vel = [0, 0]
        self.ball_launched = False
        self.ball_base_speed = self.INITIAL_BALL_SPEED
        self._reset_ball_position()

        # --- Initialize Bricks ---
        self.bricks = []
        self.brick_colors = []
        brick_width = (self.WIDTH - (self.BRICK_COLS + 1) * 4) / self.BRICK_COLS
        brick_height = 20
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                x = j * (brick_width + 4) + 4
                y = i * (brick_height + 4) + 50
                brick_rect = pygame.Rect(x, y, brick_width, brick_height)
                self.bricks.append(brick_rect)
                self.brick_colors.append(self.BRICK_COLORS[i % len(self.BRICK_COLORS)])

        # --- Initialize Game State ---
        self.lives = self.INITIAL_LIVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.bricks_destroyed_this_episode = 0
        self.consecutive_hits = 0
        self.multiplier = 1
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- 1. Handle Input & Paddle Movement ---
        paddle_moved_safely = False
        if self.ball_launched and self.ball_vel[1] > 0:
            # Calculate projected landing spot for reward shaping
            time_to_paddle = (self.paddle.top - self.ball.centery) / self.ball_vel[1]
            projected_x = self.ball.centerx + self.ball_vel[0] * time_to_paddle
            dist_before = abs(self.paddle.centerx - projected_x)
            
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED
            
            dist_after = abs(self.paddle.centerx - projected_x)
            if dist_after < dist_before:
                paddle_moved_safely = True
            else:
                reward -= 0.2  # Penalty for moving away from ball
        else: # Ball not moving towards paddle, just move
             if movement == 3: self.paddle.x -= self.PADDLE_SPEED
             if movement == 4: self.paddle.x += self.PADDLE_SPEED


        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        if space_held and not self.ball_launched:
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            self.ball_vel = [self.ball_base_speed * math.sin(angle), -self.ball_base_speed * math.cos(angle)]
            # Sound: Ball Launch

        # --- 2. Update Ball & Collisions ---
        if self.ball_launched:
            reward += 0.1  # Reward for keeping ball in play
            self.ball.x += self.ball_vel[0]
            self.ball.y += self.ball_vel[1]

            # Wall collision
            if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
                self.ball_vel[0] *= -1
                self.ball.left = max(0, self.ball.left)
                self.ball.right = min(self.WIDTH, self.ball.right)
                # Sound: Wall Bounce
            if self.ball.top <= 0:
                self.ball_vel[1] *= -1
                self.ball.top = max(0, self.ball.top)
                # Sound: Wall Bounce

            # Paddle collision
            if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
                self._handle_paddle_collision()
                self.consecutive_hits = 0
                self.multiplier = 1
                # Sound: Paddle Hit

            # Brick collision
            collided_idx = self.ball.collidelist(self.bricks)
            if collided_idx != -1:
                brick_reward = self._handle_brick_collision(collided_idx, paddle_moved_safely)
                reward += brick_reward
                # Sound: Brick Break
            
            # Bottom wall (miss)
            if self.ball.top >= self.HEIGHT:
                self.lives -= 1
                self.ball_launched = False
                self._reset_ball_position()
                self.consecutive_hits = 0
                self.multiplier = 1
                # Sound: Lose Life
        else:
            self._reset_ball_position()

        # --- 3. Update Particles ---
        self._update_particles()

        # --- 4. Check Termination ---
        self.steps += 1
        win = len(self.bricks) == 0
        lose = self.lives <= 0
        timeout = self.steps >= self.MAX_STEPS
        terminated = win or lose or timeout

        if terminated:
            self.game_over = True
            if win:
                reward += 100
            if lose:
                reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _handle_paddle_collision(self):
        self.ball.bottom = self.paddle.top
        
        offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2.0)
        self.ball_vel[0] += offset * 2.5 # Give player control over angle
        self.ball_vel[1] *= -1

        # Normalize speed
        current_speed = math.hypot(*self.ball_vel)
        if current_speed == 0: return # Avoid division by zero
        speed_factor = self.ball_base_speed / current_speed
        self.ball_vel = [v * speed_factor for v in self.ball_vel]

    def _handle_brick_collision(self, idx, safe_play):
        brick_rect = self.bricks[idx]
        
        # Determine bounce direction
        # Simplified AABB collision response
        overlap = self.ball.clip(brick_rect)
        if overlap.width < overlap.height:
            self.ball_vel[0] *= -1
        else:
            self.ball_vel[1] *= -1
        
        # Score and Multiplier
        self.consecutive_hits += 1
        self.multiplier = 1 + self.consecutive_hits // 3
        
        brick_base_reward = 1.0
        brick_multiplier_reward = 5.0 * self.multiplier

        if safe_play:
            # Reduce reward for "safe" hits to encourage riskier plays
            brick_multiplier_reward *= 0.8
        
        self.score += 10 * self.multiplier

        # Create particles
        self._create_particles(brick_rect.center, self.brick_colors[idx])
        
        # Remove brick
        del self.bricks[idx]
        del self.brick_colors[idx]

        # Difficulty scaling
        self.bricks_destroyed_this_episode += 1
        if self.bricks_destroyed_this_episode % 10 == 0:
            self.ball_base_speed += 0.5

        return brick_base_reward + brick_multiplier_reward

    def _reset_ball_position(self):
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top - 2
        self.ball_vel = [0, 0]

    def _create_particles(self, pos, color):
        for _ in range(10):
            vel = [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)]
            lifetime = self.np_random.integers(10, 20)
            size = self.np_random.integers(2, 5)
            particle_rect = pygame.Rect(pos[0], pos[1], size, size)
            self.particles.append([particle_rect, vel, lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0].x += p[1][0]
            p[0].y += p[1][1]
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Bricks
        for i, brick in enumerate(self.bricks):
            pygame.draw.rect(self.screen, self.brick_colors[i], brick)
            pygame.draw.rect(self.screen, self.COLOR_BG, brick, 1)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        pygame.draw.line(self.screen, self.COLOR_PADDLE_HL, self.paddle.topleft, self.paddle.topright, 2)
        
        # Ball
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.5)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), glow_radius, self.COLOR_BALL_GLOW + (100,))
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), glow_radius, self.COLOR_BALL_GLOW + (100,))
        # Main ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p[2] / 20.0))))
            color = p[3] + (alpha,)
            temp_surf = pygame.Surface(p[0].size, pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, temp_surf.get_rect())
            self.screen.blit(temp_surf, p[0].topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        life_text = self.font_small.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(life_text, (self.WIDTH - 160, 15))
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 80 + i * 20, 22, 7, (255, 80, 80))
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 80 + i * 20, 22, 7, (255, 80, 80))
        
        # Multiplier
        if self.multiplier > 1:
            mult_text = self.font_multiplier.render(f"{self.multiplier}x", True, self.COLOR_TEXT)
            text_rect = mult_text.get_rect(center=(self.WIDTH / 2, 25))
            self.screen.blit(mult_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_left": len(self.bricks),
            "multiplier": self.multiplier,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Pygame setup for human play
    pygame.display.set_caption("Arcade Brick Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    running = True

    while running:
        movement = 0  # 0=none, 3=left, 4=right
        space_held = 0 # 0=released, 1=held

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        action = [movement, space_held, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(30) # Match the auto-advance rate

    env.close()