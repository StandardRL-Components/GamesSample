
# Generated: 2025-08-28T04:35:14.442162
# Source Brief: brief_05298.md
# Brief Index: 5298

        
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

    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A retro arcade brick-breaker. Control the paddle to bounce the ball and destroy all the bricks. You have 3 lives."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PADDLE = (220, 220, 220)
    COLOR_BALL = (255, 255, 0)
    COLOR_UI_TEXT = (200, 200, 220)
    BRICK_COLORS = [
        (200, 70, 70), (200, 130, 70), (190, 190, 70), 
        (70, 200, 70), (70, 70, 200), (130, 70, 200)
    ]

    # Game parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 12
    PADDLE_SPEED = 8
    BALL_RADIUS = 6
    INITIAL_BALL_SPEED = 4.0
    BRICK_ROWS = 6
    BRICK_COLS = 15
    BRICK_WIDTH = SCREEN_WIDTH // BRICK_COLS
    BRICK_HEIGHT = 18
    BRICK_TOP_OFFSET = 50
    
    MAX_STEPS = 30 * 60 # 60 seconds at 30fps

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
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_launched = None
        self.ball_speed = None
        self.bricks = None
        self.brick_colors = None
        self.particles = None
        
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.destroyed_bricks_count = 0
        self.game_over_message = ""
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.destroyed_bricks_count = 0
        self.game_over_message = ""
        
        # Paddle
        paddle_y = self.SCREEN_HEIGHT - 30
        paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        # Ball
        self.ball_launched = False
        self.ball_speed = self.INITIAL_BALL_SPEED
        self._reset_ball()

        # Bricks
        self.bricks = []
        self.brick_colors = []
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                brick_x = j * self.BRICK_WIDTH
                brick_y = i * self.BRICK_HEIGHT + self.BRICK_TOP_OFFSET
                brick = pygame.Rect(brick_x, brick_y, self.BRICK_WIDTH, self.BRICK_HEIGHT)
                self.bricks.append(brick)
                self.brick_colors.append(self.BRICK_COLORS[i % len(self.BRICK_COLORS)])

        # Particles
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS,
            self.paddle.top - self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_launched = False

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = -0.001  # Small time penalty to encourage speed
        
        # --- Handle Input ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        if not self.ball_launched and space_held:
            # SFX: ball_launch
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
            self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.ball_speed

        # --- Update Game Logic ---
        self._update_particles()

        if self.ball_launched:
            self.ball.move_ip(self.ball_vel)
            
            # Wall collisions
            if self.ball.left <= 0 or self.ball.right >= self.SCREEN_WIDTH:
                self.ball_vel.x *= -1
                self.ball.left = np.clip(self.ball.left, 0, self.SCREEN_WIDTH - self.ball.width)
                self.ball.right = np.clip(self.ball.right, 0, self.SCREEN_WIDTH)
                # SFX: wall_bounce
            if self.ball.top <= 0:
                self.ball_vel.y *= -1
                self.ball.top = np.clip(self.ball.top, 0, self.SCREEN_HEIGHT - self.ball.height)
                # SFX: wall_bounce

            # Paddle collision
            if self.ball.colliderect(self.paddle) and self.ball_vel.y > 0:
                reward += 0.1
                # SFX: paddle_hit
                self.ball_vel.y *= -1
                self.ball.bottom = self.paddle.top
                
                # Add control based on hit location
                offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel.x += offset * 2.0
                self.ball_vel.normalize_ip()
                self.ball_vel *= self.ball_speed
                
                self._create_particles(self.ball.midbottom, self.COLOR_PADDLE, 10, 2)

            # Brick collisions
            collided_brick_index = self.ball.collidelist(self.bricks)
            if collided_brick_index != -1:
                reward += 1.0
                # SFX: brick_break
                brick = self.bricks.pop(collided_brick_index)
                color = self.brick_colors.pop(collided_brick_index)
                self.score += 10
                self.destroyed_bricks_count += 1
                
                self._create_particles(brick.center, color, 20, 3)

                # Determine bounce direction
                prev_ball_center = pygame.Vector2(self.ball.center) - self.ball_vel
                
                # Simplified overlap check
                dx = self.ball.centerx - brick.centerx
                dy = self.ball.centery - brick.centery
                
                if abs(dx / brick.width) > abs(dy / brick.height):
                    self.ball_vel.x *= -1
                else:
                    self.ball_vel.y *= -1
                
                # Difficulty scaling
                if self.destroyed_bricks_count in [25, 50]:
                    self.ball_speed += 1.0
                    self.ball_vel.normalize_ip()
                    self.ball_vel *= self.ball_speed

            # Missed ball
            if self.ball.top > self.SCREEN_HEIGHT:
                # SFX: lose_life
                self.lives -= 1
                reward -= 10.0
                self._reset_ball()
        else:
            # Ball follows paddle
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top

        self.steps += 1
        
        # --- Termination ---
        terminated = False
        if self.lives <= 0:
            terminated = True
            reward -= 100.0
            self.game_over_message = "GAME OVER"
        elif not self.bricks:
            terminated = True
            reward += 100.0
            self.score += 1000 # Victory bonus
            self.game_over_message = "YOU WIN!"
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw bricks
        for i, brick in enumerate(self.bricks):
            color = self.brick_colors[i]
            inner_rect = brick.inflate(-2, -2)
            pygame.draw.rect(self.screen, color, inner_rect, border_radius=2)
            
        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            p_pos = (int(p['pos'].x), int(p['pos'].y))
            p_size = int(p['size'] * (p['life'] / p['max_life']))
            if p_size > 0:
                p_color_alpha = p['color'] + (int(255 * (p['life'] / p['max_life'])),)
                temp_surf = pygame.Surface((p_size*2, p_size*2), pygame.SRCALPHA)
                pygame.draw.rect(temp_surf, p_color_alpha, (0, 0, p_size*2, p_size*2), border_radius=1)
                self.screen.blit(temp_surf, (p_pos[0] - p_size, p_pos[1] - p_size))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        for i in range(self.lives):
            x = self.SCREEN_WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            y = 10 + self.BALL_RADIUS
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        
        # Game Over message
        if self.game_over_message:
            msg_surf = self.font_large.render(self.game_over_message, True, self.COLOR_PADDLE)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_remaining": len(self.bricks),
        }

    def _create_particles(self, pos, color, count, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            max_life = self.np_random.integers(10, 20)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'color': color,
                'size': self.np_random.integers(2, 5),
                'life': max_life,
                'max_life': max_life
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It's not part of the Gymnasium environment but is useful for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Brick Breaker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated:
            # On game over, wait for a key press to reset
            keys = pygame.key.get_pressed()
            if any(keys):
                obs, info = env.reset()
                terminated = False
            continue

        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to Display ---
        # The observation is (H, W, C), but pygame surfaces are (W, H)
        # So we need to transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Tick the Clock ---
        clock.tick(30) # Run at 30 FPS

    env.close()