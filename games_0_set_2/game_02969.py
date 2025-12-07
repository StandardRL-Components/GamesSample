
# Generated: 2025-08-28T06:33:34.245744
# Source Brief: brief_02969.md
# Brief Index: 2969

        
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
        "Controls: ←→ to move the paddle. Try to clear all the bricks before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-inspired Breakout game. Use the paddle to bounce the ball and destroy all the bricks. Earn points for each brick and try to win before you run out of lives or time."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    UI_HEIGHT = 50
    
    # Colors (Synthwave/Retro Palette)
    COLOR_BG = (13, 15, 43)
    COLOR_GRID = (25, 28, 64)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_PADDLE_GLOW = (236, 64, 122)
    COLOR_BALL = (255, 236, 95)
    COLOR_BALL_GLOW = (255, 171, 145)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (236, 64, 122)
    BRICK_COLORS = [
        (255, 82, 82), (255, 145, 77), (255, 241, 118), 
        (38, 198, 218), (126, 87, 194)
    ]

    # Game Parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 16
    PADDLE_SPEED = 12
    BALL_RADIUS = 8
    BALL_SPEED = 6.0
    INITIAL_LIVES = 3
    TIME_LIMIT_SECONDS = 60
    FPS = 60 # Internal simulation rate

    # Brick Layout
    BRICK_ROWS = 5
    BRICK_COLS = 10
    BRICK_WIDTH = 58
    BRICK_HEIGHT = 20
    BRICK_H_SPACING = 4
    BRICK_V_SPACING = 4
    BRICK_AREA_TOP = UI_HEIGHT + 30

    class Particle:
        def __init__(self, x, y, color, lifetime, size, velocity):
            self.x = x
            self.y = y
            self.color = color
            self.lifetime = lifetime
            self.max_lifetime = lifetime
            self.size = size
            self.vx, self.vy = velocity

        def update(self):
            self.x += self.vx
            self.y += self.vy
            self.vy += 0.1 # Gravity
            self.lifetime -= 1
            return self.lifetime > 0

        def draw(self, surface):
            if self.lifetime > 0:
                alpha = int(255 * (self.lifetime / self.max_lifetime))
                temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, self.color + (alpha,), (self.size, self.size), self.size)
                surface.blit(temp_surf, (int(self.x - self.size), int(self.y - self.size)), special_flags=pygame.BLEND_RGBA_ADD)


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
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        self.paddle = None
        self.ball = None
        self.ball_vel = [0, 0]
        self.bricks = []
        self.particles = []
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.time_remaining = 0
        self.game_over_message = ""
        self.horizontal_bounce_counter = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS,
            self.paddle.top - self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )
        
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = [self.BALL_SPEED * math.cos(angle), self.BALL_SPEED * math.sin(angle)]

        self.bricks = []
        grid_width = self.BRICK_COLS * (self.BRICK_WIDTH + self.BRICK_H_SPACING) - self.BRICK_H_SPACING
        start_x = (self.SCREEN_WIDTH - grid_width) / 2
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                brick_x = start_x + j * (self.BRICK_WIDTH + self.BRICK_H_SPACING)
                brick_y = self.BRICK_AREA_TOP + i * (self.BRICK_HEIGHT + self.BRICK_V_SPACING)
                color = self.BRICK_COLORS[i % len(self.BRICK_COLORS)]
                self.bricks.append({"rect": pygame.Rect(brick_x, brick_y, self.BRICK_WIDTH, self.BRICK_HEIGHT), "color": color})
        
        self.particles = []
        self.lives = self.INITIAL_LIVES
        self.score = 0
        self.steps = 0
        self.time_remaining = self.TIME_LIMIT_SECONDS * self.FPS
        self.game_over = False
        self.game_over_message = ""
        self.horizontal_bounce_counter = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.001  # Small penalty for time passing
        
        self._handle_input(action)
        reward += self._update_game_state()
        
        self.steps += 1
        self.time_remaining -= 1

        terminated, terminal_reward, self.game_over_message = self._check_termination()
        if terminated:
            reward += terminal_reward
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

    def _update_game_state(self):
        # Update Ball Position
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

        # Update Particles
        self.particles = [p for p in self.particles if p.update()]
        
        # Handle Collisions
        return self._handle_collisions()

    def _handle_collisions(self):
        event_reward = 0

        # Ball vs. Walls
        if self.ball.left <= 0 or self.ball.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            self.ball.x = np.clip(self.ball.x, 0, self.SCREEN_WIDTH - self.ball.width)
            self._create_spark(self.ball.center, self.COLOR_BALL, 5)
            self.horizontal_bounce_counter += 1
        if self.ball.top <= self.UI_HEIGHT:
            self.ball_vel[1] *= -1
            self.ball.y = np.clip(self.ball.y, self.UI_HEIGHT, self.SCREEN_HEIGHT - self.ball.height)
            self._create_spark(self.ball.center, self.COLOR_BALL, 5)

        # Ball vs. Paddle
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball.bottom = self.paddle.top
            
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            offset = np.clip(offset, -0.9, 0.9) # Limit extreme angles
            
            new_vx = self.BALL_SPEED * offset
            new_vy = -math.sqrt(max(0.1, self.BALL_SPEED**2 - new_vx**2))
            self.ball_vel = [new_vx, new_vy]
            
            self._create_spark(self.ball.center, self.COLOR_PADDLE_GLOW, 10)
            self.horizontal_bounce_counter = 0 # Reset on paddle hit
            # Sound: paddle_hit.wav

        # Ball vs. Bricks
        hit_brick_idx = self.ball.collidelistall([b['rect'] for b in self.bricks])
        if hit_brick_idx:
            brick_data = self.bricks[hit_brick_idx[0]]
            brick = brick_data["rect"]
            
            self._create_explosion(brick.center, brick_data["color"])
            
            # Simple bounce logic: check overlap to determine bounce direction
            overlap_x = min(self.ball.right, brick.right) - max(self.ball.left, brick.left)
            overlap_y = min(self.ball.bottom, brick.bottom) - max(self.ball.top, brick.top)

            if overlap_x > overlap_y:
                self.ball_vel[1] *= -1
                self.ball.y += self.ball_vel[1] # Push out
            else:
                self.ball_vel[0] *= -1
                self.ball.x += self.ball_vel[0] # Push out

            del self.bricks[hit_brick_idx[0]]
            event_reward += 1.0
            self.score += 10
            self.horizontal_bounce_counter = 0 # Reset on brick hit
            # Sound: brick_destroy.wav

        # Ball vs. Bottom (Miss)
        if self.ball.top >= self.SCREEN_HEIGHT:
            self.lives -= 1
            event_reward -= 5.0
            self._create_explosion(self.paddle.center, self.COLOR_PADDLE_GLOW)
            if self.lives > 0:
                self._reset_ball()
            # Sound: lose_life.wav
            
        # Anti-softlock
        if self.horizontal_bounce_counter > 15:
            self.ball_vel[1] += self.np_random.uniform(-0.5, 0.5)
            self.ball_vel[0] *= 0.9 # Dampen horizontal
            self.horizontal_bounce_counter = 0

        return event_reward

    def _check_termination(self):
        if self.lives <= 0:
            return True, -50.0, "GAME OVER"
        if not self.bricks:
            return True, 50.0, "YOU WIN!"
        if self.time_remaining <= 0:
            return True, -20.0, "TIME UP"
        return False, 0.0, ""

    def _reset_ball(self):
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top - 5
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = [self.BALL_SPEED * math.cos(angle), self.BALL_SPEED * math.sin(angle)]

    def _create_explosion(self, position, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifetime = self.np_random.integers(15, 30)
            size = self.np_random.integers(2, 5)
            self.particles.append(self.Particle(position[0], position[1], color, lifetime, size, velocity))

    def _create_spark(self, position, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2)
            velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifetime = self.np_random.integers(10, 20)
            size = self.np_random.integers(1, 3)
            self.particles.append(self.Particle(position[0], position[1], color, lifetime, size, velocity))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_grid()
        self._render_bricks()
        for p in self.particles:
            p.draw(self.screen)
        self._render_paddle()
        if self.lives > 0:
            self._render_ball()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.UI_HEIGHT), (x, self.SCREEN_HEIGHT))
        for y in range(self.UI_HEIGHT, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_bricks(self):
        for brick_data in self.bricks:
            rect = brick_data["rect"]
            color = brick_data["color"]
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, 2) # Border

    def _render_paddle(self):
        # Glow
        glow_rect = self.paddle.inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PADDLE_GLOW + (100,), glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)
        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)
    
    def _render_ball(self):
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW + (100,))
        # Ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        pygame.draw.rect(self.screen, self.COLOR_BG, (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_PADDLE_GLOW, (0, self.UI_HEIGHT - 2), (self.SCREEN_WIDTH, self.UI_HEIGHT - 2), 2)

        # Score
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, (10, 15))

        # Time
        time_sec = self.time_remaining // self.FPS
        time_text = f"TIME: {time_sec}"
        self._draw_text(time_text, (self.SCREEN_WIDTH // 2 - 50, 15))

        # Lives
        lives_text = "LIVES:"
        self._draw_text(lives_text, (self.SCREEN_WIDTH - 200, 15))
        for i in range(self.lives):
            life_rect = pygame.Rect(self.SCREEN_WIDTH - 110 + i * 25, 18, 20, 8)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_rect, border_radius=2)

    def _draw_text(self, text, pos):
        text_surf = self.font_ui.render(text, True, self.COLOR_TEXT)
        shadow_surf = self.font_ui.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (pos[0] + 1, pos[1] + 1))
        self.screen.blit(text_surf, pos)

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
        self.screen.blit(s, (0, 0))

        text_surf = self.font_game_over.render(self.game_over_message, True, self.COLOR_TEXT)
        shadow_surf = self.font_game_over.render(self.game_over_message, True, self.COLOR_PADDLE_GLOW)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(shadow_surf, text_rect.move(3, 3))
        self.screen.blit(text_surf, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_remaining": len(self.bricks),
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

if __name__ == '__main__':
    # This block allows you to run the game and play it manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Breakout")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # No space or shift
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}. Info: {info}")
            # Wait for a moment then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()