# Generated: 2025-08-28T02:29:17.166815
# Source Brief: brief_04462.md
# Brief Index: 4462

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


# Set Pygame to run in headless mode, which is required for the environment
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ← to move the paddle left, → to move right."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist arcade game. Survive the onslaught of a bouncing ball for 60 seconds using a simple paddle."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.TIME_LIMIT_SECONDS = 60
        self.TIME_LIMIT_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (0, 0, 0)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TRAIL_START = (200, 200, 200)

        # Game element properties
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 8  # Pixels per frame
        self.BALL_RADIUS = 8
        self.BALL_TRAIL_LENGTH = 5

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
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24)
            self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            # Fallback font if Consolas is not available
            self.font_ui = pygame.font.SysFont(None, 30)
            self.font_game_over = pygame.font.SysFont(None, 60)

        # --- State Variables ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.paddle_pos = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed_base = 0
        self.ball_trail = None
        
        # Initialize state variables
        # self.reset() is not called here to avoid issues with validation
        # during subclassing, but we ensure state is set.
        
        # Run self-check
        # self.validate_implementation() # This is removed as it's a helper not part of the core env
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.paddle_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - self.PADDLE_HEIGHT * 2)
        
        self.ball_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 3)
        
        # Set a random initial angle, avoiding purely vertical/horizontal paths
        angle = self.np_random.uniform(0.4, 2.7) + self.np_random.choice([0, math.pi])
        self.ball_speed_base = self.np_random.uniform(4.0, 7.0)
        self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.ball_speed_base

        self.ball_trail = deque(maxlen=self.BALL_TRAIL_LENGTH)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Reward Initialization ---
        reward = 0.1  # Continuous reward for surviving
        
        # --- Action Handling ---
        movement = action[0]
        
        moved = False
        if movement in [3, 4]:
            moved = True
            is_moving_left = movement == 3
            ball_is_to_left = self.ball_pos.x < self.paddle_pos.x

            # Penalize moving away from the ball
            if (is_moving_left and not ball_is_to_left) or (not is_moving_left and ball_is_to_left):
                reward -= 2.0

        if movement == 3:  # Left
            self.paddle_pos.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle_pos.x += self.PADDLE_SPEED
            
        # Clamp paddle to screen bounds
        self.paddle_pos.x = max(self.PADDLE_WIDTH / 2, min(self.WIDTH - self.PADDLE_WIDTH / 2, self.paddle_pos.x))

        # --- Game Logic Update ---
        self.steps += 1
        
        # FIX: pygame.Vector2 does not have a .copy() method.
        # Create a new Vector2 instance to copy it.
        self.ball_trail.append(pygame.Vector2(self.ball_pos))
        
        # Increase ball speed every 10 seconds (600 steps)
        difficulty_multiplier = 1.0 + (self.steps // 600) * 0.1
        current_speed = self.ball_vel.length()
        if current_speed > 0:
            self.ball_vel.scale_to_length(self.ball_speed_base * difficulty_multiplier)
        
        self.ball_pos += self.ball_vel

        # --- Collisions ---
        # Left/Right walls
        if self.ball_pos.x <= self.BALL_RADIUS:
            self.ball_vel.x *= -1
            self.ball_pos.x = self.BALL_RADIUS
            # // sfx: wall_bounce
        elif self.ball_pos.x >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel.x *= -1
            self.ball_pos.x = self.WIDTH - self.BALL_RADIUS
            # // sfx: wall_bounce

        # Top wall
        if self.ball_pos.y <= self.BALL_RADIUS:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            # // sfx: wall_bounce

        # Paddle collision
        paddle_rect = pygame.Rect(self.paddle_pos.x - self.PADDLE_WIDTH / 2, self.paddle_pos.y - self.PADDLE_HEIGHT / 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        if self.ball_vel.y > 0 and paddle_rect.collidepoint(self.ball_pos):
            self.ball_vel.y *= -1
            self.ball_pos.y = paddle_rect.top - self.BALL_RADIUS

            # Add "spin" based on where the ball hits the paddle for better game feel
            hit_offset = (self.ball_pos.x - self.paddle_pos.x) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += hit_offset * 2.5
            
            self.score += 1
            reward += 1.0
            # // sfx: paddle_hit

        # --- Termination Check ---
        terminated = False
        
        # Loss condition: ball hits bottom
        if self.ball_pos.y >= self.HEIGHT - self.BALL_RADIUS:
            self.game_over = True
            terminated = True
            reward -= 100
            # // sfx: game_over
            
        # Win condition: time runs out
        if not self.game_over and self.steps >= self.TIME_LIMIT_STEPS:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100
            # // sfx: game_win

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_observation(self):
        # On the first call to reset, some state variables might be None
        if self.paddle_pos is None:
            self.reset()
            
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render trail with fading alpha
        for i, pos in enumerate(self.ball_trail):
            alpha = int(150 * (i + 1) / (self.BALL_TRAIL_LENGTH + 1))
            color = (*self.COLOR_TRAIL_START[:3], alpha)
            radius = int(self.BALL_RADIUS * 0.8 * (i + 1) / (self.BALL_TRAIL_LENGTH + 1))
            if radius > 0:
                self._draw_transparent_circle(int(pos.x), int(pos.y), radius, color)

        # Render ball with anti-aliasing
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

        # Render paddle
        paddle_rect = (
            int(self.paddle_pos.x - self.PADDLE_WIDTH / 2),
            int(self.paddle_pos.y - self.PADDLE_HEIGHT / 2),
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)

    def _draw_transparent_circle(self, x, y, radius, color):
        temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, color, (radius, radius), radius)
        self.screen.blit(temp_surf, (x - radius, y - radius))

    def _render_ui(self):
        # Render score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render timer
        remaining_seconds = max(0, (self.TIME_LIMIT_STEPS - self.steps) // self.FPS)
        time_str = f"{remaining_seconds // 60:02d}:{remaining_seconds % 60:02d}"
        time_text = self.font_ui.render(f"TIME: {time_str}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            msg_text = self.font_game_over.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()
        super().close()

    def render(self):
        # This is the only required render mode for the environment
        return self._get_observation()

if __name__ == "__main__":
    # To play the game manually, you can run this file.
    # This requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping keyboard keys to MultiDiscrete actions
    key_to_action = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Setup for human playback
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    while not done:
        # --- Human Input ---
        movement_action = 0 # Default is no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        
        action = [movement_action, 0, 0] # Space and Shift are not used

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already the rendered screen, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)
        
    env.close()
    print(f"Game Over! Final Score: {info['score']}")