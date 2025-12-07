# Generated: 2025-08-27T14:54:03.327435
# Source Brief: brief_00826.md
# Brief Index: 826

        
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

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: ← to move left, → to move right."

    # Must be a short, user-facing description of the game:
    game_description = "Clear all the bricks within the time limit in this fast-paced, grid-based arcade game."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_AREA_INSET = 10
    MAX_STEPS = 2000  # Approx 67 seconds at 30 FPS

    # Colors
    COLOR_BG = (20, 20, 40)
    COLOR_GRID = (30, 30, 50)
    COLOR_BORDER = (100, 100, 120)
    COLOR_PADDLE = (220, 220, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    BRICK_COLORS = [
        (255, 50, 50), (255, 150, 50), (255, 255, 50),
        (50, 255, 50), (50, 150, 255), (150, 50, 255)
    ]

    # Game element properties
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 8
    BALL_SPEED_INITIAL = 6.0
    BRICK_ROWS = 3
    BRICKS_PER_ROW = 10
    BRICK_WIDTH = (SCREEN_WIDTH - 2 * GAME_AREA_INSET) / BRICKS_PER_ROW - 2
    BRICK_HEIGHT = 20

    # Rewards
    REWARD_BREAK_BRICK = 1.0
    REWARD_WIN = 50.0
    REWARD_LOSE = -50.0
    REWARD_PADDLE_MOVE = -0.02
    REWARD_SURVIVAL = 0.01 # Changed from 0.1 to keep rewards small


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
        self.font_large = pygame.font.SysFont('monospace', 30, bold=True)
        self.font_small = pygame.font.SysFont('monospace', 20, bold=True)

        self.game_area = pygame.Rect(
            self.GAME_AREA_INSET,
            self.GAME_AREA_INSET,
            self.SCREEN_WIDTH - 2 * self.GAME_AREA_INSET,
            self.SCREEN_HEIGHT - 2 * self.GAME_AREA_INSET
        )

        # State variables are initialized in reset()
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.bricks = None
        self.score = 0
        self.steps = 0
        self.total_bricks = 0
        self.ball_trail = None

        # self.reset() is called by the environment wrapper, no need to call it here.
        # self.validate_implementation() # This can be useful for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle = pygame.Rect(
            self.SCREEN_WIDTH / 2 - self.PADDLE_WIDTH / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 30,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS,
            self.paddle.top - self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )

        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upwards
        self.ball_vel = [
            self.BALL_SPEED_INITIAL * math.cos(angle),
            self.BALL_SPEED_INITIAL * math.sin(angle)
        ]

        self._create_bricks()
        self.total_bricks = len(self.bricks)

        self.steps = 0
        self.score = 0
        self.ball_trail = deque(maxlen=8)

        return self._get_observation(), self._get_info()

    def _create_bricks(self):
        self.bricks = []
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICKS_PER_ROW):
                brick_x = self.game_area.left + j * (self.BRICK_WIDTH + 2) + 1
                brick_y = self.game_area.top + 50 + i * (self.BRICK_HEIGHT + 2) + 1
                color = self.BRICK_COLORS[i % len(self.BRICK_COLORS)]
                self.bricks.append({
                    "rect": pygame.Rect(brick_x, brick_y, self.BRICK_WIDTH, self.BRICK_HEIGHT),
                    "color": color
                })

    def step(self, action):
        reward = self.REWARD_SURVIVAL
        
        # --- Handle Input ---
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            reward += self.REWARD_PADDLE_MOVE
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            reward += self.REWARD_PADDLE_MOVE

        self.paddle.left = max(self.game_area.left, self.paddle.left)
        self.paddle.right = min(self.game_area.right, self.paddle.right)

        # --- Update Game Logic ---
        self._move_ball()
        brick_reward = self._handle_collisions()
        reward += brick_reward
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        truncated = False
        if not self.bricks:
            terminated = True
            reward += self.REWARD_WIN
            # print("WIN!")
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            reward += self.REWARD_LOSE
            # print("LOSE - Time up!")

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _move_ball(self):
        self.ball_trail.append(self.ball.center)
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

    def _handle_collisions(self):
        # Walls
        if self.ball.left <= self.game_area.left or self.ball.right >= self.game_area.right:
            self.ball_vel[0] *= -1
            self.ball.left = max(self.ball.left, self.game_area.left)
            self.ball.right = min(self.ball.right, self.game_area.right)
            # sfx: wall_bounce

        if self.ball.top <= self.game_area.top:
            self.ball_vel[1] *= -1
            self.ball.top = max(self.ball.top, self.game_area.top)
            # sfx: wall_bounce
        
        # In this version, ball bounces off bottom wall too
        if self.ball.bottom >= self.game_area.bottom:
            self.ball_vel[1] *= -1
            self.ball.bottom = min(self.ball.bottom, self.game_area.bottom)
            # sfx: wall_bounce

        # Paddle
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball.bottom = self.paddle.top
            
            # Change angle based on where it hits the paddle
            dist_from_center = self.ball.centerx - self.paddle.centerx
            normalized_dist = dist_from_center / (self.PADDLE_WIDTH / 2)
            
            angle_offset = normalized_dist * (math.pi / 4) # Max 45 degree change
            current_speed = math.hypot(*self.ball_vel)
            
            new_angle = -math.atan2(self.ball_vel[1], self.ball_vel[0]) + angle_offset
            
            self.ball_vel[0] = current_speed * math.cos(new_angle)
            self.ball_vel[1] = -abs(current_speed * math.sin(new_angle)) # Ensure it always goes up
            # sfx: paddle_hit

        # Bricks
        brick_reward = 0
        for brick_data in self.bricks[:]:
            if self.ball.colliderect(brick_data["rect"]):
                self.bricks.remove(brick_data)
                self.score += 1
                brick_reward += self.REWARD_BREAK_BRICK

                # Simple bounce logic: reverse vertical velocity
                # This is common in breakout games and feels good.
                self.ball_vel[1] *= -1
                # sfx: brick_break
                break # Only break one brick per frame
        
        return brick_reward

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
        
        # Game border
        pygame.draw.rect(self.screen, self.COLOR_BORDER, self.game_area, 2)

        # Bricks
        for brick_data in self.bricks:
            pygame.draw.rect(self.screen, brick_data["color"], brick_data["rect"])
            pygame.draw.rect(self.screen, self.COLOR_BG, brick_data["rect"], 1)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Ball trail
        if self.ball_trail:
            for i, pos in enumerate(self.ball_trail):
                alpha = int(255 * (i / len(self.ball_trail)) * 0.5)
                # Use a surface with per-pixel alpha for transparency
                trail_surf = pygame.Surface((self.BALL_RADIUS * 2, self.BALL_RADIUS * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(trail_surf, self.BALL_RADIUS, self.BALL_RADIUS, self.BALL_RADIUS, (*self.COLOR_BALL, alpha))
                self.screen.blit(trail_surf, (int(pos[0]) - self.BALL_RADIUS, int(pos[1]) - self.BALL_RADIUS))

        # Ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BG)


    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.game_area.left + 10, 5))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 30
        time_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.game_area.right - time_text.get_width() - 10, 5))

        # Bricks remaining
        bricks_left = len(self.bricks)
        bricks_text = self.font_large.render(f"{bricks_left}", True, self.COLOR_TEXT)
        text_rect = bricks_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 18))
        self.screen.blit(bricks_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bricks_left": len(self.bricks)
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
        # We need to reset first to initialize everything
        self.reset()
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # For human play, we want a real display.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a real pygame screen for human play
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Brick Breaker")
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # Action buffer
    action = env.action_space.sample()
    action[0] = 0 # No movement initially
    
    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0 # No-op
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            terminated = False
            truncated = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Control the frame rate

    env.close()