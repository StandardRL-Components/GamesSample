
# Generated: 2025-08-28T06:13:10.310136
# Source Brief: brief_05830.md
# Brief Index: 5830

        
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
        "Controls: ↑/↓ to move the paddle. Guide the red ball to the green zone on the right."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Use your paddle to deflect the ball upwards, navigating it to the goal before time runs out. Hitting the bottom wall ends the game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    UI_HEIGHT = 50

    GRID_W = 12
    GRID_H = 8
    
    MAX_TIME = 1200 # 20 seconds at 60 steps/sec equivalent

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (40, 50, 60)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_PADDLE_GLOW = (200, 200, 255, 50)
    COLOR_BALL = (255, 80, 80)
    COLOR_BALL_GLOW = (255, 100, 100, 80)
    COLOR_GOAL = (80, 200, 120)
    COLOR_GOAL_GLOW = (100, 220, 140, 40)
    COLOR_TRAJECTORY = (255, 255, 100, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_timer = pygame.font.SysFont("monospace", 24, bold=True)

        # Grid/Cell dimensions
        self.grid_area_height = self.SCREEN_HEIGHT - self.UI_HEIGHT
        self.cell_w = self.SCREEN_WIDTH / self.GRID_W
        self.cell_h = self.grid_area_height / self.GRID_H
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.paddle_y = 0
        self.ball_pos = [0, 0]
        self.ball_vy = 0
        self.time_left = 0
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_left = self.MAX_TIME
        
        self.paddle_y = self.np_random.integers(1, self.GRID_H - 2)
        self.ball_pos = [1, self.np_random.integers(2, self.GRID_H - 3)] # Start in col 1
        self.ball_vy = 1 # Start moving down
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- 1. Update Game Logic ---
        self.steps += 1
        self.time_left -= 1
        reward = -0.01  # Small penalty for each step to encourage speed

        # Handle paddle movement
        if movement == 1:  # Up
            self.paddle_y = max(0, self.paddle_y - 1)
        elif movement == 2:  # Down
            self.paddle_y = min(self.GRID_H - 1, self.paddle_y + 1)
        
        # --- 2. Update Ball Position ---
        # Horizontal movement
        self.ball_pos[0] += 1
        
        # Vertical movement
        self.ball_pos[1] += self.ball_vy
        
        # --- 3. Handle Collisions & Interactions ---
        
        # Paddle collision
        if self.ball_pos[0] == 1 and self.ball_pos[1] == self.paddle_y:
            # sfx: paddle_hit.wav
            self.ball_vy = -1 # Redirect upwards
            self.ball_pos[1] = self.paddle_y # Correct position
            reward += 1.0
        
        # Top wall collision
        if self.ball_pos[1] < 0:
            # sfx: wall_bounce.wav
            self.ball_pos[1] = 0
            self.ball_vy = 1 # Bounce down
            
        # --- 4. Check Termination Conditions ---
        terminated = False
        
        # Bottom wall collision (lose)
        if self.ball_pos[1] >= self.GRID_H:
            # sfx: lose_game.wav
            terminated = True
        
        # Reached end zone (win)
        if self.ball_pos[0] >= self.GRID_W - 1:
            # sfx: win_game.wav
            win_bonus = 100 * (self.time_left / self.MAX_TIME)
            reward += max(0, win_bonus) # Ensure bonus is non-negative
            terminated = True
            
        # Time ran out (lose)
        if self.time_left <= 0:
            terminated = True

        # Add penalty for being near the bottom
        if self.ball_pos[1] >= self.GRID_H - 2:
            reward -= 0.2

        self.game_over = terminated
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _grid_to_pixel(self, grid_x, grid_y):
        """Converts grid coordinates to pixel coordinates for the center of the cell."""
        px = grid_x * self.cell_w + self.cell_w / 2
        py = self.UI_HEIGHT + grid_y * self.cell_h + self.cell_h / 2
        return int(px), int(py)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw end zone
        goal_rect = pygame.Rect((self.GRID_W - 1) * self.cell_w, self.UI_HEIGHT, self.cell_w, self.grid_area_height)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, goal_rect)
        pygame.gfxdraw.box(self.screen, goal_rect, self.COLOR_GOAL_GLOW)
        
        # Draw grid lines
        for i in range(1, self.GRID_W):
            x = i * self.cell_w
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.UI_HEIGHT), (x, self.SCREEN_HEIGHT), 1)
        for i in range(1, self.GRID_H):
            y = self.UI_HEIGHT + i * self.cell_h
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

        # Draw trajectory preview
        self._render_trajectory()

        # Draw paddle
        paddle_px, paddle_py = self._grid_to_pixel(0, self.paddle_y)
        paddle_rect = pygame.Rect(
            paddle_px - self.cell_w / 4,
            paddle_py - self.cell_h / 2,
            self.cell_w / 2,
            self.cell_h
        )
        # Glow
        glow_rect = paddle_rect.inflate(12, 12)
        pygame.gfxdraw.box(self.screen, glow_rect, self.COLOR_PADDLE_GLOW)
        # Main body
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=4)
        
        # Draw ball
        ball_px, ball_py = self._grid_to_pixel(self.ball_pos[0], self.ball_pos[1])
        ball_radius = int(min(self.cell_w, self.cell_h) * 0.35)
        self._draw_glow_circle(self.screen, (ball_px, ball_py), ball_radius, self.COLOR_BALL, self.COLOR_BALL_GLOW)

    def _render_trajectory(self):
        sim_pos = list(self.ball_pos)
        sim_vy = self.ball_vy
        points = [self._grid_to_pixel(sim_pos[0], sim_pos[1])]
        
        for _ in range(2):
            if sim_pos[0] >= self.GRID_W - 1: break
            
            sim_pos[0] += 1
            sim_pos[1] += sim_vy
            
            # Simulate top wall bounce
            if sim_pos[1] < 0:
                sim_pos[1] = 0
                sim_vy = 1
            
            # Don't simulate past bottom wall
            if sim_pos[1] >= self.GRID_H:
                sim_pos[1] = self.GRID_H
                points.append(self._grid_to_pixel(sim_pos[0], sim_pos[1]))
                break
            
            points.append(self._grid_to_pixel(sim_pos[0], sim_pos[1]))

        if len(points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_TRAJECTORY, False, points)

    def _draw_glow_circle(self, surface, pos, radius, color, glow_color):
        # Glow
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius * 1.8), glow_color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius * 1.8), glow_color)
        # Main circle
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)
    
    def _render_ui(self):
        # UI Background
        pygame.draw.rect(self.screen, (0,0,0, 150), (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.UI_HEIGHT-1), (self.SCREEN_WIDTH, self.UI_HEIGHT-1), 1)

        # Score display
        score_text = f"SCORE: {self.score:.2f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        shadow_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (12, 16))
        self.screen.blit(score_surf, (10, 14))

        # Timer display
        time_sec = self.time_left / (self.MAX_TIME / 20.0)
        time_text = f"TIME: {max(0, time_sec):05.2f}"
        time_surf = self.font_timer.render(time_text, True, self.COLOR_TEXT)
        shadow_surf = self.font_timer.render(time_text, True, self.COLOR_TEXT_SHADOW)
        time_rect = time_surf.get_rect(topright=(self.SCREEN_WIDTH - 12, 12))
        shadow_rect = shadow_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 14))
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(time_surf, time_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "ball_pos": list(self.ball_pos),
            "paddle_y": self.paddle_y
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


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Paddle Maze")
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0

    # Game loop
    while not terminated:
        action = [0, 0, 0] # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        # We need to step the environment to advance the game state
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Since auto_advance is False, we control the step rate here
        clock.tick(15) # Play at 15 steps per second for a human

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()