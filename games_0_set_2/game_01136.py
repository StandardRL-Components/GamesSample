
# Generated: 2025-08-27T16:08:58.630772
# Source Brief: brief_01136.md
# Brief Index: 1136

        
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
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to change the snake's direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade snake game. Eat the red pellets to grow and increase your score. "
        "Reach a score of 100 to win, but avoid colliding with your own body or running out of time!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 10  # Snake moves 10 times per second
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE
        self.MAX_TIME_SECONDS = 60
        self.WIN_SCORE = 100

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_SNAKE_HEAD = (100, 255, 100)
        self.COLOR_SNAKE_BODY_1 = (0, 200, 0)
        self.COLOR_SNAKE_BODY_2 = (0, 150, 0)
        self.COLOR_FOOD = (255, 80, 80)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TIME_BAR = (100, 255, 100)
        self.COLOR_TIME_BAR_WARN = (255, 255, 100)
        self.COLOR_TIME_BAR_DANGER = (255, 100, 100)
        self.COLOR_OVERLAY = (0, 0, 0, 180) # RGBA

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
        self.font_large = pygame.font.Font(None, 72)
        self.font_small = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.snake_body = None
        self.snake_direction = None
        self.food_pos = None
        self.score = 0
        self.steps = 0
        self.time_remaining = 0
        self.game_over = False
        self.game_over_reason = ""
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_reason = ""
        self.time_remaining = self.FPS * self.MAX_TIME_SECONDS

        # Initialize snake in the center
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_body = deque([
            (start_x - 2, start_y),
            (start_x - 1, start_y),
            (start_x, start_y),
        ])
        self.snake_direction = (1, 0)  # Moving right

        # Place initial food
        self._place_food()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update direction based on action, preventing 180-degree turns
        new_direction = self.snake_direction
        if movement == 1 and self.snake_direction != (0, 1):  # Up
            new_direction = (0, -1)
        elif movement == 2 and self.snake_direction != (0, -1):  # Down
            new_direction = (0, 1)
        elif movement == 3 and self.snake_direction != (1, 0):  # Left
            new_direction = (-1, 0)
        elif movement == 4 and self.snake_direction != (-1, 0):  # Right
            new_direction = (1, 0)
        self.snake_direction = new_direction

        # Update game logic
        self.steps += 1
        self.time_remaining -= 1
        
        old_dist_to_food = self._manhattan_distance(self.snake_body[-1], self.food_pos)

        # Move snake
        head = self.snake_body[-1]
        new_head = (
            (head[0] + self.snake_direction[0]) % self.GRID_WIDTH,
            (head[1] + self.snake_direction[1]) % self.GRID_HEIGHT
        )

        reward = 0
        terminated = False

        # Check for self-collision
        # We check against the body excluding the very last segment, which will move
        if new_head in list(self.snake_body)[:-1]:
            # sfx: player_die
            self.game_over = True
            self.game_over_reason = "SELF-COLLISION"
            terminated = True
            reward = -50
        else:
            self.snake_body.append(new_head)
            
            # Check for food
            if new_head == self.food_pos:
                # sfx: eat_food
                self.score += 10
                reward += 10
                self._place_food()
                if self.score >= self.WIN_SCORE:
                    # sfx: win_game
                    self.game_over = True
                    self.game_over_reason = "YOU WIN!"
                    terminated = True
                    reward += 100 # Total reward for winning step is 110
            else:
                self.snake_body.popleft() # Move snake by removing tail
            
            # Distance-based reward
            new_dist_to_food = self._manhattan_distance(new_head, self.food_pos)
            if new_dist_to_food < old_dist_to_food:
                reward += 0.1
            else:
                reward -= 0.15 # Penalize moving away more

        # Check for time out
        if self.time_remaining <= 0 and not terminated:
            # sfx: time_out
            self.game_over = True
            self.game_over_reason = "TIME OUT"
            terminated = True
            reward = -50

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _place_food(self):
        while True:
            x = self.np_random.integers(0, self.GRID_WIDTH)
            y = self.np_random.integers(0, self.GRID_HEIGHT)
            if (x, y) not in self.snake_body:
                self.food_pos = (x, y)
                break

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid()
        self._render_food()
        self._render_snake()
        
        # Render UI overlay
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_grid(self):
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_snake(self):
        if not self.snake_body:
            return
        
        # Draw body segments
        num_segments = len(self.snake_body)
        for i, segment in enumerate(self.snake_body):
            rect = pygame.Rect(
                segment[0] * self.CELL_SIZE, 
                segment[1] * self.CELL_SIZE, 
                self.CELL_SIZE, 
                self.CELL_SIZE
            )
            
            # Gradient color for the body
            interp = i / max(1, num_segments - 1)
            color = (
                int(self.COLOR_SNAKE_BODY_2[0] + interp * (self.COLOR_SNAKE_BODY_1[0] - self.COLOR_SNAKE_BODY_2[0])),
                int(self.COLOR_SNAKE_BODY_2[1] + interp * (self.COLOR_SNAKE_BODY_1[1] - self.COLOR_SNAKE_BODY_2[1])),
                int(self.COLOR_SNAKE_BODY_2[2] + interp * (self.COLOR_SNAKE_BODY_1[2] - self.COLOR_SNAKE_BODY_2[2])),
            )
            pygame.draw.rect(self.screen, color, rect.inflate(-2, -2))

        # Draw head
        head = self.snake_body[-1]
        head_rect = pygame.Rect(
            head[0] * self.CELL_SIZE, 
            head[1] * self.CELL_SIZE, 
            self.CELL_SIZE, 
            self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, head_rect.inflate(-2, -2))

        # Draw eyes on head
        eye_size = 3
        dx, dy = self.snake_direction
        eye_pos1, eye_pos2 = None, None
        center_x = head_rect.centerx
        center_y = head_rect.centery
        if dx == 1: # Right
            eye_pos1 = (center_x + 4, center_y - 4)
            eye_pos2 = (center_x + 4, center_y + 4)
        elif dx == -1: # Left
            eye_pos1 = (center_x - 4, center_y - 4)
            eye_pos2 = (center_x - 4, center_y + 4)
        elif dy == 1: # Down
            eye_pos1 = (center_x - 4, center_y + 4)
            eye_pos2 = (center_x + 4, center_y + 4)
        elif dy == -1: # Up
            eye_pos1 = (center_x - 4, center_y - 4)
            eye_pos2 = (center_x + 4, center_y - 4)
        
        if eye_pos1 and eye_pos2:
            pygame.draw.circle(self.screen, (0,0,0), eye_pos1, eye_size)
            pygame.draw.circle(self.screen, (0,0,0), eye_pos2, eye_size)

    def _render_food(self):
        if not self.food_pos:
            return
        
        # Pulsing effect for the food
        pulse = (math.sin(self.steps * 0.5) + 1) / 2  # Varies between 0 and 1
        radius = int(self.CELL_SIZE * 0.3 + pulse * self.CELL_SIZE * 0.1)
        center_x = int(self.food_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
        center_y = int(self.food_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_FOOD)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_FOOD)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time bar
        time_ratio = self.time_remaining / (self.FPS * self.MAX_TIME_SECONDS)
        bar_width = 200
        bar_height = 20
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 15

        time_color = self.COLOR_TIME_BAR
        if time_ratio < 0.5: time_color = self.COLOR_TIME_BAR_WARN
        if time_ratio < 0.2: time_color = self.COLOR_TIME_BAR_DANGER
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, time_color, (bar_x, bar_y, int(bar_width * time_ratio), bar_height))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill(self.COLOR_OVERLAY)
        self.screen.blit(overlay, (0, 0))

        text_surface = self.font_large.render(self.game_over_reason, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text_surface, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": self.time_remaining // self.FPS
        }

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

    def close(self):
        pygame.quit()