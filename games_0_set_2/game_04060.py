
# Generated: 2025-08-28T01:16:14.016392
# Source Brief: brief_04060.md
# Brief Index: 4060

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to change the snake's direction. "
        "Try to eat the red food to grow and increase your score."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic arcade game. Guide the growing snake to eat food while avoiding "
        "collisions with the walls and its own body. Reach a score of 50 to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_DIM = 12
    CELL_SIZE = 30
    GRID_WIDTH = GRID_DIM * CELL_SIZE
    GRID_HEIGHT = GRID_DIM * CELL_SIZE
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_SNAKE_BODY = (40, 200, 40)
    COLOR_SNAKE_HEAD = (100, 255, 100)
    COLOR_FOOD = (220, 50, 50)
    COLOR_TEXT = (255, 255, 255)
    COLOR_GAMEOVER_BG = (0, 0, 0, 180)
    
    TARGET_SCORE = 50
    MAX_STEPS = 1000

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
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_gameover = pygame.font.SysFont("monospace", 48, bold=True)
        
        # State variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.snake_pos = None
        self.snake_dir = None
        self.food_pos = None
        self.last_reward = 0
        self.win_condition = False

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        
        # Initialize snake in the middle
        start_x, start_y = self.GRID_DIM // 2, self.GRID_DIM // 2
        self.snake_pos = deque([[start_x, start_y], [start_x - 1, start_y], [start_x - 2, start_y]])
        self.snake_dir = (1, 0)  # Start moving right
        
        self._place_food()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Determine new direction, preventing 180-degree turns
        current_dx, current_dy = self.snake_dir
        if movement == 1 and current_dy == 0:  # Up
            self.snake_dir = (0, -1)
        elif movement == 2 and current_dy == 0:  # Down
            self.snake_dir = (0, 1)
        elif movement == 3 and current_dx == 0:  # Left
            self.snake_dir = (-1, 0)
        elif movement == 4 and current_dx == 0:  # Right
            self.snake_dir = (1, 0)
        # If movement is 0 (no-op) or an invalid turn, continue in the current direction.

        # Calculate new head position
        head_x, head_y = self.snake_pos[0]
        dx, dy = self.snake_dir
        new_head = [head_x + dx, head_y + dy]

        # Initialize reward and termination
        reward = 0.1  # Survival reward for each step
        terminated = False

        # Check for collisions
        # 1. Wall collision
        if not (0 <= new_head[0] < self.GRID_DIM and 0 <= new_head[1] < self.GRID_DIM):
            reward = -100
            terminated = True
            # sfx: wall_thud.wav
        # 2. Self collision
        elif new_head in list(self.snake_pos):
            reward = -100
            terminated = True
            # sfx: self_bite.wav

        if terminated:
            self.game_over = True
        else:
            # Move snake
            self.snake_pos.appendleft(new_head)
            
            # Check for food consumption
            if new_head == self.food_pos:
                self.score += 10
                reward = 10
                # sfx: eat_food.wav
                if self.score >= self.TARGET_SCORE:
                    reward = 100
                    terminated = True
                    self.game_over = True
                    self.win_condition = True
                    # sfx: win_jingle.wav
                else:
                    self._place_food()
            else:
                # If no food is eaten, remove the tail
                self.snake_pos.pop()

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True
            reward = -10 # Penalty for running out of time

        self.last_reward = reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _place_food(self):
        """Finds a random empty cell and places the food there."""
        possible_positions = []
        snake_set = {tuple(pos) for pos in self.snake_pos}
        for x in range(self.GRID_DIM):
            for y in range(self.GRID_DIM):
                if (x, y) not in snake_set:
                    possible_positions.append([x, y])
        
        if possible_positions:
            self.food_pos = random.choice(possible_positions)
        else:
            # This case (no empty space) should only happen if the snake fills the whole board
            self.game_over = True
            self.win_condition = True


    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_DIM + 1):
            start_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.GRID_DIM + 1):
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_WIDTH, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
            
        # Draw food
        if self.food_pos:
            food_rect = pygame.Rect(
                self.GRID_OFFSET_X + self.food_pos[0] * self.CELL_SIZE,
                self.GRID_OFFSET_Y + self.food_pos[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            pygame.draw.ellipse(self.screen, self.COLOR_FOOD, food_rect.inflate(-4, -4))

        # Draw snake
        if self.snake_pos:
            # Body
            for i, pos in enumerate(list(self.snake_pos)[1:]):
                body_rect = pygame.Rect(
                    self.GRID_OFFSET_X + pos[0] * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + pos[1] * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_SNAKE_BODY, body_rect.inflate(-2, -2), border_radius=4)
            # Head
            head_pos = self.snake_pos[0]
            head_rect = pygame.Rect(
                self.GRID_OFFSET_X + head_pos[0] * self.CELL_SIZE,
                self.GRID_OFFSET_Y + head_pos[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, head_rect.inflate(-2, -2), border_radius=6)

    def _render_ui(self):
        # Render score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Render step count
        steps_text = self.font_main.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        steps_rect = steps_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(steps_text, steps_rect)
        
        # Render Game Over/Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_GAMEOVER_BG)
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win_condition else "GAME OVER"
            color = (100, 255, 100) if self.win_condition else (255, 100, 100)
            
            gameover_text = self.font_gameover.render(message, True, color)
            text_rect = gameover_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(gameover_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_pos) if self.snake_pos else 0,
            "food_pos": self.food_pos if self.food_pos else [-1, -1]
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