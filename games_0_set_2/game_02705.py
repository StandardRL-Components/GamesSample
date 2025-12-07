
# Generated: 2025-08-28T05:40:48.319767
# Source Brief: brief_02705.md
# Brief Index: 2705

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to direct the snake. "
        "The snake cannot reverse its direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic arcade game. Guide the snake to eat the food, growing longer with each bite. "
        "Avoid hitting the walls or the snake's own body. Reach a score of 50 to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_W, SCREEN_H = 640, 400
    GRID_SIZE = 20
    GRID_W, GRID_H = SCREEN_W // GRID_SIZE, SCREEN_H // GRID_SIZE
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 45)
    COLOR_WALL = (80, 80, 100)
    COLOR_SNAKE_BODY = (40, 200, 40)
    COLOR_SNAKE_HEAD = (100, 255, 100)
    COLOR_FOOD = (255, 50, 50)
    COLOR_FOOD_GLOW = (255, 100, 100, 50)
    COLOR_TEXT = (220, 220, 220)
    
    MAX_STEPS = 1000
    WIN_SCORE = 50

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Initialize state variables to be defined in reset()
        self.snake = None
        self.direction = None
        self.food_pos = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_dist_to_food = 0
        
        # Initialize state
        self.reset()
        
        # self.validate_implementation() # Optional: call for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initial snake position and direction
        center_x, center_y = self.GRID_W // 2, self.GRID_H // 2
        self.snake = deque([(center_x - i, center_y) for i in range(3)])
        self.direction = (1, 0)  # Start moving right

        self._place_food()
        self.prev_dist_to_food = self._get_dist_to_food()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle Action and Update Direction
        self._handle_action(action)
        
        # 2. Move Snake
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        self.snake.appendleft(new_head)

        # 3. Calculate Reward & Check Termination
        reward = 0
        terminated = False
        
        # Check for food consumption
        if new_head == self.food_pos:
            self.score += 1
            reward = 10
            # # Sound effect placeholder:
            # print("PLAY_SOUND: EAT")
            if self.score >= self.WIN_SCORE:
                reward += 100
                terminated = True
                self.game_over = True
            else:
                self._place_food()
        else:
            self.snake.pop()  # Remove tail if no food eaten

        # Check for collisions (after moving and potentially growing)
        if not terminated and self._check_collision(new_head):
            reward = -10
            terminated = True
            self.game_over = True
            # # Sound effect placeholder:
            # print("PLAY_SOUND: CRASH")
        
        # Check for max steps
        self.steps += 1
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        # Distance-based reward (if no major event occurred)
        if reward == 0 and not terminated:
            current_dist = self._get_dist_to_food()
            if current_dist < self.prev_dist_to_food:
                reward = 0.1
            elif current_dist > self.prev_dist_to_food:
                reward = -0.1
            self.prev_dist_to_food = current_dist
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement = action[0]
        # action[1] (space) and action[2] (shift) are ignored as per brief
        
        new_direction = self.direction
        if movement == 1 and self.direction != (0, 1):  # Up
            new_direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1):  # Down
            new_direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):  # Left
            new_direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0):  # Right
            new_direction = (1, 0)
        # movement == 0 is a no-op, continue in the same direction
        
        self.direction = new_direction

    def _place_food(self):
        possible_locations = set((x, y) for x in range(self.GRID_W) for y in range(self.GRID_H))
        snake_locations = set(self.snake)
        valid_locations = list(possible_locations - snake_locations)
        if not valid_locations:
            # This should only happen if the snake fills the entire screen
            self.game_over = True
            self.food_pos = (-1, -1) # Place food off-screen
        else:
            self.food_pos = self.np_random.choice(valid_locations, axis=0)
            self.food_pos = tuple(self.food_pos)


    def _check_collision(self, head):
        # Wall collision
        if not (0 <= head[0] < self.GRID_W and 0 <= head[1] < self.GRID_H):
            return True
        # Self-collision
        if head in list(self.snake)[1:]:
            return True
        return False

    def _get_dist_to_food(self):
        head = self.snake[0]
        return abs(head[0] - self.food_pos[0]) + abs(head[1] - self.food_pos[1])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_W, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_H))
        for y in range(0, self.SCREEN_H, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_W, y))

        # Draw food with a glow
        food_rect = pygame.Rect(
            self.food_pos[0] * self.GRID_SIZE, 
            self.food_pos[1] * self.GRID_SIZE,
            self.GRID_SIZE, self.GRID_SIZE
        )
        glow_radius = int(self.GRID_SIZE * 0.75)
        pygame.gfxdraw.filled_circle(
            self.screen,
            food_rect.centerx,
            food_rect.centery,
            glow_radius,
            self.COLOR_FOOD_GLOW
        )
        pygame.draw.rect(self.screen, self.COLOR_FOOD, food_rect)

        # Draw snake
        for i, segment in enumerate(self.snake):
            rect = pygame.Rect(
                segment[0] * self.GRID_SIZE,
                segment[1] * self.GRID_SIZE,
                self.GRID_SIZE, self.GRID_SIZE
            )
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE_BODY
            pygame.draw.rect(self.screen, color, rect, border_radius=3)

    def _render_ui(self):
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake),
            "food_pos": self.food_pos,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(info, dict)
        assert info['score'] == 0
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        # Test mechanics
        self.reset()
        initial_len = len(self.snake)
        self.food_pos = (self.snake[0][0] + 1, self.snake[0][1]) # Place food in front
        self.step(self.action_space.sample()) # Action doesn't matter, will move right
        assert len(self.snake) == initial_len + 1, "Snake did not grow after eating"
        assert self.score == 1, "Score did not increment after eating"
        
        print("✓ Implementation validated successfully")