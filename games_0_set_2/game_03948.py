
# Generated: 2025-08-28T00:54:51.425860
# Source Brief: brief_03948.md
# Brief Index: 3948

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to change the snake's direction. Your goal is to eat the red pellets."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic arcade game. Guide the snake to eat food and grow longer. Avoid hitting walls or your own tail. Eating food near your tail gives bonus points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 20
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100
        self.NUM_FOOD = 5

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_WALL = (70, 70, 150)
        self.COLOR_SNAKE = (50, 205, 50)
        self.COLOR_SNAKE_HEAD = (150, 255, 150)
        self.COLOR_FOOD = (255, 50, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_GAMEOVER = (255, 0, 0)
        self.COLOR_WIN = (255, 215, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_large = pygame.font.Font(None, 72)
        self.font_small = pygame.font.Font(None, 36)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.snake = collections.deque()
        self.direction = (1, 0) # Start moving right
        self.food = []

        # --- Initialize state and validate ---
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # Initialize snake in the center
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake = collections.deque([(start_x - i, start_y) for i in range(3)])
        self.direction = (1, 0) # (dx, dy)

        # Initialize food
        self.food = []
        for _ in range(self.NUM_FOOD):
            self._spawn_food()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        
        # --- Update Direction ---
        current_direction = self.direction
        if movement == 1 and current_direction != (0, 1): # Up
            self.direction = (0, -1)
        elif movement == 2 and current_direction != (0, -1): # Down
            self.direction = (0, 1)
        elif movement == 3 and current_direction != (1, 0): # Left
            self.direction = (-1, 0)
        elif movement == 4 and current_direction != (-1, 0): # Right
            self.direction = (1, 0)
        # Action 0 (no-op) means continue in the current direction

        # --- Calculate reward for moving towards/away from food ---
        head_pos = self.snake[0]
        dist_before, _ = self._find_closest_food(head_pos)
        
        # --- Move Snake ---
        new_head = (head_pos[0] + self.direction[0], head_pos[1] + self.direction[1])
        self.snake.appendleft(new_head)
        
        dist_after, _ = self._find_closest_food(new_head)
        
        if dist_after < dist_before:
            reward += 0.01 # Small reward for getting closer
        elif dist_after > dist_before:
            reward -= 0.02 # Small penalty for moving away

        # --- Check for Food Consumption ---
        eaten_food_index = -1
        for i, food_pos in enumerate(self.food):
            if new_head == food_pos:
                eaten_food_index = i
                break
        
        if eaten_food_index != -1:
            # Snake eats food
            eaten_pos = self.food.pop(eaten_food_index)
            # Check if food was "risky" (adjacent to body, not including new head or old head)
            if self._is_risky_food(eaten_pos, collections.deque(list(self.snake)[2:])):
                reward += 2.0
                self.score += 2
            else:
                reward += 1.0
                self.score += 1
            
            # Snake grows, so we don't pop the tail
            self._spawn_food()
            # Placeholder comment for sound effect
            # play_sound("eat")
        else:
            # No food eaten, snake moves, so pop the tail
            self.snake.pop()

        # --- Check for Collisions and Termination ---
        self.steps += 1
        reward += 0.001 # Small survival reward per step
        
        # Wall collision
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            self.game_over = True
            reward = -1.0 # Terminal penalty
            # play_sound("crash")
        
        # Self collision
        elif new_head in collections.deque(list(self.snake)[1:]):
            self.game_over = True
            reward = -1.0 # Terminal penalty
            # play_sound("crash")

        # Win condition
        if self.score >= self.WIN_SCORE:
            self.win = True
            self.game_over = True
            reward += 10.0 # Win bonus
            # play_sound("win")

        # Max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw walls (visual only, collision is grid-based)
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), self.GRID_SIZE // 4)

        # Draw food
        for fx, fy in self.food:
            rect = pygame.Rect(fx * self.GRID_SIZE, fy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            center_x, center_y = rect.center
            radius = int(self.GRID_SIZE * 0.4)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_FOOD)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_FOOD)

        # Draw snake
        for i, (sx, sy) in enumerate(self.snake):
            rect = pygame.Rect(sx * self.GRID_SIZE, sy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE
            
            # Create a subtle gradient effect
            inner_rect = rect.inflate(-self.GRID_SIZE * 0.2, -self.GRID_SIZE * 0.2)
            pygame.draw.rect(self.screen, color, inner_rect, border_radius=3)
            
            # Add an eye to the head
            if i == 0:
                eye_offset_x = self.direction[0] * self.GRID_SIZE * 0.25
                eye_offset_y = self.direction[1] * self.GRID_SIZE * 0.25
                eye_pos = (int(rect.centerx + eye_offset_x), int(rect.centery + eye_offset_y))
                pygame.draw.circle(self.screen, self.COLOR_BG, eye_pos, self.GRID_SIZE // 8)

    def _render_ui(self):
        # Render score
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Render game over/win message
        if self.game_over:
            if self.win:
                message = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                message = "GAME OVER"
                color = self.COLOR_GAMEOVER
            
            text_surface = self.font_large.render(message, True, color)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(40, 40)
            bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surface.fill((0, 0, 0, 150))
            self.screen.blit(bg_surface, bg_rect.topleft)
            
            self.screen.blit(text_surface, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake),
        }

    def _spawn_food(self):
        occupied_cells = set(self.snake) | set(self.food)
        while True:
            x = self.np_random.integers(0, self.GRID_WIDTH)
            y = self.np_random.integers(0, self.GRID_HEIGHT)
            if (x, y) not in occupied_cells:
                self.food.append((x, y))
                break

    def _is_risky_food(self, food_pos, snake_body):
        fx, fy = food_pos
        for sx, sy in snake_body:
            if abs(fx - sx) + abs(fy - sy) == 1: # Manhattan distance is 1
                return True
        return False

    def _find_closest_food(self, pos):
        if not self.food:
            return float('inf'), None
        
        min_dist = float('inf')
        closest_food = None
        
        for food_pos in self.food:
            dist = abs(pos[0] - food_pos[0]) + abs(pos[1] - food_pos[1])
            if dist < min_dist:
                min_dist = dist
                closest_food = food_pos
                
        return min_dist, closest_food

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