
# Generated: 2025-08-27T21:45:28.175900
# Source Brief: brief_01541.md
# Brief Index: 1541

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to change the snake's direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade classic! Guide the ever-growing snake to eat food, but be careful not to crash into the walls or its own tail."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 20
    GRID_HEIGHT = 15
    CELL_SIZE = 26
    MAX_STEPS = 1000
    WIN_SCORE = 50

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_SNAKE_BODY = (0, 180, 0)
    COLOR_SNAKE_HEAD = (100, 255, 100)
    COLOR_FOOD = (220, 50, 50)
    COLOR_FOOD_HL = (255, 150, 150)
    COLOR_WALL = (100, 100, 120)
    COLOR_TEXT = (255, 255, 255)
    
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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Grid rendering offsets for centering
        self.grid_render_width = self.GRID_WIDTH * self.CELL_SIZE
        self.grid_render_height = self.GRID_HEIGHT * self.CELL_SIZE
        self.x_offset = (self.SCREEN_WIDTH - self.grid_render_width) // 2
        self.y_offset = (self.SCREEN_HEIGHT - self.grid_render_height) // 2
        
        # Initialize state variables (will be properly set in reset)
        self.snake_body = collections.deque()
        self.direction = (0, 0)
        self.food_pos = (0, 0)
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_body = collections.deque([
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y)
        ])
        self.direction = (1, 0)  # Start moving right

        self._place_food()

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Determine new direction, preventing reversal
        new_direction = self.direction
        if movement == 1 and self.direction != (0, 1):    # UP
            new_direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1): # DOWN
            new_direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):  # LEFT
            new_direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0): # RIGHT
            new_direction = (1, 0)
        # If movement is 0 (no-op), keep current direction
        self.direction = new_direction

        # Calculate new head position
        head = self.snake_body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Initialize reward and termination status
        reward = -0.01  # Small penalty per step to encourage efficiency
        terminated = False

        # Check for wall collision
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            terminated = True
            reward = -100
            # SFX: Wall bump/crash
        # Check for self collision
        elif new_head in self.snake_body:
            terminated = True
            reward = -100
            # SFX: Self-bite/error sound

        if terminated:
            self.game_over = True
        else:
            # Move snake
            self.snake_body.appendleft(new_head)

            # Check for food consumption
            if new_head == self.food_pos:
                self.score += 10
                reward = 10
                # SFX: Food eat/collect
                self._place_food()
                # Don't pop tail, snake grows
            else:
                self.snake_body.pop()

        self.steps += 1

        # Check for win condition
        if not terminated and self.score >= self.WIN_SCORE:
            terminated = True
            reward = 100
            self.game_over = True
            # SFX: Win jingle

        # Check for max steps
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _place_food(self):
        all_cells = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))
        snake_cells = set(self.snake_body)
        empty_cells = list(all_cells - snake_cells)
        
        if not empty_cells:
            # Snake has filled the screen, a rare win/draw condition
            self.game_over = True
            self.food_pos = (-1, -1) # Place off-screen
        else:
            idx = self.np_random.integers(0, len(empty_cells))
            self.food_pos = empty_cells[idx]
    
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
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Draw grid border
        pygame.draw.rect(
            self.screen, self.COLOR_WALL, 
            (self.x_offset - 2, self.y_offset - 2, self.grid_render_width + 4, self.grid_render_height + 4), 2, border_radius=5
        )

        # Draw grid lines for a retro feel
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(
                    self.x_offset + x * self.CELL_SIZE, self.y_offset + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw food with a highlight for better visibility
        food_rect = pygame.Rect(
            self.x_offset + self.food_pos[0] * self.CELL_SIZE, self.y_offset + self.food_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_FOOD, food_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_FOOD_HL, food_rect.inflate(-self.CELL_SIZE*0.6, -self.CELL_SIZE*0.6), border_radius=4)

        # Draw snake with a border effect for better definition
        for i, segment in enumerate(self.snake_body):
            rect = pygame.Rect(
                self.x_offset + segment[0] * self.CELL_SIZE, self.y_offset + segment[1] * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE_BODY
            
            # Draw a slightly larger, darker background for a border effect
            border_rect = rect.inflate(2, 2)
            pygame.draw.rect(self.screen, (0, 50, 0), border_rect, border_radius=5)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)

    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        text_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (15, 10))

        if self.game_over:
            outcome_text = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            outcome_color = self.COLOR_SNAKE_HEAD if self.score >= self.WIN_SCORE else self.COLOR_FOOD
            
            outcome_surface = self.font.render(outcome_text, True, outcome_color)
            text_rect = outcome_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 150))
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(outcome_surface, text_rect)

    def close(self):
        pygame.font.quit()
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")