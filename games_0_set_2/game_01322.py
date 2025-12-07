
# Generated: 2025-08-27T16:45:38.499864
# Source Brief: brief_01322.md
# Brief Index: 1322

        
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
        "Controls: Arrow keys to change direction. Don't hit the walls or yourself!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Classic arcade snake. Eat the red food to grow longer. Survive as long as you can and aim for a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.GRID_WIDTH = 20
        self.GRID_HEIGHT = 20
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
        # Centering the 400x400 grid in the 640x400 window
        self.CELL_SIZE = self.SCREEN_HEIGHT // self.GRID_HEIGHT
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - (self.GRID_WIDTH * self.CELL_SIZE)) // 2
        
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_SNAKE = (40, 220, 110)
        self.COLOR_SNAKE_HEAD = (150, 255, 180)
        self.COLOR_FOOD = (255, 80, 80)
        self.COLOR_DANGER = (255, 255, 0, 150) # With alpha for glow
        self.COLOR_UI_TEXT = (240, 240, 240)
        
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
        self.font_ui = pygame.font.Font(None, 28)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snake_body = None
        self.direction = None
        self.food_pos = None
        self.pending_growth = 0
        self.danger_flash_pos = None

        # Initialize state variables
        self.reset()

        # Run validation check
        # self.validate_implementation() # Commented out for submission as per instructions
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.danger_flash_pos = None
        self.pending_growth = 0

        # Initial snake position and direction
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_body = deque([
            (start_x, start_y),
            (start_x, start_y + 1),
            (start_x, start_y + 2)
        ])
        self.direction = (0, -1)  # Start moving UP

        self._spawn_food()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self._update_direction(movement)
        
        # Predict next head position
        head_x, head_y = self.snake_body[0]
        dx, dy = self.direction
        next_head_pos = (head_x + dx, head_y + dy)

        # Check for termination conditions before moving
        collided_wall = not (0 <= next_head_pos[0] < self.GRID_WIDTH and 0 <= next_head_pos[1] < self.GRID_HEIGHT)
        collided_self = next_head_pos in list(self.snake_body)

        if collided_wall or collided_self:
            self.game_over = True
            self.danger_flash_pos = next_head_pos # Set flash position for rendering
            # Sound effect placeholder: # sfx_death.play()
            return self._get_observation(), -1.0, True, False, self._get_info()
        
        # Move snake
        self.snake_body.appendleft(next_head_pos)
        
        # Check for food consumption
        reward = -0.02 # Small penalty for each step to encourage efficiency

        if next_head_pos == self.food_pos:
            self.pending_growth += 1
            self.score += 1
            # Sound effect placeholder: # sfx_eat.play()

            # Check for risky move reward
            is_risky = False
            for neighbor in [(next_head_pos[0]+dx, next_head_pos[1]+dy) for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]]:
                if neighbor in list(self.snake_body)[2:]: # Check against body, excluding new head and neck
                    is_risky = True
                    break
            
            if is_risky:
                reward += 2.0
            else:
                reward += 1.0

            if self.score >= self.WIN_SCORE:
                self.game_over = True
                reward += 100.0 # Win bonus
                # Sound effect placeholder: # sfx_win.play()
            else:
                self._spawn_food()
        
        # Handle snake growth
        if self.pending_growth > 0:
            self.pending_growth -= 1
        else:
            self.snake_body.pop()

        self.steps += 1
        
        # Check for max steps termination
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated always False
            self._get_info()
        )

    def _update_direction(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1 and self.direction != (0, 1): # Up
            self.direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1): # Down
            self.direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0): # Left
            self.direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0): # Right
            self.direction = (1, 0)
        # If movement is 0 (no-op), direction remains unchanged

    def _spawn_food(self):
        available_spots = []
        snake_set = set(self.snake_body)
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in snake_set:
                    available_spots.append((x, y))
        
        if not available_spots:
            # No space left, this is effectively a win condition or a draw
            self.game_over = True
        else:
            self.food_pos = self.np_random.choice(available_spots)
            # Convert numpy tuple to standard tuple if necessary
            self.food_pos = (int(self.food_pos[0]), int(self.food_pos[1]))

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, py))

        # Draw food
        if self.food_pos:
            food_rect = pygame.Rect(
                self.GRID_OFFSET_X + self.food_pos[0] * self.CELL_SIZE,
                self.food_pos[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            pygame.gfxdraw.filled_circle(
                self.screen,
                food_rect.centerx,
                food_rect.centery,
                int(self.CELL_SIZE * 0.45),
                self.COLOR_FOOD
            )
            pygame.gfxdraw.aacircle(
                self.screen,
                food_rect.centerx,
                food_rect.centery,
                int(self.CELL_SIZE * 0.45),
                self.COLOR_FOOD
            )

        # Draw snake
        if self.snake_body:
            # Body
            for i, segment in enumerate(list(self.snake_body)[1:]):
                segment_rect = pygame.Rect(
                    self.GRID_OFFSET_X + segment[0] * self.CELL_SIZE,
                    segment[1] * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_SNAKE, segment_rect.inflate(-2, -2), border_radius=4)
            
            # Head
            head_pos = self.snake_body[0]
            head_rect = pygame.Rect(
                self.GRID_OFFSET_X + head_pos[0] * self.CELL_SIZE,
                head_pos[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, head_rect.inflate(-2, -2), border_radius=4)

        # Draw danger flash
        if self.danger_flash_pos:
            flash_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill(self.COLOR_DANGER)
            flash_rect = pygame.Rect(
                self.GRID_OFFSET_X + self.danger_flash_pos[0] * self.CELL_SIZE,
                self.danger_flash_pos[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            self.screen.blit(flash_surface, flash_rect.topleft)
            self.danger_flash_pos = None # Only show for one frame

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            end_text_str = "GAME OVER"
            if self.score >= self.WIN_SCORE:
                end_text_str = "YOU WIN!"
            
            end_font = pygame.font.Font(None, 60)
            end_text = end_font.render(end_text_str, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_body)
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
        assert info['score'] == 0
        assert info['snake_length'] == 3
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test game logic assertions from brief
        self.reset()
        assert len(self.snake_body) == 3
        assert self.GRID_WIDTH == 20 and self.GRID_HEIGHT == 20
        
        # Test food spawn
        food_pos_before = self.food_pos
        self.pending_growth = 1 # Simulate eating
        self._spawn_food()
        assert self.food_pos != food_pos_before
        assert self.food_pos not in self.snake_body

        print("✓ Implementation validated successfully")