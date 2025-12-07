
# Generated: 2025-08-28T01:23:07.734495
# Source Brief: brief_04090.md
# Brief Index: 4090

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to change direction. Guide the snake to eat apples and grow."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Classic arcade snake. Eat apples to grow longer and increase your score. Avoid hitting the walls or your own tail. Reach a score of 50 to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 20
        self.CELL_SIZE = self.SCREEN_HEIGHT // self.GRID_SIZE  # 400 / 20 = 20
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2
        self.WIN_SCORE = 50
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_WALL = (80, 80, 90)
        self.COLOR_SNAKE_BODY = (50, 200, 50)
        self.COLOR_SNAKE_HEAD = (100, 255, 100)
        self.COLOR_APPLE = (255, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_GAMEOVER = (255, 0, 0)
        self.COLOR_WIN = self.COLOR_SNAKE_HEAD

        # Fonts
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_gameover = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game state variables (to be initialized in reset)
        self.snake_body = []
        self.snake_direction = (0, 0)
        self.apple_pos = (0, 0)
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initial snake position and direction
        center = self.GRID_SIZE // 2
        self.snake_body = [(center, center), (center, center + 1), (center, center + 2)]
        self.snake_direction = (0, -1)  # Start moving UP

        self._place_apple()
        
        return self._get_observation(), self._get_info()

    def _place_apple(self):
        occupied_cells = set(self.snake_body)
        available_cells = []
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if (x, y) not in occupied_cells:
                    available_cells.append((x, y))

        if not available_cells:
            # No space left, effectively a win/draw state
            self.game_over = True
            self.apple_pos = (-1, -1) # Place it off-screen
        else:
            # Use the seeded random number generator for reproducibility
            idx = self.np_random.integers(0, len(available_cells))
            self.apple_pos = available_cells[idx]
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self._update_direction(movement)
        
        head = self.snake_body[0]
        dist_before = abs(head[0] - self.apple_pos[0]) + abs(head[1] - self.apple_pos[1])

        new_head = (head[0] + self.snake_direction[0], head[1] + self.snake_direction[1])
        self.snake_body.insert(0, new_head)
        
        dist_after = abs(new_head[0] - self.apple_pos[0]) + abs(new_head[1] - self.apple_pos[1])

        reward = 0
        terminated = False

        # Check for apple eating
        if new_head == self.apple_pos:
            # SFX: EAT
            self.score += 1
            reward += 1.0
            if self.score >= self.WIN_SCORE:
                terminated = True
                self.game_over = True
                reward += 100.0  # Win reward
            else:
                self._place_apple()
        else:
            self.snake_body.pop()

        # Distance-based reward if not a terminal state
        if not terminated:
            if dist_after < dist_before:
                reward += 0.1
            else:
                reward -= 0.1

        # Check for termination conditions
        if not (0 <= new_head[0] < self.GRID_SIZE and 0 <= new_head[1] < self.GRID_SIZE):
            # SFX: COLLISION
            terminated = True
            self.game_over = True
            reward = -100.0  # Wall collision penalty
        elif new_head in self.snake_body[1:]:
            # SFX: COLLISION
            terminated = True
            self.game_over = True
            reward = -100.0  # Self-collision penalty
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True
        
        # Handle win condition if grid is full
        if self.apple_pos == (-1, -1) and not terminated:
             terminated = True
             self.game_over = True
             reward += 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_direction(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1 and self.snake_direction != (0, 1):  # UP, not coming from DOWN
            self.snake_direction = (0, -1)
        elif movement == 2 and self.snake_direction != (0, -1): # DOWN, not coming from UP
            self.snake_direction = (0, 1)
        elif movement == 3 and self.snake_direction != (1, 0): # LEFT, not coming from RIGHT
            self.snake_direction = (-1, 0)
        elif movement == 4 and self.snake_direction != (-1, 0): # RIGHT, not coming from LEFT
            self.snake_direction = (1, 0)
        # movement == 0 (no-op) keeps current direction
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid background and border
        grid_rect = pygame.Rect(self.X_OFFSET, self.Y_OFFSET, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        pygame.draw.rect(self.screen, self.COLOR_WALL, grid_rect, 3)

        # Draw apple
        if self.apple_pos != (-1, -1):
            apple_x = self.X_OFFSET + self.apple_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
            apple_y = self.Y_OFFSET + self.apple_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
            pygame.draw.circle(self.screen, self.COLOR_APPLE, (apple_x, apple_y), self.CELL_SIZE // 2 - 2)

        # Draw snake
        for i, segment in enumerate(self.snake_body):
            seg_x = self.X_OFFSET + segment[0] * self.CELL_SIZE
            seg_y = self.Y_OFFSET + segment[1] * self.CELL_SIZE
            rect = pygame.Rect(seg_x + 1, seg_y + 1, self.CELL_SIZE - 2, self.CELL_SIZE - 2)
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE_BODY
            pygame.draw.rect(self.screen, color, rect, border_radius=3)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font_main.render(f"STEPS: {self.steps}", True, self.COLOR_TEXT)
        steps_rect = steps_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(steps_text, steps_rect)

        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_GAMEOVER

            game_over_text = self.font_gameover.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            bg_surf = pygame.Surface((text_rect.width + 20, text_rect.height + 20), pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 180))
            bg_rect = bg_surf.get_rect(center=text_rect.center)
            
            self.screen.blit(bg_surf, bg_rect)
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")