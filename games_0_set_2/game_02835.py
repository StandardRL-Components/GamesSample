
# Generated: 2025-08-28T06:06:06.980229
# Source Brief: brief_02835.md
# Brief Index: 2835

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to change the snake's direction. The snake moves one step per action."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Classic arcade snake. Grow your snake by eating food pellets, but avoid hitting the walls or your own tail. Reach a length of 20 to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.CELL_SIZE
        self.WIN_LENGTH = 20
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_WALL = (50, 50, 70)
        self.COLOR_SNAKE_HEAD = (0, 255, 127) # Spring Green
        self.COLOR_SNAKE_TAIL = (0, 100, 0)   # Dark Green
        self.COLOR_FOOD = (255, 69, 0)       # OrangeRed
        self.COLOR_FOOD_GLOW = (255, 140, 0) # DarkOrange
        self.COLOR_TEXT = (240, 240, 240)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font = pygame.font.Font(None, 36)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snake_body = deque()
        self.direction = (0, 0)
        self.food_pos = (0, 0)
        self.np_random = None

        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            # Fallback for when seed is not provided
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Initial snake state
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_body = deque([
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y)
        ])
        self.direction = (1, 0)  # Start moving right

        # Spawn initial food
        self._spawn_food()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Direction ---
        new_direction = self.direction
        if movement == 1 and self.direction != (0, 1):    # Up
            new_direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1):  # Down
            new_direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):   # Left
            new_direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0):  # Right
            new_direction = (1, 0)
        # movement == 0 (no-op) means continue in the current direction
        self.direction = new_direction

        # --- Calculate New Head Position ---
        head_x, head_y = self.snake_body[0]
        new_head_pos = (head_x + self.direction[0], head_y + self.direction[1])

        # --- Calculate Distance-to-Food Reward ---
        dist_before = abs(head_x - self.food_pos[0]) + abs(head_y - self.food_pos[1])
        dist_after = abs(new_head_pos[0] - self.food_pos[0]) + abs(new_head_pos[1] - self.food_pos[1])
        
        reward = 0.1  # Survival reward
        if dist_after < dist_before:
            pass # Implicit reward for getting closer
        elif dist_after > dist_before:
            reward -= 0.2 # Penalty for moving away from food
        
        # --- Collision Detection ---
        terminated = False
        # Wall collision
        if not (0 <= new_head_pos[0] < self.GRID_WIDTH and 0 <= new_head_pos[1] < self.GRID_HEIGHT):
            self.game_over = True
            terminated = True
            reward = -100.0
            # Sfx: impact_wall
        # Self collision
        elif new_head_pos in self.snake_body:
            self.game_over = True
            terminated = True
            reward = -100.0
            # Sfx: impact_self

        # --- Game Logic Update ---
        if not self.game_over:
            # Food consumption
            if new_head_pos == self.food_pos:
                self.snake_body.appendleft(new_head_pos)
                self.score += 1
                reward += 1.0
                # Sfx: eat_food
                
                # Check for win condition
                if len(self.snake_body) >= self.WIN_LENGTH:
                    self.game_over = True
                    terminated = True
                    reward += 100.0
                    # Sfx: win_game
                else:
                    self._spawn_food()
            else:
                # Move snake by adding new head and removing tail
                self.snake_body.appendleft(new_head_pos)
                self.snake_body.pop()

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_food(self):
        possible_locations = set(
            (x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)
        )
        snake_locations = set(self.snake_body)
        valid_locations = list(possible_locations - snake_locations)
        
        if not valid_locations:
            # This case means the snake has filled the screen, which is a win
            self.game_over = True
        else:
            choice_index = self.np_random.integers(0, len(valid_locations))
            self.food_pos = valid_locations[choice_index]


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw food with a glow effect
        food_rect = pygame.Rect(
            self.food_pos[0] * self.CELL_SIZE,
            self.food_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        center_x, center_y = food_rect.center
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(self.CELL_SIZE * 0.6), self.COLOR_FOOD_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(self.CELL_SIZE * 0.4), self.COLOR_FOOD)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(self.CELL_SIZE * 0.4), self.COLOR_FOOD)


        # Draw snake with gradient
        num_segments = len(self.snake_body)
        head_color = pygame.Color(self.COLOR_SNAKE_HEAD)
        tail_color = pygame.Color(self.COLOR_SNAKE_TAIL)

        for i, segment in enumerate(self.snake_body):
            seg_rect = pygame.Rect(
                segment[0] * self.CELL_SIZE,
                segment[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            
            # Interpolate color from head to tail
            t = i / max(1, num_segments - 1)
            color = head_color.lerp(tail_color, t)
            
            # Render as slightly smaller circles for a rounded look
            radius = self.CELL_SIZE // 2
            pygame.draw.circle(self.screen, color, seg_rect.center, radius)

            # Draw eyes on the head
            if i == 0:
                eye_radius = int(self.CELL_SIZE * 0.1)
                eye_offset_x = 0
                eye_offset_y = 0
                if self.direction == (1, 0) or self.direction == (-1, 0): # Horizontal
                    eye_offset_y = int(self.CELL_SIZE * 0.25)
                else: # Vertical
                    eye_offset_x = int(self.CELL_SIZE * 0.25)
                
                eye1_pos = (seg_rect.centerx - eye_offset_x, seg_rect.centery - eye_offset_y)
                eye2_pos = (seg_rect.centerx + eye_offset_x, seg_rect.centery + eye_offset_y)
                pygame.draw.circle(self.screen, (0,0,0), eye1_pos, eye_radius)
                pygame.draw.circle(self.screen, (0,0,0), eye2_pos, eye_radius)


    def _render_ui(self):
        length_text = f"Length: {len(self.snake_body)} / {self.WIN_LENGTH}"
        text_surface = self.font.render(length_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        if self.game_over:
            outcome_text = "YOU WIN!" if len(self.snake_body) >= self.WIN_LENGTH else "GAME OVER"
            outcome_surface = self.font.render(outcome_text, True, self.COLOR_TEXT)
            text_rect = outcome_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(outcome_surface, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "length": len(self.snake_body)
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

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless for this test
    
    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:")
    print(f"  Info: {info}")

    # Run for a few steps with random actions
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print("Episode finished. Resetting.")
            obs, info = env.reset()
    
    env.close()