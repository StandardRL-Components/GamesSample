
# Generated: 2025-08-28T03:03:24.662109
# Source Brief: brief_01901.md
# Brief Index: 1901

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
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
        "Controls: Use arrow keys to change the snake's direction. Your goal is to eat the red food to grow longer."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade game. Guide the growing snake to eat food. Avoid hitting the walls or your own tail!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE
        self.MAX_STEPS = 1000
        self.WIN_LENGTH = 20
        self.INITIAL_SNAKE_LENGTH = 3

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_SNAKE = (50, 205, 50)
        self.COLOR_SNAKE_HEAD = (124, 252, 0)
        self.COLOR_FOOD = (255, 69, 0)
        self.COLOR_TEXT = (240, 240, 240)
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.snake_body = None
        self.snake_direction = None
        self.food_pos = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.prev_dist_to_food_manhattan = 0
        self.prev_dist_to_food_euclidean = 0

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        # Initialize snake
        start_x, start_y = self.GRID_WIDTH // 4, self.GRID_HEIGHT // 2
        self.snake_body = deque([
            (start_x - i, start_y) for i in range(self.INITIAL_SNAKE_LENGTH)
        ])
        self.snake_direction = (1, 0)  # Moving right

        # Spawn food
        self._spawn_food()

        # Initialize distance metrics for reward
        head = self.snake_body[0]
        self.prev_dist_to_food_manhattan = abs(head[0] - self.food_pos[0]) + abs(head[1] - self.food_pos[1])
        self.prev_dist_to_food_euclidean = math.hypot(head[0] - self.food_pos[0], head[1] - self.food_pos[1])

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _spawn_food(self):
        while True:
            x = self.np_random.integers(0, self.GRID_WIDTH)
            y = self.np_random.integers(0, self.GRID_HEIGHT)
            if (x, y) not in self.snake_body:
                self.food_pos = (x, y)
                break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Determine new direction
        dx, dy = self.snake_direction
        if movement == 1 and self.snake_direction != (0, 1):  # Up
            dx, dy = 0, -1
        elif movement == 2 and self.snake_direction != (0, -1):  # Down
            dx, dy = 0, 1
        elif movement == 3 and self.snake_direction != (1, 0):  # Left
            dx, dy = -1, 0
        elif movement == 4 and self.snake_direction != (-1, 0):  # Right
            dx, dy = 1, 0
        # movement == 0 is a no-op, so we keep the current direction

        self.snake_direction = (dx, dy)
        
        # Update snake position
        head = self.snake_body[0]
        new_head = (head[0] + dx, head[1] + dy)
        
        # Initialize step variables
        reward = 0.0
        terminated = False
        food_eaten = False

        # Check for collisions
        if (
            new_head in self.snake_body or
            not (0 <= new_head[0] < self.GRID_WIDTH) or
            not (0 <= new_head[1] < self.GRID_HEIGHT)
        ):
            terminated = True
            reward = -100.0  # Collision penalty
        
        # Check for victory
        elif len(self.snake_body) + 1 >= self.WIN_LENGTH:
             terminated = True
             reward = 100.0 # Victory bonus
             self.snake_body.appendleft(new_head) # Grow one last time
        
        # Check max steps
        elif self.steps >= self.MAX_STEPS - 1:
            terminated = True
        
        if not terminated:
            self.snake_body.appendleft(new_head)

            # Check for food consumption
            if new_head == self.food_pos:
                food_eaten = True
                reward += 10.0  # Food bonus
                self.score += 10
                self._spawn_food()
            else:
                self.snake_body.pop() # Remove tail if no food eaten

            # Calculate distance-based rewards
            new_dist_manhattan = abs(new_head[0] - self.food_pos[0]) + abs(new_head[1] - self.food_pos[1])
            new_dist_euclidean = math.hypot(new_head[0] - self.food_pos[0], new_head[1] - self.food_pos[1])
            
            # Manhattan distance reward
            if new_dist_manhattan < self.prev_dist_to_food_manhattan:
                reward += 0.1
            else:
                reward -= 0.2

            # Euclidean proximity reward
            if new_dist_euclidean < self.prev_dist_to_food_euclidean and self.prev_dist_to_food_euclidean <= 5.0:
                 reward += 0.5
            
            # Update previous distances
            self.prev_dist_to_food_manhattan = new_dist_manhattan
            self.prev_dist_to_food_euclidean = new_dist_euclidean

            # Step penalty
            reward -= 0.01

        self.steps += 1
        self.game_over = terminated
        self.score += reward # Accumulate reward into score

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw food
        food_rect = pygame.Rect(
            self.food_pos[0] * self.CELL_SIZE,
            self.food_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_FOOD, food_rect)
        
        # Draw snake
        for i, segment in enumerate(self.snake_body):
            segment_rect = pygame.Rect(
                segment[0] * self.CELL_SIZE,
                segment[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE
            pygame.draw.rect(self.screen, color, segment_rect)
            # Add a small border for visual clarity
            pygame.draw.rect(self.screen, self.COLOR_BG, segment_rect, 1)

    def _render_ui(self):
        score_text = self.font.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        length_text = self.font.render(f"Length: {len(self.snake_body)} / {self.WIN_LENGTH}", True, self.COLOR_TEXT)
        text_rect = length_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(length_text, text_rect)

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
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_body),
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
        # We need to reset to have a valid state for observation
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Snake Gym Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        action = np.array([0, 0, 0])  # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False
                    continue
                
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print("Game Over! Press 'R' to restart.")

    env.close()