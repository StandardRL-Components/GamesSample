
# Generated: 2025-08-28T03:07:27.064523
# Source Brief: brief_01925.md
# Brief Index: 1925

        
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
    user_guide = "Controls: ↑↓←→ to change the snake's direction."

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a growing snake through a grid-based arena, consuming food to maximize length without colliding with yourself or the walls."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Grid Dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE

        # Game constants
        self.MAX_LENGTH = 20
        self.MAX_STEPS = 2000
        self.INITIAL_SNAKE_LENGTH = 3

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (30, 45, 60)
        self.COLOR_SNAKE = (0, 200, 100)
        self.COLOR_SNAKE_HEAD = (100, 255, 150)
        self.COLOR_FOOD = (255, 80, 80)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEAD_FLASH = (255, 255, 255)

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
        try:
            self.font = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 24)
        except IOError:
            self.font = pygame.font.SysFont("arial", 36)
            self.font_small = pygame.font.SysFont("arial", 24)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snake_body = deque()
        self.direction = (0, 0)
        self.food_pos = (0, 0)
        self.last_dist_to_food = 0.0
        self.head_flash_timer = 0
        self.np_random = None

        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Optional: run validation on init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = self.INITIAL_SNAKE_LENGTH
        self.game_over = False
        self.head_flash_timer = 0

        # Initialize snake in the center, moving right
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_body = deque(
            [(start_x - i, start_y) for i in range(self.INITIAL_SNAKE_LENGTH)]
        )
        self.direction = (1, 0)  # (dx, dy) -> Right

        self._place_food()
        self.last_dist_to_food = self._get_dist_to_food()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update direction, preventing reversal
        if movement == 1 and self.direction != (0, 1):    # Up
            self.direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1):  # Down
            self.direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):   # Left
            self.direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0):  # Right
            self.direction = (1, 0)
        # movement == 0 (no-op) maintains current direction

        # Move snake
        head = self.snake_body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Check for collisions
        terminated = False
        reward = 0

        # Wall collision
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            terminated = True
            reward = -100
        # Self collision (check against body, excluding the old tail that will move)
        elif new_head in list(self.snake_body)[:-1]:
            terminated = True
            reward = -100
        
        if terminated:
            self.game_over = True
        else:
            self.snake_body.appendleft(new_head)
            self.head_flash_timer = 1 # Flash head for 1 frame on move

            # Check for food consumption
            if new_head == self.food_pos:
                self.score += 1
                reward += 1
                # Victory condition
                if self.score >= self.MAX_LENGTH:
                    terminated = True
                    reward += 100
                    self.game_over = True
                else:
                    self._place_food()
            else:
                self.snake_body.pop() # Remove tail if no food eaten

            # Calculate non-terminal rewards if game is still running
            if not terminated:
                # Survival reward
                reward += 0.1

                # Distance-based reward
                new_dist = self._get_dist_to_food()
                if new_dist < self.last_dist_to_food:
                    # Reward for getting closer is implicitly handled by penalizing moving away
                    pass
                else:
                    reward -= 5 # Penalty for moving away or staying same distance
                self.last_dist_to_food = new_dist

        self.steps += 1
        # Max steps termination
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _place_food(self):
        occupied_spaces = set(self.snake_body)
        available_spaces = []
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in occupied_spaces:
                    available_spaces.append((x, y))
        
        if not available_spaces:
            # This case means the snake filled the screen, which is a win.
            # Handled by the score check, but as a fallback:
            self.game_over = True
            self.food_pos = (-1, -1) # Off-screen
        else:
            self.food_pos = self.np_random.choice(available_spaces)
            # np_random.choice returns an array, convert to tuple
            self.food_pos = (int(self.food_pos[0]), int(self.food_pos[1]))


    def _get_dist_to_food(self):
        head = self.snake_body[0]
        return math.hypot(head[0] - self.food_pos[0], head[1] - self.food_pos[1])

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
        # Draw grid lines
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
        pygame.gfxdraw.aacircle(self.screen, food_rect.centerx, food_rect.centery, self.CELL_SIZE // 2 - 2, self.COLOR_FOOD)

        # Draw snake
        for i, segment in enumerate(self.snake_body):
            rect = pygame.Rect(
                segment[0] * self.CELL_SIZE,
                segment[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            
            if i == 0: # Head
                if self.head_flash_timer > 0:
                    pygame.draw.rect(self.screen, self.COLOR_HEAD_FLASH, rect, border_radius=5)
                    self.head_flash_timer -= 1
                else:
                    pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, rect, border_radius=5)
            else: # Body
                pygame.draw.rect(self.screen, self.COLOR_SNAKE, rect, border_radius=3)

    def _render_ui(self):
        score_text = self.font.render(f"Length: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            status_text_str = "VICTORY!" if self.score >= self.MAX_LENGTH else "GAME OVER"
            status_text = self.font.render(status_text_str, True, self.COLOR_UI_TEXT)
            text_rect = status_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(status_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_body),
            "food_pos": self.food_pos,
        }

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'human' to see the game being played.
    # Note: The environment is designed for headless rendering (rgb_array),
    # but we can display it for debugging/demonstration purposes.
    render_mode = "human" # "human" or "rgb_array"
    
    env = GameEnv()
    
    if render_mode == "human":
        # For human play, we need a display
        pygame.display.set_caption("Snake Gym Environment")
        human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    done = False
    
    # Map keyboard keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Store the last movement action
    last_movement = 0

    while not done:
        action = [last_movement, 0, 0] # Default action is to continue in the same direction

        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key in key_to_action:
                        last_movement = key_to_action[event.key]
                    if event.key == pygame.K_r: # Reset on 'r' key
                        obs, info = env.reset()
                        last_movement = 0
            
            action[0] = last_movement

        # In a real RL loop, an agent would choose the action here.
        # For human play, we take it from the keyboard.
        # For a simple bot, we could use env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if render_mode == "human":
            # The observation is a numpy array, convert it back to a Pygame surface to display
            # The observation is (H, W, C), but pygame blit needs (W, H, C)
            # The internal _get_observation already transposes, so we need to reverse it for display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(10) # Control game speed for human play

    env.close()
    print(f"Game Over! Final Score (Length): {info['score']}, Steps: {info['steps']}")