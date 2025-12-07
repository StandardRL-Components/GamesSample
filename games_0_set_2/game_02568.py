
# Generated: 2025-08-27T20:45:58.123263
# Source Brief: brief_02568.md
# Brief Index: 2568

        
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
        "Controls: Use ↑↓←→ to change the snake's direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a growing snake to eat food and avoid collisions in this classic arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CELL_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE  # 32
    GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE # 20

    WIN_SCORE = 50
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_SNAKE_BODY = (50, 205, 50) # LimeGreen
    COLOR_SNAKE_HEAD = (124, 252, 0) # LawnGreen
    COLOR_FOOD = (255, 69, 0) # OrangeRed
    COLOR_TEXT = (255, 255, 255)
    COLOR_UI_BG = (10, 10, 10, 180) # Semi-transparent black
    COLOR_DEATH_FLASH = (255, 0, 0, 150) # Semi-transparent red

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 50, bold=True)

        # Game state variables are initialized in reset()
        self.snake_body = None
        self.snake_direction = None
        self.food_pos = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_status = False
        self.last_dist_to_food = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_status = False

        # Initialize snake in the center
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_body = deque([
            (start_x - 2, start_y),
            (start_x - 1, start_y),
            (start_x, start_y),
        ])
        self.snake_direction = (1, 0)  # Start moving right

        # Spawn initial food
        self._spawn_food()
        
        # Calculate initial distance to food for reward shaping
        head_pos = self.snake_body[-1]
        self.last_dist_to_food = self._get_distance(head_pos, self.food_pos)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is already over, do nothing and return the final state
            return self._get_observation(), 0.0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held and shift_held are ignored as per the brief
        
        # Update game logic
        self._update_direction(movement)
        
        reward, terminated = self._update_game_state()
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            # No penalty for timeout, just end the episode
        
        self.game_over = terminated
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_direction(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
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

    def _update_game_state(self):
        # Move snake
        head_x, head_y = self.snake_body[-1]
        move_x, move_y = self.snake_direction
        new_head = (head_x + move_x, head_y + move_y)
        
        # Calculate distance-based reward
        current_dist_to_food = self._get_distance(new_head, self.food_pos)
        reward = 0.1 if current_dist_to_food < self.last_dist_to_food else -0.1
        self.last_dist_to_food = current_dist_to_food

        # Check for collisions
        if (
            new_head in self.snake_body or
            not (0 <= new_head[0] < self.GRID_WIDTH) or
            not (0 <= new_head[1] < self.GRID_HEIGHT)
        ):
            self.win_status = False
            return -100.0, True  # Collision, game over

        # Update snake body
        self.snake_body.append(new_head)

        # Check for food consumption
        if new_head == self.food_pos:
            self.score += 1
            reward += 1.0  # Reward for eating food
            # Check for win condition
            if self.score >= self.WIN_SCORE:
                self.win_status = True
                return 100.0, True # Win, game over
            else:
                self._spawn_food() # Spawn new food, snake grows
        else:
            self.snake_body.popleft() # No food, move tail

        return reward, False # No terminal event

    def _spawn_food(self):
        while True:
            self.food_pos = (
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            )
            if self.food_pos not in self.snake_body:
                break
    
    @staticmethod
    def _get_distance(pos1, pos2):
        # Manhattan distance is appropriate for a grid world
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_observation(self):
        # Clear screen with background color
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (H, W, C format)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw snake body
        for segment in list(self.snake_body)[:-1]:
            rect = pygame.Rect(
                segment[0] * self.CELL_SIZE,
                segment[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_BODY, rect)
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, rect, 1) # Outline

        # Draw snake head
        head = self.snake_body[-1]
        head_rect = pygame.Rect(
            head[0] * self.CELL_SIZE,
            head[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, head_rect)

        # Draw food (antialiased for quality)
        food_center_x = int(self.food_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
        food_center_y = int(self.food_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        food_radius = int(self.CELL_SIZE / 2 * 0.8)
        pygame.gfxdraw.filled_circle(self.screen, food_center_x, food_center_y, food_radius, self.COLOR_FOOD)
        pygame.gfxdraw.aacircle(self.screen, food_center_x, food_center_y, food_radius, self.COLOR_FOOD)

    def _render_ui(self):
        # Score display
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        
        # Create a semi-transparent background for the score
        ui_panel = pygame.Surface((score_rect.width + 20, score_rect.height + 10), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (score_rect.left - 10, score_rect.top - 5))
        self.screen.blit(score_text, score_rect)
        
        # Game over / Win message
        if self.game_over:
            # Red flash on death
            if not self.win_status:
                flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                flash_surface.fill(self.COLOR_DEATH_FLASH)
                self.screen.blit(flash_surface, (0, 0))

            message = "YOU WIN!" if self.win_status else "GAME OVER"
            color = self.COLOR_SNAKE_HEAD if self.win_status else self.COLOR_FOOD
            
            text_surface = self.font_large.render(message, True, color)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Text shadow for readability
            shadow_surface = self.font_large.render(message, True, self.COLOR_UI_BG)
            self.screen.blit(shadow_surface, text_rect.move(4, 4))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_body),
            "food_pos": self.food_pos,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        self.reset()
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

if __name__ == "__main__":
    # This block allows you to run the file directly to test the environment
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    # --- Manual Play ---
    # Create a window to display the game
    pygame.display.set_caption("Snake Gym Environment")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op (continue direction)
        
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
                elif event.key == pygame.K_r: # Reset key
                    obs, info = env.reset()
                    done = False
                    continue # Skip step on reset frame
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                
                # Since auto_advance is False, we step the environment on each key press
                if not done:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Tick the clock to keep the window responsive
        clock.tick(30)

    env.close()