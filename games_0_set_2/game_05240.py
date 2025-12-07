
# Generated: 2025-08-28T04:24:19.357216
# Source Brief: brief_05240.md
# Brief Index: 5240

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string for Snake
    user_guide = (
        "Controls: Use arrow keys (mapped to the first action component) to change the snake's direction."
    )

    # User-facing description of the Snake game
    game_description = (
        "Navigate a growing snake to eat glowing food pellets and get a high score, but don't run into yourself!"
    )

    # Frames only advance when an action is received, suitable for turn-based games.
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 50

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_SNAKE_HEAD = (100, 255, 100)
        self.COLOR_SNAKE_BODY = (50, 205, 50)
        self.COLOR_FOOD = (255, 255, 0)
        self.COLOR_FOOD_GLOW = (255, 255, 150)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_GAMEOVER_BG = pygame.Color(0, 0, 0, 180)

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
        self.font_large = pygame.font.Font(None, 72)
        self.font_small = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.snake_body = []
        self.food_pos = (0, 0)
        self.direction = (1, 0)
        self.next_direction = (1, 0)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # Initial snake position and direction
        start_x = self.GRID_WIDTH // 2
        start_y = self.GRID_HEIGHT // 2
        self.snake_body = [
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y),
        ]
        self.direction = (1, 0)  # Moving right
        self.next_direction = (1, 0)

        self._spawn_food()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update game logic
        self.steps += 1
        reward = 0.0
        
        # Handle input and prevent 180-degree turns
        if movement == 1 and self.direction != (0, 1):    # Up
            self.next_direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1):  # Down
            self.next_direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):   # Left
            self.next_direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0):  # Right
            self.next_direction = (1, 0)
        # If movement is 0 (no-op), self.next_direction remains unchanged.

        # Update direction for this step
        self.direction = self.next_direction

        # Move snake
        head = self.snake_body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        self.snake_body.insert(0, new_head)

        # Check for collisions
        # Wall collision
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            self.game_over = True
            reward = -100.0
            # SFX: Play fail sound
        # Self collision
        elif new_head in self.snake_body[1:]:
            self.game_over = True
            reward = -100.0
            # SFX: Play fail sound
        
        # Check for food
        if not self.game_over:
            if new_head == self.food_pos:
                self.score += 1
                reward += 1.0  # Event-based reward for eating food
                # SFX: Play eat sound
                if self.score >= self.WIN_SCORE:
                    self.win = True
                    self.game_over = True
                    reward += 100.0 # Goal-oriented reward for winning
                else:
                    self._spawn_food()
            else:
                self.snake_body.pop() # Remove tail if no food was eaten

            # Continuous reward for surviving the step
            reward += 0.1

        # Check for max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _spawn_food(self):
        possible_positions = []
        snake_set = set(self.snake_body)
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in snake_set:
                    possible_positions.append((x, y))
        
        if not possible_positions:
            # Handle the unlikely case where the snake fills the screen
            self.game_over = True
            return

        self.food_pos = self.np_random.choice(possible_positions)
        self.food_pos = (self.food_pos[0], self.food_pos[1])

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
        # Draw grid
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw food with glow effect
        food_px = (int((self.food_pos[0] + 0.5) * self.CELL_SIZE), int((self.food_pos[1] + 0.5) * self.CELL_SIZE))
        glow_radius = int(self.CELL_SIZE * 0.6)
        food_radius = int(self.CELL_SIZE * 0.4)
        pygame.gfxdraw.filled_circle(self.screen, food_px[0], food_px[1], glow_radius, self.COLOR_FOOD_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, food_px[0], food_px[1], food_radius, self.COLOR_FOOD)
        pygame.gfxdraw.aacircle(self.screen, food_px[0], food_px[1], food_radius, self.COLOR_FOOD)

        # Draw snake
        if not self.snake_body:
            return

        # Draw body segments
        for i, segment in enumerate(self.snake_body[1:]):
            rect = pygame.Rect(segment[0] * self.CELL_SIZE, segment[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_BODY, rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, rect.inflate(-4, -4), border_radius=4) # Inner dot

        # Draw head
        head_rect = pygame.Rect(self.snake_body[0][0] * self.CELL_SIZE, self.snake_body[0][1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, head_rect, border_radius=6)

    def _render_ui(self):
        # Render score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Render game over/win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_GAMEOVER_BG)
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_SNAKE_HEAD if self.win else self.COLOR_FOOD
            
            text_surface = self.font_large.render(message, True, color)
            text_rect = text_surface.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_body),
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

    def close(self):
        pygame.font.quit()
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # For human play
    import sys
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op

    # Game loop for human play
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
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
                    action[0] = 0
                else: # Any other key is a no-op step
                    action[0] = 0
                
                # Step the environment with the chosen action
                obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation to the display screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # Wait for a moment before closing or resetting
            pygame.time.wait(2000)

    env.close()
    sys.exit()