
# Generated: 2025-08-28T02:46:53.893464
# Source Brief: brief_04564.md
# Brief Index: 4564

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to change the snake's direction. "
        "Try to eat the red food to grow longer."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade snake game. Control a growing snake to consume food and "
        "reach a target length of 50. Avoid colliding with yourself or the walls."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 20
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.GRID_SIZE

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
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_SNAKE = (46, 204, 113)
        self.COLOR_SNAKE_HEAD = (88, 214, 141)
        self.COLOR_FOOD = (231, 76, 60)
        self.COLOR_FOOD_GLOW = (231, 76, 60, 50)
        self.COLOR_UI = (236, 240, 241)
        self.COLOR_BOUNDARY = (52, 73, 94)

        # Game constants
        self.MAX_STEPS = 1000
        self.WIN_LENGTH = 50
        self.FOOD_RESPAWN_TIMER = 50
        
        # Initialize state variables
        self.snake_body = None
        self.snake_direction = None
        self.food_pos = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.food_timer = 0
        self.np_random = None

        self.reset()

        # Run validation check
        # self.validate_implementation() # Comment out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.game_over = False
        self.score = 3 # Initial length

        # Center snake
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_body = deque([
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y),
        ])
        self.snake_direction = (1, 0)  # Start moving right
        
        self._spawn_food()
        self.food_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update direction based on action, preventing 180-degree turns
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
        
        # Update game logic
        self.steps += 1
        self.food_timer += 1
        
        # Move snake
        head = self.snake_body[0]
        new_head = (head[0] + self.snake_direction[0], head[1] + self.snake_direction[1])
        self.snake_body.appendleft(new_head)

        # Initialize reward and termination
        reward = 0.1  # Survival reward
        terminated = False

        # Check for food consumption
        if new_head == self.food_pos:
            reward += 1.0  # Eat food reward
            self.score += 1
            # Snake grows, so we don't pop the tail
            self._spawn_food()
            self.food_timer = 0
            # sfx: positive beep
        else:
            self.snake_body.pop() # Remove tail if no food eaten
        
        # Check if food should respawn due to time
        if self.food_timer >= self.FOOD_RESPAWN_TIMER:
            self._spawn_food()
            self.food_timer = 0
            # sfx: neutral warp sound

        # Check termination conditions
        # 1. Self-collision
        if new_head in list(self.snake_body)[1:]:
            reward = -100.0
            terminated = True
            # sfx: crunch/fail sound
        
        # 2. Boundary collision
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            reward = -5.0
            terminated = True
            # sfx: thud/fail sound

        # 3. Win condition
        if len(self.snake_body) >= self.WIN_LENGTH:
            reward = 100.0
            terminated = True
            # sfx: win fanfare

        # 4. Max steps
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_food(self):
        """Spawns food in a random location not occupied by the snake."""
        possible_positions = set(
            (x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)
        )
        snake_positions = set(self.snake_body)
        available_positions = list(possible_positions - snake_positions)
        
        if not available_positions:
            # No space left, this is effectively a win but we'll just place it anywhere
            self.food_pos = (0,0)
        else:
            self.food_pos = random.choice(available_positions)

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
        # Draw boundary
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 2)
        
        # Draw food with glow effect
        if self.food_pos:
            food_x = self.food_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2
            food_y = self.food_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2
            glow_radius = int(self.GRID_SIZE * 0.75)
            food_radius = self.GRID_SIZE // 2
            
            # Draw semi-transparent glow
            pygame.gfxdraw.filled_circle(self.screen, food_x, food_y, glow_radius, self.COLOR_FOOD_GLOW)
            # Draw main food circle
            pygame.gfxdraw.filled_circle(self.screen, food_x, food_y, food_radius, self.COLOR_FOOD)
            pygame.gfxdraw.aacircle(self.screen, food_x, food_y, food_radius, self.COLOR_FOOD)

        # Draw snake
        if self.snake_body:
            # Draw body segments
            for i, segment in enumerate(list(self.snake_body)[1:]):
                seg_rect = pygame.Rect(
                    segment[0] * self.GRID_SIZE,
                    segment[1] * self.GRID_SIZE,
                    self.GRID_SIZE,
                    self.GRID_SIZE,
                )
                pygame.draw.rect(self.screen, self.COLOR_SNAKE, seg_rect)
                pygame.draw.rect(self.screen, self.COLOR_BG, seg_rect, 1) # Outline

            # Draw head
            head = self.snake_body[0]
            head_rect = pygame.Rect(
                head[0] * self.GRID_SIZE,
                head[1] * self.GRID_SIZE,
                self.GRID_SIZE,
                self.GRID_SIZE,
            )
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, head_rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, head_rect, 1) # Outline

    def _render_ui(self):
        # Display score (length)
        score_text = self.font_ui.render(f"Length: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Display game over message
        if self.game_over:
            win_message = "YOU WIN!" if len(self.snake_body) >= self.WIN_LENGTH else "GAME OVER"
            over_text = self.font_game_over.render(win_message, True, self.COLOR_UI)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "length": len(self.snake_body) if self.snake_body else 0
        }

    def close(self):
        pygame.font.quit()
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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert "score" in info and "steps" in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        assert "score" in info and "steps" in info

        # Test game mechanics
        self.reset()
        initial_length = len(self.snake_body)
        self.food_pos = (self.snake_body[0][0] + 1, self.snake_body[0][1]) # Place food in front
        self.step(self.action_space.sample()) # Take a step to eat it
        assert len(self.snake_body) == initial_length + 1, "Snake did not grow after eating"
        assert self.score == initial_length + 1, "Score did not update after eating"

        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import sys
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Snake Gym Environment")
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Map Pygame keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    print(env.user_guide)
    
    while not terminated:
        action_movement = 0 # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action_movement = key_to_action[event.key]
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        # Create the full action tuple
        action = [action_movement, 0, 0] # Space and Shift are not used
        
        # Only step if an action was taken (for human play)
        if action_movement != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action_movement}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Length: {info['length']}")

        # Render the observation to the human-visible screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(10) # Control human play speed

    print("Game Over!")
    env.close()
    pygame.quit()
    sys.exit()