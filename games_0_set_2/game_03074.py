
# Generated: 2025-08-27T22:17:01.796962
# Source Brief: brief_03074.md
# Brief Index: 3074

        
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

    # User-facing control string
    user_guide = "Controls: Use arrow keys (↑↓←→) to change the snake's direction."

    # User-facing description of the game
    game_description = "Guide a growing snake through a grid-based arena to eat food and reach a target length."

    # Frames advance only on action
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_SNAKE_BODY = (0, 200, 100)
    COLOR_SNAKE_HEAD = (100, 255, 150)
    COLOR_FOOD = (255, 80, 80)
    COLOR_FOOD_BORDER = (200, 50, 50)
    COLOR_TEXT = (255, 255, 255)
    COLOR_BORDER = (80, 80, 80)

    # Game parameters
    INITIAL_SNAKE_LENGTH = 3
    INITIAL_FOOD_COUNT = 5
    WIN_LENGTH = 20
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font = pygame.font.Font(None, 36)

        # Initialize state variables
        self.snake_body = None
        self.direction = None
        self.food_positions = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_distance_to_food = 0.0
        
        # This will be seeded by the environment
        self.np_random = None

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.game_over = False

        # Initialize snake
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_body = deque(
            [[start_x - i, start_y] for i in range(self.INITIAL_SNAKE_LENGTH)]
        )
        self.direction = (1, 0)  # Start moving right

        # Initialize score
        self.score = len(self.snake_body)
        
        # Initialize food
        self.food_positions = []
        for _ in range(self.INITIAL_FOOD_COUNT):
            self._spawn_food()

        # Initialize reward-related state
        self.last_distance_to_food = self._get_closest_food_distance()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self._update_direction(movement)

        # Store pre-move state for reward calculation
        dist_before_move = self._get_closest_food_distance()
        
        # Move the snake
        head = self.snake_body[0]
        new_head = [head[0] + self.direction[0], head[1] + self.direction[1]]
        self.snake_body.appendleft(new_head)
        
        # Check for game events
        reward = 0
        ate_food = self._check_food_collision()

        if not ate_food:
            self.snake_body.pop() # Remove tail if no food was eaten
        else:
            reward += 10 # +10 for eating food
            self.score = len(self.snake_body)
            self._spawn_food()

        # Distance-based reward
        dist_after_move = self._get_closest_food_distance()
        if dist_after_move < dist_before_move:
            reward += 0.1
        else:
            reward -= 0.1
        self.last_distance_to_food = dist_after_move

        # Check for termination conditions
        terminated = False
        if self._check_wall_collision() or self._check_self_collision():
            reward = -100 # -100 for collision
            self.game_over = True
            terminated = True
        elif self.score >= self.WIN_LENGTH:
            reward = 100 # +100 for winning
            self.game_over = True
            terminated = True

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _update_direction(self, movement):
        # 1=up, 2=down, 3=left, 4=right
        if movement == 1 and self.direction != (0, 1): # Prevent reversing
            self.direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1):
            self.direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):
            self.direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0):
            self.direction = (1, 0)
        # movement == 0 is a no-op, continue in current direction

    def _get_available_cells(self):
        occupied_cells = {tuple(pos) for pos in self.snake_body}
        occupied_cells.update(tuple(pos) for pos in self.food_positions)
        available_cells = []
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in occupied_cells:
                    available_cells.append([x, y])
        return available_cells

    def _spawn_food(self):
        available_cells = self._get_available_cells()
        if available_cells:
            # Use the seeded random number generator
            idx = self.np_random.integers(0, len(available_cells))
            new_food_pos = available_cells[idx]
            self.food_positions.append(new_food_pos)

    def _check_food_collision(self):
        head = self.snake_body[0]
        for i, food_pos in enumerate(self.food_positions):
            if head[0] == food_pos[0] and head[1] == food_pos[1]:
                # sfx: eat_sound.play()
                del self.food_positions[i]
                return True
        return False

    def _check_wall_collision(self):
        head = self.snake_body[0]
        if not (0 <= head[0] < self.GRID_WIDTH and 0 <= head[1] < self.GRID_HEIGHT):
            # sfx: crash_sound.play()
            return True
        return False

    def _check_self_collision(self):
        head = self.snake_body[0]
        for segment in list(self.snake_body)[1:]:
            if head[0] == segment[0] and head[1] == segment[1]:
                # sfx: crash_sound.play()
                return True
        return False
    
    def _get_closest_food_distance(self):
        if not self.food_positions:
            return 0
        head = self.snake_body[0]
        min_dist = float('inf')
        for food in self.food_positions:
            dist = math.hypot(head[0] - food[0], head[1] - food[1])
            if dist < min_dist:
                min_dist = dist
        return min_dist

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
        # Draw grid lines
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw border
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 2)

        # Draw food
        for pos in self.food_positions:
            x = int(pos[0] * self.GRID_SIZE + self.GRID_SIZE / 2)
            y = int(pos[1] * self.GRID_SIZE + self.GRID_SIZE / 2)
            radius = int(self.GRID_SIZE / 2 * 0.8)
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, self.COLOR_FOOD)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_FOOD_BORDER)

        # Draw snake
        if self.snake_body:
            # Body
            for segment in list(self.snake_body)[1:]:
                rect = pygame.Rect(
                    segment[0] * self.GRID_SIZE, 
                    segment[1] * self.GRID_SIZE, 
                    self.GRID_SIZE, 
                    self.GRID_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_SNAKE_BODY, rect)
                pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1) # Outline
            
            # Head
            head = self.snake_body[0]
            head_rect = pygame.Rect(
                head[0] * self.GRID_SIZE, 
                head[1] * self.GRID_SIZE, 
                self.GRID_SIZE, 
                self.GRID_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, head_rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, head_rect, 1) # Outline


    def _render_ui(self):
        score_text = f"Length: {self.score}"
        text_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_body) if self.snake_body else 0,
            "food_remaining": len(self.food_positions) if self.food_positions else 0,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space (pre-reset)
        # Can't get observation before reset, so we call reset first
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test reset output
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a different screen for display if running manually
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Gym Environment")
    
    terminated = False
    clock = pygame.time.Clock()
    
    # Game loop
    current_movement = 0 # 0 = no-op
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    current_movement = 1
                elif event.key == pygame.K_DOWN:
                    current_movement = 2
                elif event.key == pygame.K_LEFT:
                    current_movement = 3
                elif event.key == pygame.K_RIGHT:
                    current_movement = 4
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    current_movement = 0

        # Construct the action based on key presses
        # Space and Shift are not used in this game
        action = [current_movement, 0, 0]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # The environment moves one step per action, so we reset the movement
        # to no-op unless another key is pressed.
        current_movement = 0
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print("Game Over! Press 'R' to restart.")
            
        # Since auto_advance is False, we control the speed of human play here.
        clock.tick(10) # 10 moves per second

    env.close()