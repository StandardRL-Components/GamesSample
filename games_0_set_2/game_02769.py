
# Generated: 2025-08-28T05:54:16.907868
# Source Brief: brief_02769.md
# Brief Index: 2769

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to change the snake's direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Classic arcade snake. Eat the red food to grow longer and increase your score. Avoid hitting the walls or your own tail."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

    # Colors
    COLOR_BG = (15, 15, 15)
    COLOR_GRID = (40, 40, 40)
    COLOR_WALL = (80, 80, 80)
    COLOR_SNAKE_BODY = (40, 180, 99)
    COLOR_SNAKE_HEAD = (88, 214, 141)
    COLOR_SNAKE_OUTLINE = (20, 100, 50)
    COLOR_FOOD = (231, 76, 60)
    COLOR_FOOD_OUTLINE = (120, 40, 31)
    COLOR_TEXT = (236, 240, 241)

    # Game settings
    MAX_SCORE = 100
    MAX_STEPS = 1000

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
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snake_pos = []
        self.snake_direction = (0, 0)
        self.food_pos = (0, 0)
        self.last_distance_to_food = 0
        self.ate_food_this_step = False

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ate_food_this_step = False

        # Initial snake position and direction
        center_x, center_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_pos = [
            (center_x, center_y),
            (center_x - 1, center_y),
            (center_x - 2, center_y),
        ]
        self.snake_direction = (1, 0)  # Start moving right

        self._place_food()
        self.last_distance_to_food = self._get_distance_to_food()

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held and shift_held are ignored as per the brief's action mapping
        
        reward = 0
        self.ate_food_this_step = False

        if not self.game_over:
            self._handle_movement(movement)
            self._move_snake()
            
            # Check for food consumption
            if self.snake_pos[0] == self.food_pos:
                self.score += 1
                self.ate_food_this_step = True
                reward += 1.0
                # Sound: Nom nom
                if self.score < self.MAX_SCORE:
                    self._place_food()
            else:
                self.snake_pos.pop() # Remove tail if no food was eaten

        # Calculate distance-based reward
        current_distance = self._get_distance_to_food()
        if current_distance < self.last_distance_to_food:
            reward += 0.1
        elif current_distance > self.last_distance_to_food:
            reward -= 0.1
        self.last_distance_to_food = current_distance
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.score >= self.MAX_SCORE:
                reward = 100.0  # Win reward
            elif self.game_over:
                reward = -50.0  # Lose reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_movement(self, movement):
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

    def _move_snake(self):
        head_x, head_y = self.snake_pos[0]
        dir_x, dir_y = self.snake_direction
        
        new_head = (head_x + dir_x, head_y + dir_y)
        
        # Check for wall collision
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            self.game_over = True
            # Sound: Thud
            return
            
        # Check for self-collision
        # The last element is the tail, which will move, so we can collide with it.
        if new_head in self.snake_pos[:-1]:
            self.game_over = True
            # Sound: Crunch
            return

        self.snake_pos.insert(0, new_head)

    def _place_food(self):
        while True:
            x = self.np_random.integers(0, self.GRID_WIDTH)
            y = self.np_random.integers(0, self.GRID_HEIGHT)
            if (x, y) not in self.snake_pos:
                self.food_pos = (x, y)
                break

    def _get_distance_to_food(self):
        if not self.snake_pos: return 0
        head_x, head_y = self.snake_pos[0]
        food_x, food_y = self.food_pos
        return abs(head_x - food_x) + abs(head_y - food_y)

    def _check_termination(self):
        if self.game_over:
            return True
        if self.score >= self.MAX_SCORE:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw food
        food_rect = pygame.Rect(
            self.food_pos[0] * self.GRID_SIZE,
            self.food_pos[1] * self.GRID_SIZE,
            self.GRID_SIZE,
            self.GRID_SIZE
        )
        pygame.gfxdraw.filled_circle(
            self.screen,
            food_rect.centerx,
            food_rect.centery,
            int(self.GRID_SIZE * 0.45),
            self.COLOR_FOOD
        )
        pygame.gfxdraw.aacircle(
            self.screen,
            food_rect.centerx,
            food_rect.centery,
            int(self.GRID_SIZE * 0.45),
            self.COLOR_FOOD_OUTLINE
        )

        # Draw snake
        if not self.snake_pos: return

        # Draw body
        for i, segment in enumerate(self.snake_pos):
            rect = pygame.Rect(
                segment[0] * self.GRID_SIZE,
                segment[1] * self.GRID_SIZE,
                self.GRID_SIZE,
                self.GRID_SIZE,
            )
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE_BODY
            
            # Use smaller rects for a segmented look
            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, color, inner_rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_OUTLINE, inner_rect, 1, border_radius=4)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self._check_termination():
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.MAX_SCORE:
                end_text = "YOU WIN!"
            else:
                end_text = "GAME OVER"

            text_surface = self.font.render(end_text, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_pos),
            "food_pos": self.food_pos,
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
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Game loop for human play
    while running:
        action = [0, 0, 0] # Default action: no-op (maintain direction)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                
                if not terminated:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
        
        # Display the game screen
        screen_for_display = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_for_display.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we control the speed for human play
        env.clock.tick(10) # 10 moves per second

    env.close()