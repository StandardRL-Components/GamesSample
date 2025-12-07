
# Generated: 2025-08-28T01:33:13.089691
# Source Brief: brief_04146.md
# Brief Index: 4146

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to change the snake's direction. "
        "Try to eat the red food pellets to grow and score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic Snake game. Guide the snake to eat food, growing longer with each bite. "
        "Avoid colliding with the walls or your own tail. The game ends if you crash, "
        "reach the max score, or run out of steps."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE_X = 32
        self.GRID_SIZE_Y = 20
        self.CELL_WIDTH = self.WIDTH // self.GRID_SIZE_X
        self.CELL_HEIGHT = self.HEIGHT // self.GRID_SIZE_Y

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("monospace", 24, bold=True)
        except pygame.error:
            self.font = pygame.font.Font(None, 30)


        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_SNAKE_BODY = (50, 200, 50)
        self.COLOR_SNAKE_HEAD = (150, 255, 150)
        self.COLOR_FOOD = (255, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snake_body = None
        self.snake_direction = None
        self.food_pos = None
        self.last_distance_to_food = 0
        self.max_steps = 1000
        self.win_score = 100

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Initialize snake
        start_x = self.GRID_SIZE_X // 2
        start_y = self.GRID_SIZE_Y // 2
        self.snake_body = deque([
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y)
        ])
        self.snake_direction = (1, 0)  # Start moving right

        # Place initial food
        self._place_food()
        
        # Calculate initial distance for reward
        head_pos = self.snake_body[0]
        self.last_distance_to_food = self._get_distance(head_pos, self.food_pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self._update_direction(movement)
        
        self.steps += 1
        
        # Move snake
        head_x, head_y = self.snake_body[0]
        dir_x, dir_y = self.snake_direction
        new_head = (head_x + dir_x, head_y + dir_y)
        self.snake_body.appendleft(new_head)

        # Check for events and termination
        ate_food = False
        terminated = False
        
        # 1. Food collision
        if new_head == self.food_pos:
            # SFX: eat_sound.play()
            self.score += 1
            ate_food = True
            if self.score < self.win_score:
                self._place_food()
        else:
            self.snake_body.pop()

        # 2. Wall collision
        if not (0 <= new_head[0] < self.GRID_SIZE_X and 0 <= new_head[1] < self.GRID_SIZE_Y):
            # SFX: crash_sound.play()
            terminated = True
        
        # 3. Self collision
        body_iter = iter(self.snake_body)
        next(body_iter) # skip head
        for segment in body_iter:
            if new_head == segment:
                # SFX: crash_sound.play()
                terminated = True
                break
        
        # 4. Win condition
        if self.score >= self.win_score:
            # SFX: win_sound.play()
            terminated = True
            
        # 5. Max steps reached
        if self.steps >= self.max_steps:
            terminated = True
            
        self.game_over = terminated
        
        # Calculate reward
        reward = self._calculate_reward(new_head, ate_food, terminated)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_direction(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        current_dir_x, current_dir_y = self.snake_direction
        
        if movement == 1 and current_dir_y == 0:  # Up
            self.snake_direction = (0, -1)
        elif movement == 2 and current_dir_y == 0:  # Down
            self.snake_direction = (0, 1)
        elif movement == 3 and current_dir_x == 0:  # Left
            self.snake_direction = (-1, 0)
        elif movement == 4 and current_dir_x == 0:  # Right
            self.snake_direction = (1, 0)

    def _calculate_reward(self, new_head, ate_food, terminated):
        reward = -0.01

        if ate_food:
            reward += 10
        else:
            new_distance = self._get_distance(new_head, self.food_pos)
            if new_distance < self.last_distance_to_food:
                reward += 2
            elif new_distance > self.last_distance_to_food:
                reward -= 1
            self.last_distance_to_food = new_distance

        if terminated:
            if self.score >= self.win_score:
                reward += 100
            else:
                reward -= 10
        
        return reward

    def _get_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _place_food(self):
        while True:
            x = self.np_random.integers(0, self.GRID_SIZE_X)
            y = self.np_random.integers(0, self.GRID_SIZE_Y)
            if (x, y) not in self.snake_body:
                self.food_pos = (x, y)
                return

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(0, self.WIDTH, self.CELL_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        for i, segment in enumerate(self.snake_body):
            rect = pygame.Rect(
                segment[0] * self.CELL_WIDTH,
                segment[1] * self.CELL_HEIGHT,
                self.CELL_WIDTH,
                self.CELL_HEIGHT
            )
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE_BODY
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        if not (self.score >= self.win_score and self.game_over):
            food_x = int(self.food_pos[0] * self.CELL_WIDTH + self.CELL_WIDTH / 2)
            food_y = int(self.food_pos[1] * self.CELL_HEIGHT + self.CELL_HEIGHT / 2)
            radius = int(self.CELL_WIDTH / 2.5)
            
            pygame.gfxdraw.aacircle(self.screen, food_x, food_y, radius, self.COLOR_FOOD)
            pygame.gfxdraw.filled_circle(self.screen, food_x, food_y, radius, self.COLOR_FOOD)

    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        text_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        shadow_surface = self.font.render(score_text, True, self.COLOR_TEXT_SHADOW)
        
        self.screen.blit(shadow_surface, (12, 12))
        self.screen.blit(text_surface, (10, 10))

        if self.game_over:
            message = "GAME OVER"
            if self.score >= self.win_score:
                message = "YOU WIN!"
            
            end_text_surface = self.font.render(message, True, self.COLOR_TEXT)
            end_shadow_surface = self.font.render(message, True, self.COLOR_TEXT_SHADOW)
            text_rect = end_text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            shadow_rect = end_shadow_surface.get_rect(center=(self.WIDTH / 2 + 2, self.HEIGHT / 2 + 2))
            
            self.screen.blit(end_shadow_surface, shadow_rect)
            self.screen.blit(end_text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_body),
            "food_pos": self.food_pos if not self.game_over else (-1, -1),
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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

if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" 

    env = GameEnv()
    
    print("--- Testing Reset ---")
    obs, info = env.reset()
    print("Reset successful.")
    print("Initial Info:", info)
    assert obs.shape == (400, 640, 3)
    assert info['score'] == 0
    
    print("\n--- Testing Steps ---")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print("Episode finished.")
            break
            
    print(f"\nTotal reward over 10 steps: {total_reward:.2f}")
    env.close()