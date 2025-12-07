
# Generated: 2025-08-28T03:56:09.151735
# Source Brief: brief_05090.md
# Brief Index: 5090

        
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
        "Controls: Arrow keys to change direction. Survive and eat the food to grow."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade snake game. Eat the red food to grow your snake, but don't run into the walls or your own tail! The goal is to reach a length of 25."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.CELL_SIZE

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
        
        # Fonts
        try:
            self.font_large = pygame.font.Font(None, 60)
            self.font_small = pygame.font.Font(None, 32)
        except IOError:
            self.font_large = pygame.font.SysFont("sans", 60)
            self.font_small = pygame.font.SysFont("sans", 32)

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (30, 35, 40)
        self.COLOR_WALL = (60, 65, 70)
        self.COLOR_SNAKE_HEAD = (50, 255, 50)
        self.COLOR_SNAKE_BODY_1 = (0, 200, 0)
        self.COLOR_SNAKE_BODY_2 = (0, 150, 0)
        self.COLOR_FOOD = (255, 50, 50)
        self.COLOR_FOOD_SHADOW = (150, 30, 30)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_OVERLAY = (0, 0, 0, 180)

        # Game constants
        self.TARGET_LENGTH = 25
        self.MAX_STEPS = 1000
        self.INITIAL_LENGTH = 3
        
        # Initialize state variables
        self.snake_pos = []
        self.snake_direction = (0, 0)
        self.food_pos = (0, 0)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.last_reward = 0.0
        self.last_distance_to_food = 0

        self.reset()

        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = self.INITIAL_LENGTH
        self.game_over = False
        self.win = False
        self.last_reward = 0.0

        # Place snake in center, moving right
        center_x, center_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_pos = [(center_x - i, center_y) for i in range(self.INITIAL_LENGTH)]
        self.snake_direction = (1, 0) # (dx, dy) -> Right

        # Place food
        self._place_food()
        self.last_distance_to_food = self._manhattan_distance(self.snake_pos[0], self.food_pos)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Direction ---
        new_direction = self.snake_direction
        if movement == 1 and self.snake_direction != (0, 1): # Up
            new_direction = (0, -1)
        elif movement == 2 and self.snake_direction != (0, -1): # Down
            new_direction = (0, 1)
        elif movement == 3 and self.snake_direction != (1, 0): # Left
            new_direction = (-1, 0)
        elif movement == 4 and self.snake_direction != (-1, 0): # Right
            new_direction = (1, 0)
        self.snake_direction = new_direction

        # --- Move Snake ---
        head_x, head_y = self.snake_pos[0]
        dir_x, dir_y = self.snake_direction
        new_head = (head_x + dir_x, head_y + dir_y)
        
        # --- Initialize rewards and flags ---
        reward = 0.1 # Survival reward
        ate_food = False
        collision = False
        
        # --- Check for Food Consumption ---
        if new_head == self.food_pos:
            ate_food = True
            self.score += 1
            reward += 10.0
            # snake grows by not removing tail
            self._place_food() # sound: eat_sfx()
        else:
            self.snake_pos.pop() # remove tail
        
        self.snake_pos.insert(0, new_head)

        # --- Reward for moving towards/away from food ---
        current_distance = self._manhattan_distance(new_head, self.food_pos)
        if not ate_food:
            if current_distance < self.last_distance_to_food:
                pass # Implicitly rewarded by not being penalized
            elif current_distance > self.last_distance_to_food:
                reward -= 5.0
        self.last_distance_to_food = current_distance

        # --- Check for Termination ---
        # Wall collision
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            collision = True # sound: collision_sfx()
        # Self collision
        if new_head in self.snake_pos[1:]:
            collision = True # sound: collision_sfx()

        self.steps += 1
        terminated = False
        
        if collision:
            reward = -100.0
            self.game_over = True
            terminated = True
        elif self.score >= self.TARGET_LENGTH:
            reward = 100.0
            self.game_over = True
            self.win = True
            terminated = True # sound: win_sfx()
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            
        self.last_reward = reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _place_food(self):
        while True:
            pos = (
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            )
            if pos not in self.snake_pos:
                self.food_pos = pos
                break
    
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
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw food
        food_rect = pygame.Rect(
            self.food_pos[0] * self.CELL_SIZE,
            self.food_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        pygame.gfxdraw.filled_circle(
            self.screen,
            food_rect.centerx,
            food_rect.centery,
            self.CELL_SIZE // 2 - 1,
            self.COLOR_FOOD
        )
        pygame.gfxdraw.aacircle(
            self.screen,
            food_rect.centerx,
            food_rect.centery,
            self.CELL_SIZE // 2 - 1,
            self.COLOR_FOOD
        )

        # Draw snake
        for i, segment in enumerate(self.snake_pos):
            rect = pygame.Rect(
                segment[0] * self.CELL_SIZE,
                segment[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            if i == 0: # Head
                color = self.COLOR_SNAKE_HEAD
            else: # Body
                color = self.COLOR_SNAKE_BODY_1 if i % 2 == 0 else self.COLOR_SNAKE_BODY_2
            
            pygame.draw.rect(self.screen, color, rect.inflate(-2, -2), border_radius=4)
            # Add a small highlight
            pygame.draw.rect(self.screen, (255, 255, 255, 30), rect.inflate(-6, -6), 1, border_radius=3)
    
    def _render_ui(self):
        # Display score
        score_text = self.font_small.render(f"Length: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Display steps
        steps_text = self.font_small.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        # Display game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                message = "YOU WIN!"
            else:
                message = "GAME OVER"
            
            end_text = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win": self.win,
            "food_pos": self.food_pos,
            "snake_head_pos": self.snake_pos[0],
            "last_reward": self.last_reward,
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

# Example of how to run the environment for visualization
if __name__ == '__main__':
    import time
    
    # Set this to True to control the game with your keyboard
    MANUAL_PLAY = True

    env = GameEnv()
    obs, info = env.reset()
    
    # Override screen for display
    env.screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption(env.game_description)

    terminated = False
    total_reward = 0
    
    # --- Keyboard mapping ---
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while not terminated:
        movement_action = 0 # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if MANUAL_PLAY and event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    movement_action = key_to_action[event.key]
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        if MANUAL_PLAY:
            # For manual play, we only step when a key is pressed
            if movement_action != 0:
                action = [movement_action, 0, 0]
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Score: {info['score']}")
        else:
            # For agent play, step continuously
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
        # Render the observation to the display window
        rendered_obs = env._get_observation()
        # The observation is (H, W, C), but pygame surfaces expect (W, H)
        # and surfarray.make_surface transposes it back.
        surf = pygame.surfarray.make_surface(np.transpose(rendered_obs, (1, 0, 2)))
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(10) # Slow down for visibility

    print(f"--- FINAL ---")
    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Win: {info['win']}")
    time.sleep(3)
    env.close()