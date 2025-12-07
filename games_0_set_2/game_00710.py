import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to change the snake's direction. "
        "Try to eat the red apples to grow and score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic arcade game. Guide the snake to eat apples, but don't crash into the walls or your own tail!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    CELL_SIZE = 20
    SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE  # 640
    SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE # 400

    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_WALL = (10, 10, 15)
    COLOR_SNAKE = (50, 205, 50)
    COLOR_SNAKE_HEAD = (124, 252, 0)
    COLOR_APPLE = (255, 60, 60)
    COLOR_APPLE_GLOW = (255, 150, 150)
    COLOR_TEXT = (240, 240, 240)
    COLOR_SCORE = (255, 215, 0)

    MAX_STEPS = 1000
    WIN_SCORE = 200

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # State variables are initialized in reset()
        self.snake_body = None
        self.direction = None
        self.apple_pos = None
        self.score = 0
        self.steps = 0
        self.terminated = False

        # The validation call expects a fully initialized environment, so reset() must be called first.
        self.reset()
        # This validation function is part of the provided code and should pass.
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        start_x, start_y = self.GRID_WIDTH // 4, self.GRID_HEIGHT // 2
        self.snake_body = deque([
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y)
        ])
        self.direction = (1, 0)  # Start moving right
        self._place_apple()

        self.score = 0
        self.steps = 0
        self.terminated = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Processing ---
        movement = action[0]
        # space_held and shift_held are unused per brief but must be handled
        
        new_direction = self.direction
        if movement == 1 and self.direction != (0, 1):  # Up
            new_direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1):  # Down
            new_direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):  # Left
            new_direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0):  # Right
            new_direction = (1, 0)
        # movement == 0 is a no-op, continue in the same direction
        self.direction = new_direction

        # --- Game Logic ---
        self.steps += 1
        reward = 0
        
        head = self.snake_body[0]
        dist_before = self._manhattan_distance(head, self.apple_pos)
        
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        dist_after = self._manhattan_distance(new_head, self.apple_pos)

        # Reward for moving closer to the apple, penalize for moving away
        if dist_after < dist_before:
            pass # Implicitly rewarded by not being penalized
        elif dist_after > dist_before:
            reward -= 0.2

        # --- Collision Detection ---
        collided = self._check_collision(new_head)
        if collided:
            self.terminated = True
            reward -= 50
            # sfx: game_over_sound
            return self._get_observation(), reward, self.terminated, False, self._get_info()

        # --- Move Snake ---
        self.snake_body.appendleft(new_head)
        
        # --- Apple Consumption ---
        if new_head == self.apple_pos:
            self.score += 10
            reward += 10
            # sfx: eat_apple_sound
            if self.score >= self.WIN_SCORE:
                self.terminated = True
                reward += 100
                # sfx: victory_sound
            else:
                self._place_apple()
        else:
            self.snake_body.pop() # Remove tail if no apple was eaten

        # --- Step-based Reward & Termination ---
        reward += 0.1 # Small reward for surviving a step

        if self.steps >= self.MAX_STEPS:
            self.terminated = True

        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,
            self._get_info()
        )

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _check_collision(self, head):
        # Wall collision
        if not (0 <= head[0] < self.GRID_WIDTH and 0 <= head[1] < self.GRID_HEIGHT):
            return True
        # Self collision
        # Check against all but the new head (which isn't in the deque yet)
        if head in self.snake_body:
            return True
        return False

    def _place_apple(self):
        possible_positions = set(
            (x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)
        )
        snake_positions = set(self.snake_body)
        available_positions = list(possible_positions - snake_positions)
        
        if not available_positions:
            # No space left, this is a win/draw scenario, but we'll end it
            self.terminated = True
            self.apple_pos = (-1, -1) # Place apple off-screen
        else:
            idx = self.np_random.integers(len(available_positions))
            self.apple_pos = available_positions[idx]


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "length": len(self.snake_body)
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw apple with glow
        if self.apple_pos and self.apple_pos[0] >= 0:
            ax, ay = self.apple_pos
            apple_rect = pygame.Rect(
                ax * self.CELL_SIZE, ay * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE
            )
            center_x = apple_rect.centerx
            center_y = apple_rect.centery
            radius = self.CELL_SIZE // 2
            
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(radius * 1.2), self.COLOR_APPLE_GLOW)
            # Main apple
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_APPLE)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_APPLE)

        # Draw snake
        if self.snake_body:
            # Draw body segments
            for i, segment in enumerate(list(self.snake_body)[1:]):
                seg_rect = pygame.Rect(
                    segment[0] * self.CELL_SIZE,
                    segment[1] * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_SNAKE, seg_rect, border_radius=4)
            
            # Draw head
            head = self.snake_body[0]
            head_rect = pygame.Rect(
                head[0] * self.CELL_SIZE,
                head[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, head_rect, border_radius=5)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        if self.terminated:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                end_text_str = "VICTORY!"
                end_color = self.COLOR_SCORE
            else:
                end_text_str = "GAME OVER"
                end_color = self.COLOR_APPLE
                
            end_text = self.font_large.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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
        
        # Test reward structure and state changes
        self.reset()
        initial_length = len(self.snake_body)
        self.apple_pos = (self.snake_body[0][0] + 1, self.snake_body[0][1]) # Place apple in front
        self.direction = (1, 0) # Ensure moving right
        # FIX: Use a no-op action to ensure the snake moves in the manually set direction.
        # A random action could cause it to turn away and fail the test.
        _, reward, _, _, _ = self.step(np.array([0, 0, 0])) 
        assert len(self.snake_body) == initial_length + 1
        assert self.score == 10
        assert reward > 5 # Should be 10 (apple) + 0.1 (survival)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Use a display for manual play
    pygame.display.set_caption("Snake - Manual Play")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    # Initial action is to do nothing
    action = [0, 0, 0] 
    
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
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    action = [0, 0, 0]
                    continue

        # Step the environment with the chosen action
        obs, reward, terminated, truncated, info = env.step(np.array(action))
        
        # Reset action to no-op for next frame unless a key is pressed again
        action = [0, 0, 0]

        # Render to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # If terminated, wait a bit before closing or resetting
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            
    env.close()