
# Generated: 2025-08-28T02:13:09.145853
# Source Brief: brief_04374.md
# Brief Index: 4374

        
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
        "Controls: Arrow keys to change direction. Survive and eat food to grow."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Classic arcade snake. Navigate a growing snake to eat food pellets. "
        "Avoid colliding with walls or your own tail. Reach a score of 100 to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 20
    GRID_HEIGHT = 15
    CELL_SIZE = 24
    
    GRID_AREA_WIDTH = GRID_WIDTH * CELL_SIZE
    GRID_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_AREA_HEIGHT) // 2
    
    MAX_STEPS = 1000
    WIN_SCORE = 100
    INITIAL_SNAKE_LENGTH = 5

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 60)
    COLOR_SNAKE_HEAD = (80, 255, 80)
    COLOR_SNAKE_BODY = (50, 200, 50)
    COLOR_FOOD_OUTER = (255, 80, 80)
    COLOR_FOOD_INNER = (255, 150, 150)
    COLOR_TEXT = (255, 255, 255)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.snake_body = deque()
        self.food_pos = (0, 0)
        self.direction = (0, 0)
        self.steps = 0
        self.score = 0
        self.terminated = False
        self.np_random = None

        self.reset()
        
        # self.validate_implementation() # Optional: Call to run self-tests

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.terminated = False
        
        # Initialize snake in the center
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_body = deque(
            [(start_x - i, start_y) for i in range(self.INITIAL_SNAKE_LENGTH)]
        )
        self.direction = (1, 0)  # Start moving right
        
        self._spawn_food()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, self.terminated, False, self._get_info()

        movement = action[0]
        self._update_direction(movement)

        reward = 0
        
        old_head = self.snake_body[0]
        dist_before = self._calculate_distance(old_head, self.food_pos)

        new_head = (old_head[0] + self.direction[0], old_head[1] + self.direction[1])

        dist_after = self._calculate_distance(new_head, self.food_pos)

        # Distance-based reward
        if dist_after < dist_before:
            reward += 0.1
        else:
            reward -= 0.15 # Slightly penalize moving away more

        # Unsafe move reward
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (new_head[0] + dx, new_head[1] + dy)
            if neighbor in list(self.snake_body)[1:]: # Check against body, excluding head
                reward -= 5
                break

        # Check for collisions
        if (
            new_head[0] < 0 or new_head[0] >= self.GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= self.GRID_HEIGHT or
            new_head in self.snake_body
        ):
            self.terminated = True
            reward -= 100
            # Sound: game_over.wav
        else:
            self.snake_body.appendleft(new_head)
            
            # Check for food consumption
            if new_head == self.food_pos:
                self.score += 1
                reward += 10
                # Sound: eat_food.wav
                if self.score >= self.WIN_SCORE:
                    self.terminated = True
                    reward += 100 # Win bonus
                    # Sound: win_game.wav
                else:
                    self._spawn_food()
            else:
                self.snake_body.pop()

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.terminated = True

        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,
            self._get_info()
        )

    def _update_direction(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1 and self.direction != (0, 1):  # Up
            self.direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1):  # Down
            self.direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):  # Left
            self.direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0):  # Right
            self.direction = (1, 0)
        # If movement is 0 (no-op), direction remains unchanged

    def _spawn_food(self):
        possible_positions = []
        snake_set = set(self.snake_body)
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in snake_set:
                    possible_positions.append((x, y))
        
        if not possible_positions:
            # No space left, game is effectively won/drawn
            self.terminated = True
        else:
            choice_index = self.np_random.integers(0, len(possible_positions))
            self.food_pos = possible_positions[choice_index]

    def _calculate_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

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
            "snake_length": len(self.snake_body),
        }

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_OFFSET_Y), (px, self.GRID_OFFSET_Y + self.GRID_AREA_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_AREA_WIDTH, py))

        # Draw food
        food_px = int(self.GRID_OFFSET_X + self.food_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
        food_py = int(self.GRID_OFFSET_Y + self.food_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        radius = int(self.CELL_SIZE * 0.4)
        pygame.gfxdraw.filled_circle(self.screen, food_px, food_py, radius, self.COLOR_FOOD_OUTER)
        pygame.gfxdraw.aacircle(self.screen, food_px, food_py, radius, self.COLOR_FOOD_OUTER)
        pygame.gfxdraw.filled_circle(self.screen, food_px, food_py, int(radius * 0.6), self.COLOR_FOOD_INNER)
        
        # Draw snake
        for i, segment in enumerate(self.snake_body):
            px = self.GRID_OFFSET_X + segment[0] * self.CELL_SIZE
            py = self.GRID_OFFSET_Y + segment[1] * self.CELL_SIZE
            rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
            
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE_BODY
            border_radius = 5 if i == 0 else 3
            
            pygame.draw.rect(self.screen, color, rect.inflate(-4, -4), border_radius=border_radius)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        if self.terminated:
            if self.score >= self.WIN_SCORE:
                end_text = "YOU WIN!"
            else:
                end_text = "GAME OVER"
            
            end_surf = self.font_main.render(end_text, True, self.COLOR_TEXT)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            
            # Create a semi-transparent background for the text
            bg_rect = end_rect.inflate(20, 20)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 150))
            self.screen.blit(bg_surf, bg_rect)
            
            self.screen.blit(end_surf, end_rect)

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


# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Snake")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op

    while running:
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
                    action[0] = 0
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated:
            # Since auto_advance is False, we call step() on each frame to advance the game
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Reset action to no-op unless another key is pressed
            action[0] = 0

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(5) # Control game speed for human play

    env.close()