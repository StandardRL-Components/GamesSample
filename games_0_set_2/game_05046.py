
# Generated: 2025-08-28T03:48:18.200646
# Source Brief: brief_05046.md
# Brief Index: 5046

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Guide a growing snake to eat food pellets. Avoid hitting the walls or your own tail. Reach a length of 20 to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.CELL_SIZE = self.HEIGHT // self.GRID_SIZE
        self.GAME_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GAME_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.X_OFFSET = (self.WIDTH - self.GAME_WIDTH) // 2
        self.Y_OFFSET = (self.HEIGHT - self.GAME_HEIGHT) // 2

        self.MAX_STEPS = 1000
        self.WIN_LENGTH = 20
        self.INITIAL_FOOD_COUNT = 5
        self.INITIAL_SNAKE_LENGTH = 3

        # Colors
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_GRID = (50, 50, 60)
        self.COLOR_SNAKE_BODY = (0, 180, 0)
        self.COLOR_SNAKE_HEAD = (100, 255, 100)
        self.COLOR_FOOD = (255, 80, 80)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_FLASH = (255, 255, 255)
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont('monospace', 36, bold=True)
            self.font_small = pygame.font.SysFont('monospace', 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 48)
            self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snake = []
        self.direction = (1, 0)
        self.food_pos = []
        self.collision_point = None
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = self.INITIAL_SNAKE_LENGTH
        self.game_over = False
        self.collision_point = None

        # Initialize snake
        start_x, start_y = self.GRID_SIZE // 2, self.GRID_SIZE // 2
        self.snake = [(start_x - i, start_y) for i in range(self.INITIAL_SNAKE_LENGTH)]
        self.direction = (1, 0)  # Moving right

        # Initialize food
        self.food_pos = []
        for _ in range(self.INITIAL_FOOD_COUNT):
            self._spawn_food()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # 1. UPDATE DIRECTION
        new_direction = self.direction
        if movement == 1 and self.direction != (0, 1):  # Up
            new_direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1):  # Down
            new_direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):  # Left
            new_direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0):  # Right
            new_direction = (1, 0)
        self.direction = new_direction

        # 2. CALCULATE PRE-MOVE STATE FOR REWARD
        dist_before, _ = self._get_closest_food_info()
        
        # 3. CALCULATE NEW HEAD POSITION
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # 4. CHECK FOR TERMINATION CONDITIONS
        terminated = False
        reward = 0

        # Wall collision
        if not (0 <= new_head[0] < self.GRID_SIZE and 0 <= new_head[1] < self.GRID_SIZE):
            reward = -100
            terminated = True
            self.game_over = True
            self.collision_point = new_head
        # Self collision
        elif new_head in self.snake:
            reward = -100
            terminated = True
            self.game_over = True
            self.collision_point = new_head
        
        # 5. UPDATE GAME STATE IF NOT TERMINATED
        if not terminated:
            self.snake.insert(0, new_head)

            # Food consumption
            if new_head in self.food_pos:
                # SFX: Nom
                reward = 1.0
                self.score += 1
                self.food_pos.remove(new_head)
                self._spawn_food()
                
                # Win condition
                if self.score >= self.WIN_LENGTH:
                    reward = 100.0
                    terminated = True
                    self.game_over = True
            else:
                self.snake.pop() # Remove tail if no food was eaten
                # Movement-based reward
                dist_after, _ = self._get_closest_food_info()
                if dist_after < dist_before:
                    reward = 0.1
                else:
                    reward = -0.1
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True # End due to time limit
            if reward == 0: # Avoid overwriting collision/win rewards
                reward = -10 # Small penalty for running out of time

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _spawn_food(self):
        possible_spawns = set((x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE))
        occupied_spawns = set(self.snake) | set(self.food_pos)
        available_spawns = list(possible_spawns - occupied_spawns)
        
        if available_spawns:
            if self.np_random:
                idx = self.np_random.integers(0, len(available_spawns))
                new_food_pos = available_spawns[idx]
            else:
                new_food_pos = random.choice(available_spawns)
            self.food_pos.append(new_food_pos)

    def _get_closest_food_info(self):
        if not self.food_pos:
            return float('inf'), None
        
        head = self.snake[0]
        closest_dist = float('inf')
        closest_food = None
        for food in self.food_pos:
            dist = abs(head[0] - food[0]) + abs(head[1] - food[1]) # Manhattan distance
            if dist < closest_dist:
                closest_dist = dist
                closest_food = food
        return closest_dist, closest_food

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_SIZE + 1):
            px = self.X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.Y_OFFSET), (px, self.Y_OFFSET + self.GAME_HEIGHT))
        for y in range(self.GRID_SIZE + 1):
            py = self.Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.X_OFFSET, py), (self.X_OFFSET + self.GAME_WIDTH, py))

        # Draw food
        for fx, fy in self.food_pos:
            rect = pygame.Rect(self.X_OFFSET + fx * self.CELL_SIZE, self.Y_OFFSET + fy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_FOOD, rect)
            pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, self.CELL_SIZE // 2 - 2, (255, 255, 255, 50))


        # Draw snake
        for i, (sx, sy) in enumerate(self.snake):
            rect = pygame.Rect(self.X_OFFSET + sx * self.CELL_SIZE, self.Y_OFFSET + sy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE_BODY
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # Draw collision flash
        if self.game_over and self.collision_point:
            cx, cy = self.collision_point
            center_x = int(self.X_OFFSET + (cx + 0.5) * self.CELL_SIZE)
            center_y = int(self.Y_OFFSET + (cy + 0.5) * self.CELL_SIZE)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, self.CELL_SIZE, self.COLOR_FLASH)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, self.CELL_SIZE, self.COLOR_FLASH)

    def _render_ui(self):
        # Render score (snake length)
        score_text = self.font_large.render(f"LENGTH: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Render steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (20, 60))

        # Render game over message
        if self.game_over:
            message = ""
            if self.score >= self.WIN_LENGTH:
                message = "YOU WIN!"
            elif self.collision_point:
                message = "GAME OVER"
            elif self.steps >= self.MAX_STEPS:
                message = "TIME UP"
            
            if message:
                game_over_text = self.font_large.render(message, True, self.COLOR_FLASH)
                text_rect = game_over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
                self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake),
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Snake Gym Environment")
    
    terminated = False
    running = True
    
    # Game loop
    while running:
        action = [0, 0, 0] # Default action is no-op (continue direction)
        
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
                    # Map keys to actions
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we only step on key presses.
        # This loop will just handle rendering and quitting.
        
    env.close()