
# Generated: 2025-08-28T01:06:58.611836
# Source Brief: brief_04008.md
# Brief Index: 4008

        
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
        "Controls: Use arrow keys to change the snake's direction. "
        "Try to eat the red food to grow longer."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a growing snake through a procedurally generated maze. "
        "Eat food to increase your length and score, but avoid colliding with walls or your own tail. "
        "Reach a length of 25 to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 40, 25
        self.BLOCK_SIZE = self.WIDTH // self.GRID_W
        self.TARGET_LENGTH = 25
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_WALL = (40, 40, 60)
        self.COLOR_SNAKE = (50, 205, 50)
        self.COLOR_SNAKE_HEAD = (150, 255, 150)
        self.COLOR_SNAKE_FLASH = (255, 255, 255)
        self.COLOR_FOOD = (220, 50, 50)
        self.COLOR_UI = (240, 240, 240)

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
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 28)

        # Initialize state variables to None, to be set in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.grid = None
        self.snake = None
        self.direction = None
        self.food_pos = None
        self.head_flash_timer = None
        self.np_random = None

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.head_flash_timer = 0
        
        self.grid = self._generate_maze(self.GRID_W, self.GRID_H)
        
        # Find a valid starting position for the snake
        start_pos = self._find_empty_cell()
        self.snake = deque([start_pos])
        self.direction = (1, 0) # Start moving right
        
        self._place_food()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        terminated = False

        # --- Reward calculation prep ---
        old_head = self.snake[0]
        old_dist_food = abs(old_head[0] - self.food_pos[0]) + abs(old_head[1] - self.food_pos[1])
        
        # --- Update game logic ---
        # 1. Update direction based on action
        new_direction = self.direction
        if movement == 1 and self.direction != (0, 1): new_direction = (0, -1) # Up
        elif movement == 2 and self.direction != (0, -1): new_direction = (0, 1) # Down
        elif movement == 3 and self.direction != (1, 0): new_direction = (-1, 0) # Left
        elif movement == 4 and self.direction != (-1, 0): new_direction = (1, 0) # Right
        self.direction = new_direction

        # 2. Move snake
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        self.snake.appendleft(new_head)
        
        # 3. Check for events
        # Food consumption
        if new_head == self.food_pos:
            # Snake grows, so we don't pop the tail
            self.score = len(self.snake)
            reward += 10
            # sfx: positive chime
            self.head_flash_timer = 3 # frames
            if len(self.snake) >= self.TARGET_LENGTH:
                terminated = True
                self.game_over = True
                reward += 100 # Win bonus
            else:
                self._place_food()
        else:
            self.snake.pop()

        # Wall collision
        if (not (0 <= new_head[0] < self.GRID_W and 0 <= new_head[1] < self.GRID_H)) or self.grid[new_head[1], new_head[0]] == 1:
            terminated = True
            self.game_over = True
            reward = -100
            # sfx: crash sound

        # Self collision
        # Check against the rest of the body (from index 1 onwards)
        for i in range(1, len(self.snake)):
            if new_head == self.snake[i]:
                terminated = True
                self.game_over = True
                reward = -100
                # sfx: painful squish
                break

        # 4. Step-based rewards
        if not terminated:
            reward += 0.1 # Survival reward
            
            # Reward for moving towards food
            new_dist_food = abs(new_head[0] - self.food_pos[0]) + abs(new_head[1] - self.food_pos[1])
            if new_dist_food < old_dist_food and old_dist_food <= 5:
                reward += 0.5

            # Penalty for moving towards a nearby wall
            is_near_wall = False
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    check_x, check_y = new_head[0] + dx, new_head[1] + dy
                    if 0 <= check_x < self.GRID_W and 0 <= check_y < self.GRID_H and self.grid[check_y, check_x] == 1:
                        if abs(dx) + abs(dy) <= 2:
                            is_near_wall = True
                            break
                if is_near_wall: break
            
            if is_near_wall:
                reward -= 0.2


        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        if self.head_flash_timer > 0:
            self.head_flash_timer -= 1

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
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
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "length": len(self.snake),
        }

    def _render_game(self):
        # Draw maze walls
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self.grid[y, x] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, (x * self.BLOCK_SIZE, y * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE))

        # Draw food
        food_rect = pygame.Rect(self.food_pos[0] * self.BLOCK_SIZE, self.food_pos[1] * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
        center_x = food_rect.centerx
        center_y = food_rect.centery
        radius = self.BLOCK_SIZE // 2 - 1
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_FOOD)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_FOOD)
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            rect = pygame.Rect(segment[0] * self.BLOCK_SIZE, segment[1] * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
            color = self.COLOR_SNAKE
            if i == 0: # Head
                color = self.COLOR_SNAKE_FLASH if self.head_flash_timer > 0 else self.COLOR_SNAKE_HEAD
            
            # Inset rectangle for segmented look
            pygame.draw.rect(self.screen, color, rect.inflate(-2, -2))

    def _render_ui(self):
        score_text = f"Length: {len(self.snake)} / {self.TARGET_LENGTH}"
        text_surface = self.font_ui.render(score_text, True, self.COLOR_UI)
        self.screen.blit(text_surface, (10, 10))
    
    def _generate_maze(self, width, height):
        # Use a 2D grid, 1 for wall, 0 for path.
        # Ensure odd dimensions for classic maze generation algorithm
        maze_w = (width // 2) * 2 - 1
        maze_h = (height // 2) * 2 - 1
        
        grid = np.ones((height, width), dtype=np.uint8)
        
        # Randomized DFS
        stack = deque()
        start_x, start_y = (1, 1)
        grid[start_y, start_x] = 0
        stack.append((start_x, start_y))
        
        while stack:
            cx, cy = stack[-1]
            
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 1 <= nx < maze_w and 1 <= ny < maze_h and grid[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice([tuple(n) for n in neighbors])
                # Carve path
                grid[ny, nx] = 0
                grid[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Create some loops by removing a small percentage of walls
        num_walls_to_remove = int(maze_w * maze_h * 0.05)
        for _ in range(num_walls_to_remove):
            rx = self.np_random.integers(1, maze_w - 1)
            ry = self.np_random.integers(1, maze_h - 1)
            if grid[ry, rx] == 1:
                # Ensure it's not creating a 2x2 open space which looks bad
                if (grid[ry-1, rx] + grid[ry+1, rx] == 0) or \
                   (grid[ry, rx-1] + grid[ry, rx+1] == 0):
                    grid[ry, rx] = 0
                    
        return grid

    def _find_empty_cell(self):
        while True:
            x = self.np_random.integers(0, self.GRID_W)
            y = self.np_random.integers(0, self.GRID_H)
            
            is_on_snake = False
            if self.snake:
                 for segment in self.snake:
                    if (x, y) == segment:
                        is_on_snake = True
                        break
            
            if self.grid[y, x] == 0 and not is_on_snake:
                return (x, y)

    def _place_food(self):
        self.food_pos = self._find_empty_cell()

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
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up the display window
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Snake Maze")
    
    terminated = False
    running = True
    clock = pygame.time.Clock()
    
    # Game loop
    action = np.array([0, 0, 0]) # Start with no-op
    
    print(env.user_guide)

    while running:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    action = np.array([0, 0, 0])

        if not terminated:
            # --- User controls ---
            keys = pygame.key.get_pressed()
            move_action = 0 # No-op
            if keys[pygame.K_UP]: move_action = 1
            elif keys[pygame.K_DOWN]: move_action = 2
            elif keys[pygame.K_LEFT]: move_action = 3
            elif keys[pygame.K_RIGHT]: move_action = 4
            
            # Only step when a key is pressed for turn-based feel
            if move_action != 0:
                action[0] = move_action
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step: {info['steps']}, Length: {info['length']}, Reward: {reward:.2f}, Terminated: {terminated}")
                action[0] = 0 # Reset to no-op after one step
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Limit frame rate
        clock.tick(15) # Slower tick for better turn-based playability
        
    env.close()