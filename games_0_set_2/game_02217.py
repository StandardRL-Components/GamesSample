import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ↑↓←→ to change the snake's direction."
    )

    # Short, user-facing description of the game
    game_description = (
        "Guide a growing snake to eat food and reach a target length, avoiding walls and its own tail."
    )

    # Frames do not auto-advance; the game is turn-based.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.CELL_SIZE
        
        self.MAX_STEPS = 1000
        self.WIN_LENGTH = 20

        # Colors
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_GRID = (30, 45, 65)
        self.COLOR_SNAKE_HEAD = (100, 255, 100)
        self.COLOR_SNAKE_TAIL = (20, 120, 20)
        self.COLOR_FOOD = (255, 80, 80)
        self.COLOR_FOOD_GLOW = (255, 150, 150)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_WALL = (10, 15, 25)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        try:
            self.font_large = pygame.font.SysFont('Consolas', 32, bold=True)
            self.font_small = pygame.font.SysFont('Consolas', 20)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 40)
            self.font_small = pygame.font.Font(None, 28)

        # --- State Variables ---
        # These are initialized in reset()
        self.snake_body = None
        self.direction = None
        self.pending_direction = None
        self.food_pos = None
        self.steps = 0
        self.score = 0
        self.game_over_reason = ""
        
        # Run validation check
        # self.validate_implementation() # This is called for verification, but not needed for the final env


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        
        # Initial snake position and direction
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_body = deque([
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y)
        ])
        self.direction = (1, 0)  # Moving right
        self.pending_direction = self.direction
        self.score = len(self.snake_body)
        
        # Place initial food
        self._place_food()
        
        self.game_over_reason = ""

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        # space_held and shift_held are ignored per the brief
        
        reward = 0.0
        terminated = False
        
        # --- Update Direction ---
        new_direction = self.pending_direction
        if movement == 1: new_direction = (0, -1)  # Up
        elif movement == 2: new_direction = (0, 1)   # Down
        elif movement == 3: new_direction = (-1, 0)  # Left
        elif movement == 4: new_direction = (1, 0)   # Right
        # movement == 0 (no-op) means continue in the current direction

        # Prevent the snake from reversing
        if new_direction[0] != -self.direction[0] or new_direction[1] != -self.direction[1]:
            self.pending_direction = new_direction
        
        self.direction = self.pending_direction

        # --- Calculate Reward for Food Proximity ---
        head_pos = self.snake_body[0]
        dist_before = math.hypot(head_pos[0] - self.food_pos[0], head_pos[1] - self.food_pos[1])
        
        # --- Update Snake Position ---
        new_head = (head_pos[0] + self.direction[0], head_pos[1] + self.direction[1])
        
        dist_after = math.hypot(new_head[0] - self.food_pos[0], new_head[1] - self.food_pos[1])

        # Reward for moving away from food
        if dist_after > dist_before:
            reward -= 0.2
            
        # --- Check for Collisions ---
        # Wall collision
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            terminated = True
            reward = -100.0
            self.game_over_reason = "Wall Collision!"
        # Self collision
        elif new_head in self.snake_body:
            terminated = True
            reward = -100.0
            self.game_over_reason = "Self Collision!"
        
        # --- Process Step if not Terminated ---
        if not terminated:
            self.snake_body.appendleft(new_head)
            
            # --- Check for Food Consumption ---
            if new_head == self.food_pos:
                reward += 10.0 # Increased reward for eating food
                self.score += 1
                if self.score >= self.WIN_LENGTH:
                    terminated = True
                    reward = 100.0
                    self.game_over_reason = "You Win!"
                else:
                    self._place_food()
            else:
                self.snake_body.pop() # Remove tail if no food was eaten
            
            # Survival reward
            reward += 0.1

        # --- Check for Max Steps ---
        self.steps += 1
        truncated = False
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True # Using terminated as per new Gym API for time limits
            self.game_over_reason = "Time Limit Reached"

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _place_food(self):
        """Places food in a random empty cell."""
        empty_cells = []
        snake_set = set(self.snake_body)
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in snake_set:
                    empty_cells.append((x, y))
        
        if not empty_cells:
            # No space left, this is a rare edge case, treat as a win
            self.food_pos = (-1, -1) # Off-screen
        else:
            # FIX: Use np_random.integers to get an index and select a tuple from the list.
            # This avoids creating a numpy array for food_pos, which caused the ValueError.
            idx = self.np_random.integers(len(empty_cells))
            self.food_pos = empty_cells[idx]

    def _get_observation(self):
        """Renders the game state to the screen surface and returns it as a numpy array."""
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        
        # --- Grid Lines ---
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # --- Food ---
        # FIX: The comparison now works because self.food_pos is a tuple.
        if self.food_pos != (-1, -1):
            food_rect = pygame.Rect(
                self.food_pos[0] * self.CELL_SIZE, 
                self.food_pos[1] * self.CELL_SIZE, 
                self.CELL_SIZE, 
                self.CELL_SIZE
            )
            # Draw a subtle glow effect
            glow_radius = int(self.CELL_SIZE * 0.7)
            pygame.gfxdraw.filled_circle(
                self.screen, 
                food_rect.centerx, 
                food_rect.centery, 
                glow_radius, 
                self.COLOR_FOOD_GLOW
            )
            # Draw the main food circle
            pygame.gfxdraw.filled_circle(
                self.screen, 
                food_rect.centerx, 
                food_rect.centery, 
                int(self.CELL_SIZE * 0.4), 
                self.COLOR_FOOD
            )

        # --- Snake ---
        if self.snake_body:
            num_segments = len(self.snake_body)
            for i, segment in enumerate(self.snake_body):
                rect = pygame.Rect(
                    segment[0] * self.CELL_SIZE, 
                    segment[1] * self.CELL_SIZE, 
                    self.CELL_SIZE, 
                    self.CELL_SIZE
                )
                
                # Gradient color from head to tail
                if num_segments > 1:
                    t = i / (num_segments - 1)
                else:
                    t = 0
                
                color = (
                    int(self.COLOR_SNAKE_HEAD[0] * (1 - t) + self.COLOR_SNAKE_TAIL[0] * t),
                    int(self.COLOR_SNAKE_HEAD[1] * (1 - t) + self.COLOR_SNAKE_TAIL[1] * t),
                    int(self.COLOR_SNAKE_HEAD[2] * (1 - t) + self.COLOR_SNAKE_TAIL[2] * t),
                )
                
                pygame.draw.rect(self.screen, color, rect)
                # Add a border to make segments distinct
                pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1)

        # --- UI ---
        score_text = self.font_large.render(f"Length: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.game_over_reason:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            reason_text = self.font_large.render(self.game_over_reason, True, self.COLOR_TEXT)
            text_rect = reason_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(reason_text, text_rect)

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        """Returns a dictionary with auxiliary diagnostic information."""
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_body) if self.snake_body else 0,
            "food_pos": self.food_pos,
            "snake_head_pos": self.snake_body[0] if self.snake_body else None,
        }
    
    def close(self):
        """Clean up Pygame resources."""
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {obs.shape}"
        assert obs.dtype == np.uint8
        
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # --- Example Usage and Visualization ---
    env = GameEnv()
    try:
        env.validate_implementation()
    except Exception as e:
        print(f"Validation failed: {e}")
        env.close()
        exit()

    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Snake Gym Environment")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Action defaults to no-op (continue current direction)
        action = env.action_space.sample()
        action[0] = 0 # No-op
        action[1] = 0 # Released
        action[2] = 0 # Released

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
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    terminated = False
                    total_reward = 0
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"Game Over. Reason: {info.get('game_over_reason', 'Unknown')}")


        # Render the observation from the environment
        # Pygame uses (W, H), numpy uses (H, W), so we need to transpose
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # The game logic is turn-based, so we can add a small delay for human playability
        clock.tick(10) # 10 FPS for human play

    env.close()
    print(f"Game Over. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")