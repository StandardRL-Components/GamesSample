import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
from collections import deque
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
        "Guide the growing snake to eat the red food. Reach a score of 100 to win. Avoid hitting the walls or the snake's own body."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 2500
        self.WIN_SCORE = 100

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_SNAKE_BODY = (40, 200, 120)
        self.COLOR_SNAKE_HEAD = (100, 255, 180)
        self.COLOR_FOOD = (255, 80, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_GAMEOVER = (255, 50, 50)
        self.COLOR_WIN = (255, 215, 0)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Initialize state variables
        self.snake_body = None
        self.food_pos = None
        self.direction = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        # Initialize state via reset. A seed is needed for the first reset.
        # self.reset() will be called by the environment wrapper, but we do it here
        # to ensure all state variables are initialized for validation.
        # We don't need to capture the output here.
        self.reset(seed=0)

        # Validate implementation after full initialization
        # self.validate_implementation() # Optional validation call
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # Initialize snake
        start_x, start_y = self.GRID_W // 2, self.GRID_H // 2
        self.snake_body = deque([
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y),
        ])
        self.direction = (1, 0)  # Start moving right

        # Place initial food
        self._place_food()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # If game is over, do nothing and return current state
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update direction based on action, preventing reversal
        new_direction = self.direction
        if movement == 1 and self.direction != (0, 1):    # Up
            new_direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1):  # Down
            new_direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):   # Left
            new_direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0):  # Right
            new_direction = (1, 0)
        self.direction = new_direction

        # Get current head and calculate next position
        head_x, head_y = self.snake_body[0]
        next_head_pos = (head_x + self.direction[0], head_y + self.direction[1])

        # Calculate distance to food for reward shaping
        dist_before = self._distance_to_food(self.snake_body[0])

        # Move snake by adding new head
        self.snake_body.appendleft(next_head_pos)

        # Initialize reward and termination flag for this step
        reward = 0
        terminated = False

        # Check for termination conditions (collision)
        if self._check_collision():
            self.game_over = True
            terminated = True
            reward = -100  # Penalty for dying
        else:
            # Check for food consumption
            if next_head_pos == self.food_pos:
                self.score += 10
                reward = 10  # Reward for eating food
                self._place_food()
                
                # Check for win condition
                if self.score >= self.WIN_SCORE:
                    self.win = True
                    self.game_over = True
                    terminated = True
                    reward = 100 # Big reward for winning
            else:
                # If no food is eaten, the tail is removed
                self.snake_body.pop()
                
                # Continuous reward shaping if no other event occurred
                dist_after = self._distance_to_food(next_head_pos)
                if dist_after < dist_before:
                    reward = 0.1
                else:
                    reward = -0.1
        
        # Update step counter and check for max steps
        self.steps += 1
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True # Treat max steps as game over

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _distance_to_food(self, pos):
        return abs(pos[0] - self.food_pos[0]) + abs(pos[1] - self.food_pos[1])

    def _check_collision(self):
        head = self.snake_body[0]
        # Wall collision
        if not (0 <= head[0] < self.GRID_W and 0 <= head[1] < self.GRID_H):
            return True
        # Self collision
        for i in range(1, len(self.snake_body)):
            if head == self.snake_body[i]:
                return True
        return False

    def _place_food(self):
        empty_cells = []
        snake_pos_set = set(self.snake_body)
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                if (x, y) not in snake_pos_set:
                    empty_cells.append((x, y))
        
        if not empty_cells: # Should not happen unless snake fills screen
             self.game_over = True
             self.win = True # Consider this a win
             return

        # Use np_random to select an index, then get the tuple from the list.
        # This avoids creating a numpy array from the tuple, which caused the error.
        food_idx = self.np_random.integers(len(empty_cells))
        self.food_pos = empty_cells[food_idx]

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
        # Draw grid
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw food
        if self.food_pos is not None:
            fx, fy = self.food_pos
            food_rect = pygame.Rect(fx * self.GRID_SIZE, fy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_FOOD, food_rect)

        # Draw snake
        if self.snake_body:
            # Body
            for i, segment in enumerate(list(self.snake_body)[1:]):
                sx, sy = segment
                body_rect = pygame.Rect(sx * self.GRID_SIZE, sy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_SNAKE_BODY, body_rect)
            
            # Head
            hx, hy = self.snake_body[0]
            head_rect = pygame.Rect(hx * self.GRID_SIZE, hy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, head_rect)

    def _render_ui(self):
        # Render score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render game over/win message
        if self.game_over:
            if self.win:
                msg_text = self.font_large.render("YOU WIN!", True, self.COLOR_WIN)
            else:
                msg_text = self.font_large.render("GAME OVER", True, self.COLOR_GAMEOVER)
            
            text_rect = msg_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            # Add a semi-transparent background for readability
            overlay = pygame.Surface((text_rect.width + 40, text_rect.height + 40), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (text_rect.x - 20, text_rect.y - 20))
            
            self.screen.blit(msg_text, text_rect)

    def _get_info(self):
        # The snake_body might be None during the first reset call in __init__
        # before it's initialized.
        snake_len = len(self.snake_body) if self.snake_body is not None else 0
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": snake_len,
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to game actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Create a Pygame window to display the game
    # Unset the dummy driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    current_action = np.array([0, 0, 0]) # Default action is no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    current_action[0] = key_to_action[event.key]
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    done = False
                    current_action = np.array([0, 0, 0])

        if not done:
            # Step the environment with the current action
            obs, reward, terminated, truncated, info = env.step(current_action)
            done = terminated or truncated
            # Reset movement action to no-op after one step
            current_action[0] = 0

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Run at 10 FPS for a classic snake feel

    env.close()