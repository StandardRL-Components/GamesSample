import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
from collections import deque
import os
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
        "Classic arcade snake. Eat the red food to grow longer and increase your score. Avoid hitting the walls or your own tail. Reach a length of 50 to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 32
        self.GRID_HEIGHT = 20
        self.CELL_SIZE = 20
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 50

        # Colors
        self.COLOR_BG = (15, 15, 15)
        self.COLOR_GRID = (30, 30, 30)
        self.COLOR_SNAKE_BODY = (40, 180, 99)
        self.COLOR_SNAKE_HEAD = (88, 214, 141)
        self.COLOR_FOOD = (236, 112, 99)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_BOUNDARY = (100, 100, 100)

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
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 36)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 24)
        except IOError:
            self.font_large = pygame.font.SysFont("sans", 36)
            self.font_small = pygame.font.SysFont("sans", 24)
        
        # Initialize state variables that are not reset
        self.snake_body = None
        self.food_pos = None
        self.snake_direction = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.previous_distance_to_food = None

        self.reset()
        
        # This validation is commented out during submission, but useful for development
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize snake
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake_body = deque([
            (start_x - 2, start_y),
            (start_x - 1, start_y),
            (start_x, start_y),
        ])
        self.snake_direction = (1, 0)  # Start moving right

        # Place initial food
        self._place_food()
        
        # For reward calculation
        head = self.snake_body[-1]
        self.previous_distance_to_food = self._manhattan_distance(head, self.food_pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update snake direction based on action, preventing 180-degree turns
        if movement == 1 and self.snake_direction != (0, 1):    # Up
            self.snake_direction = (0, -1)
        elif movement == 2 and self.snake_direction != (0, -1):  # Down
            self.snake_direction = (0, 1)
        elif movement == 3 and self.snake_direction != (1, 0):   # Left
            self.snake_direction = (-1, 0)
        elif movement == 4 and self.snake_direction != (-1, 0):  # Right
            self.snake_direction = (1, 0)
        # movement == 0 is a no-op, continue in the same direction

        # Update game logic
        self.steps += 1
        reward = 0
        terminated = False
        
        # Move snake
        head = self.snake_body[-1]
        new_head = (head[0] + self.snake_direction[0], head[1] + self.snake_direction[1])
        
        # Calculate distance-based reward
        current_distance = self._manhattan_distance(new_head, self.food_pos)
        if current_distance < self.previous_distance_to_food:
            reward += 1  # Closer to food
        else:
            reward -= 1  # Further from food or same distance
        self.previous_distance_to_food = current_distance

        # Check for collisions
        # Wall collision
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            self.game_over = True
            reward = -100
        # Self collision
        elif new_head in self.snake_body:
            self.game_over = True
            reward = -100

        if not self.game_over:
            self.snake_body.append(new_head)
            
            # Check for food consumption
            if np.array_equal(new_head, self.food_pos):
                # SFX: Nom nom
                self.score += 1
                reward += 10
                if self.score >= self.WIN_SCORE:
                    # SFX: Victory fanfare
                    reward = 100
                    self.game_over = True
                else:
                    self._place_food()
                    # Recalculate distance after new food is placed
                    self.previous_distance_to_food = self._manhattan_distance(new_head, self.food_pos)
            else:
                self.snake_body.popleft() # Move snake by removing tail

        # Check termination conditions
        truncated = self.steps >= self.MAX_STEPS
        if self.game_over:
            terminated = True
        if truncated and not terminated:
            reward = -50 # Penalty for timeout
        
        terminated = self.game_over or truncated

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _place_food(self):
        possible_positions = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))
        snake_positions = set(self.snake_body)
        empty_positions = list(possible_positions - snake_positions)
        if not empty_positions:
            # Snake has filled the screen, an unlikely win condition
            self.game_over = True
            self.food_pos = (-1, -1) # Place off-screen
        else:
            # self.np_random.choice on a list of tuples returns a numpy array,
            # which causes a ValueError on tuple comparison.
            # Instead, we select an index and then get the tuple from the list
            # to preserve the correct data type.
            idx = self.np_random.integers(0, len(empty_positions))
            self.food_pos = empty_positions[idx]

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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
        # Draw snake
        for i, segment in enumerate(self.snake_body):
            rect = pygame.Rect(
                segment[0] * self.CELL_SIZE, 
                segment[1] * self.CELL_SIZE, 
                self.CELL_SIZE, 
                self.CELL_SIZE
            )
            color = self.COLOR_SNAKE_HEAD if i == len(self.snake_body) - 1 else self.COLOR_SNAKE_BODY
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, tuple(int(c*0.8) for c in color), rect, width=2, border_radius=4) # Outline

        # Draw food
        if self.food_pos[0] >= 0: # Only draw if food is on screen
            food_x = int((self.food_pos[0] + 0.5) * self.CELL_SIZE)
            food_y = int((self.food_pos[1] + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.4)
            pygame.gfxdraw.filled_circle(self.screen, food_x, food_y, radius, self.COLOR_FOOD)
            pygame.gfxdraw.aacircle(self.screen, food_x, food_y, radius, tuple(int(c*0.8) for c in self.COLOR_FOOD))

    def _render_ui(self):
        # Display score
        score_text = f"SCORE: {self.score}"
        text_surface = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))
        
        # Display game over message
        if self.game_over:
            win = self.score >= self.WIN_SCORE
            message = "YOU WIN!" if win else "GAME OVER"
            color = self.COLOR_SNAKE_HEAD if win else self.COLOR_FOOD
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            go_text_surf = self.font_large.render(message, True, color)
            go_text_rect = go_text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(go_text_surf, go_text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_body),
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    # It will not open a window due to the headless setup
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Use a dictionary to map pygame keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Create a display for human play
    pygame.display.init()
    display_surf = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Gym Environment")

    # Main game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default action is no-op for movement
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action[0] = key_to_action[event.key]
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)
            
        # The environment's observation is already a rendered frame
        # We just need to display it
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        
        display_surf.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we control the speed here for human play
        env.clock.tick(10) # 10 moves per second

    env.close()