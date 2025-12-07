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
    """
    A Gymnasium environment for a classic Snake game with a retro arcade visual style.

    The player controls a snake on a grid, aiming to eat food pellets to grow longer.
    The game ends if the snake collides with a wall or its own body, or if it reaches
    the target score of 100.
    """
    metadata = {"render_modes": ["rgb_array", "human"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to change the snake's direction. "
        "Survive and eat the red pellets to grow."
    )

    # Short, user-facing description of the game
    game_description = (
        "Classic arcade snake. Eat the red pellets to grow your snake, but don't "
        "run into the walls or your own tail! Reach a score of 100 to win."
    )

    # Frames only advance when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.GRID_WIDTH = 32
        self.GRID_HEIGHT = 20
        self.CELL_SIZE = 20
        self.SCREEN_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.SCREEN_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 100

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_SNAKE_BODY = (50, 205, 50)
        self.COLOR_SNAKE_HEAD = (124, 252, 0)
        self.COLOR_FOOD = (255, 60, 60)
        self.COLOR_FOOD_GLOW = (150, 30, 30)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_GAMEOVER = (255, 80, 80)
        self.COLOR_WIN = (255, 215, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 36)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 24)
        except IOError:
            self.font_large = pygame.font.SysFont("monospace", 36)
            self.font_small = pygame.font.SysFont("monospace", 24)

        # --- Game State Variables (initialized in reset) ---
        self.snake = None
        self.direction = None
        self.food_pos = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.np_random = None

        # --- Initialize and Validate ---
        # self.reset() is called by the validation function
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Initialize snake in the center
        start_x, start_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.snake = deque([(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)])
        self.direction = (1, 0)  # Start moving right

        # Place the first food
        self._place_food()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Unpack and process action ---
        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused

        # Change direction based on movement, preventing 180-degree turns
        if movement == 1 and self.direction != (0, 1):  # Up
            self.direction = (0, -1)
        elif movement == 2 and self.direction != (0, -1):  # Down
            self.direction = (0, 1)
        elif movement == 3 and self.direction != (1, 0):  # Left
            self.direction = (-1, 0)
        elif movement == 4 and self.direction != (-1, 0):  # Right
            self.direction = (1, 0)
        # movement == 0 is a no-op, continue in the current direction

        # --- 2. Update game logic ---
        self.steps += 1
        reward = 0
        terminated = False

        # Calculate new head position
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # --- 3. Check for termination conditions ---
        # Wall collision
        if not (0 <= new_head[0] < self.GRID_WIDTH and 0 <= new_head[1] < self.GRID_HEIGHT):
            reward = -50.0
            terminated = True
            # sfx: wall_thud.wav

        # Self-collision (check against body, excluding the old tail which will move)
        elif new_head in list(self.snake)[:-1]:
            reward = -50.0
            terminated = True
            # sfx: self_bite.wav

        # Max steps reached
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            # sfx: timeout.wav
        
        if terminated:
            self.game_over = True
        else:
            # --- 4. Move snake and check for food ---
            self.snake.appendleft(new_head)

            # Food consumption
            if new_head == self.food_pos:
                self.score += 1
                reward = 1.0
                # sfx: eat_food.wav

                # Check for win condition
                if self.score >= self.WIN_SCORE:
                    reward += 100.0
                    terminated = True
                    self.game_over = True
                    # sfx: win_fanfare.wav
                else:
                    self._place_food()
            else:
                # No food eaten, remove tail segment and give survival reward
                self.snake.pop()
                reward = 0.01 # Small reward for surviving a step

        # --- 5. Return step information ---
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _place_food(self):
        """Places food in a random location not occupied by the snake."""
        possible_positions = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))
        snake_positions = set(self.snake)
        empty_positions = list(possible_positions - snake_positions)

        if not empty_positions:
            # Snake has filled the screen, this is a win/draw scenario
            self.food_pos = (-1, -1) # Place food off-screen
        else:
            idx = self.np_random.integers(0, len(empty_positions))
            self.food_pos = empty_positions[idx]


    def _get_observation(self):
        # Clear screen with background color
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_grid()
        self._render_food()
        self._render_snake()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (and transpose for correct shape)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake),
        }

    def _render_grid(self):
        """Draws the background grid lines."""
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_snake(self):
        """Draws the snake on the grid."""
        if not self.snake:
            return

        # Draw body
        for segment in list(self.snake)[1:]:
            x, y = segment
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SNAKE_BODY, rect)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1) # Outline

        # Draw head
        head_x, head_y = self.snake[0]
        head_rect = pygame.Rect(head_x * self.CELL_SIZE, head_y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_SNAKE_HEAD, head_rect)
        pygame.draw.rect(self.screen, self.COLOR_BG, head_rect, 1) # Outline


    def _render_food(self):
        """Draws the food pellet on the grid with a glow effect."""
        if self.food_pos[0] < 0: return

        cx = int((self.food_pos[0] + 0.5) * self.CELL_SIZE)
        cy = int((self.food_pos[1] + 0.5) * self.CELL_SIZE)
        radius = int(self.CELL_SIZE * 0.4)
        glow_radius = int(radius * 1.5)

        # Draw glow using anti-aliased circle
        pygame.gfxdraw.aacircle(self.screen, cx, cy, glow_radius, self.COLOR_FOOD_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, glow_radius, self.COLOR_FOOD_GLOW)

        # Draw main circle
        pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, self.COLOR_FOOD)
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, self.COLOR_FOOD)

    def _render_ui(self):
        """Renders the score and game over/win messages."""
        # Render score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Render game over or win message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_GAMEOVER

            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = msg_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 128))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(msg_surf, msg_rect)


    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        assert self.score == 0
        assert len(self.snake) == 3

        # Test observation space (after reset)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        # Test food placement
        self.reset()
        initial_snake_len = len(self.snake)
        initial_score = self.score
        self.snake = deque([(self.food_pos[0] - self.direction[0], self.food_pos[1] - self.direction[1])])
        obs, reward, term, trunc, info = self.step(np.array([0, 0, 0])) # Move into food
        assert info["score"] == initial_score + 1
        # The snake starts at length 3, is manually set to 1, then eats and becomes 2.
        # The test compares the final length (2) to the initial length (3).
        # The correct assertion is that 2 == 3 - 1.
        assert info["snake_length"] == initial_snake_len - 1
        assert reward > 0

        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'human' to play the game with keyboard controls
    render_mode = "human" # or "rgb_array"

    env = GameEnv(render_mode="rgb_array")
    
    if render_mode == "human":
        # In human mode, we need a real display
        pygame.display.set_caption("Gymnasium Snake")
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    obs, info = env.reset()
    terminated = False
    total_reward = 0

    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default action: no-op, no buttons
        
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: # Press 'r' to reset
                        obs, info = env.reset()
                        total_reward = 0
                        terminated = False

            # Map keyboard keys to actions
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            # Step the environment
            if not terminated:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

            # Render the observation to the display
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Control frame rate
            env.clock.tick(10) # Slower for human playability
        
        else: # rgb_array mode for training/testing
            if terminated:
                print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}, Steps: {info['steps']}")
                obs, info = env.reset()
                total_reward = 0
                terminated = False
            
            action = env.action_space.sample() # Take random actions
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

    env.close()