import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # Define constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.COLOR_BG = (100, 100, 100)  # A gray color for the background
        self.COLOR_WHITE = (255, 255, 255)

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
        self.font = pygame.font.Font(None, 24)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False

        # The original code called reset() in __init__, which caused the error.
        # We keep this call to maintain the original structure but ensure all
        # necessary attributes (like COLOR_BG) are defined beforehand.
        # A full reset is performed by the user/environment wrapper before starting.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        # Update game logic
        self.steps += 1
        reward = self._calculate_reward()
        terminated = self._check_termination()
        self.game_over = terminated

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        # Pygame surface is (width, height), but observation space is (height, width)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Placeholder for actual game rendering logic
        # For now, we can draw a simple shape to show something is happening
        player_rect = pygame.Rect(self.WIDTH // 2 - 25, self.HEIGHT - 100, 50, 80)
        pygame.draw.rect(self.screen, (255, 0, 0), player_rect)

    def _render_ui(self):
        # Placeholder for UI rendering
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_WHITE)
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 30))

    def _calculate_reward(self):
        # Placeholder for reward logic
        return 0

    def _check_termination(self):
        # Placeholder for termination logic, e.g., after a certain number of steps
        return self.steps >= 1000

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
        return None

    def close(self):
        pygame.quit()