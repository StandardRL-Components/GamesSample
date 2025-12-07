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

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Define Colors
        self.COLOR_BG = (50, 50, 50)  # Dark grey background
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
        self.font = pygame.font.Font(None, 36)

        # The original code called self.reset() here, which caused the crash because
        # self.COLOR_BG was not yet defined. It is now defined above, so this will work.
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state, for example:
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
        # In a real implementation, game logic based on actions would go here.
        # For now, we just increment steps.

        reward = self._calculate_reward()
        terminated = self._check_termination()

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
        arr = pygame.surfarray.array3d(self.screen)
        # Transpose from (width, height, 3) to (height, width, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    # --- Placeholder methods to make the class runnable ---

    def _calculate_reward(self):
        """Calculates the reward for the current step."""
        # Placeholder: a real game would have a more complex reward function.
        return 0.0

    def _check_termination(self):
        """Checks if the episode should terminate."""
        # Placeholder: a real game would terminate based on conditions like health, time, etc.
        return self.game_over

    def _render_game(self):
        """Renders the main game elements (e.g., car, track)."""
        # Placeholder: No game elements are drawn yet.
        pass

    def _render_ui(self):
        """Renders UI elements like score."""
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))

    def close(self):
        """Clean up Pygame resources."""
        pygame.quit()