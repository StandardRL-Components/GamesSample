import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set Pygame to run in a headless mode, essential for server-side execution
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a fast-paced arcade racer game.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing documentation
    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # Define observation and action spaces according to requirements
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # Define colors and fonts
        self.COLOR_BG = (50, 50, 50)  # Dark gray background
        self.COLOR_PLAYER = (255, 0, 0) # Red
        self.COLOR_TEXT = (255, 255, 255) # White
        self.font = pygame.font.Font(None, 36)

        # Game state variables are initialized in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the random number generator
        self.np_random = np.random.default_rng(seed)

        # Initialize all game state for a new episode
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Return the initial observation and info dictionary
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack the factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean for firing weapon
        shift_held = action[2] == 1  # Boolean for drifting

        # Update game logic based on the action
        self.steps += 1
        # (Placeholder logic: e.g., score increases with steps)
        self.score += 1

        # Check for termination conditions
        terminated = self._check_termination()
        if terminated:
            self.game_over = True

        # Calculate reward for the current step
        reward = self._calculate_reward()

        # The 'truncated' flag is always False for this environment
        truncated = False

        # Return the standard 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _get_observation(self):
        # Clear the screen with the background color
        self.screen.fill(self.COLOR_BG)

        # Render all game elements onto the screen surface
        self._render_game()

        # Render the UI overlay (score, etc.)
        self._render_ui()

        # Convert the Pygame surface to a numpy array
        # pygame.surfarray.array3d returns (width, height, 3)
        # We transpose to (height, width, 3) for Gymnasium compatibility
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        # Return a dictionary with auxiliary diagnostic information
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        """Placeholder for rendering the main game elements."""
        # Example: Draw a placeholder "player" car
        player_rect = pygame.Rect(300, 300, 40, 80)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

    def _render_ui(self):
        """Placeholder for rendering the user interface."""
        # Example: Render the current score
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

    def _calculate_reward(self):
        """Placeholder for reward calculation logic."""
        # Example: A small positive reward for each step survived
        return 0.1 if not self.game_over else 0.0

    def _check_termination(self):
        """Placeholder for checking if the episode has ended."""
        # Example: End the game after 1000 steps
        return self.steps >= 1000

    def close(self):
        """Clean up resources when the environment is closed."""
        pygame.quit()