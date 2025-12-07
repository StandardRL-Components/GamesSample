import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set up Pygame to run in a headless environment
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

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # Define colors and other constants
        self.COLOR_BG = (50, 50, 50)  # FIX: Define the missing color attribute

        # Initialize state variables
        # The reset method is called to set the initial state
        # It's called here to ensure the environment is ready after initialization
        # Note: self.reset() will be called by the environment wrapper (e.g. `gym.make`)
        # before the first step, so calling it here is for standalone instantiation.
        # However, to fix the bug, we need to ensure all attributes used in reset() are defined first.

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
        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = False  # Truncation is not used in this simple case

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
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
        # Pygame surface to numpy array, then transpose from (W, H, C) to (H, W, C)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Placeholder for rendering the main game elements (e.g., car, track)
        pass

    def _render_ui(self):
        # Placeholder for rendering UI elements (e.g., score, timer)
        pass

    def _calculate_reward(self):
        # Placeholder for reward logic
        return 0

    def _check_termination(self):
        # Placeholder for termination logic (e.g., game over after a certain time)
        return self.steps >= 1000

    def render(self):
        # The new Gymnasium API expects render() to return the rendering.
        return self._get_observation()

    def close(self):
        # Clean up resources
        pygame.quit()