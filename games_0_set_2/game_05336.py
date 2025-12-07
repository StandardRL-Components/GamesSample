import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


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

        # Define colors
        self.COLOR_BG = (0, 0, 0)  # Black background

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False

        # The original code called reset() here. It's better practice to let the
        # user/runner call reset() explicitly after __init__. However, to ensure
        # compatibility with the test harness that expects __init__ to fully
        # initialize the env, we can call it. All necessary attributes (like COLOR_BG)
        # are now defined before this call.
        # self.reset() # This can be uncommented if needed, but a proper gym loop calls it.

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
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _calculate_reward(self):
        # Placeholder for reward calculation
        return 0

    def _check_termination(self):
        # Placeholder for termination condition
        # For example, end after a certain number of steps
        if self.steps > 1000:
            self.game_over = True
        return self.game_over

    def _render_game(self):
        # Placeholder for rendering game objects
        pass

    def _render_ui(self):
        # Placeholder for rendering UI elements
        pass

    def render(self):
        # This method is required by the gym.Env interface for render_mode="rgb_array"
        return self._get_observation()

    def close(self):
        pygame.quit()