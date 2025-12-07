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
        "Controls: Arrows to move cursor. Hold Shift to cycle crystal type. Press Space to place a crystal."
    )

    # Must be a short,user-facing description of the game:
    game_description = (
        "Redirect a laser beam through a crystalline cavern to hit the target before time runs out."
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
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Define colors
        self.COLOR_BG = (10, 10, 20)  # Dark cavern color
        self.COLOR_WHITE = (255, 255, 255)

        # Define font
        self.font = pygame.font.Font(None, 24)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False

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
        
        if terminated:
            self.game_over = True

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

    def _render_game(self):
        # Placeholder for game rendering logic.
        # In a real game, this would draw crystals, lasers, targets, etc.
        pass

    def _render_ui(self):
        # Render score and steps on the screen
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_WHITE)
        self.screen.blit(steps_text, (10, 30))

    def _calculate_reward(self):
        # Placeholder for reward calculation
        return 0.0

    def _check_termination(self):
        # Placeholder for termination condition (e.g., time limit)
        return self.steps >= 1000 or self.game_over

    def render(self):
        # This method is not strictly required by the problem, but is good practice
        # for Gymnasium environments.
        return self._get_observation()

    def close(self):
        pygame.quit()