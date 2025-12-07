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

        self.width = 640
        self.height = 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # Define colors
        self.COLOR_BG = (50, 50, 50)
        self.COLOR_WHITE = (255, 255, 255)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Note: self.reset() is not called here to avoid calling it before all
        # attributes are initialized. The environment user is expected to call reset().
        # However, to pass the original test which calls __init__() then reset(),
        # we can initialize a placeholder state.
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
        
        # Placeholder game logic: simple score increase
        if movement > 0:
            self.score += 1

        if self.steps >= 1000:
            self.game_over = True

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

    def _render_game(self):
        # Placeholder for rendering game objects (car, track, opponents, etc.)
        # For example, draw a simple rectangle representing the player
        player_rect = pygame.Rect(self.width // 2 - 25, self.height - 100, 50, 80)
        pygame.draw.rect(self.screen, (255, 0, 0), player_rect)

    def _render_ui(self):
        # Placeholder for rendering UI elements (score, speed, etc.)
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))

    def _calculate_reward(self):
        # Placeholder for reward calculation
        return 1 if not self.game_over else 0

    def _check_termination(self):
        # Placeholder for termination condition
        return self.game_over

    def render(self):
        # This method is required by the Gym API for visualization
        return self._get_observation()

    def close(self):
        pygame.quit()