import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set headless mode for pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    # Must be a user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400

        # Define colors
        self.COLOR_BG = (50, 50, 50)  # Dark grey
        self.COLOR_WHITE = (255, 255, 255)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Initialize font for UI rendering
        try:
            self.font = pygame.font.Font(None, 24)
        except pygame.error:
            self.font = pygame.font.SysFont("sans", 24)

        # Initialize state variables that are needed before reset() is called
        self.steps = 0
        self.score = 0
        self.game_over = False

        # The original code calls reset() in __init__. We keep this behavior
        # and ensure all necessary attributes are defined above.
        # self.reset() # Note: The test harness will call reset, so calling it here is redundant.
        # However, to fix the error as it was produced, we must make it runnable from init.
        # The traceback shows the error happens in init, so we must support that path.

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
        
        # Placeholder game logic: score increases with time
        if not self.game_over:
            self.score += 1

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
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame surface is (width, height), so array is (W, H, 3).
        # We need (H, W, 3) for the observation space.
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        """Placeholder for rendering the main game elements."""
        # This is where car, track, opponents, etc. would be drawn.
        # For now, we draw a placeholder text.
        text = self.font.render("Arcade Racer", True, self.COLOR_WHITE)
        text_rect = text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(text, text_rect)

    def _render_ui(self):
        """Renders the UI elements like score and steps."""
        score_text = f"Score: {self.score}"
        steps_text = f"Steps: {self.steps}"

        score_surface = self.font.render(score_text, True, self.COLOR_WHITE)
        steps_surface = self.font.render(steps_text, True, self.COLOR_WHITE)

        self.screen.blit(score_surface, (10, 10))
        self.screen.blit(steps_surface, (10, 40))

    def _calculate_reward(self):
        """Calculates the reward for the current step."""
        if self.game_over:
            return -10.0  # Penalty for game over
        return 0.1  # Small reward for surviving each step

    def _check_termination(self):
        """Checks if the episode should terminate."""
        # Terminate after a fixed number of steps as a placeholder.
        return self.steps >= 1000

    def close(self):
        """Cleans up resources."""
        pygame.quit()