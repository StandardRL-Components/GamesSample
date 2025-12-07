import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set up headless Pygame
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

        self.WIDTH, self.HEIGHT = 640, 400

        # Define colors
        self.COLOR_BG = (50, 50, 50)
        self.COLOR_PLAYER = (255, 0, 0)

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

        # Initialize state variables (will be properly set in reset)
        self.player_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, subsequent steps should reset the environment
            return self.reset()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        # Update game logic
        # Simple movement logic for demonstration
        if movement == 1:  # Up
            self.player_pos[1] -= 5
        elif movement == 2:  # Down
            self.player_pos[1] += 5
        elif movement == 3:  # Left
            self.player_pos[0] -= 5
        elif movement == 4:  # Right
            self.player_pos[0] += 5

        # Keep player within screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT - 1)

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
        # pygame.surfarray.array3d returns an array of shape (width, height, 3)
        # We need to transpose it to (height, width, 3) for the observation space
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Draw a simple player character
        if self.player_pos is not None:
            pygame.draw.circle(
                self.screen, self.COLOR_PLAYER, self.player_pos.astype(int), 10
            )

    def _render_ui(self):
        # Placeholder for UI elements like score, etc.
        pass

    def _calculate_reward(self):
        # Placeholder reward logic
        return 0.0

    def _check_termination(self):
        # Placeholder termination logic, e.g., end after 1000 steps
        return self.steps >= 1000

    def render(self):
        # The 'rgb_array' mode is the only one supported,
        # and it's returned by the step and reset methods.
        return self._get_observation()

    def close(self):
        pygame.quit()