import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set the SDL_VIDEODRIVER to "dummy" for headless execution
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Hold shift to drift and press space to fire your weapon."
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
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Define colors
        self.COLOR_BG = (20, 20, 20)
        self.COLOR_PLAYER = (255, 0, 0)
        self.COLOR_TEXT = (255, 255, 255)

        # Game state variables
        self.player_pos = None
        self.player_speed = 5
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Font for UI
        self.font = pygame.font.Font(None, 36)

        # The original code called reset() here. While common, it's often better
        # to let the user/wrapper call it. However, to ensure attributes are
        # initialized for any potential internal calls before the first user reset,
        # we initialize the state here.
        self._initialize_state()

    def _initialize_state(self):
        """Initializes all game state variables."""
        self.player_pos = [self.screen_width // 2, self.screen_height // 2]
        self.steps = 0
        self.score = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self._initialize_state()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        self.steps += 1

        # Update game logic based on action
        if movement == 1:  # Up
            self.player_pos[1] -= self.player_speed
        elif movement == 2:  # Down
            self.player_pos[1] += self.player_speed
        elif movement == 3:  # Left
            self.player_pos[0] -= self.player_speed
        elif movement == 4:  # Right
            self.player_pos[0] += self.player_speed

        # Boundary checks to keep player on screen
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.screen_width - 20)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.screen_height - 20)

        # Placeholder for game logic
        # Example: gain score for surviving
        self.score += 1

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
        # Pygame surface is (W, H), surfarray is (W, H, C).
        # We need to transpose to (H, W, C) for Gymnasium.
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        """Renders the main game elements."""
        # Draw player (a simple rectangle)
        if self.player_pos:
            pygame.draw.rect(
                self.screen, self.COLOR_PLAYER, (*self.player_pos, 20, 20)
            )

    def _render_ui(self):
        """Renders the UI elements like score and steps."""
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 40))

    def _calculate_reward(self):
        """Calculates the reward for the current step."""
        # Simple reward for surviving
        return 1

    def _check_termination(self):
        """Checks if the episode should terminate."""
        # End episode after a fixed number of steps
        return self.steps >= 1000

    def render(self):
        # This method is required by the gym.Env interface, but since we're
        # in headless mode, we just return the observation.
        return self._get_observation()

    def close(self):
        pygame.quit()