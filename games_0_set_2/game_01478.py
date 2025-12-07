import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set up Pygame to run headlessly
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

        # Colors
        self.COLOR_BG = (50, 50, 50)
        self.COLOR_PLAYER = (255, 0, 0)
        self.COLOR_BOOST = (0, 255, 255)
        self.COLOR_TEXT = (255, 255, 255)

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

        # Game state variables (initialized in reset)
        self.player_pos = None
        self.player_speed = 5
        self.boost_pos = None
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

        # Place a boost item randomly
        self._place_boost()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _place_boost(self):
        """Places a boost item at a random location on the screen."""
        self.boost_pos = self.np_random.integers(
            low=[20, 20], high=[self.WIDTH - 20, self.HEIGHT - 20], size=2
        )

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # Boolean - Placeholder for firing
        # shift_held = action[2] == 1  # Boolean - Placeholder for drifting

        # Update game logic
        self._handle_movement(movement)

        # Check for boost collection
        if self.boost_pos is not None and np.linalg.norm(self.player_pos - self.boost_pos) < 25: # player radius 15 + boost radius 10
            self.score += 10
            self._place_boost()
            reward = 1.0
        else:
            reward = 0.0

        self.steps += 1
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

    def _handle_movement(self, movement):
        """Updates player position based on the movement action."""
        if movement == 1:  # Up
            self.player_pos[1] -= self.player_speed
        elif movement == 2:  # Down
            self.player_pos[1] += self.player_speed
        elif movement == 3:  # Left
            self.player_pos[0] -= self.player_speed
        elif movement == 4:  # Right
            self.player_pos[0] += self.player_speed

        # Keep player on screen
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)

    def _calculate_reward(self):
        """Calculates the reward for the current step."""
        # This is now handled directly in the step method for clarity.
        # This method is kept for potential future complex reward logic.
        return 0.0

    def _check_termination(self):
        """Checks if the episode should terminate."""
        return self.steps >= 1000

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame's default is (width, height, 3), we need (height, width, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the main game elements."""
        # Draw boost
        if self.boost_pos is not None:
            pygame.draw.circle(self.screen, self.COLOR_BOOST, self.boost_pos, 10)

        # Draw player
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, self.player_pos.astype(int), 15)

    def _render_ui(self):
        """Renders the UI elements like score and steps."""
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        """Cleans up Pygame resources."""
        pygame.quit()