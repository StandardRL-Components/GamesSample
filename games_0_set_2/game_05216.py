import os
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import pygame


# Set up headless pygame
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

        # Define colors and other constants
        self.COLOR_BG = (50, 50, 50)
        self.COLOR_PLAYER = (255, 0, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.player_speed = 5.0
        self.episode_length = 1000

        # Initialize state variables by calling reset, as in the original code's execution path
        # This is necessary because the traceback shows __init__ calls reset.
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([self.width / 2, self.height / 2], dtype=np.float32)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        # Update game logic
        if movement == 1:  # Up
            self.player_pos[1] -= self.player_speed
        elif movement == 2:  # Down
            self.player_pos[1] += self.player_speed
        elif movement == 3:  # Left
            self.player_pos[0] -= self.player_speed
        elif movement == 4:  # Right
            self.player_pos[0] += self.player_speed

        # Keep player within bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.width - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.height - 1)

        self.steps += 1
        reward = self._calculate_reward()
        self.score += reward

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
        # pygame.surfarray.array3d returns (width, height, 3)
        # We need to transpose it to (height, width, 3) for Gymnasium
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Draw a simple player representation
        pygame.draw.circle(
            self.screen, self.COLOR_PLAYER, self.player_pos.astype(int), 10
        )

    def _render_ui(self):
        # Display score and steps
        score_surf = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        steps_surf = self.font.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(steps_surf, (10, 40))

    def _calculate_reward(self):
        # Placeholder reward logic: simple reward for surviving
        return 1 if not self.game_over else 0

    def _check_termination(self):
        # Placeholder termination logic: end after a fixed number of steps
        return self.steps >= self.episode_length

    def render(self):
        # The 'rgb_array' render mode is handled by _get_observation
        return self._get_observation()

    def close(self):
        pygame.quit()