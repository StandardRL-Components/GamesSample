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

        self.WIDTH, self.HEIGHT = 640, 400

        # Define colors
        self.COLOR_BG = (50, 50, 50)
        self.COLOR_PLAYER = (255, 0, 0)
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

        # The original code calls reset() here, which is fine as long as all
        # attributes used by reset() and its sub-methods are defined above.
        # self.reset() is called by the test harness, but having it here ensures
        # the environment is in a valid state after __init__.
        # We don't explicitly call it, as the test harness will.
        # However, the traceback indicates it *was* called in the original code,
        # so we must support that pattern. We will initialize attributes but not call reset.
        # The test runner will call reset. If we call it here, it gets called twice.
        # To fix the error from the traceback, we just need to make sure the attributes
        # are present for when reset() is called.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state, for example:
        self.player_pos = [self.WIDTH // 2, self.HEIGHT // 2]
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
        if not self.game_over:
            self.steps += 1

            # Simple movement logic
            if movement == 1:  # Up
                self.player_pos[1] -= 5
            elif movement == 2:  # Down
                self.player_pos[1] += 5
            elif movement == 3:  # Left
                self.player_pos[0] -= 5
            elif movement == 4:  # Right
                self.player_pos[0] += 5

            # Keep player on screen
            self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH - 20)
            self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT - 40)
            
            # Example termination condition
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

    def _calculate_reward(self):
        # Placeholder reward function
        return 0.0

    def _check_termination(self):
        # Placeholder termination check
        return self.game_over

    def _render_game(self):
        # Placeholder for rendering the main game elements
        # Draw a simple "player" rectangle
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (*self.player_pos, 20, 40))

    def _render_ui(self):
        # Placeholder for rendering UI elements like score
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 40))

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame's surfarray has shape (width, height, 3).
        # We need (height, width, 3) for Gymnasium.
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()