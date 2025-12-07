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

        # Define screen dimensions
        self.screen_width = 640
        self.screen_height = 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # Define colors
        self.COLOR_BG = (50, 50, 50)  # Dark gray background
        self.COLOR_TEXT = (255, 255, 255) # White text

        # The original code called reset() here, which is fine, but we will
        # initialize the state variables directly to avoid issues with
        # uninitialized attributes during the first reset call.
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
        # pygame.surfarray.array3d returns an array in (width, height, channels)
        # We need to transpose it to (height, width, channels)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Placeholder for game rendering logic
        # For example, draw a player car
        player_rect = pygame.Rect(self.screen_width // 2 - 20, self.screen_height - 60, 40, 50)
        pygame.draw.rect(self.screen, (255, 0, 0), player_rect) # Red car

    def _render_ui(self):
        # Render score and steps
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 40))

    def _calculate_reward(self):
        # Placeholder for reward calculation
        return 0.0

    def _check_termination(self):
        # Placeholder for termination condition
        # For example, end the game after a certain number of steps
        return self.steps >= 500

    def close(self):
        pygame.quit()

# Example of how to use the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    print("Observation space shape:", env.observation_space.shape)
    print("Action space shape:", env.action_space.nvec)

    terminated = False
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")
        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, info = env.reset()

    env.close()
    print("Environment closed.")