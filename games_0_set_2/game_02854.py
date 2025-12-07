import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set Pygame to run in headless mode
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

        # Define constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.COLOR_BG = (50, 50, 50)  # Dark gray background

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

        # Initialize state variables - they will be properly set in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # Initialize all game state, for example:
        self.steps = 0
        self.score = 0
        self.game_over = False

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
        self.steps += 1
        reward = self._calculate_reward()
        self.score += reward
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
        # Pygame surface is (width, height), but observation space is (height, width)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Placeholder for game rendering logic
        # For example, draw a simple player rectangle
        player_rect = pygame.Rect(self.WIDTH // 2 - 20, self.HEIGHT - 60, 40, 50)
        pygame.draw.rect(self.screen, (255, 0, 0), player_rect)

    def _render_ui(self):
        # Placeholder for UI rendering (score, etc.)
        score_text = self.font.render(f"Score: {self.score:.0f}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font.render(f"Steps: {self.steps}", True, (255, 255, 255))
        self.screen.blit(steps_text, (10, 40))

    def _calculate_reward(self):
        # Placeholder for reward calculation
        # Simple reward for surviving
        return 0.1

    def _check_termination(self):
        # Placeholder for termination condition
        # End game after a fixed number of steps
        return self.steps >= 1000

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # Example of how to use the environment
    env = GameEnv()
    obs, info = env.reset(seed=42)
    print("Observation space shape:", env.observation_space.shape)
    print("Action space shape:", env.action_space.nvec)

    terminated = False
    total_reward = 0
    step_count = 0

    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        if terminated or truncated:
            break

    print(f"Finished after {step_count} steps.")
    print(f"Final score: {info['score']}")
    print(f"Total reward: {total_reward}")

    env.close()