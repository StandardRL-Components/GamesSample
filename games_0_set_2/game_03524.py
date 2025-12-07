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

        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
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

        # Game state is initialized in reset. The original code called self.reset() here,
        # which is fine as long as all attributes used by reset() are defined first.
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.player_speed = 5

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        # Update game logic
        self.steps += 1

        # Simple player movement
        if movement == 1:  # up
            self.player_pos[1] -= self.player_speed
        elif movement == 2:  # down
            self.player_pos[1] += self.player_speed
        elif movement == 3:  # left
            self.player_pos[0] -= self.player_speed
        elif movement == 4:  # right
            self.player_pos[0] += self.player_speed

        # Keep player on screen
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)

        # Example: gain score for moving
        if movement != 0:
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

    def _calculate_reward(self):
        # Placeholder reward logic
        return 1

    def _check_termination(self):
        # Placeholder termination logic
        if self.steps >= 1000:
            self.game_over = True
        return self.game_over

    def _render_game(self):
        # Placeholder game rendering
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, self.player_pos, 10)

    def _render_ui(self):
        # Placeholder UI rendering
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
        # Transpose from (width, height, 3) to (height, width, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()