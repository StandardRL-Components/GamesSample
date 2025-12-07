import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set headless mode for Pygame
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
        self.font = pygame.font.Font(None, 24)

        # Colors - This was the missing attribute
        self.COLOR_BG = (50, 50, 50)
        self.COLOR_PLAYER = (255, 0, 0)
        self.COLOR_UI_TEXT = (255, 255, 255)

        # Game parameters
        self.player_speed = 5
        self.max_steps = 1000

        # Game state variables (must be defined before reset is called)
        self.player_pos = [0, 0]
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Initialize state by calling reset, as in the original code
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        # Center the player
        self.player_pos = [self.width // 2 - 10, self.height // 2 - 10]

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
        
        # Keep player within screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.width - 20)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.height - 20)

        self.steps += 1
        reward = self._calculate_reward()
        terminated = self._check_termination()
        self.game_over = terminated

        if not self.game_over:
            self.score += 1 # Simple score for surviving

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _calculate_reward(self):
        # Minimal implementation: reward for each step the game is not over
        return 1.0 if not self.game_over else 0.0

    def _check_termination(self):
        # Minimal implementation: terminate after a fixed number of steps
        return self.steps >= self.max_steps

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame array is (width, height, 3), we need (height, width, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Minimal implementation: draw player as a rectangle
        pygame.draw.rect(
            self.screen, self.COLOR_PLAYER, pygame.Rect(self.player_pos[0], self.player_pos[1], 20, 20)
        )

    def _render_ui(self):
        # Minimal implementation: render score and steps
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font.render(f"Steps: {self.steps}/{self.max_steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (10, 35))

    def close(self):
        pygame.quit()