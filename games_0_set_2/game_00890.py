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
        "Controls: ←→ to move, ↑↓ to rotate. Press space to hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist, grid-based block breaker. Clear lines by strategically placing falling tetrominoes."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and game dimensions
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

        # Define colors and font
        self.COLOR_BG = (20, 20, 20)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PLAYER = (200, 50, 50)
        self.FONT = pygame.font.Font(None, 24)

        # Initialize state variables
        self.player_pos = [0, 0]
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # The initial reset is now safe because constants are defined
        # Note: self.reset() is not called here, as per Gymnasium's new API guidelines.
        # The user is expected to call reset() after creating the environment.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [self.screen_width // 2, self.screen_height // 2]

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # Boolean (placeholder)
        # shift_held = action[2] == 1  # Boolean (placeholder)

        # Update game logic based on action
        if movement == 1:  # Up
            self.player_pos[1] -= 5
        elif movement == 2:  # Down
            self.player_pos[1] += 5
        elif movement == 3:  # Left
            self.player_pos[0] -= 5
        elif movement == 4:  # Right
            self.player_pos[0] += 5
        
        # Keep player on screen
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.screen_width - 20)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.screen_height - 20)

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
            self._get_info()
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
        # Pygame array is (width, height, channels), needs to be (height, width, channels)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Placeholder: draw a rectangle representing the player/piece
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (*self.player_pos, 20, 20))

    def _render_ui(self):
        # Display score and steps
        text_surface = self.FONT.render(
            f"Score: {self.score} | Steps: {self.steps}", True, self.COLOR_TEXT
        )
        self.screen.blit(text_surface, (10, 10))

    def _calculate_reward(self):
        # Placeholder reward logic
        return 0

    def _check_termination(self):
        # Placeholder termination logic (e.g., after a certain number of steps)
        return self.steps >= 1000

    def close(self):
        pygame.quit()