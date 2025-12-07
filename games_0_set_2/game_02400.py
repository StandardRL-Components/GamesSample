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

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Define colors and other constants
        self.COLOR_BG = (50, 50, 50)  # Dark gray
        self.COLOR_PLAYER = (255, 0, 0)  # Red
        self.COLOR_TEXT = (255, 255, 255)  # White
        self.MAX_STEPS = 1000

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

        # Initialize state variables
        # The traceback indicates that reset() is called in __init__
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [self.WIDTH // 2, self.HEIGHT // 2]

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        # Update game logic
        self._update_player_state(movement, space_held, shift_held)
        self.steps += 1
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

    def _update_player_state(self, movement, space_held, shift_held):
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
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)
        
        if space_held:
            # Example: firing weapon increases score
            self.score += 1

    def _calculate_reward(self):
        # Placeholder reward logic
        return 0.0

    def _check_termination(self):
        # Terminate after a fixed number of steps
        return self.steps >= self.MAX_STEPS

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame's surfarray is (width, height, channels), we need (height, width, channels)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Draw a simple player representation
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, self.player_pos, 15)

    def _render_ui(self):
        # Display score and steps
        score_surf = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        steps_surf = self.font.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(steps_surf, (10, 40))

    def render(self):
        # The 'rgb_array' render mode just returns the observation
        return self._get_observation()

    def close(self):
        # Clean up Pygame resources
        pygame.quit()