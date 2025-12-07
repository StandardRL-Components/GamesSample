import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
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

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.COLOR_BG = (50, 50, 50)
        self.COLOR_TRACK = (120, 120, 120)
        self.COLOR_PLAYER = (255, 0, 0)
        self.COLOR_UI = (255, 255, 255)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # Initialize state variables
        self.player_pos = None
        self.player_angle = None
        self.player_speed = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Call reset to initialize game state properly
        # self.reset() is not called here to avoid calling it twice (once here, once by the wrapper)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_angle = 0.0
        self.player_speed = 0.0

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
        self.game_over = terminated

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _update_player_state(self, movement, space_held, shift_held):
        # 0: none, 1: up, 2: down, 3: left, 4: right
        if movement == 1:
            self.player_speed += 0.5
        if movement == 2:
            self.player_speed -= 0.5
        if movement == 3:
            self.player_angle += 5
        if movement == 4:
            self.player_angle -= 5

        # Apply friction and cap speed
        self.player_speed *= 0.95
        self.player_speed = np.clip(self.player_speed, -3, 8)

        # Update position
        angle_rad = math.radians(self.player_angle)
        self.player_pos[0] += self.player_speed * math.cos(angle_rad)
        self.player_pos[1] -= self.player_speed * math.sin(angle_rad) # Pygame y is inverted

        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)

    def _calculate_reward(self):
        # Placeholder reward: small reward for moving
        return 0.1 if abs(self.player_speed) > 0.1 else -0.01

    def _check_termination(self):
        # Placeholder termination: end after 1000 steps
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
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw a simple track outline
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (50, 50, self.WIDTH - 100, self.HEIGHT - 100), 10)

        # Draw the player as a rotated rectangle
        player_surf = pygame.Surface((30, 20), pygame.SRCALPHA)
        pygame.draw.polygon(player_surf, self.COLOR_PLAYER, [(0, 0), (30, 10), (0, 20)])
        rotated_surf = pygame.transform.rotate(player_surf, self.player_angle)
        new_rect = rotated_surf.get_rect(center=self.player_pos)
        self.screen.blit(rotated_surf, new_rect)

    def _render_ui(self):
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_UI)
        self.screen.blit(steps_text, (10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()