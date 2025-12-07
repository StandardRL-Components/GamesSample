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

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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

        # Colors
        self.COLOR_BG = (25, 25, 25)
        self.COLOR_PLAYER = (255, 0, 0)
        self.COLOR_TEXT = (255, 255, 255)

        # Player attributes
        self.player_pos = None
        self.player_angle = 0
        self.player_speed = 0
        self.player_turn_speed = 5.0
        self.player_acceleration = 0.2
        self.player_max_speed = 5.0
        self.player_friction = 0.98

        # Initialize state variables by calling reset
        # Note: The original code called reset() here, which is what led to the
        # traceback. We keep this structure to match the execution flow.
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64)
        self.player_angle = 0
        self.player_speed = 0
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
        self._handle_input(movement)
        self._update_player_state()

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

    def _handle_input(self, movement):
        # movement: 0-none, 1-up, 2-down, 3-left, 4-right
        if movement == 1:  # Accelerate
            self.player_speed += self.player_acceleration
        if movement == 2:  # Brake/Reverse
            self.player_speed -= self.player_acceleration * 0.5
        if movement == 3:  # Turn Left
            self.player_angle -= self.player_turn_speed
        if movement == 4:  # Turn Right
            self.player_angle += self.player_turn_speed

    def _update_player_state(self):
        # Cap speed
        self.player_speed = np.clip(self.player_speed, -self.player_max_speed / 2, self.player_max_speed)

        # Apply friction
        self.player_speed *= self.player_friction
        if abs(self.player_speed) < 0.05:
            self.player_speed = 0

        # Update position based on angle and speed
        rad_angle = math.radians(self.player_angle)
        self.player_pos[0] += self.player_speed * math.cos(rad_angle)
        self.player_pos[1] += self.player_speed * math.sin(rad_angle)

        # Wrap player around screen edges
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT

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

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Draw the player as a rotated triangle
        player_size = 15
        px, py = self.player_pos
        rad_angle = math.radians(self.player_angle)

        p1 = (
            px + player_size * math.cos(rad_angle),
            py + player_size * math.sin(rad_angle)
        )
        p2 = (
            px + player_size * 0.5 * math.cos(rad_angle + math.radians(150)),
            py + player_size * 0.5 * math.sin(rad_angle + math.radians(150))
        )
        p3 = (
            px + player_size * 0.5 * math.cos(rad_angle - math.radians(150)),
            py + player_size * 0.5 * math.sin(rad_angle - math.radians(150))
        )
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3])

    def _render_ui(self):
        # Render score and steps
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 40))

    def _calculate_reward(self):
        # Placeholder reward function
        return 0.0

    def _check_termination(self):
        # Placeholder termination condition (e.g., after 1000 steps)
        return self.steps >= 1000

    def close(self):
        pygame.quit()