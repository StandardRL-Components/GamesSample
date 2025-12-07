import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import math
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

    # Must be a user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True or False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.COLOR_BG = (50, 50, 50)  # Fix: Added missing attribute
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
        self.font = pygame.font.Font(None, 24)

        # Initialize state variables (to be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_angle = 0.0
        self.player_speed = 0.0

        # The original code called reset() in __init__, which caused the error.
        # We keep this structure to match the execution environment.
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.player_angle = -math.pi / 2  # Start facing up
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

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _update_player_state(self, movement, space_held, shift_held):
        # Placeholder physics
        turn_rate = 0.1
        acceleration = 0.2
        braking = 0.5
        friction = 0.04
        max_speed = 5.0

        # Turning
        if movement == 3:  # Left
            self.player_angle -= turn_rate
        if movement == 4:  # Right
            self.player_angle += turn_rate

        # Acceleration/Braking
        if movement == 1:  # Up
            self.player_speed += acceleration
        elif movement == 2:  # Down
            self.player_speed -= braking

        # Apply friction and clamp speed
        self.player_speed *= 1 - friction
        self.player_speed = np.clip(self.player_speed, -max_speed / 2, max_speed)

        # Update position
        self.player_pos[0] += self.player_speed * math.cos(self.player_angle)
        self.player_pos[1] += self.player_speed * math.sin(self.player_angle)

        # Keep player on screen (wrap around)
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT

    def _calculate_reward(self):
        # Placeholder reward logic
        return 0.0

    def _check_termination(self):
        # Placeholder termination condition (e.g., time limit)
        if self.steps >= 1000:
            self.game_over = True
        return self.game_over

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
        # Draw a simple triangular player ship
        player_size = 12
        px, py = self.player_pos
        angle = self.player_angle

        p1 = (px + player_size * math.cos(angle), py + player_size * math.sin(angle))
        p2 = (
            px + player_size * math.cos(angle + 2.5),
            py + player_size * math.sin(angle + 2.5),
        )
        p3 = (
            px + player_size * math.cos(angle - 2.5),
            py + player_size * math.sin(angle - 2.5),
        )
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3])

    def _render_ui(self):
        # Display score and steps
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 30))

    def close(self):
        pygame.quit()