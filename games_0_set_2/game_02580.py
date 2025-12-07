import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import os
import os
import pygame


# Set headless mode for pygame
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

        # Colors
        self.COLOR_BG = (100, 100, 100)  # Grey background
        self.COLOR_TRACK = (50, 50, 50)   # Dark grey track
        self.COLOR_PLAYER = (255, 0, 0)   # Red player
        self.COLOR_UI_TEXT = (255, 255, 255) # White text

        # Game constants
        self.CAR_WIDTH = 20
        self.CAR_LENGTH = 40
        self.MAX_SPEED = 10.0
        self.ACCELERATION = 0.2
        self.BRAKE_FORCE = 0.4
        self.TURN_SPEED = 0.05 # radians per step
        self.FRICTION = 0.98
        self.MAX_STEPS = 2000

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

        # Player state (will be properly initialized in reset)
        self.player_pos = None
        self.player_angle = None
        self.player_speed = None

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Initialize state variables - The original code called reset() here.
        # This is kept to match the original structure, but all attributes are
        # now defined *before* this call to prevent the AttributeError.
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Reset player state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - 100], dtype=np.float64)
        self.player_angle = -math.pi / 2  # Start facing up
        self.player_speed = 0.0

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        # --- Update game logic ---
        if not self.game_over:
            # 1. Handle player input
            turn_direction = 0
            if movement == 3:  # Left
                turn_direction = -1
            elif movement == 4:  # Right
                turn_direction = 1

            if movement == 1:  # Up (accelerate)
                self.player_speed = min(self.MAX_SPEED, self.player_speed + self.ACCELERATION)
            elif movement == 2:  # Down (brake)
                self.player_speed = max(0, self.player_speed - self.BRAKE_FORCE)

            # 2. Update player physics
            if self.player_speed > 0.1:
                turn_rate = self.TURN_SPEED * (1.5 if shift_held else 1.0)
                self.player_angle += turn_direction * turn_rate

            self.player_speed *= self.FRICTION
            self.player_pos[0] += self.player_speed * math.cos(self.player_angle)
            self.player_pos[1] += self.player_speed * math.sin(self.player_angle)

            # Simple wrap-around boundaries
            self.player_pos[0] %= self.WIDTH
            self.player_pos[1] %= self.HEIGHT

            # Firing weapon (placeholder)
            if space_held:
                pass

        # Update game state
        self.steps += 1
        reward = self._calculate_reward()
        terminated = self._check_termination()
        if terminated:
            self.game_over = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _calculate_reward(self):
        # Placeholder reward: small reward for moving
        if self.game_over:
            return 0.0
        return self.player_speed / self.MAX_SPEED * 0.1

    def _check_termination(self):
        # End episode after a fixed number of steps
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
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw a simple oval track
        track_rect = pygame.Rect(50, 50, self.WIDTH - 100, self.HEIGHT - 100)
        pygame.draw.ellipse(self.screen, self.COLOR_TRACK, track_rect)
        inner_track_rect = track_rect.inflate(-100, -100)
        pygame.draw.ellipse(self.screen, self.COLOR_BG, inner_track_rect)

        # Draw the player's car
        car_surface = pygame.Surface((self.CAR_LENGTH, self.CAR_WIDTH), pygame.SRCALPHA)
        car_surface.fill(self.COLOR_PLAYER)
        rotated_car = pygame.transform.rotate(car_surface, -math.degrees(self.player_angle))
        car_rect = rotated_car.get_rect(center=tuple(self.player_pos))
        self.screen.blit(rotated_car, car_rect)

    def _render_ui(self):
        # Render score and steps
        score_text = self.font.render(f"Score: {self.score:.0f}", True, self.COLOR_UI_TEXT)
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()