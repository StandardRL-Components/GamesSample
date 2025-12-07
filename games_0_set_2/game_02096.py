import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import math
import os
import pygame


# Set up Pygame to run headlessly
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

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

        self.render_mode = render_mode
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
        self.font = pygame.font.Font(None, 24)

        # Game constants
        self.COLOR_BG = (50, 50, 50)
        self.COLOR_TRACK = (100, 100, 100)
        self.COLOR_CAR = (255, 0, 0)
        self.COLOR_TEXT = (255, 255, 255)

        # Car properties
        self.car_width = 20
        self.car_height = 10
        self.max_speed = 5.0
        self.acceleration = 0.1
        self.braking = 0.2
        self.turn_speed = 0.05
        self.drift_turn_multiplier = 1.5
        self.friction = 0.02

        # Game state variables (will be initialized in reset)
        self.car_pos = None
        self.car_angle = None
        self.car_speed = None
        self.steps = 0
        self.score = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.car_pos = np.array([self.screen_width / 2, self.screen_height / 2 + 100], dtype=np.float64)
        self.car_angle = -math.pi / 2  # Pointing upwards
        self.car_speed = 0.0
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
        # space_held = action[1] == 1  # Firing, ignored for this fix
        shift_held = action[2] == 1  # Drift

        # Update car physics
        turn_rate = self.turn_speed * (self.drift_turn_multiplier if shift_held else 1.0)
        if movement == 3:  # Left
            self.car_angle -= turn_rate
        if movement == 4:  # Right
            self.car_angle += turn_rate

        if movement == 1:  # Up
            self.car_speed += self.acceleration
        elif movement == 2:  # Down
            self.car_speed -= self.braking

        if self.car_speed > 0:
            self.car_speed -= self.friction
        elif self.car_speed < 0:
            self.car_speed += self.friction
        self.car_speed = np.clip(self.car_speed, -self.max_speed / 2, self.max_speed)

        self.car_pos[0] += self.car_speed * math.cos(self.car_angle)
        self.car_pos[1] += self.car_speed * math.sin(self.car_angle)

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
            self._get_info()
        )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()

    def close(self):
        pygame.quit()

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
        # Draw a simple oval track
        track_rect = pygame.Rect(50, 50, self.screen_width - 100, self.screen_height - 100)
        pygame.draw.ellipse(self.screen, self.COLOR_TRACK, track_rect)
        inner_track_rect = track_rect.inflate(-80, -80)
        pygame.draw.ellipse(self.screen, self.COLOR_BG, inner_track_rect)

        # Draw the car
        car_surface = pygame.Surface((self.car_width, self.car_height), pygame.SRCALPHA)
        car_surface.fill(self.COLOR_CAR)
        
        rotated_car = pygame.transform.rotate(car_surface, -math.degrees(self.car_angle))
        new_rect = rotated_car.get_rect(center=tuple(self.car_pos))
        self.screen.blit(rotated_car, new_rect.topleft)

    def _render_ui(self):
        score_text = self.font.render(f"Score: {self.score:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 35))

    def _calculate_reward(self):
        # Placeholder reward: small reward for moving
        return 0.1 if abs(self.car_speed) > 0.1 else 0.0

    def _check_termination(self):
        # Terminate after a fixed number of steps or if car goes off-screen
        is_off_screen = not self.screen.get_rect().collidepoint(self.car_pos)
        max_steps_reached = self.steps >= 1500
        return is_off_screen or max_steps_reached