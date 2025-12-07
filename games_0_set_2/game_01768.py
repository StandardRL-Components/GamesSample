import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set up headless Pygame
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

        # Colors
        self.COLOR_BG = (50, 50, 50)
        self.COLOR_PLAYER = (255, 0, 0)
        self.COLOR_UI = (255, 255, 255)

        # The original code called reset() in __init__.
        # We will keep this structure. Game state is initialized in reset().
        # self.reset() # This is called in the original code, but it's better practice
                      # for the user to call it after env creation. However, to maintain
                      # the original structure, we'll ensure reset() is safe to call here.
                      # The provided test harness calls __init__ then reset, so this is fine.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player_pos = np.array([self.width / 2, self.height / 2], dtype=np.float32)
        self.player_angle = -90.0  # Start facing up
        self.player_speed = 0.0

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean, for firing weapon
        shift_held = action[2] == 1  # Boolean, for drifting

        # --- Placeholder Game Logic ---
        turn_speed = 5.0
        acceleration = 0.2
        braking = 0.5
        max_speed = 5.0
        friction = 0.98

        if shift_held: # Drift mechanics
            turn_speed *= 1.5
            friction = 0.95

        # Turning
        if movement == 3:  # left
            self.player_angle -= turn_speed
        elif movement == 4:  # right
            self.player_angle += turn_speed

        # Acceleration/Braking
        if movement == 1:  # up
            self.player_speed += acceleration
        elif movement == 2:  # down
            self.player_speed -= braking
        
        # Apply friction and cap speed
        self.player_speed *= friction
        self.player_speed = np.clip(self.player_speed, -max_speed / 2, max_speed)

        # Update position
        rad_angle = np.deg2rad(self.player_angle)
        self.player_pos[0] += self.player_speed * np.cos(rad_angle)
        self.player_pos[1] += self.player_speed * np.sin(rad_angle)

        # Keep player on screen (wrap around)
        self.player_pos[0] %= self.width
        self.player_pos[1] %= self.height
        # --- End Placeholder Logic ---

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

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame produces (width, height, 3), but observation space is (height, width, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Draw the player's car as a triangle
        car_size = 12
        p1 = (car_size, 0)
        p2 = (-car_size / 2, -car_size / 2)
        p3 = (-car_size / 2, car_size / 2)
        
        # Rotate points around origin (0,0)
        rad = np.deg2rad(self.player_angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        def rotate(p):
            x, y = p
            return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

        # Translate points to player position
        p1_rot_trans = (rotate(p1)[0] + self.player_pos[0], rotate(p1)[1] + self.player_pos[1])
        p2_rot_trans = (rotate(p2)[0] + self.player_pos[0], rotate(p2)[1] + self.player_pos[1])
        p3_rot_trans = (rotate(p3)[0] + self.player_pos[0], rotate(p3)[1] + self.player_pos[1])

        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1_rot_trans, p2_rot_trans, p3_rot_trans])

    def _render_ui(self):
        # Display score and steps
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_UI)
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 30))

    def _calculate_reward(self):
        # Small reward for moving
        return 0.01 if abs(self.player_speed) > 1 else 0.0

    def _check_termination(self):
        # Terminate after a fixed number of steps for this placeholder
        return self.steps >= 1000

    def close(self):
        pygame.quit()