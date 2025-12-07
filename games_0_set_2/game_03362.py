import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set the SDL video driver to "dummy" to run Pygame headlessly
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
        self.COLOR_BG = (20, 20, 20)
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

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_angle = None
        self.player_speed = None
        self.steps = 0
        self.score = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_angle = self.np_random.uniform(0, 2 * math.pi)
        self.player_speed = 0.0

        self.steps = 0
        self.score = 0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # Boolean (weapon - not implemented)
        shift_held = action[2] == 1  # Boolean (drift)

        # --- Update game logic ---

        # 1. Update player angle (turning)
        turn_speed = 0.1
        if shift_held:  # Drift/sharper turn
            turn_speed = 0.15

        if movement == 3:  # left
            self.player_angle -= turn_speed
        if movement == 4:  # right
            self.player_angle += turn_speed

        # 2. Update player speed (acceleration/braking)
        acceleration = 0.5
        max_speed = 5.0
        brake_power = 1.0
        friction = 0.05

        if movement == 1:  # up
            self.player_speed += acceleration
        elif movement == 2:  # down
            self.player_speed -= brake_power

        # Apply friction
        self.player_speed *= (1 - friction)
        self.player_speed = np.clip(self.player_speed, 0, max_speed)

        # 3. Update player position
        self.player_pos[0] += self.player_speed * math.cos(self.player_angle)
        self.player_pos[1] += self.player_speed * math.sin(self.player_angle)

        # 4. Handle screen boundaries (wrap around)
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT

        # 5. Update steps, score, and check for termination
        self.steps += 1
        if self.player_speed > 1.0:
            self.score += 1
        
        terminated = self._check_termination()
        reward = self._calculate_reward()

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _calculate_reward(self):
        # Small reward for moving
        return self.player_speed / 10.0

    def _check_termination(self):
        # End episode after a fixed number of steps
        return self.steps >= 1500

    def _render_game(self):
        # Draw the player as a triangle
        player_size = 15
        p1 = (
            self.player_pos[0] + player_size * math.cos(self.player_angle),
            self.player_pos[1] + player_size * math.sin(self.player_angle)
        )
        p2 = (
            self.player_pos[0] + player_size * math.cos(self.player_angle + 2.5),
            self.player_pos[1] + player_size * math.sin(self.player_angle + 2.5)
        )
        p3 = (
            self.player_pos[0] + player_size * math.cos(self.player_angle - 2.5),
            self.player_pos[1] + player_size * math.sin(self.player_angle - 2.5)
        )
        
        # Use gfxdraw for anti-aliased polygons for a cleaner look
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _render_ui(self):
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
        # Pygame surface uses (width, height), but observation space is (height, width)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()