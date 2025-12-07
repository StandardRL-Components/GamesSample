import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:38:57.681751
# Source Brief: brief_01193.md
# Brief Index: 1193
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium Environment: Light Keeper

    **Objective:** Maneuver a light source to keep a moving target illuminated.
    **Challenge:** The light source's radius oscillates, and the target moves sinusoidally.
    **Win Condition:** Maintain over 75% illumination on the target for the entire 60-second duration.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `action[0]`: Movement (0:None, 1:Up, 2:Down, 3:Left, 4:Right)
    - `action[1]`: Space Button (Unused)
    - `action[2]`: Shift Button (Unused)

    **Observation Space:** `Box(0, 255, (400, 640, 3), uint8)`
    - An RGB image of the game screen.

    **Rewards:**
    - `+1.0` per frame with >= 75% illumination.
    - `-0.1` per frame with < 75% illumination.
    - `+100.0` bonus at the end for a perfect run (100% of frames meeting the threshold).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Maneuver a light source to keep a moving target illuminated as your light's radius and the target's position change."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move your light source."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_DURATION_S = 60
    MAX_STEPS = MAX_DURATION_S * FPS

    # --- Visuals ---
    COLOR_BG = (20, 20, 30)
    COLOR_TARGET = (255, 255, 255)
    COLOR_LIGHT = (255, 255, 0)
    COLOR_UI = (200, 200, 220)
    COLOR_UI_SUCCESS = (100, 255, 100)

    # --- Game Mechanics ---
    PLAYER_SPEED = 5
    TARGET_RADIUS = 30
    TARGET_Y = 200
    TARGET_AMPLITUDE = (SCREEN_WIDTH // 2) - TARGET_RADIUS - 40
    TARGET_PERIOD_S = 20
    TARGET_PERIOD_STEPS = TARGET_PERIOD_S * FPS

    LIGHT_MIN_RADIUS = 25
    LIGHT_MAX_RADIUS = 50
    LIGHT_PERIOD_S = 5
    LIGHT_PERIOD_STEPS = LIGHT_PERIOD_S * FPS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)

        # --- State Variables ---
        self.player_pos = None
        self.target_pos = None
        self.light_radius = None
        self.illumination_pct = None
        self.steps = None
        self.score = None
        self.frames_failed_to_illuminate = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.frames_failed_to_illuminate = 0
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)

        # --- Calculate Initial State of Dynamic Elements ---
        self._update_target_position()
        self._update_light_radius()
        self._calculate_illumination()

        return self._get_observation(), self._get_info()

    def step(self, action):
        # --- Unpack Action ---
        movement = action[0]

        self.steps += 1

        # --- Update Player Position ---
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED  # Up
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED  # Down
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED  # Left
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED  # Right

        # --- Clamp Player to Screen ---
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.SCREEN_WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.SCREEN_HEIGHT)

        # --- Update Dynamic Game Elements ---
        self._update_target_position()
        self._update_light_radius()
        self._calculate_illumination()

        # --- Calculate Reward ---
        reward = self._calculate_reward()
        self.score += reward

        # --- Check Termination ---
        terminated = self.steps >= self.MAX_STEPS
        if terminated:
            # Win condition: perfect illumination for the whole duration
            if self.frames_failed_to_illuminate == 0:
                # SFX: Play win jingle
                reward += 100.0  # Add terminal bonus
                self.score += 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _update_target_position(self):
        angle = 2 * math.pi * self.steps / self.TARGET_PERIOD_STEPS
        x_offset = self.TARGET_AMPLITUDE * math.sin(angle)
        self.target_pos = np.array([self.SCREEN_WIDTH / 2 + x_offset, self.TARGET_Y], dtype=np.float32)

    def _update_light_radius(self):
        angle = 2 * math.pi * self.steps / self.LIGHT_PERIOD_STEPS
        radius_range = self.LIGHT_MAX_RADIUS - self.LIGHT_MIN_RADIUS
        self.light_radius = self.LIGHT_MIN_RADIUS + (radius_range / 2) * (1 + math.sin(angle))

    def _calculate_illumination(self):
        dist = np.linalg.norm(self.player_pos - self.target_pos)
        intersection_area = self._get_circle_intersection_area(dist, self.light_radius, self.TARGET_RADIUS)
        target_area = math.pi * self.TARGET_RADIUS**2
        self.illumination_pct = (intersection_area / target_area) if target_area > 0 else 0

    def _get_circle_intersection_area(self, d, r1, r2):
        if d >= r1 + r2:
            return 0.0  # Circles are separate
        if d <= abs(r1 - r2):
            return math.pi * min(r1, r2)**2  # One circle contains the other

        r1_sq, r2_sq, d_sq = r1**2, r2**2, d**2
        
        # Clamp arguments to acos to prevent math domain errors from float inaccuracy
        arg1 = max(-1.0, min(1.0, (d_sq + r1_sq - r2_sq) / (2 * d * r1)))
        arg2 = max(-1.0, min(1.0, (d_sq + r2_sq - r1_sq) / (2 * d * r2)))

        term1 = r1_sq * math.acos(arg1)
        term2 = r2_sq * math.acos(arg2)
        term3 = 0.5 * math.sqrt(max(0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))

        return term1 + term2 - term3

    def _calculate_reward(self):
        if self.illumination_pct >= 0.75:
            # SFX: Play subtle positive tick sound
            return 1.0
        else:
            # SFX: Play subtle negative tick sound
            self.frames_failed_to_illuminate += 1
            return -0.1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "illumination_pct": self.illumination_pct,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def _render_game(self):
        # --- Draw Target (White Circle) ---
        target_x, target_y = int(self.target_pos[0]), int(self.target_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, target_x, target_y, self.TARGET_RADIUS, self.COLOR_TARGET)
        pygame.gfxdraw.aacircle(self.screen, target_x, target_y, self.TARGET_RADIUS, self.COLOR_TARGET)

        # --- Draw Light Source & Illumination Effect ---
        player_x, player_y = int(self.player_pos[0]), int(self.player_pos[1])
        light_rad = int(self.light_radius)
        
        # Create a temporary surface for additive blending to create a glow/light effect
        light_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)

        # Draw a soft glow
        for i in range(4):
            glow_radius = light_rad + i * 6
            glow_alpha = 40 - i * 10
            pygame.gfxdraw.filled_circle(light_surface, player_x, player_y, glow_radius, (*self.COLOR_LIGHT, glow_alpha))

        # Draw the main light circle with some transparency
        pygame.gfxdraw.filled_circle(light_surface, player_x, player_y, light_rad, (*self.COLOR_LIGHT, 150))
        pygame.gfxdraw.aacircle(light_surface, player_x, player_y, light_rad, self.COLOR_LIGHT)

        # Blit the light onto the main screen, adding its color to what's underneath
        self.screen.blit(light_surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw a small crosshair at the light's center for easier aiming
        pygame.draw.line(self.screen, self.COLOR_BG, (player_x - 5, player_y), (player_x + 5, player_y), 2)
        pygame.draw.line(self.screen, self.COLOR_BG, (player_x, player_y - 5), (player_x, player_y + 5), 2)

    def _render_ui(self):
        # --- Timer (Top-Left) ---
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"Time: {time_left:.1f}s"
        time_surface = self.font_ui.render(time_text, True, self.COLOR_UI)
        self.screen.blit(time_surface, (10, 10))

        # --- Illumination % (Top-Right) ---
        illum_text = f"Illumination: {self.illumination_pct * 100:.1f}%"
        is_success = self.illumination_pct >= 0.75
        illum_color = self.COLOR_UI_SUCCESS if is_success else self.COLOR_UI
        illum_surface = self.font_ui.render(illum_text, True, illum_color)
        self.screen.blit(illum_surface, (self.SCREEN_WIDTH - illum_surface.get_width() - 10, 10))

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Example Usage ---
    env = GameEnv()
    env.reset()
    
    # --- Manual Play Controls ---
    # Arrow Keys: Move
    # Q: Quit
    
    running = True
    terminated = False
    
    # This section requires a display. If running headlessly, it will fail.
    try:
        display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Light Keeper")
        print("\n--- Manual Control ---")
        print("ACTION      | KEY")
        print("----------------------")
        print("Move Up     | UP")
        print("Move Down   | DOWN")
        print("Move Left   | LEFT")
        print("Move Right  | RIGHT")
        print("Quit        | Q")
    except pygame.error as e:
        print(f"\nCould not set up display for manual play: {e}")
        print("This is expected in a headless environment. The environment itself is functional.")
        running = False # Skip manual play loop

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render for human viewing ---
        # The environment's _get_observation already renders to its internal screen
        # We just need to blit that to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished! Final Score: {info['score']:.2f}")
            # Wait a moment before resetting
            pygame.time.wait(2000)
            env.reset()
            terminated = False

    env.close()