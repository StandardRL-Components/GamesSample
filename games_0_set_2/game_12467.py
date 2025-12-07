import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:55:53.945194
# Source Brief: brief_02467.md
# Brief Index: 2467
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player balances the flow of two
    oscillating water pumps to equally fill four reservoirs within a time limit.

    The goal is to manage the frequency of two pumps to achieve a balanced
    fill level across four reservoirs without depleting the system's pressure.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]` (Movement):
        - 0: No-op
        - 1: Increase Pump 1 frequency
        - 2: Decrease Pump 1 frequency
        - 3: Decrease Pump 2 frequency
        - 4: Increase Pump 2 frequency
    - `action[1]` (Space): Unused
    - `action[2]` (Shift): Unused

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - Win: +100 (all reservoirs between 49 and 50 units)
    - Loss: -100 (time up or pressure at 0)
    - Continuous: +0.1 for each reservoir within 1 unit of the average level.
    - Event: +5 if all reservoirs are within 1 unit of each other.

    **Termination:**
    - Episode ends when the win condition is met.
    - Episode ends when time limit (90s) is reached.
    - Episode ends when resource (pressure) level reaches 0.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Balance two oscillating water pumps to fill four reservoirs equally. "
        "Manage system pressure and race against the clock to succeed."
    )
    user_guide = (
        "Controls: ↑ to increase Pump 1 frequency, ↓ to decrease it. "
        "→ to increase Pump 2 frequency, ← to decrease it."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    DT = 1.0 / FPS

    TIME_LIMIT = 90.0  # seconds
    MAX_STEPS = int(TIME_LIMIT * FPS)

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_STATIC = (70, 80, 90)
    COLOR_WATER = (50, 150, 255)
    COLOR_WATER_LIT = (120, 200, 255)
    COLOR_PUMP = (150, 160, 170)
    COLOR_PUMP_INDICATOR = (255, 180, 0)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_HIGHLIGHT = (255, 255, 100)
    COLOR_SUCCESS = (100, 255, 100)
    COLOR_FAIL = (255, 100, 100)

    # Game Parameters
    PUMP_OUTPUT = 10.0  # units/sec
    PUMP_FREQ_MIN = 0.1
    PUMP_FREQ_MAX = 1.0
    PUMP_FREQ_STEP = 0.1
    RESERVOIR_MAX_CAPACITY = 50.0
    RESOURCE_DEPLETION_RATE = 0.1

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.reservoir_levels = []
        self.resource_level = 0.0
        self.pump_freqs = []
        self.pump_phases = []
        self.time_elapsed = 0.0
        self.action_feedback_timers = []

        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # validation is done by the test suite

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.reservoir_levels = [0.0, 0.0, 0.0, 0.0]
        self.resource_level = 100.0
        self.pump_freqs = [0.5, 0.5]
        self.pump_phases = [0.0, 0.0]
        self.time_elapsed = 0.0
        self.action_feedback_timers = [0.0, 0.0]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_elapsed += self.DT

        # --- 1. Handle Actions ---
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        freq_changed = [False, False]
        # Action 1/2: Pump 1 Freq
        if movement == 1:
            self.pump_freqs[0] = min(self.PUMP_FREQ_MAX, self.pump_freqs[0] + self.PUMP_FREQ_STEP)
            freq_changed[0] = True
        elif movement == 2:
            self.pump_freqs[0] = max(self.PUMP_FREQ_MIN, self.pump_freqs[0] - self.PUMP_FREQ_STEP)
            freq_changed[0] = True
        # Action 3/4: Pump 2 Freq
        elif movement == 4: # Right for increase
            self.pump_freqs[1] = min(self.PUMP_FREQ_MAX, self.pump_freqs[1] + self.PUMP_FREQ_STEP)
            freq_changed[1] = True
        elif movement == 3: # Left for decrease
            self.pump_freqs[1] = max(self.PUMP_FREQ_MIN, self.pump_freqs[1] - self.PUMP_FREQ_STEP)
            freq_changed[1] = True
        
        if freq_changed[0]:
            self.action_feedback_timers[0] = 0.3 # seconds
            # sfx: pump_freq_change_1.wav
        if freq_changed[1]:
            self.action_feedback_timers[1] = 0.3 # seconds
            # sfx: pump_freq_change_2.wav

        self.action_feedback_timers = [max(0, t - self.DT) for t in self.action_feedback_timers]

        # --- 2. Update Game State ---
        # Update pump phases for oscillation
        for i in range(2):
            self.pump_phases[i] += self.pump_freqs[i] * 2 * math.pi * self.DT
        
        # Calculate water distribution
        split_ratio_1 = (math.sin(self.pump_phases[0]) + 1) / 2
        water_to_res0 = self.PUMP_OUTPUT * split_ratio_1 * self.DT
        water_to_res1 = self.PUMP_OUTPUT * (1 - split_ratio_1) * self.DT

        split_ratio_2 = (math.sin(self.pump_phases[1]) + 1) / 2
        water_to_res2 = self.PUMP_OUTPUT * split_ratio_2 * self.DT
        water_to_res3 = self.PUMP_OUTPUT * (1 - split_ratio_2) * self.DT
        
        # Update reservoir levels
        self.reservoir_levels[0] = min(self.RESERVOIR_MAX_CAPACITY, self.reservoir_levels[0] + water_to_res0)
        self.reservoir_levels[1] = min(self.RESERVOIR_MAX_CAPACITY, self.reservoir_levels[1] + water_to_res1)
        self.reservoir_levels[2] = min(self.RESERVOIR_MAX_CAPACITY, self.reservoir_levels[2] + water_to_res2)
        self.reservoir_levels[3] = min(self.RESERVOIR_MAX_CAPACITY, self.reservoir_levels[3] + water_to_res3)

        # Update resource level
        level_diff = max(self.reservoir_levels) - min(self.reservoir_levels)
        depletion = self.RESOURCE_DEPLETION_RATE * level_diff * self.DT
        self.resource_level = max(0, self.resource_level - depletion)

        # --- 3. Check Termination Conditions ---
        reward = 0
        terminated = False

        # Win condition
        if all(self.RESERVOIR_MAX_CAPACITY - 1 <= level <= self.RESERVOIR_MAX_CAPACITY for level in self.reservoir_levels):
            self.win_state = True
            self.game_over = True
            terminated = True
            reward = 100.0
            # sfx: win.wav

        # Loss conditions
        if not terminated and (self.resource_level <= 0 or self.time_elapsed >= self.TIME_LIMIT):
            self.win_state = False
            self.game_over = True
            terminated = True
            reward = -100.0
            # sfx: lose.wav

        # --- 4. Calculate Reward (if not terminated) ---
        if not terminated:
            # Continuous reward for balance
            avg_level = sum(self.reservoir_levels) / 4.0 if len(self.reservoir_levels) > 0 else 0
            for level in self.reservoir_levels:
                if abs(level - avg_level) <= 1.0:
                    reward += 0.1
            
            # Event reward for achieving near-perfect balance
            if max(self.reservoir_levels) - min(self.reservoir_levels) <= 1.0:
                reward += 5.0
        
        self.score += reward

        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed,
            "resource_level": self.resource_level,
            "reservoir_levels": self.reservoir_levels,
            "pump_freqs": self.pump_freqs,
        }

    def _render_game(self):
        self._render_reservoirs()
        self._render_pumps()
        self._render_resource_bar()

    def _render_reservoirs(self):
        res_width = 80
        res_height = 200
        spacing = 40
        total_width = 4 * res_width + 3 * spacing
        start_x = (self.SCREEN_WIDTH - total_width) // 2
        y_pos = 150

        for i in range(4):
            x = start_x + i * (res_width + spacing)
            
            # Water level
            fill_height = int((self.reservoir_levels[i] / self.RESERVOIR_MAX_CAPACITY) * res_height)
            water_rect = pygame.Rect(x, y_pos + res_height - fill_height, res_width, fill_height)
            pygame.draw.rect(self.screen, self.COLOR_WATER, water_rect)

            # Water surface wave
            if fill_height > 0:
                wave_y = y_pos + res_height - fill_height
                for wx in range(res_width):
                    offset = math.sin(self.time_elapsed * 5 + wx * 0.2) * 2
                    pygame.gfxdraw.pixel(self.screen, int(x + wx), int(wave_y + offset), self.COLOR_WATER_LIT)

            # Reservoir outline
            pygame.draw.rect(self.screen, self.COLOR_STATIC, (x, y_pos, res_width, res_height), 2)

    def _render_pumps(self):
        pump_y = 80
        # Pump 1 (controls reservoirs 0 & 1)
        pump1_x_start = (self.SCREEN_WIDTH - (4 * 80 + 3 * 40)) // 2 + 40
        pump1_x_end = pump1_x_start + 80
        self._render_pump_visual(pump1_x_start, pump1_x_end, pump_y, 0)

        # Pump 2 (controls reservoirs 2 & 3)
        pump2_x_start = (self.SCREEN_WIDTH - (4 * 80 + 3 * 40)) // 2 + 2 * (80 + 40) + 40
        pump2_x_end = pump2_x_start + 80
        self._render_pump_visual(pump2_x_start, pump2_x_end, pump_y, 1)

    def _render_pump_visual(self, x_start, x_end, y, pump_index):
        # Pump base
        pygame.draw.rect(self.screen, self.COLOR_PUMP, (x_start - 20, y - 10, x_end - x_start + 40, 20), border_radius=5)
        # Track for indicator
        pygame.draw.line(self.screen, self.COLOR_STATIC, (x_start, y), (x_end, y), 2)
        
        # Moving indicator
        indicator_pos_ratio = (math.sin(self.pump_phases[pump_index]) + 1) / 2
        indicator_x = x_start + (x_end - x_start) * indicator_pos_ratio
        pygame.gfxdraw.filled_circle(self.screen, int(indicator_x), y, 8, self.COLOR_PUMP_INDICATOR)
        pygame.gfxdraw.aacircle(self.screen, int(indicator_x), y, 8, self.COLOR_PUMP_INDICATOR)

    def _render_resource_bar(self):
        bar_x, bar_y, bar_w, bar_h = 20, 20, 250, 25
        
        # Color interpolation
        r = int(max(0, min(255, 255 * 2 * (1 - self.resource_level / 100.0))))
        g = int(max(0, min(255, 255 * 2 * (self.resource_level / 100.0))))
        color = (r, g, 50)

        # Bar fill
        fill_w = (self.resource_level / 100.0) * bar_w
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_w, bar_h))
        
        # Bar outline
        pygame.draw.rect(self.screen, self.COLOR_STATIC, (bar_x, bar_y, bar_w, bar_h), 2)
        
        # Text
        text_surf = self.font_small.render(f"Pressure", True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (bar_x + 5, bar_y + 4))
        
        percent_text = f"{self.resource_level:.0f}%"
        percent_surf = self.font_small.render(percent_text, True, self.COLOR_TEXT)
        self.screen.blit(percent_surf, (bar_x + bar_w - percent_surf.get_width() - 5, bar_y + 4))

    def _render_ui(self):
        # Timer
        time_left = max(0, self.TIME_LIMIT - self.time_elapsed)
        timer_text = f"Time: {time_left:.1f}s"
        timer_surf = self.font_main.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 20, 20))

        # Score
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 20, 50))

        # Reservoir Levels
        res_width = 80
        spacing = 40
        total_width = 4 * res_width + 3 * spacing
        start_x = (self.SCREEN_WIDTH - total_width) // 2
        y_pos = 125
        for i in range(4):
            x = start_x + i * (res_width + spacing)
            level_text = f"{self.reservoir_levels[i]:.1f}"
            level_surf = self.font_main.render(level_text, True, self.COLOR_TEXT)
            self.screen.blit(level_surf, (x + (res_width - level_surf.get_width()) // 2, y_pos - 25))

        # Pump Frequencies
        # Pump 1
        freq1_text = f"P1: {self.pump_freqs[0]:.1f} Hz"
        color1 = self.COLOR_TEXT_HIGHLIGHT if self.action_feedback_timers[0] > 0 else self.COLOR_TEXT
        freq1_surf = self.font_main.render(freq1_text, True, color1)
        self.screen.blit(freq1_surf, (start_x + res_width // 2, 50))
        
        # Pump 2
        freq2_text = f"P2: {self.pump_freqs[1]:.1f} Hz"
        color2 = self.COLOR_TEXT_HIGHLIGHT if self.action_feedback_timers[1] > 0 else self.COLOR_TEXT
        freq2_surf = self.font_main.render(freq2_text, True, color2)
        self.screen.blit(freq2_surf, (start_x + 2 * (res_width + spacing) + res_width // 2, 50))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.win_state:
            text = "SUCCESS"
            color = self.COLOR_SUCCESS
        else:
            text = "FAILURE"
            color = self.COLOR_FAIL
        
        text_surf = self.font_large.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing.
    # To run with a display, you might need to comment out the os.environ line at the top.
    env = GameEnv()
    obs, info = env.reset()
    
    # The dummy driver does not support display, so we create a new display here.
    # This requires running the script in an environment with a display server.
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Pump Balancer")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            action = [0, 0, 0] # Default action: no-op
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the observation from the environment to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Episode finished. Total reward: {total_reward}")
                print("Press 'r' to reset.")

            clock.tick(GameEnv.FPS)
    except pygame.error as e:
        print(f"Could not create display: {e}")
        print("This is expected if you are running in a headless environment.")
        print("The GameEnv class itself is designed to run headlessly.")

    env.close()