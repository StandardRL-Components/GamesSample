import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:53:28.546907
# Source Brief: brief_00729.md
# Brief Index: 729
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent must balance the water levels in two
    tanks by controlling their respective inflow/outflow sliders.

    **Visuals:**
    - Clean, minimalist aesthetic with a dark background.
    - Two tanks with animated, wavy water.
    - Sliders next to each tank provide direct control feedback.
    - A central "sync bonus" indicator glows when sliders are aligned.
    - Clear UI for score and other metrics.

    **Gameplay:**
    - The agent controls two sliders using directional actions.
    - A slider position > 50 causes its tank to fill, < 50 causes it to drain.
    - Points are awarded for keeping water levels near the 50% target.
    - Bonus points are awarded for keeping the two sliders at similar positions.
    - The episode ends if a tank overflows/empties, the score goal is met, or
      the step limit is reached.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]` (Movement):
        - 0: No-op
        - 1: Move left slider up
        - 2: Move left slider down
        - 3: Move right slider down
        - 4: Move right slider up
    - `actions[1]` (Space):
        - 1: Instantly set both sliders to the neutral 50% position.
    - `actions[2]` (Shift):
        - 1: Instantly set both sliders to their average position.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Balance the water levels in two tanks by controlling their inflow and outflow sliders. "
        "Earn bonus points for keeping the sliders synchronized."
    )
    user_guide = (
        "Use arrow keys to adjust the sliders for each tank. "
        "Press space to reset sliders to 50% and shift to sync them to their average position."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # --- Visual Constants ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_WATER = (60, 130, 220)
        self.COLOR_UI_PRIMARY = (220, 225, 230)
        self.COLOR_UI_SECONDARY = (100, 110, 120)
        self.COLOR_BONUS = (50, 255, 150)
        self.COLOR_BONUS_GLOW = (50, 255, 150, 50) # RGBA for glow
        self.COLOR_DANGER = (255, 80, 80)

        # --- Fonts ---
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_info = pygame.font.SysFont("Consolas", 16)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 30, bold=True)
            self.font_info = pygame.font.SysFont(None, 22)

        # --- Game Layout ---
        self.tank_width, self.tank_height = 160, 300
        self.tank_y = self.screen_height - self.tank_height - 20
        self.tank_left_rect = pygame.Rect(100, self.tank_y, self.tank_width, self.tank_height)
        self.tank_right_rect = pygame.Rect(self.screen_width - 100 - self.tank_width, self.tank_y, self.tank_width, self.tank_height)
        self.slider_track_height = self.tank_height
        self.slider_left_rect = pygame.Rect(self.tank_left_rect.right + 20, self.tank_y, 10, self.slider_track_height)
        self.slider_right_rect = pygame.Rect(self.tank_right_rect.left - 30, self.tank_y, 10, self.slider_track_height)

        # --- Game Mechanics Constants ---
        self.TARGET_LEVEL = 50.0
        self.LEVEL_TOLERANCE = 5.0
        self.SLIDER_SYNC_TOLERANCE = 5.0
        self.SLIDER_STEP = 2.5
        self.FLOW_RATE_MULTIPLIER = 0.025
        self.WIN_SCORE = 1000
        self.MAX_STEPS = 1000
        self.REWARD_WIN = 100
        self.REWARD_LOSS = -100

        # --- State Variables (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.water_level_left = None
        self.water_level_right = None
        self.slider_pos_left = None
        self.slider_pos_right = None
        self.wave_offset = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.water_level_left = self.np_random.uniform(45, 55)
        self.water_level_right = self.np_random.uniform(45, 55)
        self.slider_pos_left = self.TARGET_LEVEL
        self.slider_pos_right = self.TARGET_LEVEL
        self.wave_offset = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._update_sliders(action)
        self._update_water_levels()

        reward = self._calculate_reward()
        self.score += reward

        terminated = self._check_termination()
        truncated = False
        if terminated:
            self.game_over = True
            terminal_reward = self.REWARD_LOSS
            if self.score >= self.WIN_SCORE:
                terminal_reward = self.REWARD_WIN
            reward += terminal_reward
            # self.score += terminal_reward # score is already updated with per-step reward

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            truncated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_sliders(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if space_held:
            # `space` resets sliders to neutral 50
            self.slider_pos_left = self.TARGET_LEVEL
            self.slider_pos_right = self.TARGET_LEVEL
            # SFX: `ui_confirm.wav`
        elif shift_held:
            # `shift` averages the sliders
            avg = (self.slider_pos_left + self.slider_pos_right) / 2
            self.slider_pos_left = avg
            self.slider_pos_right = avg
            # SFX: `ui_sync.wav`
        else:
            # Movement controls individual sliders
            if movement == 1: self.slider_pos_left += self.SLIDER_STEP  # Up
            elif movement == 2: self.slider_pos_left -= self.SLIDER_STEP # Down
            elif movement == 4: self.slider_pos_right += self.SLIDER_STEP # Right (for right slider up)
            elif movement == 3: self.slider_pos_right -= self.SLIDER_STEP # Left (for right slider down)
            # SFX: `slider_tick.wav` for any movement

        self.slider_pos_left = np.clip(self.slider_pos_left, 0, 100)
        self.slider_pos_right = np.clip(self.slider_pos_right, 0, 100)

    def _update_water_levels(self):
        delta_left = (self.slider_pos_left - self.TARGET_LEVEL) * self.FLOW_RATE_MULTIPLIER
        delta_right = (self.slider_pos_right - self.TARGET_LEVEL) * self.FLOW_RATE_MULTIPLIER

        self.water_level_left += delta_left
        self.water_level_right += delta_right
        # SFX: `water_flow_loop.wav` with volume proportional to abs(delta)

    def _calculate_reward(self):
        reward = 0
        # +1 for each tank within target range
        if abs(self.water_level_left - self.TARGET_LEVEL) <= self.LEVEL_TOLERANCE:
            reward += 1
        if abs(self.water_level_right - self.TARGET_LEVEL) <= self.LEVEL_TOLERANCE:
            reward += 1
        # +2 bonus for synchronized sliders
        if abs(self.slider_pos_left - self.slider_pos_right) <= self.SLIDER_SYNC_TOLERANCE:
            reward += 2
        return reward

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            # SFX: `win_ fanfare.wav`
            return True
        if not (0 < self.water_level_left < 100) or not (0 < self.water_level_right < 100):
            # SFX: `lose_alarm.wav`
            return True
        return False

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
            "water_level_left": self.water_level_left,
            "water_level_right": self.water_level_right,
            "slider_pos_left": self.slider_pos_left,
            "slider_pos_right": self.slider_pos_right,
        }

    def _render_game(self):
        self.wave_offset = (self.wave_offset + 0.1) % (2 * math.pi)
        self._draw_tank(self.tank_left_rect, self.water_level_left, self.slider_pos_left, self.slider_left_rect)
        self._draw_tank(self.tank_right_rect, self.water_level_right, self.slider_pos_right, self.slider_right_rect)
        self._draw_bonus_indicator()

    def _draw_tank(self, tank_rect, water_level, slider_pos, slider_rect):
        # Draw tank outline
        pygame.draw.rect(self.screen, self.COLOR_UI_PRIMARY, tank_rect, 2, border_radius=5)

        # Draw target line
        target_y = int(tank_rect.y + tank_rect.height * (1 - self.TARGET_LEVEL / 100))
        pygame.gfxdraw.hline(self.screen, tank_rect.left, tank_rect.right, target_y, self.COLOR_UI_SECONDARY)

        # Draw water with wave effect
        level_clamped = np.clip(water_level, 0, 100)
        water_height = level_clamped / 100.0 * tank_rect.height
        if water_height > 1:
            water_top_y = tank_rect.bottom - water_height
            points = [(tank_rect.left, tank_rect.bottom)]
            for x in range(tank_rect.left, tank_rect.right + 1):
                wave = math.sin(x * 0.1 + self.wave_offset) * 2
                points.append((x, water_top_y + wave))
            points.append((tank_rect.right, tank_rect.bottom))
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_WATER)

        # Draw slider track
        pygame.draw.rect(self.screen, self.COLOR_UI_SECONDARY, slider_rect, 1, border_radius=3)
        
        # Draw slider marker
        marker_height = 15
        marker_y = slider_rect.bottom - (slider_pos / 100.0 * slider_rect.height) - marker_height / 2
        marker_rect = pygame.Rect(slider_rect.centerx - 8, marker_y, 16, marker_height)
        pygame.draw.rect(self.screen, self.COLOR_UI_PRIMARY, marker_rect, 0, border_radius=4)

    def _draw_bonus_indicator(self):
        is_bonus = abs(self.slider_pos_left - self.slider_pos_right) <= self.SLIDER_SYNC_TOLERANCE
        center_x, center_y = self.screen_width // 2, 40
        
        if is_bonus:
            # Draw glow
            glow_surface = pygame.Surface((40, 40), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surface, 20, 20, 18, self.COLOR_BONUS_GLOW)
            self.screen.blit(glow_surface, (center_x - 20, center_y - 20))
            # Draw main circle
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 10, self.COLOR_BONUS)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 10, self.COLOR_BONUS)
            # SFX: `bonus_active_loop.wav`
        else:
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 8, self.COLOR_UI_SECONDARY)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 8, self.COLOR_UI_SECONDARY)

    def _render_ui(self):
        # Render Score
        score_surf = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_PRIMARY)
        self.screen.blit(score_surf, (20, 20))

        # Render Steps
        steps_surf = self.font_info.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_SECONDARY)
        self.screen.blit(steps_surf, (20, 50))
        
        # Render tank level text
        self._render_level_text(self.tank_left_rect, self.water_level_left)
        self._render_level_text(self.tank_right_rect, self.water_level_right)
        
        # Render slider value text
        self._render_slider_text(self.slider_left_rect, self.slider_pos_left)
        self._render_slider_text(self.slider_right_rect, self.slider_pos_right)

    def _render_level_text(self, tank_rect, level):
        color = self.COLOR_UI_PRIMARY
        if not (0 < level < 100):
            color = self.COLOR_DANGER
        level_text = f"{level:.1f}%"
        surf = self.font_info.render(level_text, True, color)
        text_rect = surf.get_rect(centerx=tank_rect.centerx, y=tank_rect.bottom + 5)
        self.screen.blit(surf, text_rect)

    def _render_slider_text(self, slider_rect, pos):
        pos_text = f"{pos:.1f}"
        surf = self.font_info.render(pos_text, True, self.COLOR_UI_SECONDARY)
        text_rect = surf.get_rect(centerx=slider_rect.centerx, y=slider_rect.bottom + 5)
        self.screen.blit(surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This part of the script is for manual play and will not be part of the
    # final environment used by the testing framework. We need to set up a
    # visible display here.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    
    env = GameEnv()
    # The validation call is removed from the main execution path to avoid
    # creating dummy surfaces when not needed. It's good for dev but not for
    # the main loop.
    # env.validate_implementation() 
    
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Controls ---
    # W/S: Control left slider
    # Up/Down Arrow: Control right slider
    # Space: Reset sliders to 50
    # Left Shift: Average sliders
    
    print("--- MANUAL PLAY ---")
    print("W/S: Left Slider | Arrow Up/Down: Right Slider")
    print("Space: Reset Sliders | Shift: Average Sliders")
    print("-------------------")
    
    # Create a display window for manual play
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Water Balance")

    while not done:
        # Map pygame events to gymnasium action
        movement = 0
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_UP]:
            movement = 4
        elif keys[pygame.K_DOWN]:
            movement = 3
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = np.array([movement, space, shift])
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render for human viewing
        # Pygame uses (width, height), but our obs is (height, width, 3)
        # So we need to transpose it back for display
        display_obs = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(display_obs)
        
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
    env.close()