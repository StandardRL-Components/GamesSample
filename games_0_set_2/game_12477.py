import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # --- Fixes for test failures ---
    game_description = (
        "Control the inflow and outflow valves of a water tank to keep the water level within a specific target range."
    )
    user_guide = (
        "Controls: ↑ to increase inflow, ↓ to decrease inflow. ← to decrease outflow, → to increase outflow."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TANK_WIDTH = 200
    TANK_HEIGHT = 300
    TANK_X = (SCREEN_WIDTH - TANK_WIDTH) // 2
    TANK_Y = (SCREEN_HEIGHT - TANK_HEIGHT) // 2 + 20
    MAX_WATER_LEVEL = 100.0
    MAX_VALVE_RATE = 5
    TARGET_RANGE_MIN = 45
    TARGET_RANGE_MAX = 55
    FAIL_BOUND_LOWER = 25
    FAIL_BOUND_UPPER = 75
    WIN_CONDITION_TURNS = 60
    MAX_EPISODE_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_TANK = (100, 110, 120)
    COLOR_WATER = (50, 150, 255)
    COLOR_WATER_GLOW = (150, 200, 255)
    COLOR_INFLOW = (40, 220, 110)
    COLOR_OUTFLOW = (255, 80, 80)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_TARGET_RANGE = (255, 255, 255, 30)  # RGBA for transparency
    COLOR_FAIL_BOUND = (255, 0, 0, 40)  # RGBA for transparency

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False

        self.water_level = 0.0  # For smooth visual interpolation
        self.target_water_level = 0.0  # The actual game state
        self.inflow_rate = 0
        self.outflow_rate = 0
        self.consecutive_in_range = 0

        self.bubbles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False

        self.target_water_level = 50.0
        self.water_level = self.target_water_level
        self.inflow_rate = 0
        self.outflow_rate = 0
        self.consecutive_in_range = 0

        self.bubbles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do not advance the state.
            # Return the last observation and determine term/trunc from the final state.
            is_fail_condition = self.target_water_level < self.FAIL_BOUND_LOWER or self.target_water_level > self.FAIL_BOUND_UPPER
            terminated = self.win_condition_met or is_fail_condition
            truncated = (self.steps >= self.MAX_EPISODE_STEPS) and not terminated
            return self._get_observation(), 0, terminated, truncated, self._get_info()

        # --- Action Handling ---
        movement = action[0]

        # --- Update Valve Settings ---
        # 1=up (inflow+), 2=down (inflow-), 3=left (outflow-), 4=right (outflow+)
        if movement == 1:
            self.inflow_rate = min(self.MAX_VALVE_RATE, self.inflow_rate + 1)
        elif movement == 2:
            self.inflow_rate = max(0, self.inflow_rate - 1)
        elif movement == 3:
            self.outflow_rate = max(0, self.outflow_rate - 1)
        elif movement == 4:
            self.outflow_rate = min(self.MAX_VALVE_RATE, self.outflow_rate + 1)

        # --- Update Game Logic ---
        self.steps += 1

        # Calculate water level change
        level_change = self.inflow_rate - self.outflow_rate
        self.target_water_level += level_change
        self.target_water_level = np.clip(self.target_water_level, 0, self.MAX_WATER_LEVEL)

        # Check if water is in the target range
        if self.TARGET_RANGE_MIN <= self.target_water_level <= self.TARGET_RANGE_MAX:
            self.consecutive_in_range += 1
        else:
            self.consecutive_in_range = 0

        # --- Termination, Truncation, and Reward ---
        terminated = False
        if self.target_water_level < self.FAIL_BOUND_LOWER or self.target_water_level > self.FAIL_BOUND_UPPER:
            terminated = True
        elif self.consecutive_in_range >= self.WIN_CONDITION_TURNS:
            self.win_condition_met = True
            terminated = True
        
        truncated = False
        if not terminated and self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True

        self.game_over = terminated or truncated

        reward = self._calculate_reward(terminated)
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _calculate_reward(self, terminated):
        reward = 0.0

        # Continuous reward for being in range
        if self.TARGET_RANGE_MIN <= self.target_water_level <= self.TARGET_RANGE_MAX:
            reward += 1.0

        # Bonus for reaching the win turn count
        if self.consecutive_in_range == self.WIN_CONDITION_TURNS:
            reward += 5.0

        if terminated:
            if self.win_condition_met:
                reward += 100.0
            elif self.target_water_level < self.FAIL_BOUND_LOWER or self.target_water_level > self.FAIL_BOUND_UPPER:
                reward -= 100.0

        return reward

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "water_level": self.target_water_level,
            "inflow": self.inflow_rate,
            "outflow": self.outflow_rate,
            "consecutive_in_range": self.consecutive_in_range,
        }

    def _get_observation(self):
        # --- Update Visuals ---
        self._update_visual_state()

        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        # --- Convert to numpy array ---
        # Pygame surface is (W, H), observation space is (H, W)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _update_visual_state(self):
        # Smoothly interpolate visual water level towards the target
        self.water_level += (self.target_water_level - self.water_level) * 0.2

        # Update bubbles
        if self.inflow_rate > 0 and not self.game_over:
            for _ in range(self.inflow_rate):
                # SFX: bubble.wav
                self.bubbles.append([
                    self.np_random.uniform(self.TANK_X + 5, self.TANK_X + self.TANK_WIDTH - 5),
                    self.TANK_Y + self.TANK_HEIGHT - 5,
                    self.np_random.uniform(1, 4)  # size
                ])

        new_bubbles = []
        water_surface_y = self.TANK_Y + self.TANK_HEIGHT * (1 - self.water_level / self.MAX_WATER_LEVEL)
        for bubble in self.bubbles:
            bubble[1] -= self.np_random.uniform(0.5, 1.5)  # move up
            bubble[2] *= 0.99  # shrink
            if bubble[1] > water_surface_y and bubble[2] > 0.5:
                new_bubbles.append(bubble)
        self.bubbles = new_bubbles

    def _render_game(self):
        # --- Draw Tank ---
        tank_rect = pygame.Rect(self.TANK_X, self.TANK_Y, self.TANK_WIDTH, self.TANK_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_TANK, tank_rect, 5, border_radius=5)

        # --- Draw Water ---
        water_height = self.TANK_HEIGHT * (self.water_level / self.MAX_WATER_LEVEL)
        water_rect = pygame.Rect(
            self.TANK_X,
            self.TANK_Y + self.TANK_HEIGHT - water_height,
            self.TANK_WIDTH,
            water_height
        )
        pygame.draw.rect(self.screen, self.COLOR_WATER, water_rect, border_bottom_left_radius=5,
                         border_bottom_right_radius=5)

        # --- Draw Water Glow/Surface ---
        surface_y = water_rect.top
        if water_height > 0:
            pygame.draw.line(self.screen, self.COLOR_WATER_GLOW, (self.TANK_X, surface_y),
                             (self.TANK_X + self.TANK_WIDTH, surface_y), 3)

        # --- Draw Bubbles ---
        for x, y, size in self.bubbles:
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), int(size), self.COLOR_WATER_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(size), self.COLOR_WATER_GLOW)

        # --- Draw Range Indicators ---
        self._draw_indicator_zone(self.FAIL_BOUND_LOWER, self.FAIL_BOUND_UPPER, self.COLOR_FAIL_BOUND, True)
        self._draw_indicator_zone(self.TARGET_RANGE_MIN, self.TARGET_RANGE_MAX, self.COLOR_TARGET_RANGE, False)

        # --- Draw Valves ---
        self._draw_valve_gauge(self.TANK_X - 120, self.TANK_Y + 50, "IN", self.inflow_rate, self.COLOR_INFLOW)
        self._draw_valve_gauge(self.TANK_X + self.TANK_WIDTH + 20, self.TANK_Y + 50, "OUT", self.outflow_rate,
                               self.COLOR_OUTFLOW)

    def _draw_indicator_zone(self, min_level, max_level, color, is_fail_zone):
        # Helper surface for transparency
        s = pygame.Surface((self.TANK_WIDTH, self.TANK_HEIGHT), pygame.SRCALPHA)

        if is_fail_zone:
            # Lower fail zone
            y_lower = self.TANK_HEIGHT * (1 - min_level / self.MAX_WATER_LEVEL)
            h_lower = self.TANK_HEIGHT - y_lower
            pygame.draw.rect(s, color, (0, y_lower, self.TANK_WIDTH, h_lower))

            # Upper fail zone
            h_upper = self.TANK_HEIGHT * (1 - max_level / self.MAX_WATER_LEVEL)
            pygame.draw.rect(s, color, (0, 0, self.TANK_WIDTH, h_upper))
        else:
            # Target zone
            y_max = self.TANK_HEIGHT * (1 - max_level / self.MAX_WATER_LEVEL)
            y_min = self.TANK_HEIGHT * (1 - min_level / self.MAX_WATER_LEVEL)
            h = y_min - y_max
            pygame.draw.rect(s, color, (0, y_max, self.TANK_WIDTH, h))

        self.screen.blit(s, (self.TANK_X, self.TANK_Y))

    def _draw_valve_gauge(self, x, y, label, value, color):
        gauge_width = 100
        gauge_height = 20

        self._draw_text(label, self.font_medium, x, y, align="topleft")

        # Draw gauge background
        bg_rect = pygame.Rect(x, y + 30, gauge_width, gauge_height)
        pygame.draw.rect(self.screen, self.COLOR_TANK, bg_rect, border_radius=3)

        # Draw filled portion
        fill_width = (value / self.MAX_VALVE_RATE) * gauge_width
        fill_rect = pygame.Rect(x, y + 30, fill_width, gauge_height)
        pygame.draw.rect(self.screen, color, fill_rect, border_radius=3)

        # Draw value text
        value_text = f"{value}"
        self._draw_text(value_text, self.font_small, x + gauge_width + 10, y + 30, align="topleft")

    def _render_ui(self):
        # --- Draw water level text ---
        level_text = f"{self.target_water_level:.1f}"
        self._draw_text(level_text, self.font_large, self.SCREEN_WIDTH // 2, self.TANK_Y - 15, align="center")

        # --- Draw consecutive turns in range ---
        range_text = f"IN RANGE: {self.consecutive_in_range} / {self.WIN_CONDITION_TURNS}"
        self._draw_text(range_text, self.font_medium, self.SCREEN_WIDTH // 2, self.TANK_Y + self.TANK_HEIGHT + 25,
                        align="center")

        # --- Draw Score and Steps ---
        score_text = f"SCORE: {self.score:.0f}"
        steps_text = f"STEP: {self.steps} / {self.MAX_EPISODE_STEPS}"
        self._draw_text(score_text, self.font_medium, 10, 10, align="topleft")
        self._draw_text(steps_text, self.font_medium, self.SCREEN_WIDTH - 10, 10, align="topright")

        # --- Draw Game Over/Win Message ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            if self.win_condition_met:
                # SFX: win.wav
                msg = "SYSTEM STABLE"
                sub_msg = f"Final Score: {self.score:.0f}"
            elif self.target_water_level < self.FAIL_BOUND_LOWER or self.target_water_level > self.FAIL_BOUND_UPPER:
                # SFX: lose.wav
                msg = "CRITICAL FAILURE"
                sub_msg = "Tank bounds exceeded"
            else:
                # SFX: timeout.wav
                msg = "SIMULATION END"
                sub_msg = "Maximum steps reached"

            self._draw_text(msg, self.font_large, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20, align="center")
            self._draw_text(sub_msg, self.font_medium, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20, align="center")

    def _draw_text(self, text, font, x, y, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        shadow_surface = font.render(text, True, shadow_color)

        if align == "center":
            text_rect.center = (x, y)
        elif align == "topleft":
            text_rect.topleft = (x, y)
        elif align == "topright":
            text_rect.topright = (x, y)

        # Draw shadow first, then text
        self.screen.blit(shadow_surface, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    # The main loop needs a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")

    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False

    # Pygame setup for display
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Water Tank Control")
    clock = pygame.time.Clock()

    print("\n--- Manual Control ---")
    print("W/↑: Inflow +1 | S/↓: Inflow -1")
    print("A/←: Outflow -1 | D/→: Outflow +1")
    print("R: Reset | Q: Quit")

    while not done:
        action = [0, 0, 0]  # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                continue
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_w, pygame.K_UP]:
                    action[0] = 1  # up
                elif event.key in [pygame.K_s, pygame.K_DOWN]:
                    action[0] = 2  # down
                elif event.key in [pygame.K_a, pygame.K_LEFT]:
                    action[0] = 3  # left
                elif event.key in [pygame.K_d, pygame.K_RIGHT]:
                    action[0] = 4  # right
                elif event.key == pygame.K_q:
                    done = True
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
        
        if done:
            break

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Game Over! Final Info: {info}")
            # Allow a moment to see the final screen, then reset
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Render the observation to the display window
        # Need to transpose back for pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(10)  # Limit FPS

    env.close()