import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
from gymnasium.spaces import MultiDiscrete
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    PlantNurturer Gymnasium Environment

    In this environment, the agent must nurture a procedurally generated plant
    to a target height by managing its water intake.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]`: Movement (0-4) - No effect.
    - `action[1]`: Space (0/1) - Waters the plant if held (1).
    - `action[2]`: Shift (0/1) - No effect.

    **Observation Space:** A 640x400 RGB image of the game state.

    **Reward Structure:**
    - +1 for each step of optimal watering.
    - -5 for overwatering.
    - +100 for reaching the target height.
    """

    game_description = "Nurture a procedurally generated plant to its target height by managing its water intake."
    user_guide = "Press space to water the plant. Be careful not to overwater it!"
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"]}

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    TARGET_HEIGHT = 280.0
    MAX_STEPS = 1200
    PLANT_BASE_Y = 340
    SEGMENT_LENGTH = 4.0

    # --- Colors ---
    COLOR_BG = pygame.Color("#18142c")
    COLOR_POT = pygame.Color("#8c4c3c")
    COLOR_POT_LIP = pygame.Color("#a46454")
    COLOR_SOIL_DRY = pygame.Color("#5a3a32")
    COLOR_SOIL_WET = pygame.Color("#3e2821")
    COLOR_SOIL_FLOODED = pygame.Color("#2a1b16")
    COLOR_PLANT_HEALTHY = pygame.Color("#64d41e")
    COLOR_PLANT_UNHEALTHY = pygame.Color("#c8a45a")
    COLOR_WATER_DROPLET = pygame.Color("#80c0ff")
    COLOR_UI_TEXT = pygame.Color("#e0f0ff")
    COLOR_UI_BG = pygame.Color(30, 30, 50, 180)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_timer = pygame.font.Font(None, 36)

        # Initialize state variables to be set in reset()
        self.steps = None
        self.score = None
        self.plant_height = None
        self.water_level = None
        self.plant_segments = None
        self.water_droplets = None
        self.last_space_press = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.plant_height = 10.0
        self.water_level = 0  # 0: dry, 1-2: optimal, 3-4: overwatered
        self.water_droplets = []
        self.last_space_press = False

        # Generate initial plant stem
        self.plant_segments = [(self.WIDTH // 2, self.PLANT_BASE_Y)]
        self._update_plant_segments()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_raw, shift_held = action
        space_held = space_raw == 1

        self.steps += 1
        reward = 0

        # --- Game Logic ---

        # Plant dries out over time
        if not space_held and self.water_level > 0:
            if self.np_random.random() < 0.05:  # Chance to dry out
                self.water_level -= 1

        # Watering action
        if space_held:
            # Only trigger watering logic on the rising edge of the press
            if not self.last_space_press:
                # Sfx: Water pour sound
                self.water_level += 1
                self.water_level = min(self.water_level, 4)  # Cap water level
                self._spawn_water_droplets()

                if 1 <= self.water_level <= 2:  # Optimal watering
                    growth = self.np_random.uniform(2.0, 4.0)
                    self.plant_height = min(
                        self.TARGET_HEIGHT, self.plant_height + growth
                    )
                    reward = 1.0
                    self.score += 1
                    # Sfx: Positive chime
                else:  # Overwatering (or watering when dry)
                    reward = -5.0
                    self.score -= 5
                    # Sfx: Negative buzz / squish sound
        self.last_space_press = space_held

        # Update dynamic elements
        self._update_plant_segments()
        self._update_particles()

        # --- Termination and Final Reward ---
        terminated = False
        truncated = False
        if self.plant_height >= self.TARGET_HEIGHT:
            reward += 100.0
            terminated = True
            # Sfx: Victory fanfare
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Per Gymnasium docs, time limit is termination
            # Sfx: Failure sound

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

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
            "plant_height": self.plant_height,
            "water_level": self.water_level,
        }

    # --- Rendering Methods ---

    def _render_game(self):
        self._render_pot_and_soil()
        self._render_plant()
        self._render_particles()

    def _render_pot_and_soil(self):
        # Pot body
        pot_rect = pygame.Rect(self.WIDTH // 2 - 50, self.PLANT_BASE_Y, 100, 60)
        pygame.draw.rect(
            self.screen,
            self.COLOR_POT,
            pot_rect,
            border_bottom_left_radius=10,
            border_bottom_right_radius=10,
        )
        # Pot lip
        lip_rect = pygame.Rect(self.WIDTH // 2 - 60, self.PLANT_BASE_Y - 10, 120, 15)
        pygame.draw.rect(self.screen, self.COLOR_POT_LIP, lip_rect, border_radius=5)

        # Soil
        if self.water_level == 0:
            soil_color = self.COLOR_SOIL_DRY
        elif self.water_level <= 2:
            soil_color = self.COLOR_SOIL_WET
        else:
            soil_color = self.COLOR_SOIL_FLOODED
        soil_rect = pygame.Rect(pot_rect.x + 5, pot_rect.y, pot_rect.width - 10, 10)
        pygame.draw.rect(self.screen, soil_color, soil_rect)

    def _render_plant(self):
        if len(self.plant_segments) < 2:
            return

        # Determine plant color based on health
        overwater_factor = max(0, self.water_level - 2) / 2.0  # Range 0-1
        plant_color = self.COLOR_PLANT_HEALTHY.lerp(
            self.COLOR_PLANT_UNHEALTHY, overwater_factor
        )

        # Draw main thick line for the plant stem
        pygame.draw.lines(self.screen, plant_color, False, self.plant_segments, width=8)

        # Draw a slightly brighter inner line for a highlight effect
        highlight_color = plant_color.lerp((255, 255, 255), 0.3)
        pygame.draw.lines(
            self.screen, highlight_color, False, self.plant_segments, width=3
        )

        # Draw smooth joints
        for p in self.plant_segments:
            pygame.gfxdraw.aacircle(
                self.screen, int(p[0]), int(p[1]), 4, plant_color
            )
            pygame.gfxdraw.filled_circle(
                self.screen, int(p[0]), int(p[1]), 4, plant_color
            )

    def _render_particles(self):
        for p in self.water_droplets:
            pos, life = p["pos"], p["life"]
            alpha = max(0, min(255, int(life * 2.55)))  # Fade out
            color = self.COLOR_WATER_DROPLET
            rgba_color = (color.r, color.g, color.b, alpha)
            pygame.gfxdraw.filled_circle(
                self.screen, int(pos[0]), int(pos[1]), 3, rgba_color
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(pos[0]), int(pos[1]), 3, rgba_color
            )

    def _render_ui(self):
        # UI background bar
        ui_bar = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_bar.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bar, (0, 0))

        # Height display
        height_text = f"Height: {self.plant_height:.1f} / {self.TARGET_HEIGHT:.1f}"
        height_surf = self.font_ui.render(height_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(height_surf, (15, 10))

        # Score display
        score_text = f"Score: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 15, 10))

        # Timer display
        time_left = self.MAX_STEPS - self.steps
        timer_text = f"{time_left // 60:02}:{time_left % 60:02}"
        timer_surf = self.font_timer.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH // 2 - timer_surf.get_width() // 2, 6))

    # --- Helper Methods ---

    def _update_plant_segments(self):
        target_segments = int(self.plant_height / self.SEGMENT_LENGTH)

        # Grow plant
        while len(self.plant_segments) < target_segments:
            last_point = self.plant_segments[-1]
            # Add some gentle swaying motion using a sine wave based on height
            sway = math.sin(last_point[1] / 50.0) * 3.0
            new_x = last_point[0] + self.np_random.uniform(-1.5, 1.5) + sway
            # Clamp x to prevent it from going too far off screen
            new_x = np.clip(new_x, 50, self.WIDTH - 50)
            new_y = last_point[1] - self.SEGMENT_LENGTH
            self.plant_segments.append((new_x, new_y))

        # Shrink plant (if necessary, e.g., due to a penalty)
        while (
            len(self.plant_segments) > target_segments and len(self.plant_segments) > 1
        ):
            self.plant_segments.pop()

    def _spawn_water_droplets(self):
        for _ in range(15):
            droplet = {
                "pos": [
                    self.WIDTH // 2 + self.np_random.uniform(-40, 40),
                    self.PLANT_BASE_Y - 10,
                ],
                "vel": [
                    self.np_random.uniform(-0.5, 0.5),
                    self.np_random.uniform(-3, -1),
                ],
                "life": self.np_random.uniform(80, 100),  # Lifespan in frames
            }
            self.water_droplets.append(droplet)

    def _update_particles(self):
        for p in self.water_droplets[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] -= p["vel"][
                1
            ]  # Velocity is negative, so this moves it down
            p["life"] -= 1
            if p["life"] <= 0:
                self.water_droplets.remove(p)

    def close(self):
        pygame.quit()