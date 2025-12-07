import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Reservoir Balance: A puzzle game where the player must balance water flow
    between four interconnected reservoirs to reach target fill levels.

    The game uses a MultiDiscrete([5, 2, 2]) action space:
    - actions[0]: Selects a lever (1: Up, 2: Down, 3: Left, 4: Right correspond to levers 0-3)
    - actions[1]: (Space) Increases the selected lever's flow rate.
    - actions[2]: (Shift) Decreases the selected lever's flow rate.

    The goal is to fill at least 3 of the 4 reservoirs to 80% capacity
    or more within 20 turns. Water flows according to lever settings,
    and any overflow spills into the next reservoir clockwise.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Balance water flow between four interconnected reservoirs to reach target fill levels in a turn-based puzzle."
    user_guide = "Use arrow keys (↑↓←→) to select a lever. Press space to increase its flow rate and shift to decrease it."
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 20
    NUM_RESERVOIRS = 4
    RESERVOIR_CAPACITY = 100.0
    LEVER_MIN = 0
    LEVER_MAX = 10
    WIN_THRESHOLD_PERCENT = 80.0
    WIN_CONDITION_COUNT = 3
    VISUAL_LERP_RATE = 0.15 # For smooth water level animation

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_RESERVOIR_EMPTY = (50, 60, 70)
    COLOR_WATER = (60, 120, 220)
    COLOR_WATER_BUBBLE = (180, 210, 255)
    COLOR_TARGET_LINE = (220, 50, 50)
    COLOR_LEVER_BG = (40, 50, 60)
    COLOR_LEVER_FG = (80, 200, 120)
    COLOR_LEVER_ACTIVE = (255, 220, 100)
    COLOR_TEXT = (220, 220, 230)
    COLOR_UI_PANEL = (30, 40, 50)
    COLOR_WIN = (100, 220, 100)
    COLOR_LOSS = (220, 100, 100)


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_status = False

        self.reservoir_levels = np.zeros(self.NUM_RESERVOIRS, dtype=float)
        self.visual_water_levels = np.zeros(self.NUM_RESERVOIRS, dtype=float)
        self.lever_positions = np.zeros(self.NUM_RESERVOIRS, dtype=int)
        self.selected_lever = 0
        self.has_reached_80 = [False] * self.NUM_RESERVOIRS
        self.particles = []

        # --- Pre-calculate UI element positions for efficiency ---
        self._calculate_layout()


    def _calculate_layout(self):
        """Pre-calculates positions and dimensions of UI elements."""
        self.reservoir_rects = []
        self.lever_rects = []
        margin = 40
        top_y = 60
        bottom_y = self.SCREEN_HEIGHT / 2 + 20
        res_w, res_h = 150, 120
        lever_w, lever_h = 20, res_h

        # Top-left (0)
        self.reservoir_rects.append(pygame.Rect(margin, top_y, res_w, res_h))
        self.lever_rects.append(pygame.Rect(margin + res_w + 10, top_y, lever_w, lever_h))
        # Top-right (1)
        self.reservoir_rects.append(pygame.Rect(self.SCREEN_WIDTH - margin - res_w, top_y, res_w, res_h))
        self.lever_rects.append(pygame.Rect(self.SCREEN_WIDTH - margin - res_w - 10 - lever_w, top_y, lever_w, lever_h))
        # Bottom-right (2)
        self.reservoir_rects.append(pygame.Rect(self.SCREEN_WIDTH - margin - res_w, bottom_y, res_w, res_h))
        self.lever_rects.append(pygame.Rect(self.SCREEN_WIDTH - margin - res_w - 10 - lever_w, bottom_y, lever_w, lever_h))
        # Bottom-left (3)
        self.reservoir_rects.append(pygame.Rect(margin, bottom_y, res_w, res_h))
        self.lever_rects.append(pygame.Rect(margin + res_w + 10, bottom_y, lever_w, lever_h))


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_status = False
        self.selected_lever = 0
        self.lever_positions = np.zeros(self.NUM_RESERVOIRS, dtype=int)
        self.has_reached_80 = [False] * self.NUM_RESERVOIRS
        self.particles.clear()

        # Initial water levels are random
        self.reservoir_levels = self.np_random.uniform(0, 20, size=self.NUM_RESERVOIRS)
        self.visual_water_levels = self.reservoir_levels.copy()

        return self._get_observation(), self._get_info()


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Unpack and process action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement selects a lever
        if 1 <= movement <= 4:
            self.selected_lever = movement - 1

        # Space/Shift adjusts the selected lever
        if space_held:
            self.lever_positions[self.selected_lever] += 1
        if shift_held:
            self.lever_positions[self.selected_lever] -= 1

        self.lever_positions = np.clip(self.lever_positions, self.LEVER_MIN, self.LEVER_MAX)

        # --- 2. Update game logic: Water flow ---
        initial_levels = self.reservoir_levels.copy()
        self.reservoir_levels += self.lever_positions.astype(float)

        # Handle spillover in a loop to catch cascading overflows
        spillover_occurred = True
        while spillover_occurred:
            spillover_occurred = False
            for i in range(self.NUM_RESERVOIRS):
                if self.reservoir_levels[i] > self.RESERVOIR_CAPACITY:
                    spillover_occurred = True
                    overflow = self.reservoir_levels[i] - self.RESERVOIR_CAPACITY
                    self.reservoir_levels[i] = self.RESERVOIR_CAPACITY
                    next_reservoir_index = (i + 1) % self.NUM_RESERVOIRS
                    self.reservoir_levels[next_reservoir_index] += overflow

        # Add particles for visual effect
        for i in range(self.NUM_RESERVOIRS):
            water_added = self.reservoir_levels[i] - initial_levels[i]
            if water_added > 0.1:
                self._add_particles(i, water_added)

        self.steps += 1

        # --- 3. Calculate reward and termination ---
        reward = self._calculate_reward()
        terminated = self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            num_full = sum(1 for level in self.reservoir_levels if level >= self.WIN_THRESHOLD_PERCENT)
            if num_full >= self.WIN_CONDITION_COUNT:
                self.win_status = True
                terminal_reward = 100.0
            else:
                self.win_status = False
                terminal_reward = -100.0
            reward += terminal_reward
            self.score += terminal_reward

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )


    def _calculate_reward(self):
        """Calculates the reward for the current state."""
        reward = 0.0

        # Continuous rewards for maintaining good levels
        for i in range(self.NUM_RESERVOIRS):
            if self.reservoir_levels[i] > 50.0:
                reward += 0.1
            if self.reservoir_levels[i] < 20.0:
                reward -= 0.1

        # Event-based reward for reaching the 80% threshold for the first time
        for i in range(self.NUM_RESERVOIRS):
            if not self.has_reached_80[i] and self.reservoir_levels[i] >= self.WIN_THRESHOLD_PERCENT:
                reward += 1.0
                self.has_reached_80[i] = True

        return reward


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lever_positions": self.lever_positions.tolist(),
            "reservoir_levels": self.reservoir_levels.tolist(),
        }


    def _get_observation(self):
        # Update visual state (smooth animation)
        self.visual_water_levels += (self.reservoir_levels - self.visual_water_levels) * self.VISUAL_LERP_RATE

        # --- Render all game elements ---
        self.screen.fill(self.COLOR_BG)
        self._render_reservoirs()
        self._render_levers()
        self._update_and_render_particles()
        self._render_ui()

        if self.game_over:
            self._render_game_over_screen()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)


    def _render_reservoirs(self):
        for i in range(self.NUM_RESERVOIRS):
            res_rect = self.reservoir_rects[i]

            # Draw empty reservoir background
            pygame.draw.rect(self.screen, self.COLOR_RESERVOIR_EMPTY, res_rect)

            # Draw water level
            water_height = int((self.visual_water_levels[i] / self.RESERVOIR_CAPACITY) * res_rect.height)
            water_rect = pygame.Rect(
                res_rect.left,
                res_rect.bottom - water_height,
                res_rect.width,
                max(0, water_height)
            )
            pygame.draw.rect(self.screen, self.COLOR_WATER, water_rect)

            # Draw target line
            target_y = res_rect.bottom - int((self.WIN_THRESHOLD_PERCENT / self.RESERVOIR_CAPACITY) * res_rect.height)
            pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (res_rect.left, target_y), (res_rect.right, target_y), 2)

            # Draw reservoir outline
            pygame.draw.rect(self.screen, self.COLOR_TEXT, res_rect, 2, border_radius=3)

            # Draw percentage text
            percent_text = f"{self.reservoir_levels[i]:.1f}%"
            text_surf = self.font_small.render(percent_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(res_rect.centerx, res_rect.top - 15))
            self.screen.blit(text_surf, text_rect)


    def _render_levers(self):
        for i in range(self.NUM_RESERVOIRS):
            lever_bg_rect = self.lever_rects[i]

            # Draw lever background
            pygame.draw.rect(self.screen, self.COLOR_LEVER_BG, lever_bg_rect, border_radius=5)

            # Draw lever fill
            fill_height = int((self.lever_positions[i] / self.LEVER_MAX) * lever_bg_rect.height)
            fill_rect = pygame.Rect(
                lever_bg_rect.left,
                lever_bg_rect.bottom - fill_height,
                lever_bg_rect.width,
                max(0, fill_height)
            )
            pygame.draw.rect(self.screen, self.COLOR_LEVER_FG, fill_rect, border_radius=5)

            # Highlight selected lever
            if i == self.selected_lever:
                pygame.draw.rect(self.screen, self.COLOR_LEVER_ACTIVE, lever_bg_rect, 3, border_radius=5)
            else:
                pygame.draw.rect(self.screen, self.COLOR_TEXT, lever_bg_rect, 1, border_radius=5)


    def _update_and_render_particles(self):
        """Update and draw bubble particles for visual feedback."""
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['vel'][1] -= 0.05 # Bubbles accelerate upwards
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                radius = int(p['life'] / p['max_life'] * 3)
                if radius > 0:
                    pos = (int(p['pos'][0]), int(p['pos'][1]))
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_WATER_BUBBLE)


    def _add_particles(self, reservoir_index, amount):
        res_rect = self.reservoir_rects[reservoir_index]
        num_particles = min(20, int(amount * 2))

        water_surface_y = res_rect.bottom - (self.visual_water_levels[reservoir_index] / self.RESERVOIR_CAPACITY) * res_rect.height

        for _ in range(num_particles):
            life = random.randint(20, 40)
            self.particles.append({
                'pos': [random.uniform(res_rect.left + 5, res_rect.right - 5), water_surface_y],
                'vel': [random.uniform(-0.5, 0.5), random.uniform(-1, -0.2)],
                'life': life,
                'max_life': life
            })


    def _render_ui(self):
        # UI Panel at the top
        ui_panel_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, self.COLOR_UI_PANEL, ui_panel_rect)
        pygame.draw.line(self.screen, self.COLOR_TEXT, (0, 40), (self.SCREEN_WIDTH, 40), 1)

        # Turn counter
        turn_text = f"Turn: {self.steps}/{self.MAX_STEPS}"
        turn_surf = self.font_medium.render(turn_text, True, self.COLOR_TEXT)
        self.screen.blit(turn_surf, (15, 8))

        # Score
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 15, 8))
        self.screen.blit(score_surf, score_rect)


    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))

        if self.win_status:
            text = "VICTORY"
            color = self.COLOR_WIN
        else:
            text = "FAILURE"
            color = self.COLOR_LOSS

        text_surf = self.font_large.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))

        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)


    def close(self):
        pygame.quit()