import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to push all pixels in the chosen direction."
    )

    game_description = (
        "Recreate the target image by pushing pixels around the grid. Match the pattern before the timer runs out!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_FRAMES = 1800  # 60 seconds at 30fps

    GRID_SIZE = 8
    GRID_PIXEL_SIZE = 40
    GRID_WIDTH = GRID_SIZE * GRID_PIXEL_SIZE
    GRID_HEIGHT = GRID_SIZE * GRID_PIXEL_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    TARGET_PIXEL_SIZE = 15
    TARGET_WIDTH = GRID_SIZE * TARGET_PIXEL_SIZE
    TARGET_HEIGHT = GRID_SIZE * TARGET_PIXEL_SIZE
    TARGET_X = SCREEN_WIDTH - TARGET_WIDTH - 30
    TARGET_Y = GRID_Y

    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID_BG = (45, 48, 56)
    COLOR_UI_BG = (35, 38, 46)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_UI_ACCENT = (100, 180, 255)

    PIXEL_COLORS_BRIGHT = [
        (0, 0, 0),          # 0: Empty (unused in patterns)
        (255, 80, 80),      # 1: Red
        (80, 255, 80),      # 2: Green
        (80, 150, 255),     # 3: Blue
        (255, 255, 80),     # 4: Yellow
    ]
    PIXEL_COLORS_MUTED = [
        (0, 0, 0),
        (120, 40, 40),
        (40, 120, 40),
        (40, 70, 120),
        (120, 120, 40),
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_main = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 22)
        self.font_large = pygame.font.Font(None, 60)

        self.target_patterns = self._create_patterns()

        # State variables are initialized in reset(), but we need a default non-None
        # value for the validation check that runs during __init__.
        # The reset() call later in the validation will set the proper initial state.
        self.target_grid = self.target_patterns[0].copy()
        self.current_grid = self.target_patterns[0].copy()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.moves = 0
        self.particles = []
        self.completed_rows = np.zeros(self.GRID_SIZE, dtype=bool)
        self.completed_cols = np.zeros(self.GRID_SIZE, dtype=bool)

        # This validation function is called to ensure the environment is set up correctly.
        # It needs the grid variables to be initialized to run.
        self.validate_implementation()

    def _create_patterns(self):
        patterns = []
        # Pattern 1: X
        p = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        for i in range(self.GRID_SIZE):
            p[i, i] = 1
            p[i, self.GRID_SIZE - 1 - i] = 1
        patterns.append(p)
        # Pattern 2: Checkerboard
        p = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (r + c) % 2 == 0:
                    p[r, c] = 2
        patterns.append(p)
        # Pattern 3: Frame
        p = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        p[0, :] = 3
        p[-1, :] = 3
        p[:, 0] = 3
        p[:, -1] = 3
        patterns.append(p)
        # Pattern 4: Smiley
        p = np.array([
            [0,0,4,4,4,4,0,0],
            [0,4,0,0,0,0,4,0],
            [4,0,4,0,4,0,0,4],
            [4,0,0,0,0,0,0,4],
            [4,0,4,0,0,4,0,4],
            [4,0,0,4,4,0,0,4],
            [0,4,0,0,0,0,4,0],
            [0,0,4,4,4,4,0,0],
        ], dtype=int)
        patterns.append(p)
        return patterns

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_FRAMES
        self.moves = 0
        self.particles = []

        self.target_grid = self.target_patterns[self.np_random.integers(0, len(self.target_patterns))].copy()

        self.current_grid = self.target_grid.copy()
        num_shuffles = self.np_random.integers(10, 21)
        for _ in range(num_shuffles):
            direction = self.np_random.integers(1, 5) # 1-4 for up/down/left/right
            axis = 0 if direction in [1, 2] else 1
            shift = -1 if direction in [1, 3] else 1
            self.current_grid = np.roll(self.current_grid, shift=shift, axis=axis)

        self._update_completion_state()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0

        self.time_remaining -= 1
        self.steps += 1

        old_grid_correct = np.sum(self.current_grid == self.target_grid)

        if movement != 0:
            self.moves += 1
            # sfx: push_sound
            self._perform_push(movement)

        self._update_particles()

        # --- Reward Calculation ---
        # 1. Pixel position reward
        new_grid_correct = np.sum(self.current_grid == self.target_grid)
        reward += (new_grid_correct - old_grid_correct) * 0.1

        # 2. Row/column completion reward
        new_reward, self.completed_rows, self.completed_cols = self._calculate_line_completion_reward(self.completed_rows, self.completed_cols)
        reward += new_reward

        # --- Termination Check ---
        is_complete = np.array_equal(self.current_grid, self.target_grid)
        time_up = self.time_remaining <= 0
        terminated = is_complete or time_up

        if terminated:
            self.game_over = True
            if is_complete:
                # sfx: win_jingle
                time_bonus = 100 * (self.time_remaining / self.MAX_FRAMES)
                reward += time_bonus
            # else: sfx: lose_buzzer

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _perform_push(self, direction):
        axis = 0 if direction in [1, 2] else 1
        shift = -1 if direction in [1, 3] else 1

        self.current_grid = np.roll(self.current_grid, shift=shift, axis=axis)

        # Create particles
        for i in range(self.GRID_SIZE):
            if direction == 1: # Up
                pos = [self.GRID_X + i * self.GRID_PIXEL_SIZE + self.GRID_PIXEL_SIZE / 2, self.GRID_Y + self.GRID_HEIGHT - self.GRID_PIXEL_SIZE / 2]
                vel = [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-3, -1)]
                color_idx = self.current_grid[-1, i]
            elif direction == 2: # Down
                pos = [self.GRID_X + i * self.GRID_PIXEL_SIZE + self.GRID_PIXEL_SIZE / 2, self.GRID_Y + self.GRID_PIXEL_SIZE / 2]
                vel = [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(1, 3)]
                color_idx = self.current_grid[0, i]
            elif direction == 3: # Left
                pos = [self.GRID_X + self.GRID_WIDTH - self.GRID_PIXEL_SIZE / 2, self.GRID_Y + i * self.GRID_PIXEL_SIZE + self.GRID_PIXEL_SIZE / 2]
                vel = [self.np_random.uniform(-3, -1), self.np_random.uniform(-0.5, 0.5)]
                color_idx = self.current_grid[i, -1]
            else: # Right
                pos = [self.GRID_X + self.GRID_PIXEL_SIZE / 2, self.GRID_Y + i * self.GRID_PIXEL_SIZE + self.GRID_PIXEL_SIZE / 2]
                vel = [self.np_random.uniform(1, 3), self.np_random.uniform(-0.5, 0.5)]
                color_idx = self.current_grid[i, 0]

            if color_idx > 0:
                for _ in range(3): # more particles
                    self.particles.append({
                        'pos': list(pos),
                        'vel': [v * self.np_random.uniform(0.8, 1.2) for v in vel],
                        'life': self.np_random.integers(15, 25),
                        'max_life': 25,
                        'color': self.PIXEL_COLORS_BRIGHT[color_idx]
                    })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_completion_state(self):
        self.completed_rows = np.array([np.array_equal(self.current_grid[r, :], self.target_grid[r, :]) for r in range(self.GRID_SIZE)])
        self.completed_cols = np.array([np.array_equal(self.current_grid[:, c], self.target_grid[:, c]) for c in range(self.GRID_SIZE)])

    def _calculate_line_completion_reward(self, old_rows, old_cols):
        reward = 0
        new_rows = np.array([np.array_equal(self.current_grid[r, :], self.target_grid[r, :]) for r in range(self.GRID_SIZE)])
        new_cols = np.array([np.array_equal(self.current_grid[:, c], self.target_grid[:, c]) for c in range(self.GRID_SIZE)])

        newly_completed_rows = np.sum(new_rows & ~old_rows)
        newly_completed_cols = np.sum(new_cols & ~old_cols)

        if newly_completed_rows > 0:
            reward += newly_completed_rows * 5
            # sfx: line_complete
        if newly_completed_cols > 0:
            reward += newly_completed_cols * 5
            # sfx: line_complete

        return reward, new_rows, new_cols

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT), border_radius=5)

        # Render pixels
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.current_grid[r, c]
                if color_idx == 0: continue

                is_correct = color_idx == self.target_grid[r, c]
                color = self.PIXEL_COLORS_BRIGHT[color_idx] if is_correct else self.PIXEL_COLORS_MUTED[color_idx]

                px, py = self.GRID_X + c * self.GRID_PIXEL_SIZE, self.GRID_Y + r * self.GRID_PIXEL_SIZE
                rect = (px + 2, py + 2, self.GRID_PIXEL_SIZE - 4, self.GRID_PIXEL_SIZE - 4)

                if is_correct:
                    # Glow effect
                    glow_rect = (px, py, self.GRID_PIXEL_SIZE, self.GRID_PIXEL_SIZE)
                    glow_surf = pygame.Surface((self.GRID_PIXEL_SIZE, self.GRID_PIXEL_SIZE), pygame.SRCALPHA)
                    pygame.draw.rect(glow_surf, (*color, 60), (0, 0, self.GRID_PIXEL_SIZE, self.GRID_PIXEL_SIZE), border_radius=8)
                    self.screen.blit(glow_surf, (px, py))

                pygame.draw.rect(self.screen, color, rect, border_radius=5)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(6 * (p['life'] / p['max_life']))
            if size > 0:
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (size, size), size)
                self.screen.blit(particle_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_text(self, text, font, color, position, center=False):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = position
        else:
            text_rect.topleft = position
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Left Panel
        left_panel_rect = (10, 10, self.GRID_X - 20, self.SCREEN_HEIGHT - 20)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, left_panel_rect, border_radius=5)

        self._render_text("TIME", self.font_title, self.COLOR_UI_TEXT, (left_panel_rect[0] + left_panel_rect[2]//2, 30), center=True)
        time_str = f"{self.time_remaining / 30:.2f}"
        self._render_text(time_str, self.font_main, self.COLOR_UI_ACCENT, (left_panel_rect[0] + left_panel_rect[2]//2, 60), center=True)

        self._render_text("SCORE", self.font_title, self.COLOR_UI_TEXT, (left_panel_rect[0] + left_panel_rect[2]//2, 110), center=True)
        self._render_text(f"{int(self.score)}", self.font_main, self.COLOR_UI_ACCENT, (left_panel_rect[0] + left_panel_rect[2]//2, 140), center=True)

        self._render_text("MOVES", self.font_title, self.COLOR_UI_TEXT, (left_panel_rect[0] + left_panel_rect[2]//2, 190), center=True)
        self._render_text(f"{self.moves}", self.font_main, self.COLOR_UI_ACCENT, (left_panel_rect[0] + left_panel_rect[2]//2, 220), center=True)

        # Right Panel (Target)
        right_panel_rect = (self.GRID_X + self.GRID_WIDTH + 10, 10, self.SCREEN_WIDTH - (self.GRID_X + self.GRID_WIDTH) - 20, self.SCREEN_HEIGHT - 20)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, right_panel_rect, border_radius=5)
        self._render_text("TARGET", self.font_title, self.COLOR_UI_TEXT, (right_panel_rect[0] + right_panel_rect[2]//2, 30), center=True)

        # Render target image
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.target_grid[r, c]
                if color_idx == 0: continue
                color = self.PIXEL_COLORS_BRIGHT[color_idx]
                px = self.TARGET_X + c * self.TARGET_PIXEL_SIZE
                py = self.TARGET_Y + r * self.TARGET_PIXEL_SIZE
                rect = (px + 1, py + 1, self.TARGET_PIXEL_SIZE - 2, self.TARGET_PIXEL_SIZE - 2)
                pygame.draw.rect(self.screen, color, rect, border_radius=2)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            is_win = np.array_equal(self.current_grid, self.target_grid)
            message = "COMPLETE!" if is_win else "TIME'S UP!"
            color = (100, 255, 100) if is_win else (255, 100, 100)
            self._render_text(message, self.font_large, color, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "moves": self.moves,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        This method is used to validate that the environment conforms to the Gymnasium API.
        It is called in __init__ to ensure the environment is set up correctly.
        """
        print("Beginning implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}, expected {(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)}"
        assert test_obs.dtype == np.uint8, f"Obs dtype is {test_obs.dtype}, expected np.uint8"

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Reset obs shape is {obs.shape}, expected {(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)}"
        assert isinstance(info, dict), f"Reset info is {type(info)}, expected dict"

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Step obs shape is {obs.shape}, expected {(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)}"
        assert isinstance(reward, (int, float)), f"Reward is {type(reward)}, expected int or float"
        assert isinstance(term, bool), f"Terminated is {type(term)}, expected bool"
        assert trunc is False, f"Truncated is {trunc}, expected False"
        assert isinstance(info, dict), f"Step info is {type(info)}, expected dict"

        print("✓ Implementation validated successfully")