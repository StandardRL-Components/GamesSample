
# Generated: 2025-08-28T02:09:16.992734
# Source Brief: brief_01617.md
# Brief Index: 1617

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import Counter
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Shift to toggle between the grid and the color palette. "
        "Press Space to paint the selected grid cell with the selected color."
    )

    game_description = (
        "A relaxing puzzle game where you recreate a hidden pixel art image. "
        "Select colors and paint the grid to match the preview image before you run out of time or paint."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 20
    MAX_STEPS = 1000

    # --- Colors (A nice, high-contrast palette) ---
    COLOR_BG = (20, 20, 30)
    COLOR_UI_BG = (30, 30, 40)
    COLOR_GRID_LINES = (50, 50, 60)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_ACCENT = (255, 200, 0)
    COLOR_CURSOR_GRID = (255, 255, 0)
    COLOR_CURSOR_PALETTE = (0, 200, 255)
    
    PALETTE = [
        (0, 0, 0),         # 0: Empty/BG for drawing
        (255, 0, 77),      # 1: Red
        (255, 163, 0),     # 2: Orange
        (255, 236, 39),    # 3: Yellow
        (0, 228, 54),      # 4: Green
        (41, 173, 255),    # 5: Blue
        (131, 118, 156),   # 6: Purple
        (255, 119, 168),   # 7: Pink
        (255, 255, 255),   # 8: White
        (105, 105, 105),   # 9: Gray
    ]
    
    # --- UI Layout ---
    GRID_PIXEL_SIZE = 16
    GRID_AREA_WIDTH = GRID_PIXEL_SIZE * GRID_SIZE
    GRID_AREA_HEIGHT = GRID_PIXEL_SIZE * GRID_SIZE
    GRID_TOP_LEFT = (40, (SCREEN_HEIGHT - GRID_AREA_HEIGHT) // 2)

    UI_PANEL_X = GRID_TOP_LEFT[0] + GRID_AREA_WIDTH + 40
    UI_PANEL_Y = GRID_TOP_LEFT[1]
    UI_PANEL_WIDTH = SCREEN_WIDTH - UI_PANEL_X - 40
    
    PREVIEW_SIZE = 80
    PALETTE_SWATCH_SIZE = 24
    PALETTE_COLS = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)
        
        self.target_image = None
        self.grid = None
        self.initial_color_counts = None
        self.remaining_colors = None
        self.grid_cursor = None
        self.palette_cursor = None
        self.focus_mode = None
        self.prev_shift_held = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.completed_rows = None
        self.completed_cols = None

        self.validate_implementation()

    def _generate_target_image(self):
        """Creates a simple, solvable 20x20 pixel art image."""
        pattern_type = self.np_random.integers(0, 3)
        
        # Color indices (1-9), 0 is background
        c1, c2, c3 = self.np_random.choice(range(1, 10), 3, replace=False)
        
        img = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)

        if pattern_type == 0:  # Symmetric Cross
            for i in range(self.GRID_SIZE):
                img[i, 9] = img[i, 10] = c1
                img[9, i] = img[10, i] = c1
            for i in range(6, 14):
                img[i, i] = c2
                img[i, self.GRID_SIZE - 1 - i] = c2
        elif pattern_type == 1: # Heart
            points = [
                (2,4),(3,4),(4,4),(5,3),(6,2),(7,2),(8,2),(9,3),(10,3),(11,2),(12,2),(13,2),(14,3),(15,4),(16,4),(17,4),
                (17,5),(16,6),(15,7),(14,8),(13,9),(12,10),(11,11),(10,12),(9,12),(8,11),(7,10),(6,9),(5,8),(4,7),(3,6),(2,5)
            ]
            for x, y in points:
                img[y+2, x+2] = c1
            img[7:11, 7:13] = c2
        else:  # Checkerboard
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    if (r // 4 + c // 4) % 2 == 0:
                        img[r, c] = c1
                    else:
                        img[r,c] = c2
        return img

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.target_image = self._generate_target_image()
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        color_counts = Counter(self.target_image.flatten())
        self.initial_color_counts = np.array([color_counts.get(i, 0) for i in range(len(self.PALETTE))])
        self.remaining_colors = self.initial_color_counts.copy()

        self.grid_cursor = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.palette_cursor = 1
        self.focus_mode = 'grid'
        self.prev_shift_held = False
        
        self.completed_rows = set()
        self.completed_cols = set()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # 1. Handle Input
        # Toggle focus mode on shift press (not hold)
        if shift_held and not self.prev_shift_held:
            self.focus_mode = 'palette' if self.focus_mode == 'grid' else 'grid'
        self.prev_shift_held = shift_held

        # Move cursor
        if self.focus_mode == 'grid':
            if movement == 1: self.grid_cursor[1] -= 1  # Up
            elif movement == 2: self.grid_cursor[1] += 1  # Down
            elif movement == 3: self.grid_cursor[0] -= 1  # Left
            elif movement == 4: self.grid_cursor[0] += 1  # Right
            self.grid_cursor[0] = np.clip(self.grid_cursor[0], 0, self.GRID_SIZE - 1)
            self.grid_cursor[1] = np.clip(self.grid_cursor[1], 0, self.GRID_SIZE - 1)
        else: # 'palette'
            if movement == 1: self.palette_cursor -= self.PALETTE_COLS
            elif movement == 2: self.palette_cursor += self.PALETTE_COLS
            elif movement == 3: self.palette_cursor -= 1
            elif movement == 4: self.palette_cursor += 1
            # Palette cursor wraps, skipping index 0 (background)
            self.palette_cursor = np.clip(self.palette_cursor, 1, len(self.PALETTE) - 1)

        # Paint action
        if space_held and self.focus_mode == 'grid':
            cx, cy = self.grid_cursor
            selected_color_idx = self.palette_cursor
            target_color_idx = self.target_image[cy, cx]
            
            # Only act if pixel is not already correct
            if self.grid[cy, cx] != target_color_idx:
                if self.remaining_colors[selected_color_idx] > 0:
                    # Sfx: paint_splash.wav
                    self.grid[cy, cx] = selected_color_idx
                    
                    if selected_color_idx == target_color_idx:
                        reward += 0.1
                    else:
                        reward -= 0.01
                    
                    # Check for row/column completion
                    reward += self._check_completion(cx, cy)
                else:
                    # Sfx: error.wav
                    reward -= 0.05 # Penalize trying to use an empty color

        self.score += reward
        
        # 2. Check Termination Conditions
        terminated = False
        
        # a) Win condition
        if np.array_equal(self.grid, self.target_image):
            # Sfx: victory.wav
            reward += 100
            terminated = True
        
        # b) Loss condition: Time out
        if self.steps >= self.MAX_STEPS:
            reward -= 100
            terminated = True
            
        # c) Loss condition: Unwinnable (ran out of a needed color)
        if not terminated:
            needed_colors = Counter(c for r_idx, r in enumerate(self.target_image) for c_idx, c in enumerate(r) if self.grid[r_idx, c_idx] != c)
            for color_idx, count in needed_colors.items():
                if self.remaining_colors[color_idx] < count:
                    reward -= 100
                    terminated = True
                    break
                    
        if terminated:
            self.game_over = True
            self.score += reward # Add terminal reward to total score

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_completion(self, x, y):
        """Check if a row or column was just completed and give reward."""
        reward = 0
        # Check row
        if y not in self.completed_rows:
            if np.array_equal(self.grid[y, :], self.target_image[y, :]):
                reward += 5
                self.completed_rows.add(y)
                # Sfx: line_complete.wav
        # Check column
        if x not in self.completed_cols:
            if np.array_equal(self.grid[:, x], self.target_image[:, x]):
                reward += 5
                self.completed_cols.add(x)
                # Sfx: line_complete.wav
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the main painting grid."""
        gx, gy = self.GRID_TOP_LEFT
        
        # Grid background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (gx, gy, self.GRID_AREA_WIDTH, self.GRID_AREA_HEIGHT))
        
        # Painted pixels
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.grid[r, c]
                if color_idx != 0:
                    pixel_rect = (
                        gx + c * self.GRID_PIXEL_SIZE,
                        gy + r * self.GRID_PIXEL_SIZE,
                        self.GRID_PIXEL_SIZE,
                        self.GRID_PIXEL_SIZE
                    )
                    pygame.draw.rect(self.screen, self.PALETTE[color_idx], pixel_rect)
        
        # Grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (gx + i * self.GRID_PIXEL_SIZE, gy)
            end_pos = (gx + i * self.GRID_PIXEL_SIZE, gy + self.GRID_AREA_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos)
            # Horizontal
            start_pos = (gx, gy + i * self.GRID_PIXEL_SIZE)
            end_pos = (gx + self.GRID_AREA_WIDTH, gy + i * self.GRID_PIXEL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos)

        # Grid cursor
        if self.focus_mode == 'grid':
            cx, cy = self.grid_cursor
            cursor_rect = (
                gx + cx * self.GRID_PIXEL_SIZE,
                gy + cy * self.GRID_PIXEL_SIZE,
                self.GRID_PIXEL_SIZE,
                self.GRID_PIXEL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_CURSOR_GRID, cursor_rect, 2)


    def _render_ui(self):
        """Renders the UI panel: preview, palette, timer, etc."""
        # --- Preview Image ---
        px, py = self.UI_PANEL_X, self.UI_PANEL_Y
        self._draw_text("TARGET", (px, py - 20))
        
        preview_pixel_size = self.PREVIEW_SIZE // self.GRID_SIZE
        preview_rect = (px, py, self.PREVIEW_SIZE, self.PREVIEW_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, preview_rect)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.target_image[r, c]
                if color_idx != 0:
                    pygame.draw.rect(self.screen, self.PALETTE[color_idx], (
                        px + c * preview_pixel_size,
                        py + r * preview_pixel_size,
                        preview_pixel_size, preview_pixel_size
                    ))
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, preview_rect, 1)

        # --- Palette ---
        palette_y = py + self.PREVIEW_SIZE + 40
        self._draw_text("PALETTE", (px, palette_y - 20))
        
        for i in range(1, len(self.PALETTE)):
            row = (i - 1) // self.PALETTE_COLS
            col = (i - 1) % self.PALETTE_COLS
            
            swatch_x = px + col * (self.PALETTE_SWATCH_SIZE + 10)
            swatch_y = palette_y + row * (self.PALETTE_SWATCH_SIZE + 10)
            
            pygame.draw.rect(self.screen, self.PALETTE[i], (swatch_x, swatch_y, self.PALETTE_SWATCH_SIZE, self.PALETTE_SWATCH_SIZE))
            
            # Highlight selected color
            if i == self.palette_cursor:
                cursor_color = self.COLOR_CURSOR_PALETTE if self.focus_mode == 'palette' else self.COLOR_GRID_LINES
                pygame.draw.rect(self.screen, cursor_color, (swatch_x - 2, swatch_y - 2, self.PALETTE_SWATCH_SIZE + 4, self.PALETTE_SWATCH_SIZE + 4), 2)
            
            # Color counts
            count_text = self.font_small.render(f"{self.remaining_colors[i]}", True, self.COLOR_TEXT)
            self.screen.blit(count_text, (swatch_x + self.PALETTE_SWATCH_SIZE + 5, swatch_y + 4))

        # --- Stats ---
        stats_y = palette_y + 150
        self._draw_text(f"TIME: {self.MAX_STEPS - self.steps}", (px, stats_y), self.COLOR_TEXT_ACCENT)
        self._draw_text(f"SCORE: {self.score:.2f}", (px, stats_y + 20))
        
        # --- Focus Indicator ---
        focus_text = f"MODE: {self.focus_mode.upper()}"
        focus_color = self.COLOR_CURSOR_GRID if self.focus_mode == 'grid' else self.COLOR_CURSOR_PALETTE
        self._draw_text(focus_text, (px, stats_y + 50), focus_color)
        
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            is_win = np.array_equal(self.grid, self.target_image)
            end_text = "COMPLETE!" if is_win else "GAME OVER"
            self._draw_text(end_text, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20), self.COLOR_TEXT_ACCENT, center=True, font=self.font_main)


    def _draw_text(self, text, pos, color=None, center=False, font=None):
        if color is None: color = self.COLOR_TEXT
        if font is None: font = self.font_main
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_time": self.MAX_STEPS - self.steps,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset and observation space
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # To run this, you need to install pygame (`pip install pygame`)
    # and change render_mode in __init__ to "human" if available,
    # or manually render the rgb_array.
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Painter")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished! Total reward: {total_reward}, Final Score: {info['score']}")
            # obs, info = env.reset() # Uncomment to auto-reset
            # total_reward = 0
            
        clock.tick(10) # Control manual play speed

    env.close()