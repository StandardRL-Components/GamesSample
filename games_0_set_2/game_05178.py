
# Generated: 2025-08-28T04:13:46.504384
# Source Brief: brief_05178.md
# Brief Index: 5178

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place a pixel. Shift to cycle color."
    )

    game_description = (
        "Recreate a target image on a pixel grid by placing colored pixels before time runs out."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 10
    MAX_STEPS = 600
    WIN_THRESHOLD = 75  # 75 correct pixels out of 100

    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID_BG = (45, 48, 56)
    COLOR_GRID_LINES = (65, 68, 76)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_ACCENT = (100, 255, 100)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_WIN = (70, 255, 70, 200)
    COLOR_LOSE = (255, 70, 70, 200)
    
    PIXEL_COLORS = [
        (230, 57, 70),   # Red
        (241, 150, 56),  # Orange
        (252, 211, 79),  # Yellow
        (144, 190, 109), # Green
        (67, 170, 139),  # Teal
        (86, 11, 173),   # Purple
        (29, 128, 228),  # Blue
        (255, 255, 255), # White
    ]
    EMPTY_COLOR_IDX = -1
    EMPTY_COLOR_RGB = (55, 58, 66)

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
        self.font_large = pygame.font.Font(None, 64)
        self.font_small = pygame.font.Font(None, 20)

        self.grid_state = None
        self.target_image = None
        self.cursor_pos = None
        self.selected_color_index = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.row_bonus_awarded = None
        self.col_bonus_awarded = None
        self.placement_effects = []
        
        self.reset()
        
        self.validate_implementation()

    def _generate_target_image(self):
        img = np.full((self.GRID_SIZE, self.GRID_SIZE), self.EMPTY_COLOR_IDX, dtype=int)
        
        # Use np_random for deterministic generation
        num_shapes = self.np_random.integers(2, 5)
        
        for _ in range(num_shapes):
            color = self.np_random.integers(0, len(self.PIXEL_COLORS))
            start_x = self.np_random.integers(0, self.GRID_SIZE)
            start_y = self.np_random.integers(0, self.GRID_SIZE)
            size_x = self.np_random.integers(1, self.GRID_SIZE // 2)
            size_y = self.np_random.integers(1, self.GRID_SIZE // 2)
            
            for y in range(start_y, min(self.GRID_SIZE, start_y + size_y)):
                for x in range(start_x, min(self.GRID_SIZE, start_x + size_x)):
                    if self.np_random.random() > 0.2:
                        img[y, x] = color
        
        # Ensure it's not empty
        if np.all(img == self.EMPTY_COLOR_IDX):
            img[self.GRID_SIZE // 2, self.GRID_SIZE // 2] = 0

        return img

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.target_image = self._generate_target_image()
        self.grid_state = np.full((self.GRID_SIZE, self.GRID_SIZE), self.EMPTY_COLOR_IDX, dtype=int)
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_color_index = 0
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.row_bonus_awarded = [False] * self.GRID_SIZE
        self.col_bonus_awarded = [False] * self.GRID_SIZE
        
        self.placement_effects = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Actions ---
        # Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        # Wrap cursor around grid
        self.cursor_pos[0] %= self.GRID_SIZE
        self.cursor_pos[1] %= self.GRID_SIZE

        # Color Cycle (on press)
        if shift_held and not self.prev_shift_held:
            self.selected_color_index = (self.selected_color_index + 1) % len(self.PIXEL_COLORS)
            # sfx: color_cycle.wav

        # Place Pixel (on press)
        if space_held and not self.prev_space_held:
            x, y = self.cursor_pos
            if self.grid_state[y, x] == self.EMPTY_COLOR_IDX:
                self.grid_state[y, x] = self.selected_color_index
                self.placement_effects.append({'pos': (x, y), 'timer': 5}) # Add flash effect
                # sfx: place_pixel.wav
                
                # Calculate placement reward
                if self.grid_state[y, x] == self.target_image[y, x]:
                    reward += 1.0
                    # sfx: place_correct.wav
                else:
                    reward -= 0.2
                    # sfx: place_wrong.wav
                
                # Check for row/column completion bonus
                reward += self._check_completion_bonus(x, y)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # --- Check Termination ---
        correct_pixels = np.sum(self.grid_state == self.target_image)
        
        terminated = False
        if correct_pixels >= self.WIN_THRESHOLD:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100.0  # Win bonus
            # sfx: win_game.wav
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            terminated = True
            reward -= 10.0  # Time out penalty
            # sfx: lose_game.wav
            
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_completion_bonus(self, x, y):
        bonus = 0
        # Check column
        if not self.col_bonus_awarded[x]:
            col = self.grid_state[:, x]
            target_col = self.target_image[:, x]
            if np.all(col != self.EMPTY_COLOR_IDX) and np.all(col == target_col):
                bonus += 5.0
                self.col_bonus_awarded[x] = True
                # sfx: complete_line.wav
        
        # Check row
        if not self.row_bonus_awarded[y]:
            row = self.grid_state[y, :]
            target_row = self.target_image[y, :]
            if np.all(row != self.EMPTY_COLOR_IDX) and np.all(row == target_row):
                bonus += 5.0
                self.row_bonus_awarded[y] = True
                # sfx: complete_line.wav
        
        return bonus

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Layout ---
        main_grid_size = 300
        main_grid_x = (self.SCREEN_WIDTH - main_grid_size) // 2
        main_grid_y = (self.SCREEN_HEIGHT - main_grid_size) // 2 + 20
        pixel_size = main_grid_size / self.GRID_SIZE

        target_grid_size = 100
        target_grid_x = 20
        target_grid_y = 20
        target_pixel_size = target_grid_size / self.GRID_SIZE

        # --- Update and Render Effects ---
        self.placement_effects = [eff for eff in self.placement_effects if eff['timer'] > 0]
        for effect in self.placement_effects:
            effect['timer'] -= 1

        # --- Render Main Grid ---
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (main_grid_x, main_grid_y, main_grid_size, main_grid_size))
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.grid_state[r, c]
                color = self.PIXEL_COLORS[color_idx] if color_idx != self.EMPTY_COLOR_IDX else self.EMPTY_COLOR_RGB
                rect = (main_grid_x + c * pixel_size, main_grid_y + r * pixel_size, pixel_size, pixel_size)
                pygame.draw.rect(self.screen, color, rect)

        # Render placement flash effect
        for effect in self.placement_effects:
            c, r = effect['pos']
            alpha = int(255 * (effect['timer'] / 5.0))
            flash_color = (255, 255, 255, alpha)
            flash_surface = pygame.Surface((pixel_size, pixel_size), pygame.SRCALPHA)
            flash_surface.fill(flash_color)
            self.screen.blit(flash_surface, (main_grid_x + c * pixel_size, main_grid_y + r * pixel_size))

        # Render Grid Lines
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (main_grid_x, main_grid_y + i * pixel_size), (main_grid_x + main_grid_size, main_grid_y + i * pixel_size), 1)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (main_grid_x + i * pixel_size, main_grid_y), (main_grid_x + i * pixel_size, main_grid_y + main_grid_size), 1)

        # --- Render Cursor ---
        cursor_x = main_grid_x + self.cursor_pos[0] * pixel_size
        cursor_y = main_grid_y + self.cursor_pos[1] * pixel_size
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y, pixel_size, pixel_size), 3)

        # --- Render Target Image ---
        self._render_text("Target", (target_grid_x, target_grid_y - 15), self.font_small)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (target_grid_x, target_grid_y, target_grid_size, target_grid_size))
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.target_image[r, c]
                color = self.PIXEL_COLORS[color_idx] if color_idx != self.EMPTY_COLOR_IDX else self.EMPTY_COLOR_RGB
                rect = (target_grid_x + c * target_pixel_size, target_grid_y + r * target_pixel_size, target_pixel_size, target_pixel_size)
                pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, (target_grid_x, target_grid_y, target_grid_size, target_grid_size), 1)


    def _render_ui(self):
        # --- Time Remaining ---
        time_text = f"Time: {max(0, self.MAX_STEPS - self.steps)}"
        self._render_text(time_text, (self.SCREEN_WIDTH - 150, 20), self.font_main)

        # --- Match Percentage ---
        correct_pixels = np.sum(self.grid_state == self.target_image)
        match_percent = (correct_pixels / self.WIN_THRESHOLD) * 100
        match_text = f"Match: {min(100, int(match_percent))}%"
        self._render_text(match_text, (self.SCREEN_WIDTH - 150, 50), self.font_main, color=self.COLOR_TEXT_ACCENT)

        # --- Selected Color Swatch ---
        self._render_text("Selected:", (20, self.SCREEN_HEIGHT - 60), self.font_small)
        pygame.draw.rect(self.screen, self.PIXEL_COLORS[self.selected_color_index], (20, self.SCREEN_HEIGHT - 40, 30, 30))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (20, self.SCREEN_HEIGHT - 40, 30, 30), 2)

        # --- Game Over Overlay ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            color = self.COLOR_WIN if self.win else self.COLOR_LOSE
            overlay.fill(color)
            self.screen.blit(overlay, (0, 0))
            
            message = "COMPLETE!" if self.win else "TIME UP"
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _render_text(self, text, position, font, color=COLOR_TEXT, align="topleft"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        setattr(text_rect, align, position)
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "correct_pixels": np.sum(self.grid_state == self.target_image),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a persistent window for human play
    pygame.display.set_caption("Pixel Painter")
    human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    
    print("\n" + "="*30)
    print("Pixel Painter - Manual Control")
    print(GameEnv.user_guide)
    print("Press ESC or close window to quit.")
    print("="*30 + "\n")

    while not terminated:
        # --- Human Input to Action ---
        movement, space, shift = 0, 0, 0
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total Score: {info['score']:.2f}")

        if terminated:
            print("--- GAME OVER ---")
            print(f"Final Score: {info['score']:.2f}")
            print(f"Correct Pixels: {info['correct_pixels']}")

        # --- Render to screen ---
        # The observation is (H, W, C), but pygame wants (W, H) surface
        # We can just re-use the env's internal screen surface
        surf = env.screen 
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we need to control the speed for human play
        env.clock.tick(30) # Limit to 30 FPS

    env.close()