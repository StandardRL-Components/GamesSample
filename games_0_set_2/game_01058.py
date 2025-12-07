
# Generated: 2025-08-27T15:43:55.475410
# Source Brief: brief_01058.md
# Brief Index: 1058

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to paint. Shift to cycle colors."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target image pixel by pixel before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CANVAS_GRID_SIZE = 16
    FPS = 30
    MAX_STEPS = FPS * 60  # 60-second time limit

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_UI_BG = (30, 35, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_ACCENT = (255, 200, 0)
    COLOR_PROGRESS_BAR = (50, 200, 50)
    COLOR_PROGRESS_BG = (50, 60, 80)
    COLOR_GRID_LINE = (40, 45, 60)
    
    PALETTE = [
        (0, 0, 0),         # Black
        (255, 255, 255),   # White
        (255, 0, 0),       # Red
        (0, 255, 0),       # Green
        (0, 0, 255),       # Blue
        (255, 255, 0),     # Yellow
        (0, 255, 255),     # Cyan
        (255, 0, 255),     # Magenta
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
        
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 16)
        self.font_small = pygame.font.SysFont("monospace", 12)
        
        # Game state variables are initialized in reset()
        self.target_image = None
        self.canvas = None
        self.cursor_pos = None
        self.selected_color_idx = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.target_image = self.np_random.integers(
            0, len(self.PALETTE), size=(self.CANVAS_GRID_SIZE, self.CANVAS_GRID_SIZE)
        )
        self.canvas = np.zeros_like(self.target_image)
        
        self.cursor_pos = [self.CANVAS_GRID_SIZE // 2, self.CANVAS_GRID_SIZE // 2]
        self.selected_color_idx = 1 # Default to white

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Handle Actions ---
        # Movement (continuous)
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right

        # Cursor wrap-around
        self.cursor_pos[0] %= self.CANVAS_GRID_SIZE
        self.cursor_pos[1] %= self.CANVAS_GRID_SIZE

        # Paint (edge-triggered on press)
        if space_held and not self.prev_space_held:
            # sfx: paint_sound
            cx, cy = self.cursor_pos
            target_color_idx = self.target_image[cy, cx]
            current_color_idx = self.canvas[cy, cx]

            # Only change state if the pixel is not already correct
            if current_color_idx != target_color_idx:
                if self.selected_color_idx == target_color_idx:
                    # Correct paint
                    self.canvas[cy, cx] = self.selected_color_idx
                    reward += 1.0  # Reward for fixing a pixel
                    self.score += 10
                    # sfx: correct_paint_sound
                else:
                    # Incorrect paint
                    self.canvas[cy, cx] = self.selected_color_idx
                    reward -= 0.5 # Penalty for incorrect paint
                    # sfx: wrong_paint_sound
            else:
                # Painting over an already correct pixel
                if self.selected_color_idx != target_color_idx:
                    # Ruining a correct pixel
                    self.canvas[cy, cx] = self.selected_color_idx
                    reward -= 1.5 # Heavier penalty for undoing progress
                    self.score -= 10
                else:
                    # Wasting time, small penalty
                    reward -= 0.1

        # Cycle color (edge-triggered on press)
        if shift_held and not self.prev_shift_held:
            self.selected_color_idx = (self.selected_color_idx + 1) % len(self.PALETTE)
            # sfx: color_switch_sound

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        self.steps += 1

        # --- Check Termination Conditions ---
        is_complete = np.array_equal(self.canvas, self.target_image)
        time_up = self.steps >= self.MAX_STEPS

        terminated = False
        if is_complete:
            reward += 100.0  # Big bonus for completion
            self.score += 1000
            terminated = True
            # sfx: win_jingle
        elif time_up:
            # Calculate a final penalty based on how many pixels are wrong
            num_wrong = np.count_nonzero(self.canvas != self.target_image)
            reward -= num_wrong * 0.2
            terminated = True
            # sfx: lose_buzzer
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Main Canvas ---
        canvas_rect = pygame.Rect(20, 20, 368, 368)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, canvas_rect)
        pixel_size = canvas_rect.width // self.CANVAS_GRID_SIZE

        # Draw painted pixels
        for y in range(self.CANVAS_GRID_SIZE):
            for x in range(self.CANVAS_GRID_SIZE):
                color = self.PALETTE[self.canvas[y, x]]
                pixel_rect = pygame.Rect(
                    canvas_rect.left + x * pixel_size,
                    canvas_rect.top + y * pixel_size,
                    pixel_size,
                    pixel_size
                )
                pygame.draw.rect(self.screen, color, pixel_rect)
        
        # Draw grid lines
        for i in range(self.CANVAS_GRID_SIZE + 1):
            start_x = canvas_rect.left + i * pixel_size
            start_y = canvas_rect.top + i * pixel_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (start_x, canvas_rect.top), (start_x, canvas_rect.bottom))
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (canvas_rect.left, start_y), (canvas_rect.right, start_y))

        # --- Cursor ---
        cursor_x, cursor_y = self.cursor_pos
        cursor_world_rect = pygame.Rect(
            canvas_rect.left + cursor_x * pixel_size,
            canvas_rect.top + cursor_y * pixel_size,
            pixel_size,
            pixel_size
        )
        
        # Pulsing effect for cursor
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        cursor_color = (
            255, 
            255, 
            150 + 105 * pulse
        )
        pygame.draw.rect(self.screen, cursor_color, cursor_world_rect, 3)

    def _render_ui(self):
        ui_panel_rect = pygame.Rect(408, 0, self.SCREEN_WIDTH - 408, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_panel_rect)

        # --- Target Image Preview ---
        target_title = self.font_medium.render("TARGET", True, self.COLOR_TEXT)
        self.screen.blit(target_title, (420, 15))

        target_rect = pygame.Rect(420, 40, 192, 192)
        pygame.draw.rect(self.screen, self.COLOR_BG, target_rect)
        pixel_size = target_rect.width // self.CANVAS_GRID_SIZE

        for y in range(self.CANVAS_GRID_SIZE):
            for x in range(self.CANVAS_GRID_SIZE):
                color = self.PALETTE[self.target_image[y, x]]
                pixel_rect = pygame.Rect(
                    target_rect.left + x * pixel_size,
                    target_rect.top + y * pixel_size,
                    pixel_size,
                    pixel_size
                )
                pygame.draw.rect(self.screen, color, pixel_rect)

        # --- Palette ---
        palette_title = self.font_medium.render("PALETTE", True, self.COLOR_TEXT)
        self.screen.blit(palette_title, (420, 245))
        
        color_box_size = 24
        for i, color in enumerate(self.PALETTE):
            row = i // 4
            col = i % 4
            box_rect = pygame.Rect(
                420 + col * (color_box_size + 5),
                270 + row * (color_box_size + 5),
                color_box_size,
                color_box_size
            )
            pygame.draw.rect(self.screen, color, box_rect)
            if i == self.selected_color_idx:
                pygame.draw.rect(self.screen, self.COLOR_ACCENT, box_rect, 3)
        
        # --- Timer ---
        remaining_time = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_large.render(f"{remaining_time:.1f}s", True, self.COLOR_ACCENT)
        self.screen.blit(timer_text, (420, 330))

        # --- Progress Bar ---
        correct_pixels = np.count_nonzero(self.canvas == self.target_image)
        total_pixels = self.CANVAS_GRID_SIZE * self.CANVAS_GRID_SIZE
        progress = correct_pixels / total_pixels
        
        bar_bg_rect = pygame.Rect(420, 365, 200, 20)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BG, bar_bg_rect, border_radius=4)

        bar_fill_rect = pygame.Rect(420, 365, int(200 * progress), 20)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR, bar_fill_rect, border_radius=4)
        
        progress_text = self.font_small.render(f"{correct_pixels}/{total_pixels}", True, self.COLOR_TEXT)
        text_rect = progress_text.get_rect(center=bar_bg_rect.center)
        self.screen.blit(progress_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "progress": np.count_nonzero(self.canvas == self.target_image) / (self.CANVAS_GRID_SIZE**2),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Painter")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Transpose the observation back for Pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.FPS)
        
    env.close()