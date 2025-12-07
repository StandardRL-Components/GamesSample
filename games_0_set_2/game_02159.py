
# Generated: 2025-08-28T03:55:15.124049
# Source Brief: brief_02159.md
# Brief Index: 2159

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to paint the selected color. Press Shift to cycle through colors."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target image by painting pixels on the grid before time runs out. Earn points for correct pixels and lose them for mistakes."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 100
        self.GRID_HEIGHT = 100
        self.PIXEL_SIZE = 4 # Each grid cell is 4x4 pixels
        self.GRID_AREA_WIDTH = self.GRID_WIDTH * self.PIXEL_SIZE
        self.UI_AREA_WIDTH = self.SCREEN_WIDTH - self.GRID_AREA_WIDTH
        self.MAX_STEPS = 6000 # 60 seconds * 100 steps/s
        self.TARGET_FPS = 100 # To match MAX_STEPS

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_UI_BG = (30, 35, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_HEADER = (255, 180, 0)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_BLANK = (50, 55, 70)
        self.PAINT_PALETTE = [
            (255, 60, 60),    # Red
            (60, 255, 60),    # Green
            (60, 120, 255),   # Blue
            (255, 255, 60),   # Yellow
            (255, 60, 255),   # Magenta
            (60, 255, 255),   # Cyan
        ]
        # Full palette includes the blank color at index 0
        self.FULL_PALETTE = [self.COLOR_BLANK] + self.PAINT_PALETTE
        self.BLANK_COLOR_IDX = 0

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
        self.font_main = pygame.font.Font(None, 24)
        self.font_header = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 16)

        # --- State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.cursor_pos = [0, 0]
        self.selected_paint_color_idx = 0 # Index into PAINT_PALETTE
        self.last_space_held = False
        self.last_shift_held = False
        self.target_grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.int8)
        self.player_grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.int8)
        self.correct_pixels = 0
        self.total_pixels = self.GRID_WIDTH * self.GRID_HEIGHT

        self.validate_implementation(self)

    def _generate_target_grid(self):
        """Generates a simple, procedural image on the grid."""
        grid = np.full((self.GRID_WIDTH, self.GRID_HEIGHT), self.BLANK_COLOR_IDX, dtype=np.int8)
        num_shapes = self.np_random.integers(5, 10)

        for _ in range(num_shapes):
            shape_type = self.np_random.integers(0, 2)
            color_idx = self.np_random.integers(1, len(self.FULL_PALETTE)) # Get a paint color index

            cx = self.np_random.integers(0, self.GRID_WIDTH)
            cy = self.np_random.integers(0, self.GRID_HEIGHT)

            if shape_type == 0: # Rectangle
                w = self.np_random.integers(10, self.GRID_WIDTH // 2)
                h = self.np_random.integers(10, self.GRID_HEIGHT // 2)
                x1 = max(0, cx - w // 2)
                y1 = max(0, cy - h // 2)
                x2 = min(self.GRID_WIDTH, cx + w // 2)
                y2 = min(self.GRID_HEIGHT, cy + h // 2)
                grid[x1:x2, y1:y2] = color_idx
            elif shape_type == 1: # Circle
                radius = self.np_random.integers(5, self.GRID_WIDTH // 4)
                for x in range(max(0, cx - radius), min(self.GRID_WIDTH, cx + radius)):
                    for y in range(max(0, cy - radius), min(self.GRID_HEIGHT, cy + radius)):
                        if math.hypot(x - cx, y - cy) <= radius:
                            grid[x, y] = color_idx
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_paint_color_idx = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.target_grid = self._generate_target_grid()
        self.player_grid = np.full((self.GRID_WIDTH, self.GRID_HEIGHT), self.BLANK_COLOR_IDX, dtype=np.int8)

        # A completely blank grid has some correct pixels if the target has blank spots
        self.correct_pixels = np.sum(self.player_grid == self.target_grid)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.TARGET_FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1
        reward = 0.0

        # --- Unpack and handle actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right

        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Paint (Spacebar press)
        if space_held and not self.last_space_held:
            cx, cy = self.cursor_pos
            # Add 1 to index because FULL_PALETTE has BLANK at index 0
            painted_color_idx = self.selected_paint_color_idx + 1
            old_color_idx = self.player_grid[cx, cy]
            target_color_idx = self.target_grid[cx, cy]

            if old_color_idx != painted_color_idx:
                self.player_grid[cx, cy] = painted_color_idx

                old_was_correct = (old_color_idx == target_color_idx)
                new_is_correct = (painted_color_idx == target_color_idx)

                if new_is_correct and not old_was_correct:
                    self.correct_pixels += 1
                elif not new_is_correct and old_was_correct:
                    self.correct_pixels -= 1

                if new_is_correct:
                    reward = 1.0 # Correct paint
                    # Sound: correct_paint.wav
                else:
                    reward = -0.2 # Incorrect paint
                    # Sound: incorrect_paint.wav

                self.score += reward

        # Cycle Color (Shift press)
        if shift_held and not self.last_shift_held:
            self.selected_paint_color_idx = (self.selected_paint_color_idx + 1) % len(self.PAINT_PALETTE)
            # Sound: color_switch.wav

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Check for termination ---
        terminated = False
        completion_ratio = self.correct_pixels / self.total_pixels

        if self.time_remaining <= 0:
            terminated = True
            # Sound: game_over.wav

        if completion_ratio >= 0.95:
            terminated = True
            win_reward = 100.0
            reward += win_reward
            self.score += win_reward
            # Sound: victory.wav

        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _render_game(self):
        # Draw the player's grid
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                color_idx = self.player_grid[x, y]
                color = self.FULL_PALETTE[color_idx]
                rect = pygame.Rect(
                    x * self.PIXEL_SIZE,
                    y * self.PIXEL_SIZE,
                    self.PIXEL_SIZE,
                    self.PIXEL_SIZE
                )
                # Draw slightly smaller rects to create a grid effect
                pygame.draw.rect(self.screen, color, rect.inflate(-1, -1))

        # Draw the cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            cx * self.PIXEL_SIZE,
            cy * self.PIXEL_SIZE,
            self.PIXEL_SIZE,
            self.PIXEL_SIZE
        )
        # Pulsing effect for cursor outline
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # Varies between 0 and 1
        alpha = int(100 + 155 * pulse)
        cursor_color = self.COLOR_CURSOR + (alpha,)

        # Create a temporary surface for the transparent cursor
        temp_surf = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(temp_surf, cursor_color, temp_surf.get_rect(), 2)
        self.screen.blit(temp_surf, cursor_rect.topleft)

    def _render_ui(self):
        ui_x_start = self.GRID_AREA_WIDTH

        # UI Background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (ui_x_start, 0, self.UI_AREA_WIDTH, self.SCREEN_HEIGHT))

        # --- Text Rendering Helper ---
        def draw_text(text, font, color, x, y, center=False):
            surface = font.render(text, True, color)
            rect = surface.get_rect()
            if center:
                rect.center = (x, y)
            else:
                rect.topleft = (x, y)
            self.screen.blit(surface, rect)

        y_pos = 20

        # Title
        draw_text("PIXEL PAINTER", self.font_header, self.COLOR_UI_HEADER, ui_x_start + self.UI_AREA_WIDTH // 2, y_pos, center=True)
        y_pos += 40

        # Timer
        time_sec = max(0, self.time_remaining / self.TARGET_FPS)
        draw_text(f"TIME: {time_sec:.1f}", self.font_main, self.COLOR_UI_TEXT, ui_x_start + 20, y_pos)
        y_pos += 30

        # Score
        draw_text(f"SCORE: {int(self.score)}", self.font_main, self.COLOR_UI_TEXT, ui_x_start + 20, y_pos)
        y_pos += 30

        # Completion
        completion_ratio = self.correct_pixels / self.total_pixels
        draw_text(f"COMPLETION: {completion_ratio * 100:.1f}%", self.font_main, self.COLOR_UI_TEXT, ui_x_start + 20, y_pos)
        y_pos += 50

        # Selected Color
        draw_text("SELECTED COLOR", self.font_main, self.COLOR_UI_TEXT, ui_x_start + 20, y_pos)
        y_pos += 25
        selected_color = self.PAINT_PALETTE[self.selected_paint_color_idx]
        color_rect = pygame.Rect(ui_x_start + 20, y_pos, self.UI_AREA_WIDTH - 40, 40)
        pygame.draw.rect(self.screen, selected_color, color_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, color_rect, 2) # Border
        y_pos += 60

        # Target Preview
        draw_text("TARGET", self.font_main, self.COLOR_UI_TEXT, ui_x_start + 20, y_pos)
        y_pos += 25
        preview_size = 1
        preview_width = self.GRID_WIDTH * preview_size
        preview_height = self.GRID_HEIGHT * preview_size

        preview_x_offset = ui_x_start + (self.UI_AREA_WIDTH - preview_width) // 2
        preview_y_offset = y_pos

        preview_area = pygame.Rect(preview_x_offset, preview_y_offset, preview_width, preview_height)
        pygame.draw.rect(self.screen, self.COLOR_BG, preview_area)

        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                color_idx = self.target_grid[x, y]
                color = self.FULL_PALETTE[color_idx]
                pixel_rect = pygame.Rect(
                    preview_x_offset + x * preview_size,
                    preview_y_offset + y * preview_size,
                    preview_size,
                    preview_size
                )
                self.screen.set_at(pixel_rect.topleft, color)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, preview_area, 1)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.GRID_AREA_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            completion_ratio = self.correct_pixels / self.total_pixels
            if completion_ratio >= 0.95:
                msg = "SUCCESS!"
                color = (100, 255, 100)
            else:
                msg = "TIME UP!"
                color = (255, 100, 100)
            draw_text(msg, pygame.font.Font(None, 80), color, self.GRID_AREA_WIDTH // 2, self.SCREEN_HEIGHT // 2, center=True)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        completion_ratio = self.correct_pixels / self.total_pixels
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "completion_ratio": completion_ratio
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self, _self_instance_for_static_call=None):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        this = _self_instance_for_static_call if _self_instance_for_static_call else self
        # Test action space
        assert this.action_space.shape == (3,)
        assert this.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        # We need to reset to generate the first observation
        obs, info = this.reset()
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8

        # Test reset
        obs, info = this.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = this.action_space.sample()
        obs, reward, term, trunc, info = this.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

# Example of how to run the environment, e.g., for testing
if __name__ == '__main__':
    import os
    # Set this to "dummy" to run headlessly for server-side testing
    # os.environ["SDL_VIDEODRIVER"] = "dummy" 
    
    # --- For interactive playing ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Painter")
    clock = pygame.time.Clock()

    done = False
    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        movement = 0 # no-op
        space = 0
        shift = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.TARGET_FPS)

    print(f"Game finished.")
    print(f"Final score: {info['score']:.2f}")
    print(f"Final completion: {info['completion_ratio']*100:.2f}%")

    env.close()
    pygame.quit()