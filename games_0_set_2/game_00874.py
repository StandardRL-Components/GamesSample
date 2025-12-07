
# Generated: 2025-08-27T15:03:42.977688
# Source Brief: brief_00874.md
# Brief Index: 874

        
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
    """
    A Gymnasium environment for a pixel-painting puzzle game.

    The player must replicate a target 10x10 pixel art image on their own
    canvas within a 60-second time limit. The game is won by achieving
    90% or greater accuracy.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move the cursor. Space to paint. Shift to cycle the selected color."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target pixel art image before time runs out. Select colors and paint squares to match the picture and achieve 90% accuracy to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_DIM = 10
        self.MAX_STEPS = 600  # 60 seconds at 10 steps/sec
        self.WIN_ACCURACY = 0.9
        
        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_UI_BG = (40, 45, 60)
        self.COLOR_GRID_LINE = (60, 65, 80)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_WIN = (0, 255, 128, 180)
        self.COLOR_LOSE = (255, 0, 64, 180)
        
        # Palette: Index 0 is the blank color.
        self.PALETTE = [
            (80, 80, 80),      # 0: Blank
            (230, 60, 60),     # 1: Red
            (60, 200, 60),     # 2: Green
            (60, 100, 230),    # 3: Blue
            (240, 240, 80),    # 4: Yellow
            (230, 80, 230),    # 5: Magenta
            (80, 220, 220),    # 6: Cyan
            (250, 250, 250),   # 7: White
            (10, 10, 10),      # 8: Black
        ]
        
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
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_state = ""
        self.cursor_pos = [0, 0]
        self.selected_color_index = 1
        self.target_grid = None
        self.player_grid = None
        self.accuracy = 0.0
        self.last_space_held = False
        self.last_shift_held = False
        self.np_random = None
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_state = ""
        self.cursor_pos = [self.GRID_DIM // 2, self.GRID_DIM // 2]
        self.selected_color_index = 1  # Start with the first paintable color
        self.last_space_held = False
        self.last_shift_held = False

        # Generate a new target image using paintable colors (indices 1 to end)
        self.target_grid = self.np_random.integers(
            1, len(self.PALETTE), size=(self.GRID_DIM, self.GRID_DIM), dtype=int
        )
        # Player grid starts blank (index 0)
        self.player_grid = np.full((self.GRID_DIM, self.GRID_DIM), 0, dtype=int)
        
        self.accuracy = self._calculate_accuracy()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # Use rising edge detection for discrete actions
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle color cycling (Shift)
        if shift_pressed:
            # `// sound: color_cycle.wav`
            current_paint_idx = self.selected_color_index - 1
            num_paint_colors = len(self.PALETTE) - 1
            new_paint_idx = (current_paint_idx + 1) % num_paint_colors
            self.selected_color_index = new_paint_idx + 1

        # 2. Handle cursor movement
        if movement == 1:  # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_DIM) % self.GRID_DIM
        elif movement == 2:  # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_DIM
        elif movement == 3:  # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_DIM) % self.GRID_DIM
        elif movement == 4:  # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_DIM

        # 3. Handle painting (Space)
        if space_pressed:
            # `// sound: paint_splat.wav`
            cx, cy = self.cursor_pos
            old_pixel_correct = self.player_grid[cy, cx] == self.target_grid[cy, cx]
            
            self.player_grid[cy, cx] = self.selected_color_index
            
            new_pixel_correct = self.player_grid[cy, cx] == self.target_grid[cy, cx]
            
            if new_pixel_correct and not old_pixel_correct:
                reward += 0.1  # Reward for correcting a pixel
            
            self.accuracy = self._calculate_accuracy()
            self.score += reward

        # 4. Update game state
        self.steps += 1
        
        # 5. Check for termination
        terminated = False
        if self.accuracy >= self.WIN_ACCURACY:
            # `// sound: win_jingle.wav`
            reward += 100.0
            self.score += 100.0
            terminated = True
            self.game_over = True
            self.win_state = "WIN"
        elif self.steps >= self.MAX_STEPS:
            # `// sound: lose_buzzer.wav`
            terminated = True
            self.game_over = True
            self.win_state = "LOSE"
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_accuracy(self):
        if self.target_grid is None or self.player_grid is None:
            return 0.0
        correct_pixels = np.sum(self.player_grid == self.target_grid)
        total_pixels = self.GRID_DIM * self.GRID_DIM
        return correct_pixels / total_pixels if total_pixels > 0 else 0.0

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
            "accuracy": self.accuracy,
        }

    def _render_grid(self, surface, grid_data, top_left_x, top_left_y, pixel_size):
        for y in range(self.GRID_DIM):
            for x in range(self.GRID_DIM):
                color_index = grid_data[y, x]
                color = self.PALETTE[color_index]
                rect = pygame.Rect(
                    top_left_x + x * pixel_size,
                    top_left_y + y * pixel_size,
                    pixel_size,
                    pixel_size
                )
                pygame.draw.rect(surface, color, rect)
        
        grid_total_size = self.GRID_DIM * pixel_size
        for i in range(self.GRID_DIM + 1):
            pygame.draw.line(surface, self.COLOR_GRID_LINE, (top_left_x + i * pixel_size, top_left_y), (top_left_x + i * pixel_size, top_left_y + grid_total_size))
            pygame.draw.line(surface, self.COLOR_GRID_LINE, (top_left_x, top_left_y + i * pixel_size), (top_left_x + grid_total_size, top_left_y + i * pixel_size))

    def _render_game(self):
        pixel_size = 22
        grid_width = self.GRID_DIM * pixel_size
        padding = 20
        
        target_grid_pos = (padding, (self.SCREEN_HEIGHT - grid_width) // 2)
        player_grid_pos = (padding * 2 + grid_width, (self.SCREEN_HEIGHT - grid_width) // 2)
        
        # Render Titles
        target_title = self.font_small.render("Target", True, self.COLOR_TEXT)
        self.screen.blit(target_title, (target_grid_pos[0], target_grid_pos[1] - 25))
        player_title = self.font_small.render("Your Canvas", True, self.COLOR_TEXT)
        self.screen.blit(player_title, (player_grid_pos[0], player_grid_pos[1] - 25))

        # Render Grids
        self._render_grid(self.screen, self.target_grid, target_grid_pos[0], target_grid_pos[1], pixel_size)
        self._render_grid(self.screen, self.player_grid, player_grid_pos[0], player_grid_pos[1], pixel_size)
        
        # Render Cursor
        if not self.game_over:
            cx, cy = self.cursor_pos
            cursor_rect = pygame.Rect(
                player_grid_pos[0] + cx * pixel_size,
                player_grid_pos[1] + cy * pixel_size,
                pixel_size,
                pixel_size
            )
            pulse = (math.sin(self.steps * 0.3) + 1) / 2
            line_width = int(2 + pulse * 2)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, line_width, border_radius=2)

    def _render_ui(self):
        padding = 20
        ui_panel_width = self.SCREEN_WIDTH - (padding * 3 + (self.GRID_DIM * 22) * 2)
        ui_panel_x = self.SCREEN_WIDTH - ui_panel_width - padding
        ui_panel_rect = pygame.Rect(ui_panel_x, padding, ui_panel_width, self.SCREEN_HEIGHT - padding * 2)
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_panel_rect, border_radius=10)

        current_y = ui_panel_rect.y + 20
        
        time_left = max(0, (self.MAX_STEPS - self.steps) / 10.0)
        timer_text = self.font_medium.render(f"Time: {time_left:.1f}s", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (ui_panel_x + 20, current_y))
        current_y += 40

        acc_text = self.font_medium.render(f"Accuracy: {self.accuracy:.1%}", True, self.COLOR_TEXT)
        self.screen.blit(acc_text, (ui_panel_x + 20, current_y))
        current_y += 40
        
        score_text = self.font_medium.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_panel_x + 20, current_y))
        current_y += 60

        palette_title = self.font_small.render("Color (Shift to cycle):", True, self.COLOR_TEXT)
        self.screen.blit(palette_title, (ui_panel_x + 20, current_y))
        current_y += 30
        
        color_box_size = 28
        for i, color in enumerate(self.PALETTE):
            if i == 0: continue
            row, col = divmod(i - 1, 4)
            box_x = ui_panel_x + 20 + col * (color_box_size + 5)
            box_y = current_y + row * (color_box_size + 5)
            pygame.draw.rect(self.screen, color, (box_x, box_y, color_box_size, color_box_size), border_radius=4)
            if i == self.selected_color_index:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, (box_x - 2, box_y - 2, color_box_size + 4, color_box_size + 4), 2, border_radius=5)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        color = self.COLOR_WIN if self.win_state == "WIN" else self.COLOR_LOSE
        text = "YOU WIN!" if self.win_state == "WIN" else "TIME'S UP!"
        
        overlay.fill(color)
        self.screen.blit(overlay, (0, 0))
        
        end_text_surf = self.font_large.render(text, True, self.COLOR_TEXT)
        shadow_surf = self.font_large.render(text, True, (0, 0, 0, 100))
        end_text_rect = end_text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        self.screen.blit(shadow_surf, (end_text_rect.x + 3, end_text_rect.y + 3))
        self.screen.blit(end_text_surf, end_text_rect)

    def close(self):
        pygame.font.quit()
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Painter")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    # Game loop
    while running:
        # Action defaults
        movement = 0  # no-op
        space_held = 0
        shift_held = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Movement
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            # Other actions
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
        else: # Game is over
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]: # Press R to reset
                obs, info = env.reset()
                terminated = False
        
        # Rendering
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        if terminated:
            font = pygame.font.Font(None, 36)
            text = font.render("Press 'R' to restart", True, (255, 255, 255))
            text_rect = text.get_rect(center=(env.SCREEN_WIDTH / 2, env.SCREEN_HEIGHT - 30))
            screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(10) # Run at 10 FPS to match step rate
        
    env.close()