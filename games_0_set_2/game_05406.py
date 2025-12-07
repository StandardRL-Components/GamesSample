
# Generated: 2025-08-28T04:55:03.090014
# Source Brief: brief_05406.md
# Brief Index: 5406

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle through colors. Press Space to paint."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target pixel art image on the canvas. Match 90% of the pixels before the timer runs out to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
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
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 24
        self.CELL_SIZE = 15
        self.MAX_STEPS = 600
        self.VICTORY_THRESHOLD = 0.90
        
        self.CANVAS_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.CANVAS_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.CANVAS_X_OFFSET = 30
        self.CANVAS_Y_OFFSET = (self.SCREEN_HEIGHT - self.CANVAS_HEIGHT) // 2

        self.UI_X_OFFSET = self.CANVAS_X_OFFSET + self.CANVAS_WIDTH + 30
        self.UI_WIDTH = self.SCREEN_WIDTH - self.UI_X_OFFSET - 20

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_CANVAS_BG = (40, 45, 60)
        self.COLOR_GRID_LINE = (50, 55, 70)
        self.COLOR_UI_BG = (30, 35, 50)
        self.COLOR_UI_BORDER = (60, 65, 80)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_CURSOR = (255, 255, 0)
        
        self.COLOR_PALETTE = [
            (230, 57, 70),   # Red
            (168, 218, 220), # Light Blue
            (69, 123, 157),  # Dark Blue
            (241, 250, 238), # White
        ]
        self.COLOR_EMPTY = 0 # Index for empty/background in grids

        # Fonts
        self.font_s = pygame.font.Font(None, 20)
        self.font_m = pygame.font.Font(None, 28)
        self.font_l = pygame.font.Font(None, 36)

        # Game state variables (initialized in reset)
        self.np_random = None
        self.canvas_grid = None
        self.target_image = None
        self.cursor_pos = None
        self.selected_color_idx = None
        self.steps = None
        self.score = None
        self.game_over = None
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_color_idx = 0
        
        self.target_image = self._generate_target_image()
        self.canvas_grid = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), self.COLOR_EMPTY, dtype=int)
        
        self.score = self._calculate_accuracy()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_press = action[1] == 1
        shift_press = action[2] == 1
        
        self.steps += 1
        reward = 0.0
        
        # --- Handle Actions ---
        # Shift: Cycle color
        if shift_press:
            self.selected_color_idx = (self.selected_color_idx + 1) % len(self.COLOR_PALETTE)
            # SFX: color_swap.wav

        # Movement: Move cursor
        if movement > 0:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_WIDTH - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_HEIGHT - 1)

        # Space: Paint pixel
        if space_press:
            cx, cy = self.cursor_pos
            target_color = self.target_image[cy, cx]
            new_color = self.selected_color_idx + 1  # Palette is 1-indexed in grid
            old_color = self.canvas_grid[cy, cx]

            if new_color != old_color:
                # SFX: paint_pixel.wav
                was_correct = (old_color == target_color and target_color != self.COLOR_EMPTY)
                is_correct = (new_color == target_color and target_color != self.COLOR_EMPTY)

                if is_correct and not was_correct:
                    reward += 0.1  # Correctly placed a pixel
                elif not is_correct and was_correct:
                    reward -= 0.1  # Overwrote a correct pixel with a wrong one
                elif not is_correct and not was_correct:
                    reward -= 0.01 # Placed a wrong pixel where it was already wrong/empty
                
                self.canvas_grid[cy, cx] = new_color

        # --- Update State and Check Termination ---
        self.score = self._calculate_accuracy()
        terminated = False

        if self.score >= self.VICTORY_THRESHOLD:
            reward += 100
            terminated = True
            self.game_over = True
            # SFX: victory.wav
        
        if self.steps >= self.MAX_STEPS:
            reward -= 10
            terminated = True
            self.game_over = True
            # SFX: timeout.wav

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game_area()
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_steps": self.MAX_STEPS - self.steps,
            "accuracy": self.score
        }

    def _generate_target_image(self):
        grid = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), self.COLOR_EMPTY, dtype=int)
        num_shapes = self.np_random.integers(3, 6)
        
        for _ in range(num_shapes):
            color_idx = self.np_random.integers(1, len(self.COLOR_PALETTE) + 1)
            w = self.np_random.integers(2, self.GRID_WIDTH // 2)
            h = self.np_random.integers(2, self.GRID_HEIGHT // 2)
            x = self.np_random.integers(0, self.GRID_WIDTH - w)
            y = self.np_random.integers(0, self.GRID_HEIGHT - h)
            grid[y:y+h, x:x+w] = color_idx
        return grid

    def _calculate_accuracy(self):
        target_pixels_mask = self.target_image != self.COLOR_EMPTY
        total_target_pixels = np.sum(target_pixels_mask)
        
        if total_target_pixels == 0:
            return 1.0

        correct_pixels = np.sum(self.canvas_grid[target_pixels_mask] == self.target_image[target_pixels_mask])
        return correct_pixels / total_target_pixels

    def _render_grid(self, surface, grid, cell_size, offset_x, offset_y, show_grid_lines=False):
        grid_h, grid_w = grid.shape
        for y in range(grid_h):
            for x in range(grid_w):
                color_idx = grid[y, x]
                rect = pygame.Rect(
                    offset_x + x * cell_size,
                    offset_y + y * cell_size,
                    cell_size,
                    cell_size
                )
                if color_idx == self.COLOR_EMPTY:
                    pygame.draw.rect(surface, self.COLOR_CANVAS_BG, rect)
                else:
                    pygame.draw.rect(surface, self.COLOR_PALETTE[color_idx - 1], rect)
        
        if show_grid_lines:
            for i in range(grid_w + 1):
                start = (offset_x + i * cell_size, offset_y)
                end = (offset_x + i * cell_size, offset_y + grid_h * cell_size)
                pygame.draw.line(surface, self.COLOR_GRID_LINE, start, end)
            for i in range(grid_h + 1):
                start = (offset_x, offset_y + i * cell_size)
                end = (offset_x + grid_w * cell_size, offset_y + i * cell_size)
                pygame.draw.line(surface, self.COLOR_GRID_LINE, start, end)

    def _render_game_area(self):
        # Render canvas
        self._render_grid(self.screen, self.canvas_grid, self.CELL_SIZE, self.CANVAS_X_OFFSET, self.CANVAS_Y_OFFSET, True)
        
        # Render cursor
        cursor_x = self.CANVAS_X_OFFSET + self.cursor_pos[0] * self.CELL_SIZE
        cursor_y = self.CANVAS_Y_OFFSET + self.cursor_pos[1] * self.CELL_SIZE
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)

    def _render_ui(self):
        # UI Panel
        ui_rect = pygame.Rect(self.UI_X_OFFSET, self.CANVAS_Y_OFFSET, self.UI_WIDTH, self.CANVAS_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, ui_rect, 2)
        
        y_pos = self.CANVAS_Y_OFFSET + 20

        # Target Image
        self._draw_text("Target", self.font_m, y_pos, self.UI_X_OFFSET + self.UI_WIDTH // 2)
        y_pos += 30
        target_cell_size = min((self.UI_WIDTH - 20) // self.GRID_WIDTH, (120) // self.GRID_HEIGHT)
        target_w = self.GRID_WIDTH * target_cell_size
        target_h = self.GRID_HEIGHT * target_cell_size
        target_x = self.UI_X_OFFSET + (self.UI_WIDTH - target_w) // 2
        self._render_grid(self.screen, self.target_image, target_cell_size, target_x, y_pos)
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, (target_x-1, y_pos-1, target_w+2, target_h+2), 1)
        y_pos += target_h + 20

        # Timer
        self._draw_text("Time", self.font_m, y_pos, self.UI_X_OFFSET + self.UI_WIDTH // 2)
        y_pos += 25
        time_pct = max(0, (self.MAX_STEPS - self.steps) / self.MAX_STEPS)
        bar_w = self.UI_WIDTH - 40
        bar_h = 15
        bar_x = self.UI_X_OFFSET + 20
        
        time_color = (80, 200, 120)
        if time_pct < 0.5: time_color = (255, 193, 7)
        if time_pct < 0.2: time_color = (211, 47, 47)

        pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, (bar_x, y_pos, bar_w, bar_h))
        pygame.draw.rect(self.screen, time_color, (bar_x, y_pos, bar_w * time_pct, bar_h))
        y_pos += bar_h + 20

        # Accuracy
        acc_text = f"Accuracy: {self.score:.1%}"
        self._draw_text(acc_text, self.font_m, y_pos, self.UI_X_OFFSET + self.UI_WIDTH // 2)
        y_pos += 40

        # Color Palette
        self._draw_text("Color", self.font_m, y_pos, self.UI_X_OFFSET + self.UI_WIDTH // 2)
        y_pos += 30
        palette_box_size = 25
        total_palette_width = len(self.COLOR_PALETTE) * (palette_box_size + 5) - 5
        start_x = self.UI_X_OFFSET + (self.UI_WIDTH - total_palette_width) // 2
        for i, color in enumerate(self.COLOR_PALETTE):
            rect = pygame.Rect(start_x + i * (palette_box_size + 5), y_pos, palette_box_size, palette_box_size)
            pygame.draw.rect(self.screen, color, rect)
            if i == self.selected_color_idx:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3)
    
    def _draw_text(self, text, font, y_pos, center_x):
        text_surf = font.render(text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(centerx=center_x, top=y_pos)
        self.screen.blit(text_surf, text_rect)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:")
    print(f"  Score: {info['score']:.2f}, Steps: {info['steps']}")

    # Take 10 random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print("Episode finished.")
            break
    
    # Save an image of the final state
    from PIL import Image
    img = Image.fromarray(obs)
    img.save("game_state.png")
    print("Saved final game state to game_state.png")