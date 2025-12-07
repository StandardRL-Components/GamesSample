import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor, Space to cycle colors, Shift to paint."
    )

    game_description = (
        "Recreate a hidden pixel art image by strategically coloring squares on a grid before you run out of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 10
        self.MAX_MOVES = 20
        self.WIN_ACCURACY = 0.8

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Visuals ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (60, 70, 80)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HINT = (40, 45, 50)

        # Palette: 0 is background, 1-4 are colors
        self.COLOR_PALETTE = [
            self.COLOR_BG,
            (220, 50, 50),   # Red
            (50, 220, 50),   # Green
            (50, 150, 255),  # Blue
            (255, 200, 50),  # Yellow
        ]
        self.DESATURATED_PALETTE = [
            self.COLOR_BG,
            (110, 65, 65),
            (65, 110, 65),
            (65, 95, 128),
            (128, 110, 65),
        ]
        
        # --- Predefined Target Images ---
        self.TARGET_PATTERNS = self._define_patterns()
        
        # --- State Variables ---
        self.target_image = None
        self.player_grid = None
        self.cursor_pos = None
        self.selected_color_idx = None
        self.moves_remaining = None
        self.score = None
        self.accuracy = None
        self.game_over = None
        self.steps = None
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.completed_rows = None
        self.completed_cols = None
        self.victory_bonus_awarded = None
        
        self.paint_effects = []

        # --- Grid Geometry ---
        self.cell_size = (self.SCREEN_HEIGHT - 80) // self.GRID_SIZE
        self.grid_width = self.GRID_SIZE * self.cell_size
        self.grid_height = self.GRID_SIZE * self.cell_size
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2
        
        # Initialize state for validation
        self.reset(seed=42)
        self.validate_implementation()


    def _define_patterns(self):
        heart = [
            "0033003300",
            "0333303333",
            "3333333333",
            "3333333333",
            "3333333333",
            "0333333330",
            "0033333300",
            "0003333000",
            "0000330000",
            "0000000000",
        ]
        creeper = [
            "2222222222",
            "2222222222",
            "2202202202",
            "2202202202",
            "2222002222",
            "2220000222",
            "2220000222",
            "2220220222",
            "2222222222",
            "2222222222",
        ]
        boat = [
            "0000400000",
            "0004440000",
            "0004040000",
            "0044444000",
            "0000000000",
            "1111111111",
            "0111111110",
            "0011111100",
            "0001111000",
            "0000000000",
        ]
        patterns = [heart, creeper, boat]
        return [np.array([[int(c) for c in row] for row in p]) for p in patterns]


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.target_image = self.TARGET_PATTERNS[self.np_random.choice(len(self.TARGET_PATTERNS))]
        
        self.player_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_color_idx = 1 # Start on first actual color
        self.moves_remaining = self.MAX_MOVES
        self.score = 0
        self.accuracy = 0.0
        self.game_over = False
        self.steps = 0
        
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True
        
        self.completed_rows = set()
        self.completed_cols = set()
        self.victory_bonus_awarded = False
        
        self.paint_effects = []

        self._calculate_accuracy()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_action, shift_action = action
        reward = 0.0
        
        # --- Handle Actions ---
        # 1. Cursor Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] %= self.GRID_SIZE
        self.cursor_pos[1] %= self.GRID_SIZE

        # 2. Color Cycling (on key press, not hold)
        space_held = space_action == 1
        if space_held and not self.prev_space_held:
            self.selected_color_idx = (self.selected_color_idx % (len(self.COLOR_PALETTE) - 1)) + 1
        self.prev_space_held = space_held
        
        # 3. Paint Action (on key press, not hold)
        shift_held = shift_action == 1
        if shift_held and not self.prev_shift_held and not self.game_over:
            self.moves_remaining -= 1
            cx, cy = self.cursor_pos
            
            # Place the color
            self.player_grid[cy, cx] = self.selected_color_idx
            
            self._add_paint_effect(cx, cy)
            
            # --- Calculate Reward ---
            if self.player_grid[cy, cx] == self.target_image[cy, cx]:
                reward += 1.0 # Correct color
            else:
                reward -= 1.0 # Incorrect color
            
            self._calculate_accuracy()

            # Row/Column Completion Bonus
            if cy not in self.completed_rows and np.array_equal(self.player_grid[cy, :], self.target_image[cy, :]):
                reward += 5.0
                self.completed_rows.add(cy)
            if cx not in self.completed_cols and np.array_equal(self.player_grid[:, cx], self.target_image[:, cx]):
                reward += 5.0
                self.completed_cols.add(cx)

            # Victory Bonus
            if self.accuracy >= self.WIN_ACCURACY and not self.victory_bonus_awarded:
                reward += 50.0
                self.victory_bonus_awarded = True
        
        self.prev_shift_held = shift_held

        # --- Update Game State ---
        self.steps += 1
        self.score += reward
        self._update_effects()
        
        terminated = bool(self.moves_remaining <= 0 or (self.accuracy >= self.WIN_ACCURACY))
        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_accuracy(self):
        if self.target_image is None or self.player_grid is None:
            self.accuracy = 0.0
            return
        
        correct_pixels = np.sum(self.player_grid == self.target_image)
        self.accuracy = correct_pixels / (self.GRID_SIZE * self.GRID_SIZE)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render faint hint of the target image
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_idx = self.target_image[y, x]
                if color_idx != 0:
                    rect = pygame.Rect(
                        self.grid_offset_x + x * self.cell_size,
                        self.grid_offset_y + y * self.cell_size,
                        self.cell_size, self.cell_size
                    )
                    pygame.draw.rect(self.screen, self.COLOR_HINT, rect)
    
        # Render the player's grid
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_idx = self.player_grid[y, x]
                if color_idx != 0:
                    is_correct = color_idx == self.target_image[y, x]
                    color = self.COLOR_PALETTE[color_idx] if is_correct else self.DESATURATED_PALETTE[color_idx]
                    rect = pygame.Rect(
                        self.grid_offset_x + x * self.cell_size,
                        self.grid_offset_y + y * self.cell_size,
                        self.cell_size, self.cell_size
                    )
                    pygame.draw.rect(self.screen, color, rect)

        # Render grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y + self.grid_height)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.grid_offset_x, self.grid_offset_y + i * self.cell_size)
            end_pos = (self.grid_offset_x + self.grid_width, self.grid_offset_y + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Render paint effects
        for effect in self.paint_effects:
            progress = effect['life'] / effect['max_life']
            size = int(self.cell_size * progress)
            offset = (self.cell_size - size) // 2
            rect = pygame.Rect(
                self.grid_offset_x + effect['pos'][0] * self.cell_size + offset,
                self.grid_offset_y + effect['pos'][1] * self.cell_size + offset,
                size, size
            )
            pygame.draw.rect(self.screen, effect['color'], rect, border_radius=2)
            
        # Render cursor
        if self.cursor_pos:
            cx, cy = self.cursor_pos
            pulse = 2 * math.sin(self.steps * 0.2)
            cursor_rect = pygame.Rect(
                self.grid_offset_x + cx * self.cell_size - pulse,
                self.grid_offset_y + cy * self.cell_size - pulse,
                self.cell_size + 2 * pulse,
                self.cell_size + 2 * pulse
            )
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=3)

    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        # Score
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)
        
        # Accuracy
        acc_text = self.font_small.render(f"Accuracy: {self.accuracy:.0%}", True, self.COLOR_TEXT)
        acc_rect = acc_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 55))
        self.screen.blit(acc_text, acc_rect)

        # Color Palette UI
        palette_label = self.font_small.render("Color:", True, self.COLOR_TEXT)
        self.screen.blit(palette_label, (20, self.SCREEN_HEIGHT - 45))
        
        swatch_size = 30
        for i, color in enumerate(self.COLOR_PALETTE):
            if i == 0: continue # Skip background color
            x_pos = 90 + (i-1) * (swatch_size + 10)
            y_pos = self.SCREEN_HEIGHT - 50
            rect = pygame.Rect(x_pos, y_pos, swatch_size, swatch_size)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            if i == self.selected_color_idx:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=4)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.accuracy >= self.WIN_ACCURACY:
                msg = "Congratulations!"
            else:
                msg = "Out of Moves!"
            
            end_text = self.font_main.render(msg, True, (255, 255, 255))
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, end_rect)


    def _add_paint_effect(self, x, y):
        color_idx = self.player_grid[y, x]
        is_correct = color_idx == self.target_image[y, x]
        color = self.COLOR_PALETTE[color_idx] if is_correct else self.DESATURATED_PALETTE[color_idx]
        self.paint_effects.append({
            'pos': (x, y),
            'color': color,
            'life': 10,
            'max_life': 10
        })
        
    def _update_effects(self):
        self.paint_effects = [e for e in self.paint_effects if e['life'] > 0]
        for effect in self.paint_effects:
            effect['life'] -= 1

    def _get_info(self):
        return {
            "score": self.score,
            "moves_remaining": self.moves_remaining,
            "accuracy": self.accuracy,
            "steps": self.steps,
        }
        
    def close(self):
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
        assert trunc is False
        assert isinstance(info, dict)

# Example usage for interactive play
if __name__ == '__main__':
    # The main loop needs a real display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)
    print(env.game_description)

    # Use a separate display for human interaction
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Art Restorer")
    clock = pygame.time.Clock()

    while not done:
        # --- Action mapping from keyboard to MultiDiscrete ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering for human player ---
        # The observation is already a rendered frame
        # We just need to blit it to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Limit frame rate for human play

    print(f"Game Over! Final Info: {info}")
    pygame.time.wait(2000)
    env.close()