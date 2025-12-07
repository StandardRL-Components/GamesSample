
# Generated: 2025-08-27T12:45:38.468285
# Source Brief: brief_00152.md
# Brief Index: 152

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a number, then move to an adjacent number and press Space again to combine them."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Combine numbers on a grid to reach the target value of 100 in this minimalist math puzzle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_ROWS = 5
        self.GRID_COLS = 8
        self.TARGET_NUMBER = 100
        self.MAX_STEPS = 250
        
        # Calculate grid dimensions
        self.GRID_AREA_HEIGHT = self.SCREEN_HEIGHT - 60
        self.CELL_WIDTH = self.SCREEN_WIDTH // self.GRID_COLS
        self.CELL_HEIGHT = self.GRID_AREA_HEIGHT // self.GRID_ROWS
        self.GRID_TOP_MARGIN = 60
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_CURSOR = (255, 255, 0, 150)
        self.COLOR_SELECTED = (0, 255, 150, 150)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_UI_TEXT = (180, 190, 210)
        self.COLOR_WIN = (100, 255, 150)
        self.COLOR_LOSE = (255, 100, 100)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        self.base_font_size = 18

        # State variables are initialized in reset()
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_cell = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.last_space_held = False
        self.rng = None
        
        # Visual effect state
        self.flash_effect = None # Will be {'pos': (r, c), 'value': val} for one frame

        # Initialize state
        self.reset()
        
        # Run validation
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_cell = None
        self.last_space_held = False
        self.flash_effect = None
        self._generate_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is unused per the brief

        reward = 0
        self.flash_effect = None # Clear any previous flash effect

        # --- Handle Input ---
        self._handle_movement(movement)
        
        # Combination logic on the rising edge of the space button press
        if space_held and not self.last_space_held:
            reward = self._handle_combination()
        self.last_space_held = space_held

        # --- Update Game State ---
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.win_condition:
                reward = 100.0 # Large positive reward for winning
            else:
                reward = -10.0 # Small negative reward for losing
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_grid(self):
        self.grid = [[0 for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        num_cells_to_fill = int(self.GRID_ROWS * self.GRID_COLS * 0.7) # Fill 70% of the grid
        
        for _ in range(num_cells_to_fill):
            r, c = self.rng.integers(0, self.GRID_ROWS), self.rng.integers(0, self.GRID_COLS)
            while self.grid[r][c] != 0:
                r, c = self.rng.integers(0, self.GRID_ROWS), self.rng.integers(0, self.GRID_COLS)
            
            self.grid[r][c] = self.rng.integers(1, 21) # Initial numbers between 1 and 20

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_ROWS
        elif movement == 2: # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_ROWS
        elif movement == 3: # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_COLS
        elif movement == 4: # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_COLS

    def _handle_combination(self):
        c, r = self.cursor_pos
        
        # Can't select an empty cell
        if self.grid[r][c] == 0:
            return 0

        # If nothing is selected, select the current cell
        if self.selected_cell is None:
            self.selected_cell = (c, r)
            # sfx: select_sound
            return 0
        
        # If trying to select the same cell again, deselect it
        if self.selected_cell == (c, r):
            self.selected_cell = None
            # sfx: deselect_sound
            return 0

        # If a cell is selected, attempt to combine
        sc, sr = self.selected_cell
        if self._is_adjacent((c, r), (sc, sr)):
            # --- Perform Combination ---
            old_max_val = self._get_max_grid_value()
            
            val1 = self.grid[sr][sc]
            val2 = self.grid[r][c]
            new_val = val1 + val2

            self.grid[r][c] = new_val
            self.grid[sr][sc] = 0 # Empty the source cell
            self.selected_cell = None
            self.score += new_val
            
            # Trigger visual effect for this frame
            self.flash_effect = {'pos': (r, c), 'value': new_val}
            # sfx: combine_sound

            # Check for win condition immediately
            if new_val == self.TARGET_NUMBER:
                self.win_condition = True

            # --- Calculate Reward ---
            new_max_val = self._get_max_grid_value()
            old_dist = abs(self.TARGET_NUMBER - old_max_val)
            new_dist = abs(self.TARGET_NUMBER - new_max_val)
            
            # Reward is positive if the max value got closer to the target
            reward = old_dist - new_dist
            return float(np.clip(reward, -10, 10))
        else:
            # If not adjacent, just change the selection to the new cell
            self.selected_cell = (c, r)
            # sfx: select_sound_fail
            return -0.1 # Small penalty for invalid move attempt

    def _is_adjacent(self, pos1, pos2):
        c1, r1 = pos1
        c2, r2 = pos2
        return abs(c1 - c2) + abs(r1 - r2) == 1

    def _check_termination(self):
        if self.win_condition:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        if not self._has_valid_moves():
            return True
        return False

    def _has_valid_moves(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] != 0:
                    # Check right neighbor
                    if c + 1 < self.GRID_COLS and self.grid[r][c+1] != 0:
                        return True
                    # Check down neighbor
                    if r + 1 < self.GRID_ROWS and self.grid[r+1][c] != 0:
                        return True
        return False

    def _get_max_grid_value(self):
        max_val = 0
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] > max_val:
                    max_val = self.grid[r][c]
        return max_val

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
            "cursor_pos": self.cursor_pos,
            "selected_cell": self.selected_cell,
            "max_value": self._get_max_grid_value(),
        }

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_TOP_MARGIN + r * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 2)
        for c in range(self.GRID_COLS + 1):
            x = c * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_TOP_MARGIN), (x, self.SCREEN_HEIGHT), 2)

        # Draw selected cell highlight
        if self.selected_cell is not None:
            sc, sr = self.selected_cell
            rect = pygame.Rect(sc * self.CELL_WIDTH, self.GRID_TOP_MARGIN + sr * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
            s = pygame.Surface((self.CELL_WIDTH, self.CELL_HEIGHT), pygame.SRCALPHA)
            s.fill(self.COLOR_SELECTED)
            self.screen.blit(s, rect.topleft)

        # Draw cursor
        cc, cr = self.cursor_pos
        rect = pygame.Rect(cc * self.CELL_WIDTH, self.GRID_TOP_MARGIN + cr * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 4)

        # Draw numbers and flash effect
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                value = self.grid[r][c]
                if value > 0:
                    self._render_number(c, r, value)
        
        if self.flash_effect:
            r, c = self.flash_effect['pos']
            value = self.flash_effect['value']
            center_x = int(c * self.CELL_WIDTH + self.CELL_WIDTH / 2)
            center_y = int(self.GRID_TOP_MARGIN + r * self.CELL_HEIGHT + self.CELL_HEIGHT / 2)
            radius = int(min(self.CELL_WIDTH, self.CELL_HEIGHT) * 0.7)
            flash_color = self._get_color_for_value(value, flash=True)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, flash_color)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, flash_color)
            self._render_number(c, r, value) # Redraw number on top of flash

    def _render_number(self, c, r, value):
        color = self._get_color_for_value(value)
        font_size = self.base_font_size + int(math.log(value + 1) * 4)
        font = pygame.font.SysFont("Consolas", font_size, bold=True)
        
        text_surf = font.render(str(value), True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(
            c * self.CELL_WIDTH + self.CELL_WIDTH // 2,
            self.GRID_TOP_MARGIN + r * self.CELL_HEIGHT + self.CELL_HEIGHT // 2
        ))
        
        # Draw a subtle background circle for the number
        circle_radius = int(min(self.CELL_WIDTH, self.CELL_HEIGHT) / 2 * min(1.0, value / self.TARGET_NUMBER * 1.5 + 0.3))
        pygame.gfxdraw.filled_circle(self.screen, text_rect.centerx, text_rect.centery, circle_radius, color)
        pygame.gfxdraw.aacircle(self.screen, text_rect.centerx, text_rect.centery, circle_radius, color)
        
        self.screen.blit(text_surf, text_rect)

    def _get_color_for_value(self, value, flash=False):
        # Interpolate from blue (low) to red (high)
        # Clamp value to avoid extreme colors
        norm_val = min(1.0, value / self.TARGET_NUMBER)
        
        # Blue -> Green -> Yellow -> Red
        if norm_val < 0.33:
            # Blue to Green
            p = norm_val / 0.33
            r = int(60 * (1-p) + 80 * p)
            g = int(120 * (1-p) + 220 * p)
            b = int(240 * (1-p) + 100 * p)
        elif norm_val < 0.66:
            # Green to Yellow
            p = (norm_val - 0.33) / 0.33
            r = int(80 * (1-p) + 255 * p)
            g = int(220 * (1-p) + 220 * p)
            b = int(100 * (1-p) + 50 * p)
        else:
            # Yellow to Red
            p = (norm_val - 0.66) / 0.34
            r = int(255 * (1-p) + 255 * p)
            g = int(220 * (1-p) + 80 * p)
            b = int(50 * (1-p) + 50 * p)

        if flash:
            return (255, 255, 255, 200)
        return (r, g, b)

    def _render_ui(self):
        # Score and Target display
        score_text = f"SCORE: {self.score}"
        target_text = f"TARGET: {self.TARGET_NUMBER}"
        
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        target_surf = self.font_ui.render(target_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(score_surf, (15, 15))
        self.screen.blit(target_surf, (self.SCREEN_WIDTH - target_surf.get_width() - 15, 15))

        # Game Over message
        if self.game_over:
            if self.win_condition:
                msg = "TARGET REACHED!"
                color = self.COLOR_WIN
            else:
                msg = "NO MOVES LEFT"
                color = self.COLOR_LOSE
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_surf = self.font_game_over.render(msg, True, color)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_surf, end_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # For human play
    import pygame
    
    obs, info = env.reset()
    terminated = False
    
    pygame.display.set_caption("Number Grid")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # Game loop
    running = True
    while running:
        movement = 0 # No-op
        space_press = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space_press = 1

            action = [movement, space_press, 0] # shift is unused
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for human play

    env.close()