
# Generated: 2025-08-27T23:33:23.113333
# Source Brief: brief_03498.md
# Brief Index: 3498

        
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
        "Controls: Use arrow keys to move the cursor. Press SHIFT to cycle through colors. Press SPACE to paint the selected cell."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the hidden pixel art image. Place colors strategically to match the target and achieve 90% accuracy before you run out of moves."
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

        # Game constants
        self.GRID_SIZE = 10
        self.MAX_MOVES = self.GRID_SIZE * self.GRID_SIZE
        self.WIN_ACCURACY = 0.90

        # Visual constants
        self.CELL_SIZE = 32
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_TOP_LEFT = (40, 40)

        self.REVEAL_CELL_SIZE = 15
        self.REVEAL_GRID_SIZE = self.GRID_SIZE * self.REVEAL_CELL_SIZE
        self.REVEAL_TOP_LEFT = (420, 180)

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID_LINES = (50, 50, 70)
        self.COLOR_UI_BG = (30, 30, 45)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_ACCENT = (100, 255, 100)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_HIDDEN = (60, 60, 80)
        
        self.PAINT_COLORS = [
            (220, 50, 50),   # Red
            (50, 220, 50),   # Green
            (50, 100, 220),  # Blue
            (220, 220, 50),  # Yellow
            (150, 50, 220),  # Purple
        ]
        self.FEEDBACK_CORRECT = (50, 255, 50, 150)
        self.FEEDBACK_INCORRECT = (255, 50, 50, 150)

        # Fonts
        self.font_s = pygame.font.Font(None, 22)
        self.font_m = pygame.font.Font(None, 28)
        self.font_l = pygame.font.Font(None, 48)
        self.font_xl = pygame.font.Font(None, 64)

        # State variables that persist across resets but are not part of the game logic
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Game-specific state
        self.target_image = self.np_random.integers(0, len(self.PAINT_COLORS), size=(self.GRID_SIZE, self.GRID_SIZE))
        self.player_grid = np.full((self.GRID_SIZE, self.GRID_SIZE), -1, dtype=int)
        self.cursor_pos = [0, 0]  # [col, row]
        self.selected_color_idx = 0
        self.moves_left = self.MAX_MOVES
        self.last_accuracy = 0.0
        self.last_placement_feedback = None # Stores {'pos': (col, row), 'type': 'correct'/'incorrect'}
        
        # Reset action state trackers
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self.last_placement_feedback = None # Clear feedback from previous step

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- Update game logic ---

        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE # Up
        elif movement == 2: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE # Down
        elif movement == 3: self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE # Left
        elif movement == 4: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE # Right

        # 2. Handle color cycling (on key press, not hold)
        if shift_held and not self.prev_shift_held:
            self.selected_color_idx = (self.selected_color_idx + 1) % len(self.PAINT_COLORS)
            # sfx: ui_swap.wav

        # 3. Handle color placement (on key press, not hold)
        if space_held and not self.prev_space_held:
            col, row = self.cursor_pos
            if self.player_grid[row, col] == -1: # Can only place on an empty square
                self.moves_left -= 1
                self.player_grid[row, col] = self.selected_color_idx
                
                target_color_idx = self.target_image[row, col]
                if self.selected_color_idx == target_color_idx:
                    reward = 1
                    self.last_placement_feedback = {'pos': (col, row), 'type': 'correct'}
                    # sfx: place_correct.wav
                else:
                    reward = -1
                    self.last_placement_feedback = {'pos': (col, row), 'type': 'incorrect'}
                    # sfx: place_wrong.wav
        
        self.score += reward

        # Update input state trackers for next step
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # Check termination conditions
        terminated = False
        self.last_accuracy = self._calculate_accuracy()
        
        if self.last_accuracy >= self.WIN_ACCURACY:
            terminated = True
            reward += 100
            self.game_over = True
            # sfx: win_jingle.wav
        elif self.moves_left <= 0:
            terminated = True
            reward -= 100
            self.game_over = True
            # sfx: lose_jingle.wav

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _calculate_accuracy(self):
        filled_mask = self.player_grid != -1
        if not np.any(filled_mask):
            return 0.0
        
        num_filled = np.sum(filled_mask)
        correct_placements = np.sum(self.player_grid[filled_mask] == self.target_image[filled_mask])
        
        return correct_placements / num_filled

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
            "moves_left": self.moves_left,
            "accuracy": self.last_accuracy
        }

    def _draw_text(self, text, font, color, center_pos):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=center_pos)
        self.screen.blit(text_surface, text_rect)

    def _render_game(self):
        # Draw main player grid
        grid_x, grid_y = self.GRID_TOP_LEFT
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(
                    grid_x + c * self.CELL_SIZE,
                    grid_y + r * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE
                )
                color_idx = self.player_grid[r, c]
                if color_idx != -1:
                    pygame.draw.rect(self.screen, self.PAINT_COLORS[color_idx], cell_rect)
        
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (grid_x + i * self.CELL_SIZE, grid_y)
            end_pos = (grid_x + i * self.CELL_SIZE, grid_y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (grid_x, grid_y + i * self.CELL_SIZE)
            end_pos = (grid_x + self.GRID_WIDTH, grid_y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos, 1)

        # Draw placement feedback
        if self.last_placement_feedback:
            c, r = self.last_placement_feedback['pos']
            feedback_rect = pygame.Rect(
                grid_x + c * self.CELL_SIZE,
                grid_y + r * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            feedback_color = self.FEEDBACK_CORRECT if self.last_placement_feedback['type'] == 'correct' else self.FEEDBACK_INCORRECT
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill(feedback_color)
            self.screen.blit(s, feedback_rect.topleft)

        # Draw cursor
        cur_c, cur_r = self.cursor_pos
        cursor_rect = pygame.Rect(
            grid_x + cur_c * self.CELL_SIZE,
            grid_y + cur_r * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        # UI Panel background
        ui_panel_rect = pygame.Rect(380, 0, 640 - 380, 400)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_panel_rect)
        
        # Title
        self._draw_text("Pixel Perfect", self.font_l, self.COLOR_TEXT, (510, 40))

        # Selected Color
        self._draw_text("Selected Color", self.font_m, self.COLOR_TEXT, (510, 90))
        color_box = pygame.Rect(0, 0, 80, 50)
        color_box.center = (510, 130)
        pygame.draw.rect(self.screen, self.PAINT_COLORS[self.selected_color_idx], color_box)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, color_box, 2)

        # Revealed Target Image
        self._draw_text("Target", self.font_m, self.COLOR_TEXT, (510, 180))
        rev_x, rev_y = self.REVEAL_TOP_LEFT
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(
                    rev_x + c * self.REVEAL_CELL_SIZE,
                    rev_y + r * self.REVEAL_CELL_SIZE,
                    self.REVEAL_CELL_SIZE,
                    self.REVEAL_CELL_SIZE
                )
                if self.player_grid[r, c] != -1:
                    color_idx = self.target_image[r, c]
                    pygame.draw.rect(self.screen, self.PAINT_COLORS[color_idx], cell_rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_HIDDEN, cell_rect)
        
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, (rev_x, rev_y, self.REVEAL_GRID_SIZE, self.REVEAL_GRID_SIZE), 1)

        # Stats
        self._draw_text(f"Moves Left: {self.moves_left}", self.font_m, self.COLOR_TEXT, (510, 350))
        accuracy_text = f"Accuracy: {self.last_accuracy:.1%}"
        self._draw_text(accuracy_text, self.font_m, self.COLOR_TEXT_ACCENT, (510, 375))

    def _render_game_over(self):
        overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.last_accuracy >= self.WIN_ACCURACY:
            self._draw_text("VICTORY!", self.font_xl, self.COLOR_TEXT_ACCENT, (320, 150))
            self._draw_text(f"Final Accuracy: {self.last_accuracy:.1%}", self.font_l, self.COLOR_TEXT, (320, 220))
        else:
            self._draw_text("OUT OF MOVES", self.font_xl, (220, 50, 50), (320, 150))
            self._draw_text(f"Final Accuracy: {self.last_accuracy:.1%}", self.font_l, self.COLOR_TEXT, (320, 220))
        
        self._draw_text("Press Reset to Play Again", self.font_m, self.COLOR_TEXT, (320, 280))

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

# Example usage for interactive play
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Pixel Perfect")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0  # No-op
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
        
        # Since auto_advance is False, we only step on an event
        # For interactive play, we need to decide when to step.
        # A simple approach is to step on any key press.
        
        event_occurred = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                event_occurred = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()

        # Step the environment if a key was pressed or if the mouse was clicked etc.
        # For this game, any keydown is a valid time to process an action.
        if event_occurred:
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Game Over! Score: {info['score']}, Accuracy: {info['accuracy']:.1%}")

        # Rendering
        # The observation is the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Limit frame rate for human play

    pygame.quit()