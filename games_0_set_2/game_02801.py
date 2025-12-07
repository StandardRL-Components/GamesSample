
# Generated: 2025-08-28T06:00:56.990326
# Source Brief: brief_02801.md
# Brief Index: 2801

        
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
        "Controls: Use arrow keys to shift the row/column of the selected pixel. "
        "Space selects the next pixel, Shift selects the previous one. Match the target pattern!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game where you must rearrange a grid of colored pixels to match a target pattern. "
        "Each move shifts an entire row or column, and you have a limited number of moves. Plan carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.NUM_COLORS = 5
        self.MAX_MOVES = 25
        self.SCRAMBLE_MOVES = 15

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Visuals ---
        self.COLOR_BG = pygame.Color("#1d2b53")
        self.COLOR_GRID_BG = pygame.Color("#314078")
        self.COLOR_PANEL_BG = pygame.Color("#253366")
        self.COLOR_TEXT = pygame.Color("#f1f2f6")
        self.COLOR_SELECT = pygame.Color("#ffffff")
        self.COLOR_EFFECT = pygame.Color("#ffcc00")
        self.COLOR_PALETTE = [
            pygame.Color("#29adff"), # Blue
            pygame.Color("#ff77a8"), # Pink
            pygame.Color("#00e436"), # Green
            pygame.Color("#fff024"), # Yellow
            pygame.Color("#ff004d"), # Red
        ]
        
        try:
            self.FONT_TITLE = pygame.font.Font(None, 36)
            self.FONT_UI = pygame.font.Font(None, 28)
            self.FONT_SMALL = pygame.font.Font(None, 22)
        except pygame.error:
            # Fallback if default font not found (e.g. in minimal docker)
            self.FONT_TITLE = pygame.font.SysFont("sans-serif", 36)
            self.FONT_UI = pygame.font.SysFont("sans-serif", 28)
            self.FONT_SMALL = pygame.font.SysFont("sans-serif", 22)

        # --- Layout ---
        self.PANEL_WIDTH = 220
        self.GRID_AREA_X = self.PANEL_WIDTH
        self.GRID_AREA_WIDTH = self.WIDTH - self.PANEL_WIDTH
        self.CELL_SIZE = 38
        self.GRID_PIXEL_SIZE = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_OFFSET_X = self.GRID_AREA_X + (self.GRID_AREA_WIDTH - self.GRID_PIXEL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_PIXEL_SIZE) // 2

        self.TARGET_CELL_SIZE = 16
        self.TARGET_PIXEL_SIZE = self.GRID_SIZE * self.TARGET_CELL_SIZE
        self.TARGET_OFFSET_X = (self.PANEL_WIDTH - self.TARGET_PIXEL_SIZE) // 2
        self.TARGET_OFFSET_Y = 60
        
        # --- Game State (initialized in reset) ---
        self.grid = None
        self.target_grid = None
        self.selected_pos = None
        self.moves_left = None
        self.score = None
        self.game_over = None
        self.last_correct_pixels = None
        self.last_action_effect = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.target_grid, self.grid = self._generate_puzzle()
        self.selected_pos = (0, 0)
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.last_action_effect = None
        
        self.last_correct_pixels = self._count_correct_pixels()
        self.score = self.last_correct_pixels
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        shift_pressed = action[2] == 1  # Boolean
        
        reward = 0.0
        move_made = False
        
        # --- Action Handling ---
        # Priority: Movement > Select Next > Select Previous
        if movement != 0: # 0=none, 1=up, 2=down, 3=left, 4=right
            # SFX: Play shift sound
            move_made = True
            self.moves_left -= 1
            sel_x, sel_y = self.selected_pos
            
            if movement == 1: # Up
                self.grid[:, sel_x] = np.roll(self.grid[:, sel_x], -1)
                self.last_action_effect = {'type': 'col', 'index': sel_x}
            elif movement == 2: # Down
                self.grid[:, sel_x] = np.roll(self.grid[:, sel_x], 1)
                self.last_action_effect = {'type': 'col', 'index': sel_x}
            elif movement == 3: # Left
                self.grid[sel_y, :] = np.roll(self.grid[sel_y, :], -1)
                self.last_action_effect = {'type': 'row', 'index': sel_y}
            elif movement == 4: # Right
                self.grid[sel_y, :] = np.roll(self.grid[sel_y, :], 1)
                self.last_action_effect = {'type': 'row', 'index': sel_y}

        elif space_pressed:
            # SFX: Play select sound
            current_idx = self.selected_pos[1] * self.GRID_SIZE + self.selected_pos[0]
            next_idx = (current_idx + 1) % (self.GRID_SIZE * self.GRID_SIZE)
            self.selected_pos = (next_idx % self.GRID_SIZE, next_idx // self.GRID_SIZE)
            self.last_action_effect = {'type': 'select'}
        
        elif shift_pressed:
            # SFX: Play select sound
            current_idx = self.selected_pos[1] * self.GRID_SIZE + self.selected_pos[0]
            prev_idx = (current_idx - 1 + (self.GRID_SIZE * self.GRID_SIZE)) % (self.GRID_SIZE * self.GRID_SIZE)
            self.selected_pos = (prev_idx % self.GRID_SIZE, prev_idx // self.GRID_SIZE)
            self.last_action_effect = {'type': 'select'}
        else:
             self.last_action_effect = None

        # --- Reward and State Update ---
        current_correct = self._count_correct_pixels()
        self.score = current_correct

        if move_made:
            reward = (current_correct - self.last_correct_pixels) - 0.1
            self.last_correct_pixels = current_correct

        # --- Termination Check ---
        is_solved = (current_correct == self.GRID_SIZE * self.GRID_SIZE)
        is_out_of_moves = self.moves_left <= 0
        terminated = is_solved or is_out_of_moves

        if terminated:
            self.game_over = True
            if is_solved:
                # SFX: Play win sound
                reward += 100
            elif is_out_of_moves:
                # SFX: Play lose sound
                reward -= 10
        
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
        self._render_panel()
        self._render_grid()
        self.last_action_effect = None # Consume the effect after rendering
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.MAX_MOVES - self.moves_left,
            "moves_left": self.moves_left,
            "correct_pixels": self.last_correct_pixels
        }

    def _generate_puzzle(self):
        target = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE))
        
        # Ensure the puzzle isn't trivially solved or too uniform
        while len(np.unique(target)) < self.NUM_COLORS -1:
             target = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE))

        scrambled = target.copy()
        
        for _ in range(self.SCRAMBLE_MOVES):
            axis = self.np_random.integers(0, 2) # 0 for row, 1 for col
            index = self.np_random.integers(0, self.GRID_SIZE)
            shift = self.np_random.integers(1, self.GRID_SIZE)
            
            if axis == 0: # row
                scrambled[index, :] = np.roll(scrambled[index, :], shift)
            else: # col
                scrambled[:, index] = np.roll(scrambled[:, index], shift)
        
        # Ensure it's not solved by accident
        if np.array_equal(target, scrambled):
            scrambled[0, :] = np.roll(scrambled[0, :], 1)

        return target, scrambled

    def _count_correct_pixels(self):
        return int(np.sum(self.grid == self.target_grid))

    def _render_panel(self):
        # Panel Background
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BG, (0, 0, self.PANEL_WIDTH, self.HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_GRID_BG, (self.PANEL_WIDTH, 0), (self.PANEL_WIDTH, self.HEIGHT), 2)

        # Target Grid
        title_surf = self.FONT_UI.render("TARGET", True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (self.TARGET_OFFSET_X, self.TARGET_OFFSET_Y - 30))
        
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (
            self.TARGET_OFFSET_X - 5, self.TARGET_OFFSET_Y - 5, 
            self.TARGET_PIXEL_SIZE + 10, self.TARGET_PIXEL_SIZE + 10
        ))
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color = self.COLOR_PALETTE[self.target_grid[r, c]]
                rect = pygame.Rect(
                    self.TARGET_OFFSET_X + c * self.TARGET_CELL_SIZE,
                    self.TARGET_OFFSET_Y + r * self.TARGET_CELL_SIZE,
                    self.TARGET_CELL_SIZE, self.TARGET_CELL_SIZE
                )
                pygame.draw.rect(self.screen, color, rect)

        # UI Text
        moves_text = f"Moves: {self.moves_left}"
        score_text = f"Matched: {self.score} / 100"
        
        moves_surf = self.FONT_UI.render(moves_text, True, self.COLOR_TEXT)
        score_surf = self.FONT_UI.render(score_text, True, self.COLOR_TEXT)

        ui_y_start = self.TARGET_OFFSET_Y + self.TARGET_PIXEL_SIZE + 40
        self.screen.blit(moves_surf, ((self.PANEL_WIDTH - moves_surf.get_width()) // 2, ui_y_start))
        self.screen.blit(score_surf, ((self.PANEL_WIDTH - score_surf.get_width()) // 2, ui_y_start + 40))

    def _render_grid(self):
        # Grid Background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (
            self.GRID_OFFSET_X - 5, self.GRID_OFFSET_Y - 5,
            self.GRID_PIXEL_SIZE + 10, self.GRID_PIXEL_SIZE + 10
        ))
        
        # Draw cells and effects
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color = self.COLOR_PALETTE[self.grid[r, c]]
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, color, rect)

                # Draw action effect highlight
                if self.last_action_effect:
                    eff_type = self.last_action_effect['type']
                    if (eff_type == 'row' and self.last_action_effect['index'] == r) or \
                       (eff_type == 'col' and self.last_action_effect['index'] == c):
                        pygame.gfxdraw.rectangle(self.screen, rect, self.COLOR_EFFECT)
                        
        # Draw selection highlight
        sel_x, sel_y = self.selected_pos
        sel_rect = pygame.Rect(
            int(self.GRID_OFFSET_X + sel_x * self.CELL_SIZE),
            int(self.GRID_OFFSET_Y + sel_y * self.CELL_SIZE),
            int(self.CELL_SIZE), int(self.CELL_SIZE)
        )
        
        # "Pop" effect for selection change
        thickness = 3 if self.last_action_effect and self.last_action_effect['type'] == 'select' else 2
        pygame.draw.rect(self.screen, self.COLOR_SELECT, sel_rect, thickness)
        
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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Shift")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print("      Pixel Shift Controls")
    print("="*30)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Action mapping for human play ---
        movement = 0 # no-op
        space_pressed = 0
        shift_pressed = 0

        keys = pygame.key.get_pressed()
        # These are checked on keydown event, not every frame
        
        # --- Event handling ---
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                action_taken = True
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_pressed = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift_pressed = 1
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                    action_taken = False # Don't step on reset
                
        if action_taken:
            action = [movement, space_pressed, shift_pressed]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    print("Game Over!")
    env.close()