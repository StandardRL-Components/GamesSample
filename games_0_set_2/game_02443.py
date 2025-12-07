
# Generated: 2025-08-28T04:50:46.696428
# Source Brief: brief_02443.md
# Brief Index: 2443

        
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
        "Controls: Use arrow keys to move the cursor. Press space to pick up or place a pixel. "
        "Hold shift and press an arrow key to jump to the next pixel in that direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Rearrange the pixels on the grid to match the target pattern shown above. "
        "You only have 20 moves, so plan your swaps carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.MAX_MOVES = 20
        self.NUM_SWAPS = 4  # Initial puzzle difficulty

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_LINES = (40, 50, 70)
        self.COLOR_TEXT = (220, 230, 240)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_CURSOR_SELECT = (0, 255, 128)
        self.PIXEL_COLORS = {
            1: (255, 80, 80),   # Red
            2: (80, 255, 80),   # Green
            3: (80, 120, 255),  # Blue
            4: (255, 200, 80),  # Orange
        }
        self.COLOR_EMPTY = (30, 35, 50)

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- Game State ---
        self.grid = None
        self.target_grid = None
        self.cursor_pos = None
        self.selected_pixel_color = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.last_space_held = False
        
        # --- Rendering Layout ---
        self.CELL_SIZE = 32
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = self.HEIGHT - self.GRID_HEIGHT - 20

        self.TARGET_CELL_SIZE = 12
        self.TARGET_GRID_WIDTH = self.GRID_SIZE * self.TARGET_CELL_SIZE
        self.TARGET_GRID_X = self.GRID_X
        self.TARGET_GRID_Y = 30

        # Create the static target pattern once
        self._create_target_pattern()
        
        # Initialize state variables
        self.reset()
    
    def _create_target_pattern(self):
        """Creates a fixed, static pattern to be the puzzle's goal."""
        self.target_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.target_grid[2, 2] = 1
        self.target_grid[2, 5] = 1
        self.target_grid[3, 3] = 2
        self.target_grid[3, 4] = 2
        self.target_grid[4, 2] = 3
        self.target_grid[4, 5] = 3
        self.target_grid[5, 3] = 4
        self.target_grid[5, 4] = 4

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Scramble the target grid to create the starting puzzle
        self.grid = self.target_grid.copy()
        pixel_coords = list(zip(*np.where(self.grid > 0)))

        if len(pixel_coords) >= 2:
            for _ in range(self.NUM_SWAPS):
                idx1, idx2 = self.np_random.choice(len(pixel_coords), 2, replace=False)
                coord1 = tuple(pixel_coords[idx1])
                coord2 = tuple(pixel_coords[idx2])
                self.grid[coord1], self.grid[coord2] = self.grid[coord2], self.grid[coord1]

        # Initialize game state
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_pixel_color = None
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = True # Prevent action on first frame after reset

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = 0
        grid_before_move = self.grid.copy()
        
        # --- Handle Input and Update Game Logic ---
        
        # Cursor Movement (only if a direction is pressed)
        if movement != 0:
            if shift_held: # Snap to next pixel in a direction
                self._snap_cursor(movement)
            else: # Normal 1-cell movement
                if movement == 1: self.cursor_pos[0] -= 1
                elif movement == 2: self.cursor_pos[0] += 1
                elif movement == 3: self.cursor_pos[1] -= 1
                elif movement == 4: self.cursor_pos[1] += 1
            
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)

        # Spacebar action (pick up / place) - on press (rising edge)
        space_press = space_held and not self.last_space_held
        if space_press:
            r, c = self.cursor_pos
            if self.selected_pixel_color is None:
                if self.grid[r, c] > 0:
                    # sfx: pickup_pixel
                    self.selected_pixel_color = self.grid[r, c]
                    self.grid[r, c] = 0
            else:
                if self.grid[r, c] == 0:
                    # sfx: place_pixel_success
                    self.grid[r, c] = self.selected_pixel_color
                    self.selected_pixel_color = None
                    
                    self.moves_left -= 1
                    reward -= 0.1 # Cost for making a move
                    
                    # Calculate placement reward based on change in correctness
                    correct_before = np.sum(grid_before_move == self.target_grid)
                    correct_after = np.sum(self.grid == self.target_grid)
                    reward += (correct_after - correct_before) * 5.0
                else:
                    # sfx: place_pixel_fail
                    pass

        self.last_space_held = space_held
        self.steps += 1
        self.score += reward
        
        # --- Check for Termination ---
        terminated = False
        win = np.array_equal(self.grid, self.target_grid)
        loss = self.moves_left <= 0 and not win

        if win:
            reward += 50.0
            self.score += 50.0
            terminated = True
            # sfx: puzzle_win
        elif loss:
            reward -= 50.0
            self.score -= 50.0
            terminated = True
            # sfx: puzzle_loss

        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _snap_cursor(self, direction):
        """Moves the cursor to the next available pixel in the given direction."""
        r, c = self.cursor_pos
        pixels = list(zip(*np.where(self.grid > 0)))
        if not pixels: return

        if direction == 1: # Up
            candidates = sorted([p for p in pixels if p[1] == c and p[0] < r], key=lambda p: p[0], reverse=True)
        elif direction == 2: # Down
            candidates = sorted([p for p in pixels if p[1] == c and p[0] > r], key=lambda p: p[0])
        elif direction == 3: # Left
            candidates = sorted([p for p in pixels if p[0] == r and p[1] < c], key=lambda p: p[1], reverse=True)
        elif direction == 4: # Right
            candidates = sorted([p for p in pixels if p[0] == r and p[1] > c], key=lambda p: p[1])
        else:
            return

        if candidates:
            self.cursor_pos = list(candidates[0])

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
            "moves_left": self.moves_left,
            "correct_pixels": int(np.sum(self.grid == self.target_grid))
        }

    def _render_game(self):
        # Draw Main Grid
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                color_val = self.grid[r, c]
                pixel_color = self.PIXEL_COLORS.get(color_val, self.COLOR_EMPTY)
                pygame.draw.rect(self.screen, pixel_color, cell_rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, cell_rect, 1)

        # Draw Cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_X + cursor_c * self.CELL_SIZE, self.GRID_Y + cursor_r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        cursor_color = self.COLOR_CURSOR_SELECT if self.selected_pixel_color else self.COLOR_CURSOR
        pygame.draw.rect(self.screen, cursor_color, cursor_rect, 3, border_radius=2)

        # Draw Selected Pixel following the cursor
        if self.selected_pixel_color:
            color = self.PIXEL_COLORS[self.selected_pixel_color]
            selected_rect = pygame.Rect(0, 0, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
            selected_rect.center = cursor_rect.center
            pygame.draw.rect(self.screen, color, selected_rect, border_radius=4)
            pygame.draw.rect(self.screen, tuple(min(255, c+60) for c in color), selected_rect, 2, border_radius=4)

    def _render_ui(self):
        # Draw Target Pattern
        target_label = self.font_small.render("Target Pattern", True, self.COLOR_TEXT)
        self.screen.blit(target_label, (self.TARGET_GRID_X, self.TARGET_GRID_Y - 20))
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(self.TARGET_GRID_X + c * self.TARGET_CELL_SIZE, self.TARGET_GRID_Y + r * self.TARGET_CELL_SIZE, self.TARGET_CELL_SIZE, self.TARGET_CELL_SIZE)
                color_val = self.target_grid[r, c]
                pixel_color = self.PIXEL_COLORS.get(color_val, self.COLOR_EMPTY)
                pygame.draw.rect(self.screen, pixel_color, cell_rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, cell_rect, 1)
        
        # Draw Moves Left
        moves_text = f"Moves: {self.moves_left}"
        moves_surf = self.font_main.render(moves_text, True, self.COLOR_TEXT)
        moves_rect = moves_surf.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(moves_surf, moves_rect)
        
        # Draw Score
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 20, 60))
        self.screen.blit(score_surf, score_rect)

        # Draw Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win = np.array_equal(self.grid, self.target_grid)
            msg = "PUZZLE SOLVED!" if win else "OUT OF MOVES"
            msg_color = self.COLOR_CURSOR_SELECT if win else self.PIXEL_COLORS[1]
            
            msg_surf = self.font_main.render(msg, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
            self.screen.blit(msg_surf, msg_rect)
            
            reset_surf = self.font_small.render("Call reset() to play again", True, self.COLOR_TEXT)
            reset_rect = reset_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 20))
            self.screen.blit(reset_surf, reset_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    
    pygame.display.set_caption("Pixel Puzzle Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        # In a real agent, you'd get the action from your model
        # For human play, we poll pygame events
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()

        # Map keyboard to MultiDiscrete action
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {terminated}")

        if terminated:
            print(f"--- GAME OVER (Press 'R' to reset) --- Final Score: {info['score']:.2f}")

        # Render the current state to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Control the frame rate for human play

    env.close()