
# Generated: 2025-08-28T06:49:12.733689
# Source Brief: brief_03047.md
# Brief Index: 3047

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move the selected pixel. "
        "Space cycles to the next pixel, Shift cycles to the previous."
    )

    game_description = (
        "A minimalist puzzle game. Rearrange the colored pixels on the grid "
        "to match the faint target image in the background before you run out of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.W, self.H = 640, 400
        self.observation_space = Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()

        # --- Colors & Fonts ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (50, 60, 80)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_SELECT = (255, 255, 255)
        self.COLOR_WIN = (180, 255, 180)
        self.COLOR_LOSE = (255, 180, 180)
        
        try:
            self.font_main = pygame.font.Font(None, 32)
            self.font_big = pygame.font.Font(None, 72)
        except FileNotFoundError:
            # Fallback if default font is not found in some headless environments
            self.font_main = pygame.font.SysFont("sans", 32)
            self.font_big = pygame.font.SysFont("sans", 72)

        # --- Game State (initialized in reset) ---
        self.level = None
        self.grid_size = None
        self.num_colors = None
        self.palette = None
        self.target_grid = None
        self.movable_pixels = None
        self.selected_pixel_index = None
        self.moves_left = None
        self.max_moves = None
        self.completed_rows = None
        self.completed_cols = None
        self.steps = None
        self.score = None
        self.game_over = None
        
        # --- Grid Rendering Geometry ---
        self.grid_rect = None
        self.cell_size = None
        self.grid_offset_x = None
        self.grid_offset_y = None

        self.reset()
        
        # This will be called by the user, but we can call it here for development
        # self.validate_implementation()

    def _generate_palette(self, num_colors):
        """Generates a list of visually distinct, bright colors."""
        self.palette = []
        for i in range(num_colors):
            hue = int(i * (360 / num_colors))
            color = pygame.Color(0)
            color.hsla = (hue, 100, 55, 100)
            self.palette.append(tuple(color)[:3])

    def _setup_level(self):
        """Initializes the game state for the current level."""
        # Difficulty scaling
        self.grid_size = 5 + (self.level - 1) * 2
        self.num_colors = 5 + (self.level - 1) * 2
        self.max_moves = self.grid_size * self.grid_size + 10 * self.level

        self._generate_palette(self.num_colors)
        self._generate_puzzle()

        self.completed_rows = set()
        self.completed_cols = set()
        self.moves_left = self.max_moves
        self.selected_pixel_index = 0
        
        # Calculate grid rendering geometry
        grid_render_size = min(self.W * 0.7, self.H * 0.8)
        self.cell_size = grid_render_size / self.grid_size
        actual_grid_size = self.cell_size * self.grid_size
        self.grid_offset_x = (self.W - actual_grid_size) / 2
        self.grid_offset_y = (self.H - actual_grid_size) / 2
        self.grid_rect = pygame.Rect(self.grid_offset_x, self.grid_offset_y, actual_grid_size, actual_grid_size)

    def _generate_puzzle(self):
        """Creates the target and scrambled pixel layouts."""
        num_pixels = self.grid_size * self.grid_size // 2  # Fill about half the grid
        
        # Create a list of all possible grid coordinates
        all_coords = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        self.np_random.shuffle(all_coords)

        # Create target grid (the solution)
        self.target_grid = np.full((self.grid_size, self.grid_size), -1, dtype=int)
        target_pixels = []
        for i in range(num_pixels):
            gx, gy = all_coords[i]
            color_idx = i % self.num_colors
            self.target_grid[gy, gx] = color_idx
            target_pixels.append({"pos": (gx, gy), "color_idx": color_idx})
        
        # Create scrambled movable pixels
        self.np_random.shuffle(all_coords)
        self.movable_pixels = []
        for i in range(num_pixels):
            gx, gy = all_coords[i]
            # Use the same set of colors as the target
            color_idx = target_pixels[i]["color_idx"]
            self.movable_pixels.append({"pos": (gx, gy), "color_idx": color_idx})
        
        # Sort pixels for consistent selection order
        self.movable_pixels.sort(key=lambda p: (p["pos"][1], p["pos"][0]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self._setup_level()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # 1. Handle pixel selection change
        if space_pressed and not shift_pressed:
            self.selected_pixel_index = (self.selected_pixel_index + 1) % len(self.movable_pixels)
        elif shift_pressed and not space_pressed:
            self.selected_pixel_index = (self.selected_pixel_index - 1 + len(self.movable_pixels)) % len(self.movable_pixels)
        
        # 2. Handle pixel movement
        moved = False
        if movement != 0:
            pixel_to_move = self.movable_pixels[self.selected_pixel_index]
            old_pos = pixel_to_move["pos"]
            new_pos = list(old_pos)
            
            if movement == 1: new_pos[1] -= 1  # Up
            elif movement == 2: new_pos[1] += 1  # Down
            elif movement == 3: new_pos[0] -= 1  # Left
            elif movement == 4: new_pos[0] += 1  # Right
            
            new_pos = tuple(new_pos)

            # Check if move is valid
            if (0 <= new_pos[0] < self.grid_size and
                0 <= new_pos[1] < self.grid_size and
                not self._is_position_occupied(new_pos, exclude_index=self.selected_pixel_index)):
                
                # Apply move
                pixel_to_move["pos"] = new_pos
                self.moves_left -= 1
                moved = True
                # SFX: move_pixel.wav
                
                # 3. Calculate rewards for the move
                reward += self._calculate_move_reward(self.selected_pixel_index, old_pos)

        # 4. Check for level completion
        if self._check_level_win():
            reward += 100
            self.score += reward
            self.level += 1
            self._setup_level()
            # SFX: level_complete.wav
            # The episode continues to the next level
            return self._get_observation(), reward, False, False, self._get_info()

        # 5. Update state and check for episode termination
        self.steps += 1
        self.score += reward
        terminated = False

        if self.moves_left <= 0:
            reward -= 100  # Final penalty for running out of moves
            self.score -= 100 # Adjust score directly to reflect penalty
            terminated = True
            self.game_over = True
            # SFX: game_over.wav
        
        if self.steps >= 1000:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_position_occupied(self, pos, exclude_index=-1):
        """Checks if a grid cell is occupied by another pixel."""
        for i, p in enumerate(self.movable_pixels):
            if i != exclude_index and p["pos"] == pos:
                return True
        return False

    def _calculate_move_reward(self, moved_pixel_idx, old_pos):
        """Calculates rewards for placing pixels correctly and completing rows/cols."""
        reward = 0
        pixel = self.movable_pixels[moved_pixel_idx]
        new_pos = pixel["pos"]
        color_idx = pixel["color_idx"]

        # Reward for moving a pixel to its correct spot
        if self.target_grid[new_pos[1], new_pos[0]] == color_idx:
            reward += 0.1
        
        # Penalty for moving a pixel from its correct spot (optional, but good for learning)
        if self.target_grid[old_pos[1], old_pos[0]] == color_idx:
            reward -= 0.1

        # Check for new row/column completion
        # Row
        if new_pos[1] not in self.completed_rows and self._is_row_complete(new_pos[1]):
            reward += 5
            self.completed_rows.add(new_pos[1])
            # SFX: row_complete.wav
        # Column
        if new_pos[0] not in self.completed_cols and self._is_col_complete(new_pos[0]):
            reward += 5
            self.completed_cols.add(new_pos[0])
            # SFX: col_complete.wav
        
        return reward

    def _is_row_complete(self, row_idx):
        """Checks if all pixels in a given row are in their correct target positions."""
        for x in range(self.grid_size):
            target_color = self.target_grid[row_idx, x]
            # Find which pixel is at (x, row_idx), if any
            current_pixel_color = -1
            for p in self.movable_pixels:
                if p["pos"] == (x, row_idx):
                    current_pixel_color = p["color_idx"]
                    break
            if target_color != current_pixel_color:
                return False
        return True

    def _is_col_complete(self, col_idx):
        """Checks if all pixels in a given column are in their correct target positions."""
        for y in range(self.grid_size):
            target_color = self.target_grid[y, col_idx]
            current_pixel_color = -1
            for p in self.movable_pixels:
                if p["pos"] == (col_idx, y):
                    current_pixel_color = p["color_idx"]
                    break
            if target_color != current_pixel_color:
                return False
        return True

    def _check_level_win(self):
        """Checks if all movable pixels are in their correct target positions."""
        for p in self.movable_pixels:
            if self.target_grid[p["pos"][1], p["pos"][0]] != p["color_idx"]:
                return False
        return True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_target_ghost()
        self._render_grid()
        self._render_pixels()
        self._render_selection()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_pixel_center(self, gx, gy):
        """Converts grid coordinates to screen pixel coordinates (center of cell)."""
        px = self.grid_offset_x + (gx + 0.5) * self.cell_size
        py = self.grid_offset_y + (gy + 0.5) * self.cell_size
        return int(px), int(py)

    def _render_target_ghost(self):
        """Renders the transparent target image in the background."""
        ghost_surface = pygame.Surface((self.grid_rect.width, self.grid_rect.height), pygame.SRCALPHA)
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                color_idx = self.target_grid[gy, gx]
                if color_idx != -1:
                    color = self.palette[color_idx]
                    faded_color = (color[0], color[1], color[2], 50)
                    rect = pygame.Rect(gx * self.cell_size, gy * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(ghost_surface, faded_color, rect)
        self.screen.blit(ghost_surface, self.grid_rect.topleft)

    def _render_grid(self):
        """Draws the grid lines."""
        for i in range(self.grid_size + 1):
            # Vertical lines
            start_pos = (self.grid_rect.left + i * self.cell_size, self.grid_rect.top)
            end_pos = (self.grid_rect.left + i * self.cell_size, self.grid_rect.bottom)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
            # Horizontal lines
            start_pos = (self.grid_rect.left, self.grid_rect.top + i * self.cell_size)
            end_pos = (self.grid_rect.right, self.grid_rect.top + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

    def _render_pixels(self):
        """Draws the movable pixels."""
        pixel_size = max(1, int(self.cell_size * 0.8))
        for p in self.movable_pixels:
            gx, gy = p["pos"]
            color = self.palette[p["color_idx"]]
            px, py = self._grid_to_pixel_center(gx, gy)
            rect = pygame.Rect(px - pixel_size // 2, py - pixel_size // 2, pixel_size, pixel_size)
            pygame.draw.rect(self.screen, color, rect, border_radius=int(pixel_size * 0.2))
            
    def _render_selection(self):
        """Highlights the currently selected pixel."""
        if self.game_over or not self.movable_pixels:
            return
        selected_pixel = self.movable_pixels[self.selected_pixel_index]
        gx, gy = selected_pixel["pos"]
        rect = pygame.Rect(
            self.grid_offset_x + gx * self.cell_size,
            self.grid_offset_y + gy * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, width=2)

    def _render_ui(self):
        """Renders score, moves left, and other text information."""
        # Moves Left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.W - moves_text.get_width() - 20, 20))
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        # Level
        level_text = self.font_main.render(f"Level: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.W // 2 - level_text.get_width() // 2, 20))

        if self.game_over:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            message = "YOU RAN OUT OF MOVES"
            color = self.COLOR_LOSE
            
            text_surf = self.font_big.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.W / 2, self.H / 2))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "moves_left": self.moves_left,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        assert "score" in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        assert "steps" in info

        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    # --- Pygame Interactive Loop ---
    pygame.display.set_caption("Pixel Puzzle")
    screen = pygame.display.set_mode((env.W, env.H))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action mapping from keyboard ---
        movement = 0 # no-op
        space_pressed = 0
        shift_pressed = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_pressed = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_pressed = 1

        action = [movement, space_pressed, shift_pressed]

        # --- Event handling ---
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    action_taken = True
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    done = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if action_taken and not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

        # --- Rendering ---
        # The observation is already the rendered screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30)

    env.close()