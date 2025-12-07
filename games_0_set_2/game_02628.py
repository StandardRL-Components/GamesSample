
# Generated: 2025-08-27T20:56:24.091148
# Source Brief: brief_02628.md
# Brief Index: 2628

        
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

    user_guide = (
        "Controls: ←→ to move, ↑ to rotate clockwise, SPACE to rotate counter-clockwise. "
        "↓ for soft drop, SHIFT for hard drop."
    )

    game_description = (
        "A fast-paced, falling block puzzle. Strategically place pieces to clear lines and maximize your score. "
        "The game speeds up as you clear more lines. Win by clearing 10 lines."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.BLOCK_SIZE = 20
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2

        self.MAX_STEPS = 10000
        self.WIN_CONDITION_LINES = 10

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_FLASH = (255, 255, 255)
        self.PIECE_COLORS = [
            (0, 0, 0),  # 0: Empty
            (0, 240, 240),  # 1: I (Cyan)
            (0, 0, 240),  # 2: J (Blue)
            (240, 160, 0),  # 3: L (Orange)
            (240, 240, 0),  # 4: O (Yellow)
            (0, 240, 0),  # 5: S (Green)
            (160, 0, 240),  # 6: T (Purple)
            (240, 0, 0),  # 7: Z (Red)
        ]

        # --- Tetromino Shapes ---
        # 4 rotations for each piece type
        self.SHAPES = [
            [], # 0: Empty
            [[1, 5, 9, 13], [4, 5, 6, 7]],  # I
            [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],  # J
            [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 4, 5, 7]],  # L
            [[1, 2, 5, 6]],  # O
            [[5, 6, 8, 9], [1, 5, 6, 10]],  # S
            [[1, 4, 5, 6], [1, 5, 6, 9], [4, 5, 6, 9], [1, 4, 5, 9]],  # T
            [[4, 5, 9, 10], [2, 6, 5, 9]],  # Z
        ]

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
        self.font_small = pygame.font.Font(None, 28)

        # --- Game State Initialization ---
        self.grid = None
        self.current_piece = None
        self.next_piece_type = None
        self.score = 0
        self.steps = 0
        self.lines_cleared_total = 0
        self.fall_speed = 0
        self.fall_counter = 0
        self.game_over = False
        self.last_action = np.array([0, 0, 0])
        self.line_clear_animation = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.steps = 0
        self.lines_cleared_total = 0
        self.fall_speed = 0.5
        self.fall_counter = 0
        self.game_over = False
        self.last_action = np.array([0, 0, 0])
        self.line_clear_animation = []

        self.next_piece_type = self.np_random.integers(1, len(self.SHAPES))
        self._spawn_piece()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- Action Processing ---
        movement, space_button, shift_button = action[0], action[1], action[2]
        
        # Actions that trigger on press (state change from 0 to 1)
        up_pressed = movement == 1 and self.last_action[0] != 1
        space_pressed = space_button == 1 and self.last_action[1] != 1
        
        # Actions that are continuous or event-based
        is_left = movement == 3
        is_right = movement == 4
        is_down = movement == 2
        is_shift = shift_button == 1

        # --- Handle player input ---
        if up_pressed: # Rotate Clockwise
            self._rotate(1)
        if space_pressed: # Rotate Counter-Clockwise
            self._rotate(-1)
        if is_left:
            self._move(-1)
        if is_right:
            self._move(1)

        # --- Hard Drop ---
        if is_shift:
            # Find landing position
            while self._is_valid_position(self.current_piece, offset_y=1):
                self.current_piece["y"] += 1
                reward += 0.1 # Small reward for dropping
            
            hole_penalty = self._lock_piece()
            lines_cleared = self._clear_lines()
            reward += self._calculate_line_clear_reward(lines_cleared) - hole_penalty
            self._spawn_piece()
        else:
            # --- Gravity & Soft Drop ---
            soft_drop_active = is_down
            self.fall_counter += self.fall_speed + (4.5 if soft_drop_active else 0)
            if self.fall_counter >= 5:
                self.fall_counter = 0
                if self._is_valid_position(self.current_piece, offset_y=1):
                    self.current_piece["y"] += 1
                    if soft_drop_active:
                        reward += 0.1 # Reward for actively dropping
                else:
                    # Lock piece
                    hole_penalty = self._lock_piece()
                    lines_cleared = self._clear_lines()
                    reward += self._calculate_line_clear_reward(lines_cleared) - hole_penalty
                    self._spawn_piece()

        # --- Update game state ---
        self.last_action = action
        terminated = self.game_over or self.steps >= self.MAX_STEPS or self.lines_cleared_total >= self.WIN_CONDITION_LINES

        if self.game_over:
            reward -= 100
        elif self.lines_cleared_total >= self.WIN_CONDITION_LINES:
            reward += 100

        return self._get_observation(), reward, terminated, False, self._get_info()

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
            "lines_cleared": self.lines_cleared_total,
        }

    def _spawn_piece(self):
        self.current_piece = {
            "type": self.next_piece_type,
            "rotation": 0,
            "x": self.GRID_WIDTH // 2 - 2,
            "y": 0,
        }
        self.next_piece_type = self.np_random.integers(1, len(self.SHAPES))
        if not self._is_valid_position(self.current_piece):
            self.game_over = True

    def _get_piece_coords(self, piece):
        coords = []
        shape_pattern = self.SHAPES[piece["type"]][piece["rotation"]]
        for i in range(16):
            if i in shape_pattern:
                coords.append((
                    piece["x"] + (i % 4),
                    piece["y"] + (i // 4)
                ))
        return coords

    def _is_valid_position(self, piece, offset_x=0, offset_y=0):
        test_piece = piece.copy()
        test_piece["x"] += offset_x
        test_piece["y"] += offset_y
        coords = self._get_piece_coords(test_piece)
        for x, y in coords:
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return False  # Out of bounds
            if y >= 0 and self.grid[y, x] > 0:
                return False  # Collision with existing block
        return True

    def _rotate(self, direction):
        if self.current_piece:
            original_rotation = self.current_piece["rotation"]
            num_rotations = len(self.SHAPES[self.current_piece["type"]])
            
            test_piece = self.current_piece.copy()
            test_piece["rotation"] = (original_rotation + direction) % num_rotations

            # Wall kick logic (simplified)
            for offset in [0, 1, -1, 2, -2]:
                if self._is_valid_position(test_piece, offset_x=offset):
                    self.current_piece["rotation"] = test_piece["rotation"]
                    self.current_piece["x"] += offset
                    # sfx: rotate
                    return
            # sfx: rotate_fail

    def _move(self, direction):
        if self.current_piece and self._is_valid_position(self.current_piece, offset_x=direction):
            self.current_piece["x"] += direction
            # sfx: move

    def _lock_piece(self):
        coords = self._get_piece_coords(self.current_piece)
        max_y = 0
        for x, y in coords:
            if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                self.grid[y, x] = self.current_piece["type"]
                max_y = max(max_y, y)
        
        # Calculate hole penalty
        hole_penalty = 0
        for x, y in coords:
            # Check for empty cells directly below the newly placed block parts
            for row in range(y + 1, self.GRID_HEIGHT):
                if self.grid[row, x] == 0:
                    hole_penalty += 0.2 # Each empty cell below is a hole
                else:
                    break # Stop if we hit another block in the same column
        
        self.current_piece = None
        # sfx: lock
        return hole_penalty

    def _clear_lines(self):
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] > 0):
                lines_to_clear.append(r)

        if lines_to_clear:
            # sfx: line_clear
            self.line_clear_animation = [(r, 5) for r in lines_to_clear] # row, timer
            
            for r in lines_to_clear:
                self.grid[1:r+1, :] = self.grid[0:r, :]
                self.grid[0, :] = 0
            
            self.lines_cleared_total += len(lines_to_clear)
            if self.lines_cleared_total % 5 == 0 and self.lines_cleared_total > 0:
                self.fall_speed = min(5.0, self.fall_speed + 0.1)

        return len(lines_to_clear)

    def _calculate_line_clear_reward(self, lines_cleared):
        if lines_cleared == 1:
            self.score += 100
            return 1
        elif lines_cleared == 2:
            self.score += 300
            return 2
        elif lines_cleared == 3:
            self.score += 500
            return 3
        elif lines_cleared == 4:
            self.score += 800
            return 4
        return 0

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw placed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] > 0:
                    self._draw_block(c, r, self.grid[r, c])
        
        # Draw ghost piece
        if self.current_piece and not self.game_over:
            ghost_piece = self.current_piece.copy()
            while self._is_valid_position(ghost_piece, offset_y=1):
                ghost_piece["y"] += 1
            coords = self._get_piece_coords(ghost_piece)
            for x, y in coords:
                self._draw_block(x, y, self.current_piece["type"], is_ghost=True)

        # Draw current piece
        if self.current_piece and not self.game_over:
            coords = self._get_piece_coords(self.current_piece)
            for x, y in coords:
                self._draw_block(x, y, self.current_piece["type"])

        # Draw line clear animation
        if self.line_clear_animation:
            new_animation_list = []
            for r, timer in self.line_clear_animation:
                flash_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET + r * self.BLOCK_SIZE, self.GRID_WIDTH * self.BLOCK_SIZE, self.BLOCK_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_FLASH, flash_rect)
                if timer > 0:
                    new_animation_list.append((r, timer - 1))
            self.line_clear_animation = new_animation_list

        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            y = self.GRID_Y_OFFSET + r * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH * self.BLOCK_SIZE, y))
        for c in range(self.GRID_WIDTH + 1):
            x = self.GRID_X_OFFSET + c * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.BLOCK_SIZE))

    def _draw_block(self, grid_x, grid_y, piece_type, is_ghost=False):
        if grid_y < 0: return
        
        px, py = self.GRID_X_OFFSET + grid_x * self.BLOCK_SIZE, self.GRID_Y_OFFSET + grid_y * self.BLOCK_SIZE
        color = self.PIECE_COLORS[piece_type]
        
        rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2, border_radius=3)
        else:
            light_color = tuple(min(255, c + 50) for c in color)
            dark_color = tuple(max(0, c - 50) for c in color)
            
            pygame.draw.rect(self.screen, light_color, rect.move(-1, -1), border_radius=4)
            pygame.draw.rect(self.screen, dark_color, rect.move(1, 1), border_radius=4)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.rect(self.screen, (0,0,0,100), rect, 1, border_radius=3)


    def _render_ui(self):
        # --- Right Panel ---
        right_panel_x = self.GRID_X_OFFSET + self.GRID_WIDTH * self.BLOCK_SIZE + 50
        
        # Score
        score_text = self.font_main.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (right_panel_x, 50))
        score_val = self.font_main.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_val, (right_panel_x, 80))

        # Lines
        lines_text = self.font_main.render("LINES", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (right_panel_x, 140))
        lines_val = self.font_main.render(f"{self.lines_cleared_total}/{self.WIN_CONDITION_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(lines_val, (right_panel_x, 170))
        
        # Next Piece
        next_text = self.font_main.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (right_panel_x, 230))
        
        next_piece_preview = {
            "type": self.next_piece_type,
            "rotation": 0, "x": 0, "y": 0
        }
        coords = self._get_piece_coords(next_piece_preview)
        for x, y in coords:
            px = right_panel_x + 10 + x * self.BLOCK_SIZE
            py = 270 + y * self.BLOCK_SIZE
            rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
            color = self.PIECE_COLORS[self.next_piece_type]
            pygame.draw.rect(self.screen, color, rect, border_radius=3)

        # --- Game Over / Win Message ---
        if self.game_over:
            self._draw_overlay_message("GAME OVER")
        elif self.lines_cleared_total >= self.WIN_CONDITION_LINES:
            self._draw_overlay_message("YOU WIN!")

    def _draw_overlay_message(self, message):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text = self.font_main.render(message, True, self.COLOR_FLASH)
        text_rect = text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(text, text_rect)

    def validate_implementation(self):
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

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # Use Pygame for human interaction
    pygame.display.set_caption("Falling Block Puzzle")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    running = True
    total_reward = 0
    
    action = env.action_space.sample()
    action.fill(0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if not terminated:
            # --- Map keyboard to action space ---
            keys = pygame.key.get_pressed()
            
            # Movement
            action[0] = 0 # No-op
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            # Space button
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            
            # Shift button
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Match the auto_advance rate

    pygame.quit()