
# Generated: 2025-08-28T02:24:28.864882
# Source Brief: brief_01691.md
# Brief Index: 1691

        
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
    """
    A minimalist puzzle game where the player places geometric shapes to fill a 10x10 grid.
    The game is turn-based, with actions for moving, rotating, and placing the current shape.
    The goal is to achieve the highest score by filling the grid and clearing lines.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑↓ to rotate. Press space to place the shape."
    )

    game_description = (
        "Minimalist puzzle game. Place falling geometric shapes to fill the 10x10 grid."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    CELL_SIZE = 32
    GRID_WIDTH = GRID_SIZE * CELL_SIZE
    GRID_HEIGHT = GRID_SIZE * CELL_SIZE
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH) // 2 - 120
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    MAX_STEPS = 1000
    TOTAL_SHAPES = 25 # 100 cells / 4 cells per shape

    # --- Colors ---
    COLOR_BG = (25, 25, 40)
    COLOR_GRID = (50, 50, 70)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 20)

    # --- Shapes ---
    # Shape data: color, and a list of 4 rotations.
    # Each rotation is a list of (row, col) offsets from a pivot.
    SHAPES = [
        # I shape (ID 1)
        ( (66, 214, 230), [ # Cyan
            [(1, 0), (1, 1), (1, 2), (1, 3)],
            [(0, 2), (1, 2), (2, 2), (3, 2)],
            [(2, 0), (2, 1), (2, 2), (2, 3)],
            [(0, 1), (1, 1), (2, 1), (3, 1)] ]),
        # O shape (ID 2)
        ( (240, 230, 80), [ # Yellow
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (1, 0), (1, 1)] ]),
        # T shape (ID 3)
        ( (188, 80, 240), [ # Purple
            [(0, 1), (1, 0), (1, 1), (1, 2)],
            [(0, 1), (1, 1), (2, 1), (1, 2)],
            [(1, 0), (1, 1), (1, 2), (2, 1)],
            [(0, 1), (1, 0), (1, 1), (2, 1)] ]),
        # L shape (ID 4)
        ( (240, 160, 80), [ # Orange
            [(0, 2), (1, 0), (1, 1), (1, 2)],
            [(0, 1), (1, 1), (2, 1), (2, 2)],
            [(1, 0), (1, 1), (1, 2), (2, 0)],
            [(0, 0), (0, 1), (1, 1), (2, 1)] ]),
        # J shape (ID 5)
        ( (80, 100, 240), [ # Blue
            [(0, 0), (1, 0), (1, 1), (1, 2)],
            [(0, 1), (0, 2), (1, 1), (2, 1)],
            [(1, 0), (1, 1), (1, 2), (2, 2)],
            [(0, 1), (1, 1), (2, 0), (2, 1)] ]),
        # S shape (ID 6)
        ( (80, 240, 80), [ # Green
            [(0, 1), (0, 2), (1, 0), (1, 1)],
            [(0, 1), (1, 1), (1, 2), (2, 2)],
            [(1, 1), (1, 2), (2, 0), (2, 1)],
            [(0, 0), (1, 0), (1, 1), (2, 1)] ]),
        # Z shape (ID 7)
        ( (240, 80, 80), [ # Red
            [(0, 0), (0, 1), (1, 1), (1, 2)],
            [(0, 2), (1, 1), (1, 2), (2, 1)],
            [(1, 0), (1, 1), (2, 1), (2, 2)],
            [(0, 1), (1, 0), (1, 1), (2, 0)] ])
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.shape_colors = {i + 1: color for i, (color, _) in enumerate(self.SHAPES)}
        self.shape_defs = {i + 1: rotations for i, (_, rotations) in enumerate(self.SHAPES)}
        
        self.grid = None
        self.available_shapes = []
        self.next_shapes = []
        self.current_shape_id = 0
        self.current_shape_pos = [0, 0] # row, col
        self.current_shape_rot = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None

        # self.validate_implementation() # Optional: call for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        shape_indices = list(range(1, len(self.SHAPES) + 1))
        initial_shapes = [1, 2] # I and O shapes are easier to start with
        
        all_shapes_for_episode = []
        while len(all_shapes_for_episode) < self.TOTAL_SHAPES:
            shuffled_indices = list(shape_indices)
            self.np_random.shuffle(shuffled_indices)
            all_shapes_for_episode.extend(shuffled_indices)
        
        self.available_shapes = initial_shapes + all_shapes_for_episode[:self.TOTAL_SHAPES - len(initial_shapes)]
        self.np_random.shuffle(self.available_shapes)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.next_shapes = [self.available_shapes.pop(0) for _ in range(min(3, len(self.available_shapes)))]
        self._spawn_new_shape()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0
        terminated = False
        
        if space_held:
            # Placement action is prioritized
            drop_pos = self._find_drop_position()
            self._place_shape(drop_pos)
            reward += 4 # +1 for each of the 4 cells filled
            self.score += 4
            
            cleared_lines = self._check_and_clear_lines()
            if cleared_lines > 0:
                line_reward = 5 * cleared_lines + (cleared_lines ** 2) * 5 # Bonus for multi-line clears
                reward += line_reward
                self.score += line_reward
            
            if np.all(self.grid != 0):
                # Win condition: grid is completely full
                reward += 100
                self.score += 100
                terminated = True
            else:
                if not self._spawn_new_shape():
                    terminated = True # Loss: cannot spawn new shape (no space or no shapes left)
        else:
            # Movement/rotation action
            if movement == 1: # Rotate CW (mapped from 'up')
                self._rotate_shape(1)
            elif movement == 2: # Rotate CCW (mapped from 'down')
                self._rotate_shape(-1)
            elif movement == 3: # Move Left
                self._move_shape(-1)
            elif movement == 4: # Move Right
                self._move_shape(1)
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_placed_shapes()
        if not self.game_over:
            self._render_ghost_shape()
            self._render_current_shape()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
        
    def _spawn_new_shape(self):
        if not self.next_shapes:
            return False # No more shapes left
        
        self.current_shape_id = self.next_shapes.pop(0)
        if self.available_shapes:
            self.next_shapes.append(self.available_shapes.pop(0))
        
        self.current_shape_rot = 0
        shape_coords = self.shape_defs[self.current_shape_id][self.current_shape_rot]
        min_col = min(c for r, c in shape_coords)
        max_col = max(c for r, c in shape_coords)
        shape_width = max_col - min_col + 1
        start_col = (self.GRID_SIZE - shape_width) // 2 - min_col
        self.current_shape_pos = [0, start_col]
        
        if not self._is_valid_position(self.current_shape_id, self.current_shape_rot, self.current_shape_pos):
            return False # Loss: grid is full at the top
        return True

    def _is_valid_position(self, shape_id, rotation, pos):
        shape_coords = self.shape_defs[shape_id][rotation]
        for r_off, c_off in shape_coords:
            r, c = pos[0] + r_off, pos[1] + c_off
            if not (0 <= r < self.GRID_SIZE and 0 <= c < self.GRID_SIZE):
                return False # Out of bounds
            if self.grid[r, c] != 0:
                return False # Collision with existing block
        return True

    def _rotate_shape(self, direction):
        new_rot = (self.current_shape_rot + direction) % 4
        if self._is_valid_position(self.current_shape_id, new_rot, self.current_shape_pos):
            self.current_shape_rot = new_rot
            return True
        # Wall kick attempt for better game feel
        for offset in [-1, 1, -2, 2]:
            new_pos = [self.current_shape_pos[0], self.current_shape_pos[1] + offset]
            if self._is_valid_position(self.current_shape_id, new_rot, new_pos):
                self.current_shape_rot = new_rot
                self.current_shape_pos = new_pos
                return True
        return False

    def _move_shape(self, direction):
        new_pos = [self.current_shape_pos[0], self.current_shape_pos[1] + direction]
        if self._is_valid_position(self.current_shape_id, self.current_shape_rot, new_pos):
            self.current_shape_pos = new_pos
            return True
        return False

    def _find_drop_position(self):
        pos = list(self.current_shape_pos)
        while self._is_valid_position(self.current_shape_id, self.current_shape_rot, [pos[0] + 1, pos[1]]):
            pos[0] += 1
        return pos

    def _place_shape(self, pos):
        shape_coords = self.shape_defs[self.current_shape_id][self.current_shape_rot]
        for r_off, c_off in shape_coords:
            r, c = pos[0] + r_off, pos[1] + c_off
            if 0 <= r < self.GRID_SIZE and 0 <= c < self.GRID_SIZE:
                self.grid[r, c] = self.current_shape_id

    def _check_and_clear_lines(self):
        full_rows = [r for r in range(self.GRID_SIZE) if np.all(self.grid[r, :] != 0)]
        full_cols = [c for c in range(self.GRID_SIZE) if np.all(self.grid[:, c] != 0)]
        
        if not full_rows and not full_cols:
            return 0
            
        cleared_mask = np.zeros_like(self.grid, dtype=bool)
        for r in full_rows:
            cleared_mask[r, :] = True
        for c in full_cols:
            cleared_mask[:, c] = True
            
        new_grid = np.zeros_like(self.grid)
        for c in range(self.GRID_SIZE):
            write_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if not cleared_mask[r, c]:
                    if write_row >= 0:
                        new_grid[write_row, c] = self.grid[r, c]
                        write_row -= 1
        self.grid = new_grid
        
        return len(full_rows) + len(full_cols)

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end_pos = (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
            # Horizontal lines
            start_pos = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + i * self.CELL_SIZE)
            end_pos = (self.GRID_X_OFFSET + self.GRID_WIDTH, self.GRID_Y_OFFSET + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

    def _render_block(self, r, c, color, alpha=255):
        x = self.GRID_X_OFFSET + c * self.CELL_SIZE
        y = self.GRID_Y_OFFSET + r * self.CELL_SIZE
        
        border_color = tuple(max(0, val - 40) for val in color)
        inner_color = color

        if alpha < 255:
            surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(surface, (*border_color, alpha), (0, 0, self.CELL_SIZE, self.CELL_SIZE), border_radius=4)
            pygame.draw.rect(surface, (*inner_color, alpha), (2, 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4), border_radius=3)
            self.screen.blit(surface, (x, y))
        else:
            rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, border_color, rect, border_radius=4)
            pygame.draw.rect(self.screen, inner_color, rect.inflate(-4, -4), border_radius=3)

    def _render_placed_shapes(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                shape_id = self.grid[r, c]
                if shape_id != 0:
                    color = self.shape_colors[shape_id]
                    # Desaturate color for placed blocks for visual distinction
                    gray = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
                    desat_color = (
                        int(color[0] * 0.4 + gray * 0.6),
                        int(color[1] * 0.4 + gray * 0.6),
                        int(color[2] * 0.4 + gray * 0.6)
                    )
                    self._render_block(r, c, desat_color)

    def _render_shape_at(self, shape_id, rotation, pos, alpha):
        shape_coords = self.shape_defs[shape_id][rotation]
        color = self.shape_colors[shape_id]
        for r_off, c_off in shape_coords:
            self._render_block(pos[0] + r_off, pos[1] + c_off, color, alpha)

    def _render_ghost_shape(self):
        ghost_pos = self._find_drop_position()
        self._render_shape_at(self.current_shape_id, self.current_shape_rot, ghost_pos, 60)

    def _render_current_shape(self):
        self._render_shape_at(self.current_shape_id, self.current_shape_rot, self.current_shape_pos, 255)

    def _render_ui(self):
        # --- Score Display ---
        score_text = f"{self.score:06d}"
        self._draw_text(score_text, self.font_large, (40, 40))

        # --- Next Shapes Display ---
        next_ui_x = self.GRID_X_OFFSET + self.GRID_WIDTH + 40
        next_ui_y = self.GRID_Y_OFFSET
        self._draw_text("NEXT", self.font_medium, (next_ui_x, next_ui_y))
        
        for i, shape_id in enumerate(self.next_shapes):
            color = self.shape_colors[shape_id]
            shape_coords = self.shape_defs[shape_id][0] # Default rotation for preview
            
            min_r, max_r = min(r for r,c in shape_coords), max(r for r,c in shape_coords)
            min_c, max_c = min(c for r,c in shape_coords), max(c for r,c in shape_coords)
            shape_h, shape_w = (max_r - min_r + 1), (max_c - min_c + 1)
            
            base_y, base_x, p_cell_size = next_ui_y + 50 + i * 100, next_ui_x, 20
            
            for r_off, c_off in shape_coords:
                x = base_x + (c_off - min_c) * p_cell_size + (4 - shape_w) * p_cell_size / 2
                y = base_y + (r_off - min_r) * p_cell_size + (4 - shape_h) * p_cell_size / 2
                rect = pygame.Rect(x, y, p_cell_size, p_cell_size)
                border_color = tuple(max(0, val - 40) for val in color)
                pygame.draw.rect(self.screen, border_color, rect, border_radius=3)
                pygame.draw.rect(self.screen, color, rect.inflate(-3, -3), border_radius=2)
                
        # --- Game Over Display ---
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            win = np.all(self.grid != 0)
            message = "GRID COMPLETE!" if win else "GAME OVER"
            self._draw_text(message, self.font_large, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20), center=True)
            self._draw_text(f"Final Score: {self.score}", self.font_medium, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30), center=True)

    def _draw_text(self, text, font, pos, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Geometric Grid Filler")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        action_taken = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                elif event.key == pygame.K_ESCAPE:
                    running = False
                
                # Map key presses to actions for a single step
                if not done:
                    if event.key == pygame.K_UP: action[0], action_taken = 1, True
                    elif event.key == pygame.K_DOWN: action[0], action_taken = 2, True
                    elif event.key == pygame.K_LEFT: action[0], action_taken = 3, True
                    elif event.key == pygame.K_RIGHT: action[0], action_taken = 4, True
                    elif event.key == pygame.K_SPACE: action[1], action_taken = 1, True

        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

        # Render the current game state from the observation
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()