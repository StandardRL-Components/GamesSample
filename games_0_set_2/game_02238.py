
# Generated: 2025-08-27T19:43:24.264633
# Source Brief: brief_02238.md
# Brief Index: 2238

        
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
        "Controls: Use arrow keys to move the shape cursor. Hold SHIFT to rotate the current shape. "
        "Press SPACE to place the shape. Placement automatically selects the next shape."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A geometric puzzle game. Fit the transforming shapes into the grid to fill it completely "
        "before you run out of moves. Each placement or rotation costs one move."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 10, 10
        self.UI_WIDTH = 180
        self.GAME_WIDTH = self.WIDTH - self.UI_WIDTH
        self.CELL_SIZE = min(self.GAME_WIDTH // self.GRID_COLS, self.HEIGHT // self.GRID_ROWS)
        self.GRID_WIDTH = self.CELL_SIZE * self.GRID_COLS
        self.GRID_HEIGHT = self.CELL_SIZE * self.GRID_ROWS
        self.GRID_OFFSET_X = (self.GAME_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2
        self.MAX_MOVES = 40
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_UI_BG = (40, 50, 60, 200)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_GHOST_VALID = (255, 255, 255, 128)
        self.COLOR_GHOST_INVALID = (255, 100, 100, 128)
        self.COLOR_GHOST_OUTLINE = (255, 255, 255)
        self.SHAPE_COLORS = [
            (52, 152, 219), (231, 76, 60), (46, 204, 113),
            (241, 196, 15), (155, 89, 182), (26, 188, 156), (230, 126, 34)
        ]

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 36)
        self.font_big = pygame.font.Font(None, 64)

        # --- Shape Definitions ---
        self.BASE_SHAPES = [
            {'coords': [(0, 0), (0, 1), (1, 1), (1, 0)], 'id': 1},  # Square
            {'coords': [(0, 0), (0, 1), (0, 2), (0, 3)], 'id': 2},  # I-shape
            {'coords': [(0, 0), (0, 1), (1, 1), (0, 2)], 'id': 3},  # T-shape
            {'coords': [(0, 0), (0, 1), (1, 1), (1, 2)], 'id': 4},  # S-shape
            {'coords': [(1, 0), (1, 1), (0, 1), (0, 2)], 'id': 5},  # Z-shape
            {'coords': [(0, 0), (0, 1), (0, 2), (1, 2)], 'id': 6},  # L-shape
            {'coords': [(1, 0), (1, 1), (1, 2), (0, 2)], 'id': 7},  # J-shape
        ]

        # --- State Variables ---
        self.grid = None
        self.shapes = []
        self.cursor_pos = [0, 0]
        self.current_shape_idx = 0
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.previous_space_held = False
        self.previous_shift_held = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.previous_space_held = False
        self.previous_shift_held = False
        
        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        # Select 5 unique shapes for the puzzle
        chosen_base_shapes = self.np_random.choice(self.BASE_SHAPES, size=5, replace=False)
        self.shapes = []
        for base_shape in chosen_base_shapes:
            shape_obj = {
                'base_coords': np.array(base_shape['coords']),
                'coords': np.array(base_shape['coords']),
                'id': base_shape['id'],
                'used': False
            }
            self.shapes.append(shape_obj)
        
        # Pre-fill grid with 1 or 2 shapes
        num_prefill = self.np_random.integers(1, 3)
        for _ in range(num_prefill):
            # Find an unused shape
            available_indices = [i for i, s in enumerate(self.shapes) if not s['used']]
            if not available_indices: break
            shape_to_place_idx = self.np_random.choice(available_indices)
            shape_to_place = self.shapes[shape_to_place_idx]
            
            # Try to place it randomly
            for _ in range(20): # 20 attempts
                num_rotations = self.np_random.integers(0, 4)
                rotated_coords = shape_to_place['base_coords']
                for _ in range(num_rotations):
                    rotated_coords = self._rotate_coords(rotated_coords)
                
                pos = [self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS)]
                
                if self._is_placement_valid(rotated_coords, pos):
                    self._commit_shape_to_grid(rotated_coords, pos, shape_to_place['id'])
                    shape_to_place['used'] = True
                    break
        
        self.current_shape_idx = self._find_next_available_shape_idx()

    def _find_next_available_shape_idx(self, start_idx=0):
        if all(s['used'] for s in self.shapes):
            return -1
        idx = start_idx
        for _ in range(len(self.shapes)):
            if not self.shapes[idx]['used']:
                return idx
            idx = (idx + 1) % len(self.shapes)
        return -1

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.previous_space_held
        shift_pressed = shift_held and not self.previous_shift_held
        
        step_reward = 0
        
        # Action: Move cursor (does not cost a move)
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)
        
        # Action: Rotate shape (Shift)
        if shift_pressed:
            if self.current_shape_idx != -1:
                self.moves_left -= 1
                step_reward -= 0.5 # Small cost for rotating
                current_shape = self.shapes[self.current_shape_idx]
                current_shape['coords'] = self._rotate_coords(current_shape['coords'])
                # sfx: rotate

        # Action: Place shape (Space)
        if space_pressed:
            if self.current_shape_idx != -1:
                self.moves_left -= 1
                current_shape = self.shapes[self.current_shape_idx]
                is_valid = self._is_placement_valid(current_shape['coords'], self.cursor_pos)
                
                if is_valid:
                    # sfx: place_shape
                    num_cells = len(current_shape['coords'])
                    step_reward += 5 + num_cells
                    self._commit_shape_to_grid(current_shape['coords'], self.cursor_pos, current_shape['id'])
                    current_shape['used'] = True
                    self.current_shape_idx = self._find_next_available_shape_idx()
                else:
                    # sfx: error
                    step_reward -= 2 # Penalty for invalid placement

        self.previous_space_held = space_held
        self.previous_shift_held = shift_held
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if np.all(self.grid > 0): # Win
                # sfx: win
                step_reward += 100
                self.win_message = "GRID COMPLETE!"
            else: # Loss
                # sfx: lose
                step_reward -= 100
                if self.moves_left <= 0:
                    self.win_message = "OUT OF MOVES"
                elif self.current_shape_idx == -1:
                    self.win_message = "STUCK!"
                else:
                    self.win_message = "TIME LIMIT"
        
        self.score += step_reward
        
        return self._get_observation(), step_reward, terminated, False, self._get_info()

    def _rotate_coords(self, coords):
        # 90-degree clockwise rotation matrix: [[0, 1], [-1, 0]]
        return np.dot(coords, np.array([[0, 1], [-1, 0]])).astype(int)

    def _is_placement_valid(self, shape_coords, pos):
        for x, y in shape_coords:
            grid_x, grid_y = pos[0] + x, pos[1] + y
            if not (0 <= grid_x < self.GRID_COLS and 0 <= grid_y < self.GRID_ROWS):
                return False
            if self.grid[grid_y, grid_x] != 0:
                return False
        return True

    def _commit_shape_to_grid(self, shape_coords, pos, shape_id):
        for x, y in shape_coords:
            grid_x, grid_y = pos[0] + x, pos[1] + y
            self.grid[grid_y, grid_x] = shape_id

    def _check_termination(self):
        is_full = np.all(self.grid > 0)
        no_moves = self.moves_left <= 0
        time_up = self.steps >= self.MAX_STEPS
        
        # Check for unsolvable state (no available shapes but grid not full)
        unsolvable = (self.current_shape_idx == -1 and not is_full)

        if is_full or no_moves or time_up or unsolvable:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH, y))
        for i in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT))

        # Draw filled cells
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] > 0:
                    color = self.SHAPE_COLORS[self.grid[r, c] - 1]
                    rect = pygame.Rect(
                        self.GRID_OFFSET_X + c * self.CELL_SIZE,
                        self.GRID_OFFSET_Y + r * self.CELL_SIZE,
                        self.CELL_SIZE, self.CELL_SIZE
                    )
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, tuple(int(x*0.7) for x in color), rect, 2)

        # Draw ghost shape
        if self.current_shape_idx != -1 and not self.game_over:
            current_shape = self.shapes[self.current_shape_idx]
            is_valid = self._is_placement_valid(current_shape['coords'], self.cursor_pos)
            
            for x, y in current_shape['coords']:
                grid_x, grid_y = self.cursor_pos[0] + x, self.cursor_pos[1] + y
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + grid_x * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + grid_y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                
                s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                color = self.COLOR_GHOST_VALID if is_valid else self.COLOR_GHOST_INVALID
                pygame.draw.rect(s, color, s.get_rect())
                self.screen.blit(s, rect.topleft)
                pygame.gfxdraw.rectangle(self.screen, rect, self.COLOR_GHOST_OUTLINE)

    def _render_ui(self):
        ui_panel = pygame.Surface((self.UI_WIDTH, self.HEIGHT), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        
        title_surf = self.font_title.render("Polyomino", True, self.COLOR_TEXT)
        ui_panel.blit(title_surf, (self.UI_WIDTH // 2 - title_surf.get_width() // 2, 20))
        
        moves_text = f"Moves: {self.moves_left}"
        moves_surf = self.font_main.render(moves_text, True, self.COLOR_TEXT)
        ui_panel.blit(moves_surf, (20, 70))
        
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        ui_panel.blit(score_surf, (20, 100))
        
        total_cells = self.GRID_ROWS * self.GRID_COLS
        filled_cells = np.count_nonzero(self.grid)
        completion_pct = filled_cells / total_cells if total_cells > 0 else 0
        
        bar_bg_rect = pygame.Rect(20, 140, self.UI_WIDTH - 40, 20)
        pygame.draw.rect(ui_panel, self.COLOR_GRID, bar_bg_rect, border_radius=4)
        if completion_pct > 0:
            bar_fill_rect = pygame.Rect(20, 140, max(0, (self.UI_WIDTH - 40) * completion_pct), 20)
            pygame.draw.rect(ui_panel, self.SHAPE_COLORS[0], bar_fill_rect, border_radius=4)
        
        bank_title_surf = self.font_main.render("Available Shapes:", True, self.COLOR_TEXT)
        ui_panel.blit(bank_title_surf, (20, 180))
        
        preview_y = 210
        for i, shape in enumerate(self.shapes):
            if not shape['used']:
                is_current = (i == self.current_shape_idx)
                self._render_shape_preview(ui_panel, shape, (self.UI_WIDTH // 2, preview_y), is_current)
                preview_y += 60

        self.screen.blit(ui_panel, (self.WIDTH - self.UI_WIDTH, 0))

    def _render_shape_preview(self, surface, shape, center_pos, is_current):
        preview_cell_size = 10
        coords = shape['coords']
        
        min_x = min(c[0] for c in coords)
        max_x = max(c[0] for c in coords)
        min_y = min(c[1] for c in coords)
        max_y = max(c[1] for c in coords)
        
        shape_width = (max_x - min_x + 1) * preview_cell_size
        shape_height = (max_y - min_y + 1) * preview_cell_size
        
        offset_x = center_pos[0] - shape_width // 2
        offset_y = center_pos[1] - shape_height // 2

        color = self.SHAPE_COLORS[shape['id'] - 1]
        
        for x, y in coords:
            rect = pygame.Rect(
                offset_x + (x - min_x) * preview_cell_size,
                offset_y + (y - min_y) * preview_cell_size,
                preview_cell_size, preview_cell_size
            )
            pygame.draw.rect(surface, color, rect)
            if is_current:
                pygame.draw.rect(surface, self.COLOR_GHOST_OUTLINE, rect, 1)

    def _render_game_over(self):
        overlay = pygame.Surface((self.GAME_WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text_surf = self.font_big.render(self.win_message, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.GAME_WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "completion": np.count_nonzero(self.grid) / (self.GRID_ROWS * self.GRID_COLS)
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

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


if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    print("--- Polyomino Puzzle ---")
    print(env.user_guide)
    print("------------------------")
    
    # Setup for human play
    render_surface = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Polyomino Puzzle")
    
    running = True
    while running:
        action_to_take = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                
                # Create a one-hot action for the key press
                # This simulates how an agent would interact, one action at a time
                if not env.game_over:
                    action = np.array([0, 0, 0])
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    elif event.key == pygame.K_SPACE: action[1] = 1
                    elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                    
                    if np.any(action > 0):
                        action_to_take = action

        # If an action was registered, step the environment
        if action_to_take is not None:
             obs, reward, terminated, truncated, info = env.step(action_to_take)
             print(f"Action: {action_to_take}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves: {info['moves_left']}")
             if terminated:
                 print("Game Over! Press 'R' to reset.")

        # Always render the current state
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_surface.blit(draw_surface, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)

    env.close()