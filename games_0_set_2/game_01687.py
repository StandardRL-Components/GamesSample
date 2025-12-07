
# Generated: 2025-08-28T02:22:37.867148
# Source Brief: brief_01687.md
# Brief Index: 1687

        
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
        "Controls: ←→ to move piece, ↑↓ to rotate. Press Space to drop."
    )

    game_description = (
        "Stack falling geometric tiles to build the tallest tower possible. Match 3 or more tiles of the same color vertically to clear them and score points. Be careful, an unstable tower will collapse!"
    )

    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 12
    GRID_HEIGHT = 28
    TARGET_HEIGHT = 25
    CELL_SIZE = 14
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (45, 50, 62)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TARGET_LINE = (255, 200, 0)
    
    TILE_COLORS = [
        (239, 83, 80),   # Red
        (255, 167, 38),  # Orange
        (255, 238, 88),  # Yellow
        (102, 187, 106), # Green
        (66, 165, 245),  # Blue
        (126, 87, 194),  # Purple
        (236, 64, 122),  # Pink
        (0, 188, 212),   # Cyan
    ]

    # Tetromino shapes
    SHAPES = [
        [[1, 1, 1, 1]],  # I
        [[1, 1, 0], [0, 1, 1]],  # Z
        [[0, 1, 1], [1, 1, 0]],  # S
        [[1, 1, 1], [0, 1, 0]],  # T
        [[1, 1, 1], [1, 0, 0]],  # L
        [[1, 1, 1], [0, 0, 1]],  # J
        [[1, 1], [1, 1]],  # O
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
        
        try:
            self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
            self.font_medium = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 16)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 42)
            self.font_medium = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 22)


        self.grid_pixel_width = self.GRID_WIDTH * self.CELL_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.CELL_SIZE
        self.grid_start_x = (self.SCREEN_WIDTH - self.grid_pixel_width) // 2
        self.grid_start_y = self.SCREEN_HEIGHT - self.grid_pixel_height

        self.game_over = False
        self.steps = 0
        self.score = 0
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.current_piece = None
        self.tower_height = 0
        self.particles = []
        self.wobble_magnitude = 0.0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tower_height = 0
        self.particles = []
        self.wobble_magnitude = 0.0

        self._spawn_new_piece()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_pressed, _ = action
        reward = 0
        terminated = False

        if space_pressed:
            reward, terminated = self._perform_drop()
        else:
            self._perform_positioning(movement)

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        # Decay wobble over time
        self.wobble_magnitude = max(0, self.wobble_magnitude - 0.05)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _perform_positioning(self, movement):
        # 0=none, 1=up(rot+), 2=down(rot-), 3=left, 4=right
        if movement == 1: # Rotate clockwise
            self._rotate_piece(1)
        elif movement == 2: # Rotate anti-clockwise
            self._rotate_piece(-1)
        elif movement == 3: # Move left
            self._move_piece(-1)
        elif movement == 4: # Move right
            self._move_piece(1)

    def _perform_drop(self):
        reward = 0
        terminated = False

        # 1. Find landing position
        landing_y = self._get_ghost_piece_y()
        self.current_piece['y'] = landing_y

        # 2. Check for risky placement reward
        if self._is_risky_placement():
            reward += 2.0

        # 3. Place piece on grid
        self._place_piece_on_grid()
        reward += 0.1 # Reward for successful placement

        # 4. Check for and clear matches
        cleared_tiles, matched_positions = self._check_and_clear_matches()
        if cleared_tiles > 0:
            reward += cleared_tiles * 1.0
            # sfx: match_clear.wav
            for pos, color_idx in matched_positions:
                self._create_particles(pos, color_idx)
            
            # 5. Apply gravity to tiles above cleared ones
            self._apply_gravity()

        # 6. Check for collapse (floating pieces)
        if self._check_for_collapse():
            reward = -100.0
            terminated = True
            # sfx: tower_collapse.wav
        
        if not terminated:
            # 7. Check for instability
            if self._check_instability():
                reward -= 5.0
                self.wobble_magnitude = min(5.0, self.wobble_magnitude + 1.5)
                # sfx: wobble.wav
            
            # 8. Update tower height and check for win
            self._update_tower_height()
            if self.tower_height >= self.TARGET_HEIGHT:
                reward = 100.0
                terminated = True
                # sfx: win_fanfare.wav

            # 9. Spawn new piece if game continues
            if not terminated:
                self._spawn_new_piece()
                # Check if new piece immediately causes a loss
                if self._check_collision(self.current_piece, 0, 0):
                    reward = -100.0
                    terminated = True
        
        return reward, terminated

    def _spawn_new_piece(self):
        shape_idx = self.np_random.integers(0, len(self.SHAPES))
        color_idx = self.np_random.integers(1, len(self.TILE_COLORS) + 1)
        shape = self.SHAPES[shape_idx]
        
        self.current_piece = {
            'shape': shape,
            'color': color_idx,
            'x': self.GRID_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0
        }

    def _move_piece(self, dx):
        if not self._check_collision(self.current_piece, dx, 0):
            self.current_piece['x'] += dx
            # sfx: move.wav

    def _rotate_piece(self, direction):
        original_shape = self.current_piece['shape']
        if direction == 1: # Clockwise
            new_shape = [list(row) for row in zip(*original_shape[::-1])]
        else: # Anti-clockwise
            new_shape = [list(row) for row in zip(*original_shape)][::-1]
        
        test_piece = self.current_piece.copy()
        test_piece['shape'] = new_shape

        # Wall kick
        if not self._check_collision(test_piece, 0, 0):
            self.current_piece['shape'] = new_shape
            # sfx: rotate.wav
        elif not self._check_collision(test_piece, 1, 0): # Kick right
            self.current_piece['shape'] = new_shape
            self.current_piece['x'] += 1
            # sfx: rotate.wav
        elif not self._check_collision(test_piece, -1, 0): # Kick left
            self.current_piece['shape'] = new_shape
            self.current_piece['x'] -= 1
            # sfx: rotate.wav

    def _check_collision(self, piece, dx, dy):
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    grid_x = piece['x'] + c + dx
                    grid_y = piece['y'] + r + dy
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return True # Out of bounds
                    if self.grid[grid_y, grid_x] != 0:
                        return True # Collision with existing tile
        return False

    def _place_piece_on_grid(self):
        piece = self.current_piece
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    grid_x = piece['x'] + c
                    grid_y = piece['y'] + r
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y, grid_x] = piece['color']
        # sfx: piece_land.wav

    def _check_and_clear_matches(self):
        to_clear = set()
        matched_positions = []
        total_cleared = 0

        for x in range(self.GRID_WIDTH):
            run_start_y = -1
            run_color = -1
            for y in range(self.GRID_HEIGHT):
                cell_color = self.grid[y, x]
                if cell_color != 0 and cell_color == run_color:
                    continue
                else:
                    run_length = y - run_start_y
                    if run_length >= 3:
                        for i in range(run_length):
                            pos = (run_start_y + i, x)
                            if pos not in to_clear:
                                to_clear.add(pos)
                                matched_positions.append((pos, run_color))
                                total_cleared += 1
                    
                    run_start_y = y
                    run_color = cell_color
            
            # Check last run in column
            run_length = self.GRID_HEIGHT - run_start_y
            if run_length >= 3 and run_color != 0:
                for i in range(run_length):
                    pos = (run_start_y + i, x)
                    if pos not in to_clear:
                        to_clear.add(pos)
                        matched_positions.append((pos, run_color))
                        total_cleared += 1

        for y, x in to_clear:
            self.grid[y, x] = 0
            
        return total_cleared, matched_positions

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y, x] != 0:
                    if y != empty_row:
                        self.grid[empty_row, x] = self.grid[y, x]
                        self.grid[y, x] = 0
                    empty_row -= 1

    def _check_for_collapse(self):
        if np.sum(self.grid) == 0: return False

        q = []
        visited = set()

        # Start flood fill from all base tiles
        for x in range(self.GRID_WIDTH):
            if self.grid[self.GRID_HEIGHT - 1, x] != 0:
                pos = (self.GRID_HEIGHT - 1, x)
                q.append(pos)
                visited.add(pos)
        
        head = 0
        while head < len(q):
            y, x = q[head]
            head += 1
            
            for dy, dx in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if self.grid[ny, nx] != 0 and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        q.append((ny, nx))

        total_tiles = np.count_nonzero(self.grid)
        return len(visited) < total_tiles

    def _is_risky_placement(self):
        piece = self.current_piece
        y_base = piece['y'] + len(piece['shape'])
        if y_base >= self.GRID_HEIGHT: return False

        piece_xs = {piece['x'] + c for r in piece['shape'] for c, cell in enumerate(r) if cell}
        
        for x in piece_xs:
            if 0 <= x-1 < self.GRID_WIDTH and self.grid[y_base, x-1] == 0 and \
               0 <= x-2 < self.GRID_WIDTH and self.grid[y_base, x-2] == 0:
                return True
            if 0 <= x+1 < self.GRID_WIDTH and self.grid[y_base, x+1] == 0 and \
               0 <= x+2 < self.GRID_WIDTH and self.grid[y_base, x+2] == 0:
                return True
        return False

    def _check_instability(self):
        if self.tower_height < 5: return False
        
        for y in range(self.GRID_HEIGHT - 2, self.GRID_HEIGHT - self.tower_height, -1):
            row_width = np.count_nonzero(self.grid[y, :])
            row_above_width = np.count_nonzero(self.grid[y-1, :])
            if row_width == 1 and row_above_width > 2:
                return True
        return False

    def _update_tower_height(self):
        if np.sum(self.grid) == 0:
            self.tower_height = 0
            return
        
        non_empty_rows = np.where(np.any(self.grid != 0, axis=1))[0]
        if len(non_empty_rows) > 0:
            top_row_idx = non_empty_rows.min()
            self.tower_height = self.GRID_HEIGHT - top_row_idx
        else:
            self.tower_height = 0

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tower_height": self.tower_height,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid_and_border()
        self._render_target_line()
        self._render_placed_tiles()
        if self.current_piece and not self.game_over:
            self._render_ghost_piece()
            self._render_falling_piece()
        self._update_and_render_particles()

    def _render_grid_and_border(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.grid_start_x + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.grid_start_y), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.grid_start_y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_start_x, py), (self.grid_start_x + self.grid_pixel_width, py))
        
        # Draw border
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.grid_start_x, self.grid_start_y, self.grid_pixel_width, self.grid_pixel_height), 1)

    def _render_target_line(self):
        y_pos = self.grid_start_y + (self.GRID_HEIGHT - self.TARGET_HEIGHT) * self.CELL_SIZE
        pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (self.grid_start_x, y_pos), (self.grid_start_x + self.grid_pixel_width, y_pos), 2)
        
        text = self.font_small.render(f"{self.TARGET_HEIGHT}", True, self.COLOR_TARGET_LINE)
        self.screen.blit(text, (self.grid_start_x + self.grid_pixel_width + 5, y_pos - text.get_height() // 2))

    def _render_placed_tiles(self):
        wobble_offset = math.sin(self.steps * 0.4) * self.wobble_magnitude
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_idx = self.grid[r, c]
                if color_idx != 0:
                    self._draw_tile(c, r, color_idx, offset_x=wobble_offset)

    def _get_ghost_piece_y(self):
        ghost_piece = self.current_piece.copy()
        y = ghost_piece['y']
        while not self._check_collision(ghost_piece, 0, 1):
            ghost_piece['y'] += 1
        return ghost_piece['y']

    def _render_ghost_piece(self):
        ghost_y = self._get_ghost_piece_y()
        piece = self.current_piece
        
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    color = self.TILE_COLORS[piece['color'] - 1]
                    alpha_color = (*color, 60) # RGBA
                    
                    rect = pygame.Rect(
                        self.grid_start_x + (piece['x'] + c) * self.CELL_SIZE,
                        self.grid_start_y + (ghost_y + r) * self.CELL_SIZE,
                        self.CELL_SIZE, self.CELL_SIZE
                    )
                    
                    # Create a temporary surface for transparency
                    shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                    pygame.draw.rect(shape_surf, alpha_color, shape_surf.get_rect())
                    self.screen.blit(shape_surf, rect.topleft)

    def _render_falling_piece(self):
        piece = self.current_piece
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_tile(piece['x'] + c, piece['y'] + r, piece['color'], is_falling=True)

    def _draw_tile(self, grid_c, grid_r, color_idx, is_falling=False, offset_x=0):
        color = self.TILE_COLORS[color_idx - 1]
        darker_color = tuple(max(0, val - 40) for val in color)
        
        rect = pygame.Rect(
            int(self.grid_start_x + grid_c * self.CELL_SIZE + offset_x),
            self.grid_start_y + grid_r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )

        inner_rect = rect.inflate(-3, -3)
        
        pygame.draw.rect(self.screen, darker_color, rect)
        pygame.draw.rect(self.screen, color, inner_rect)

        if is_falling:
            pygame.gfxdraw.rectangle(self.screen, rect, (*self.COLOR_TEXT, 150))

    def _create_particles(self, grid_pos, color_idx):
        y, x = grid_pos
        px = self.grid_start_x + x * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.grid_start_y + y * self.CELL_SIZE + self.CELL_SIZE // 2
        color = self.TILE_COLORS[color_idx - 1]
        
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [px, py],
                'vel': vel,
                'life': self.np_random.integers(20, 40),
                'color': color
            })

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / 30))))
                pygame.draw.circle(self.screen, (*p['color'], alpha), p['pos'], max(1, int(p['life'] * 0.1)))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Tower Height
        height_text = self.font_medium.render(f"HEIGHT: {self.tower_height}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (self.SCREEN_WIDTH - height_text.get_width() - 20, 20))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "TOWER COMPLETE!" if self.tower_height >= self.TARGET_HEIGHT else "TOWER COLLAPSED"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_TARGET_LINE if self.tower_height >= self.TARGET_HEIGHT else (255, 80, 80))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Stacker")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # --- Action mapping for human play ---
    # action = [movement, space, shift]
    # movement: 0=none, 1=up(rot+), 2=down(rot-), 3=left, 4=right
    
    while not terminated:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_UP:
                    action[0] = 1 # Rotate Clockwise
                elif event.key == pygame.K_DOWN:
                    action[0] = 2 # Rotate Anti-clockwise
                elif event.key == pygame.K_SPACE:
                    action[1] = 1 # Drop
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
        
        # If any key was pressed, send the action
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    pygame.quit()