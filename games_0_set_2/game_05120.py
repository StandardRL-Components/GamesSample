
# Generated: 2025-08-28T04:02:31.319766
# Source Brief: brief_05120.md
# Brief Index: 5120

        
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
        "Controls: ←→ to move, ↑↓ to rotate. Hold space for soft drop, press shift for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced falling block puzzle. Clear lines by filling them with blocks. "
        "Clear 15 lines to win, but don't let the stack reach the top!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock() # Not used for stepping, but can be useful
        
        # --- Visuals & Fonts ---
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 80)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_FLASH = (255, 255, 255)

        # --- Game Constants ---
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.PLAYFIELD_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.PLAYFIELD_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.GRID_X = (self.screen_width - self.PLAYFIELD_WIDTH) // 2
        self.GRID_Y = (self.screen_height - self.PLAYFIELD_HEIGHT) // 2
        self.MAX_STEPS = 10000
        self.WIN_CONDITION_LINES = 15

        # --- Tetromino Definitions ---
        self.TETROMINOES = {
            'I': {'shape': [[1, 1, 1, 1]], 'color': (66, 215, 245)},
            'J': {'shape': [[1, 0, 0], [1, 1, 1]], 'color': (0, 100, 200)},
            'L': {'shape': [[0, 0, 1], [1, 1, 1]], 'color': (245, 160, 66)},
            'O': {'shape': [[1, 1], [1, 1]], 'color': (245, 225, 66)},
            'S': {'shape': [[0, 1, 1], [1, 1, 0]], 'color': (140, 245, 66)},
            'T': {'shape': [[0, 1, 0], [1, 1, 1]], 'color': (180, 66, 245)},
            'Z': {'shape': [[1, 1, 0], [0, 1, 1]], 'color': (245, 66, 95)}
        }
        self.TETROMINO_NAMES = list(self.TETROMINOES.keys())

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = None
        self.lines_cleared = None
        self.steps = None
        self.game_over = None
        self.win = None
        
        self.drop_timer = None
        self.base_drop_interval = None
        self.current_drop_interval = None
        
        self.line_clear_animation = None
        self.particles = None
        
        self.prev_action = None
        
        # --- Initialization ---
        self.reset()
        self.validate_implementation()
    
    def _create_new_piece(self):
        piece_name = self.np_random.choice(self.TETROMINO_NAMES)
        piece_data = self.TETROMINOES[piece_name]
        return {
            'name': piece_name,
            'shape': piece_data['shape'],
            'color': piece_data['color'],
            'x': self.GRID_WIDTH // 2 - len(piece_data['shape'][0]) // 2,
            'y': 0
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        self.current_piece = self._create_new_piece()
        self.next_piece = self._create_new_piece()
        
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        
        self.drop_timer = 0
        self.base_drop_interval = 30 # 1 second at 30fps
        self.current_drop_interval = self.base_drop_interval
        
        self.line_clear_animation = None # (timer, [row_indices])
        self.particles = []
        
        self.prev_action = self.action_space.sample() * 0 # All zeros
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = -0.01 # Per-step cost

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if self.line_clear_animation:
            self.line_clear_animation = (self.line_clear_animation[0] - 1, self.line_clear_animation[1])
            if self.line_clear_animation[0] <= 0:
                self._finish_line_clear()
        elif not self.game_over and not self.win:
            # --- Handle Input ---
            self._handle_input(movement, shift_held)

            # --- Game Logic Update ---
            self.drop_timer += 2 if space_held else 1 # Soft drop doubles speed
            
            if self.drop_timer >= self.current_drop_interval:
                self.drop_timer = 0
                if not self._check_collision(self.current_piece, offset_y=1):
                    self.current_piece['y'] += 1
                else:
                    # Place piece and get rewards
                    placement_rewards = self._place_piece()
                    reward += placement_rewards

        self.prev_action = action
        
        # --- Termination Checks ---
        terminated = self.game_over or self.win or self.steps >= self.MAX_STEPS
        if terminated:
            if self.win: reward += 100
            elif self.game_over: reward -= 100

        # --- Update animations ---
        self._update_particles()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, shift_held):
        hard_drop_press = shift_held and not (self.prev_action[2] == 1)

        if hard_drop_press:
            # --- Hard Drop ---
            dy = 0
            while not self._check_collision(self.current_piece, offset_y=dy + 1):
                dy += 1
            if dy > 0:
                self.current_piece['y'] += dy
                # sfx: hard_drop_sound
                for _ in range(30):
                    px = self.current_piece['x'] + self.np_random.random() * len(self.current_piece['shape'][0])
                    py = self.current_piece['y'] + len(self.current_piece['shape'])
                    self.particles.append(Particle(self.np_random, px, py, self.current_piece['color'], is_grid_coords=True))
            self.drop_timer = self.current_drop_interval # Force placement on next logic tick
            return

        # Handle movement and rotation only if not hard dropping
        if movement == 1: # Up -> Rotate Clockwise
            self._rotate_piece(clockwise=True)
        elif movement == 2: # Down -> Rotate Counter-Clockwise
            self._rotate_piece(clockwise=False)
        elif movement == 3: # Left
            if not self._check_collision(self.current_piece, offset_x=-1):
                self.current_piece['x'] -= 1
        elif movement == 4: # Right
            if not self._check_collision(self.current_piece, offset_x=1):
                self.current_piece['x'] += 1
    
    def _place_piece(self):
        # sfx: piece_lock_sound
        total_reward = 0
        shape = self.current_piece['shape']
        color_idx = self.TETROMINO_NAMES.index(self.current_piece['name']) + 1
        for r_idx, row in enumerate(shape):
            for c_idx, cell in enumerate(row):
                if cell:
                    grid_y = self.current_piece['y'] + r_idx
                    grid_x = self.current_piece['x'] + c_idx
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y, grid_x] = color_idx
        
        # --- Check for line clears ---
        lines_cleared, cleared_rows = self._check_line_clears()
        
        if lines_cleared > 0:
            line_reward = {1: 1, 2: 3, 3: 7, 4: 15}[lines_cleared]
            self.score += line_reward
            total_reward += line_reward
            self.lines_cleared += lines_cleared
            self.line_clear_animation = (10, cleared_rows) # 10 frames of animation
            
            # Update difficulty
            difficulty_level = self.lines_cleared // 5
            self.current_drop_interval = max(5, self.base_drop_interval - difficulty_level * 3)
        
        # --- Spawn next piece ---
        self.current_piece = self.next_piece
        self.next_piece = self._create_new_piece()

        # --- Check for game over ---
        if self._check_collision(self.current_piece):
            self.game_over = True
            self.current_piece = None # Don't draw the piece that caused the loss
        
        # --- Check for win ---
        if self.lines_cleared >= self.WIN_CONDITION_LINES:
            self.win = True

        # --- Calculate placement reward (for risky gaps) ---
        total_reward += self._calculate_placement_reward()
        
        return total_reward

    def _calculate_placement_reward(self):
        # Reward +0.1 for creating/maintaining a single-cell wide well
        reward = 0
        for x in range(self.GRID_WIDTH):
            is_well = True
            for y in range(self.GRID_HEIGHT):
                if self.grid[y, x] != 0:
                    is_well = False
                    break
            if is_well:
                left_wall = x == 0 or np.any(self.grid[:, x-1])
                right_wall = x == self.GRID_WIDTH-1 or np.any(self.grid[:, x+1])
                if left_wall and right_wall:
                    reward += 0.1
        return reward

    def _check_line_clears(self):
        full_rows = [r for r in range(self.GRID_HEIGHT) if np.all(self.grid[r, :] != 0)]
        return len(full_rows), full_rows

    def _finish_line_clear(self):
        # sfx: line_clear_sound
        _, cleared_rows = self.line_clear_animation
        
        # Add particles
        for r in cleared_rows:
            for i in range(self.GRID_WIDTH * 2):
                px = (i / 2)
                py = r
                self.particles.append(Particle(self.np_random, px, py, self.COLOR_FLASH, is_grid_coords=True, life=15))

        # Remove rows and shift down
        self.grid = np.delete(self.grid, cleared_rows, axis=0)
        new_rows = np.zeros((len(cleared_rows), self.GRID_WIDTH), dtype=int)
        self.grid = np.vstack((new_rows, self.grid))
        
        self.line_clear_animation = None

    def _check_collision(self, piece, offset_x=0, offset_y=0):
        if piece is None: return True
        shape = piece['shape']
        for r_idx, row in enumerate(shape):
            for c_idx, cell in enumerate(row):
                if cell:
                    grid_y = piece['y'] + r_idx + offset_y
                    grid_x = piece['x'] + c_idx + offset_x
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return True # Wall collision
                    if self.grid[grid_y, grid_x] != 0:
                        return True # Other block collision
        return False

    def _rotate_piece(self, clockwise=True):
        if self.current_piece['name'] == 'O': return # O piece doesn't rotate
        
        original_shape = self.current_piece['shape']
        
        if clockwise:
            new_shape = [list(row) for row in zip(*original_shape[::-1])]
        else:
            new_shape = [list(row) for row in zip(*original_shape)][::-1]

        # Test rotation with wall kicks (SRS-like simple kicks)
        test_offsets = [0, -1, 1, -2, 2]
        for offset in test_offsets:
            test_piece = self.current_piece.copy()
            test_piece['shape'] = new_shape
            if not self._check_collision(test_piece, offset_x=offset):
                self.current_piece['shape'] = new_shape
                self.current_piece['x'] += offset
                # sfx: rotate_sound
                return
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, (0,0,0), (self.GRID_X, self.GRID_Y, self.PLAYFIELD_WIDTH, self.PLAYFIELD_HEIGHT))
        
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_Y), (px, self.GRID_Y + self.PLAYFIELD_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, py), (self.GRID_X + self.PLAYFIELD_WIDTH, py))

        # Draw placed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    color = self.TETROMINOES[self.TETROMINO_NAMES[self.grid[r,c]-1]]['color']
                    self._draw_cell(c, r, color)
        
        if self.current_piece:
            # Draw ghost piece
            ghost_piece = self.current_piece.copy()
            dy = 0
            while not self._check_collision(ghost_piece, offset_y=dy + 1):
                dy += 1
            ghost_piece['y'] += dy
            self._draw_piece(ghost_piece, is_ghost=True)

            # Draw current piece
            self._draw_piece(self.current_piece)

        # Draw line clear animation
        if self.line_clear_animation:
            timer, rows = self.line_clear_animation
            alpha = int(255 * (math.sin(timer / 10 * math.pi))) # Fade in/out
            flash_surface = pygame.Surface((self.PLAYFIELD_WIDTH, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_FLASH, alpha))
            for r in rows:
                self.screen.blit(flash_surface, (self.GRID_X, self.GRID_Y + r * self.CELL_SIZE))
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen, self.GRID_X, self.GRID_Y, self.CELL_SIZE)

    def _draw_piece(self, piece, is_ghost=False):
        shape = piece['shape']
        for r_idx, row in enumerate(shape):
            for c_idx, cell in enumerate(row):
                if cell:
                    self._draw_cell(piece['x'] + c_idx, piece['y'] + r_idx, piece['color'], is_ghost)
    
    def _draw_cell(self, grid_x, grid_y, color, is_ghost=False):
        px, py = self.GRID_X + grid_x * self.CELL_SIZE, self.GRID_Y + grid_y * self.CELL_SIZE
        if py < self.GRID_Y: return # Don't draw above the playfield
        rect = (px, py, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2) # Just the outline
        else:
            # Main cell
            pygame.draw.rect(self.screen, color, rect)
            # 3D effect
            dark_color = tuple(max(0, c - 50) for c in color)
            light_color = tuple(min(255, c + 50) for c in color)
            pygame.draw.line(self.screen, dark_color, (px + self.CELL_SIZE - 1, py), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1), 2)
            pygame.draw.line(self.screen, dark_color, (px, py + self.CELL_SIZE - 1), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1), 2)
            pygame.draw.line(self.screen, light_color, (px, py), (px + self.CELL_SIZE - 1, py), 2)
            pygame.draw.line(self.screen, light_color, (px, py), (px, py + self.CELL_SIZE - 1), 2)
            
    def _render_ui(self):
        # --- Score ---
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # --- Lines ---
        lines_text = self.font_main.render(f"LINES: {self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (self.screen_width - lines_text.get_width() - 20, 20))
        
        # --- Next Piece ---
        next_box_x = self.GRID_X + self.PLAYFIELD_WIDTH + 20
        next_box_y = self.GRID_Y
        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (next_box_x, next_box_y))
        
        if self.next_piece:
            shape = self.next_piece['shape']
            w, h = len(shape[0]), len(shape)
            start_x = next_box_x + (80 - w * self.CELL_SIZE) // 2
            start_y = next_box_y + 40 + (80 - h * self.CELL_SIZE) // 2
            
            for r_idx, row in enumerate(shape):
                for c_idx, cell in enumerate(row):
                    if cell:
                        px, py = start_x + c_idx * self.CELL_SIZE, start_y + r_idx * self.CELL_SIZE
                        rect = (px, py, self.CELL_SIZE, self.CELL_SIZE)
                        pygame.draw.rect(self.screen, self.next_piece['color'], rect)

        # --- Game Over / Win Text ---
        if self.game_over:
            self._render_centered_text("GAME OVER", self.font_main, (255, 80, 80))
        elif self.win:
            self._render_centered_text("YOU WIN!", self.font_main, (80, 255, 80))

    def _render_centered_text(self, text, font, color):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
        # Add a dark background for readability
        bg_rect = text_rect.inflate(20, 10)
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 180))
        self.screen.blit(bg_surf, bg_rect)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared
        }

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

class Particle:
    def __init__(self, rng, x, y, color, is_grid_coords=False, life=20):
        self.rng = rng
        self.is_grid_coords = is_grid_coords
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.max_life = life
        
        angle = self.rng.uniform(0, 2 * math.pi)
        speed = self.rng.uniform(0.1, 0.3)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.02 # gravity
        self.life -= 1
        return self.life > 0

    def draw(self, surface, grid_x_offset, grid_y_offset, cell_size):
        alpha = int(255 * (self.life / self.max_life))
        color = (*self.color, alpha)
        
        if self.is_grid_coords:
            px = grid_x_offset + self.x * cell_size
            py = grid_y_offset + self.y * cell_size
        else:
            px, py = self.x, self.y
            
        temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
        temp_surf.fill(color)
        surface.blit(temp_surf, (int(px), int(py)))