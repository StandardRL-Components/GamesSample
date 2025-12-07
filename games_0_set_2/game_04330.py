
# Generated: 2025-08-28T02:04:50.925348
# Source Brief: brief_04330.md
# Brief Index: 4330

        
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
        "Controls: ←→ to move, ↑ to rotate, ↓ for soft drop. Space for hard drop, Shift to hold piece."
    )

    game_description = (
        "A fast-paced, falling-block puzzle game. Clear lines to score points, but watch out as the speed increases! "
        "Plan ahead using the 'hold' and 'next piece' features to achieve high scores."
    )

    auto_advance = True

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    CELL_SIZE = 18
    GRID_DRAW_WIDTH = GRID_WIDTH * CELL_SIZE
    GRID_DRAW_HEIGHT = GRID_HEIGHT * CELL_SIZE
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_DRAW_WIDTH) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_DRAW_HEIGHT) // 2

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 60)
    COLOR_TEXT = (220, 220, 220)
    COLOR_DANGER = (100, 20, 20, 100) # RGBA for transparency
    
    TETROMINO_DATA = {
        0: {'shape': [[1, 1, 1, 1]], 'color': (0, 240, 240)},  # I
        1: {'shape': [[1, 1, 0], [0, 1, 1]], 'color': (240, 0, 0)},  # Z
        2: {'shape': [[0, 1, 1], [1, 1, 0]], 'color': (0, 240, 0)},  # S
        3: {'shape': [[1, 1, 1], [0, 0, 1]], 'color': (0, 0, 240)},  # J
        4: {'shape': [[1, 1, 1], [1, 0, 0]], 'color': (240, 160, 0)},  # L
        5: {'shape': [[1, 1, 1], [0, 1, 0]], 'color': (160, 0, 240)},  # T
        6: {'shape': [[1, 1], [1, 1]], 'color': (240, 240, 0)}   # O
    }
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        self._precompute_rotations()
        self.reset()
        
        self.validate_implementation()

    def _precompute_rotations(self):
        self.rotations = {}
        for idx, data in self.TETROMINO_DATA.items():
            shape = np.array(data['shape'])
            self.rotations[idx] = [shape]
            for _ in range(3):
                shape = np.rot90(shape)
                self.rotations[idx].append(shape)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        self.piece_bag = list(range(len(self.TETROMINO_DATA)))
        self.np_random.shuffle(self.piece_bag)
        
        self.held_piece_type = None
        self.can_swap = True
        
        self.fall_timer = 0
        self.fall_speed = 30 # Frames per grid cell drop
        
        self.last_space_held = False
        self.last_shift_held = False

        self.line_clear_effects = []
        self.particles = []

        self._spawn_new_piece() # Current
        self._spawn_new_piece() # Next

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = -0.01

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        # 1. Handle one-off actions
        if shift_pressed:
            self._handle_hold()

        if space_pressed:
            reward += self._handle_hard_drop()
        else:
            # 2. Handle continuous/timed actions if not hard dropping
            if movement == 1: # Rotate
                self._move(0, 0, 1)
            elif movement == 3: # Left
                self._move(-1, 0, 0)
            elif movement == 4: # Right
                self._move(1, 0, 0)
            
            soft_drop = movement == 2
            self.fall_timer += 5 if soft_drop else 1

            if self.fall_timer >= self.fall_speed:
                self.fall_timer = 0
                if not self._move(0, 1, 0): # Try to move down
                    reward += self._lock_piece()

        # 3. Update animations
        self._update_effects()

        # 4. Check termination conditions
        terminated = self.game_over or self.lines_cleared >= 100 or self.steps >= 10000
        if terminated:
            if self.lines_cleared >= 100:
                reward += 100 # Win bonus
            elif self.game_over:
                reward -= 100 # Lose penalty
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    # --- Game Logic ---

    def _spawn_new_piece(self):
        if not self.piece_bag:
            self.piece_bag = list(range(len(self.TETROMINO_DATA)))
            self.np_random.shuffle(self.piece_bag)
        
        self.current_piece = getattr(self, 'next_piece', None)
        piece_type = self.piece_bag.pop(0)
        shape = self.rotations[piece_type][0]
        
        self.next_piece = {
            'type': piece_type,
            'shape': shape,
            'color': self.TETROMINO_DATA[piece_type]['color']
        }

        if self.current_piece:
            self.current_piece['x'] = self.GRID_WIDTH // 2 - self.current_piece['shape'].shape[1] // 2
            self.current_piece['y'] = 0
            self.current_piece['rotation'] = 0
            self.can_swap = True
            
            if self._check_collision(self.current_piece['x'], self.current_piece['y'], self.current_piece['shape']):
                self.game_over = True

    def _check_collision(self, x, y, shape):
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = x + c, y + r
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return True # Out of bounds
                    if self.grid[grid_y, grid_x] != 0:
                        return True # Collides with existing block
        return False

    def _move(self, dx, dy, dr):
        next_x = self.current_piece['x'] + dx
        next_y = self.current_piece['y'] + dy
        next_rot = (self.current_piece['rotation'] + dr) % 4
        next_shape = self.rotations[self.current_piece['type']][next_rot]

        if not self._check_collision(next_x, next_y, next_shape):
            self.current_piece['x'] = next_x
            self.current_piece['y'] = next_y
            self.current_piece['rotation'] = next_rot
            self.current_piece['shape'] = next_shape
            return True
        return False

    def _handle_hold(self):
        if not self.can_swap:
            return

        if self.held_piece_type is None:
            self.held_piece_type = self.current_piece['type']
            self._spawn_new_piece()
        else:
            held_type_cache = self.held_piece_type
            self.held_piece_type = self.current_piece['type']
            
            shape = self.rotations[held_type_cache][0]
            self.current_piece = {
                'type': held_type_cache,
                'shape': shape,
                'color': self.TETROMINO_DATA[held_type_cache]['color'],
                'x': self.GRID_WIDTH // 2 - shape.shape[1] // 2,
                'y': 0,
                'rotation': 0
            }
        self.can_swap = False

    def _handle_hard_drop(self):
        # # SFX: Hard drop sound
        while self._move(0, 1, 0):
            pass # Move down until collision
        return self._lock_piece()

    def _lock_piece(self):
        reward = 0
        prev_stack_height = self._get_stack_height()
        
        piece = self.current_piece
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    self.grid[piece['y'] + r, piece['x'] + c] = piece['type'] + 1
        
        # # SFX: Piece lock sound
        cleared_count, cleared_rows = self._clear_lines()
        if cleared_count > 0:
            # # SFX: Line clear sound (different for 1, 2, 3, 4 lines)
            self.lines_cleared += cleared_count
            
            # Score update
            score_map = {1: 40, 2: 100, 3: 300, 4: 1200}
            self.score += score_map.get(cleared_count, 0) * (self.lines_cleared // 20 + 1)
            
            # Reward update
            reward += cleared_count * 1 # +1 per line
            if cleared_count > 1:
                reward += 5 # +5 bonus for multi-line
            
            # Difficulty scaling
            self.fall_speed = max(5, 30 - (self.lines_cleared // 20) * 5)
            
            # Animation
            for row_idx in cleared_rows:
                self.line_clear_effects.append({'y': row_idx, 'timer': 10})
                for i in range(20): # Spawn particles
                    self.particles.append(Particle(
                        self.GRID_X_OFFSET + self.np_random.uniform(0, self.GRID_DRAW_WIDTH),
                        self.GRID_Y_OFFSET + row_idx * self.CELL_SIZE + self.CELL_SIZE / 2,
                        self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 1),
                        self.np_random.integers(15, 30),
                        (255, 255, 255)
                    ))

        new_stack_height = self._get_stack_height()
        if new_stack_height > self.GRID_HEIGHT * 0.8 and new_stack_height > prev_stack_height:
            reward -= 2

        self._spawn_new_piece()
        return reward

    def _clear_lines(self):
        cleared_rows = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                cleared_rows.append(r)
        
        if not cleared_rows:
            return 0, []

        for r in cleared_rows:
            self.grid[r, :] = 0 # Clear the line
        
        # Shift rows down
        rows_to_move = sorted([r for r in range(self.GRID_HEIGHT) if r not in cleared_rows], reverse=True)
        new_grid = np.zeros_like(self.grid)
        new_row_idx = self.GRID_HEIGHT - 1
        for r in rows_to_move:
            new_grid[new_row_idx, :] = self.grid[r, :]
            new_row_idx -= 1
        
        self.grid = new_grid
        return len(cleared_rows), cleared_rows

    def _get_stack_height(self):
        for r in range(self.GRID_HEIGHT):
            if np.any(self.grid[r, :] != 0):
                return self.GRID_HEIGHT - r
        return 0
    
    def _update_effects(self):
        # Update line clear flash
        self.line_clear_effects = [e for e in self.line_clear_effects if e['timer'] > 0]
        for effect in self.line_clear_effects:
            effect['timer'] -= 1
        # Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background and border
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET - 4, self.GRID_Y_OFFSET - 4, self.GRID_DRAW_WIDTH + 8, self.GRID_DRAW_HEIGHT + 8))
        pygame.draw.rect(self.screen, self.COLOR_BG, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_DRAW_WIDTH, self.GRID_DRAW_HEIGHT))

        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET), (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_DRAW_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + y * self.CELL_SIZE), (self.GRID_X_OFFSET + self.GRID_DRAW_WIDTH, self.GRID_Y_OFFSET + y * self.CELL_SIZE))

        # Draw danger zone
        if self._get_stack_height() > self.GRID_HEIGHT * 0.8:
            s = pygame.Surface((self.GRID_DRAW_WIDTH, self.GRID_DRAW_HEIGHT * 0.2), pygame.SRCALPHA)
            s.fill(self.COLOR_DANGER)
            self.screen.blit(s, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET))

        # Draw placed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    color = self.TETROMINO_DATA[self.grid[r, c] - 1]['color']
                    self._draw_cell(c, r, color)
        
        if not self.game_over:
            # Draw ghost piece
            ghost_y = self.current_piece['y']
            while not self._check_collision(self.current_piece['x'], ghost_y + 1, self.current_piece['shape']):
                ghost_y += 1
            self._draw_piece(self.current_piece, (self.current_piece['x'], ghost_y), ghost=True)

            # Draw current piece
            self._draw_piece(self.current_piece, (self.current_piece['x'], self.current_piece['y']))
        
        # Draw effects
        for effect in self.line_clear_effects:
            alpha = int(255 * (effect['timer'] / 10))
            pygame.gfxdraw.box(self.screen, 
                               (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + effect['y'] * self.CELL_SIZE, self.GRID_DRAW_WIDTH, self.CELL_SIZE),
                               (255, 255, 255, alpha))
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 20, 20))
        # Lines
        lines_surf = self.font_main.render(f"LINES: {self.lines_cleared}", True, self.COLOR_TEXT)
        self.screen.blit(lines_surf, (20, 20))

        # Next Piece
        self._draw_preview_box(self.SCREEN_WIDTH - 120, 80, "NEXT", self.next_piece)
        # Held Piece
        held_piece_data = None
        if self.held_piece_type is not None:
            held_piece_data = {
                'shape': self.rotations[self.held_piece_type][0],
                'color': self.TETROMINO_DATA[self.held_piece_type]['color']
            }
        self._draw_preview_box(20, 80, "HOLD", held_piece_data)
        
        if self.game_over:
            self._draw_centered_text("GAME OVER", 80)
        elif self.lines_cleared >= 100:
            self._draw_centered_text("YOU WIN!", 80)

    def _draw_cell(self, grid_x, grid_y, color, ghost=False):
        screen_x = self.GRID_X_OFFSET + grid_x * self.CELL_SIZE
        screen_y = self.GRID_Y_OFFSET + grid_y * self.CELL_SIZE
        
        if ghost:
            pygame.gfxdraw.rectangle(self.screen, (screen_x, screen_y, self.CELL_SIZE, self.CELL_SIZE), (*color, 100))
        else:
            main_rect = (screen_x + 1, screen_y + 1, self.CELL_SIZE - 2, self.CELL_SIZE - 2)
            pygame.draw.rect(self.screen, color, main_rect)
            
            # 3D effect
            light_color = tuple(min(255, c + 50) for c in color)
            dark_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.line(self.screen, light_color, (main_rect[0], main_rect[1]), (main_rect[0] + main_rect[2], main_rect[1]), 2)
            pygame.draw.line(self.screen, light_color, (main_rect[0], main_rect[1]), (main_rect[0], main_rect[1] + main_rect[3]), 2)
            pygame.draw.line(self.screen, dark_color, (main_rect[0] + main_rect[2], main_rect[1]), (main_rect[0] + main_rect[2], main_rect[1] + main_rect[3]), 2)
            pygame.draw.line(self.screen, dark_color, (main_rect[0], main_rect[1] + main_rect[3]), (main_rect[0] + main_rect[2], main_rect[1] + main_rect[3]), 2)

    def _draw_piece(self, piece, pos, ghost=False):
        px, py = pos
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_cell(px + c, py + r, piece['color'], ghost)

    def _draw_preview_box(self, x, y, title, piece_data):
        title_surf = self.font_small.render(title, True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (x + (100 - title_surf.get_width())//2, y))
        
        box_rect = (x, y + 30, 100, 80)
        pygame.draw.rect(self.screen, self.COLOR_GRID, box_rect, 4)

        if piece_data:
            shape = piece_data['shape']
            w, h = shape.shape[1], shape.shape[0]
            
            # Center the piece in the box
            start_x = x + (100 - w * self.CELL_SIZE) // 2
            start_y = y + 30 + (80 - h * self.CELL_SIZE) // 2
            
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        screen_x = start_x + c * self.CELL_SIZE
                        screen_y = start_y + r * self.CELL_SIZE
                        main_rect = (screen_x + 1, screen_y + 1, self.CELL_SIZE - 2, self.CELL_SIZE - 2)
                        pygame.draw.rect(self.screen, piece_data['color'], main_rect)

    def _draw_centered_text(self, text, size, color=(255, 255, 255)):
        font = pygame.font.Font(None, size)
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        
        # Add a dark background for readability
        bg_rect = text_rect.inflate(20, 20)
        s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, bg_rect)
        
        self.screen.blit(text_surf, text_rect)

    # --- Gymnasium Interface ---
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
            "level": self.lines_cleared // 20
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

class Particle:
    def __init__(self, x, y, vx, vy, life, color):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.life = life
        self.color = color
        self.gravity = 0.1

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.life -= 1

    def draw(self, surface):
        alpha = max(0, min(255, int(255 * (self.life / 20))))
        pygame.gfxdraw.pixel(surface, int(self.x), int(self.y), (*self.color, alpha))