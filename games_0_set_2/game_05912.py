import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
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
        "A fast-paced, falling block puzzle game. Clear lines to score points and advance through stages with increasing speed. Win by clearing all stages."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    GRID_WIDTH = 10
    GRID_HEIGHT = 20
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_DANGER = (100, 20, 30, 100)
    COLOR_GHOST = (255, 255, 255, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SCORE = (255, 209, 102)

    SHAPES = {
        'T': [[[1, 0], [0, 1], [1, 1], [2, 1]], [[1, 0], [0, 1], [1, 1], [1, 2]], [[0, 1], [1, 1], [2, 1], [1, 2]], [[1, 0], [1, 1], [2, 1], [1, 2]]],
        'I': [[[0, 1], [1, 1], [2, 1], [3, 1]], [[2, 0], [2, 1], [2, 2], [2, 3]], [[0, 2], [1, 2], [2, 2], [3, 2]], [[1, 0], [1, 1], [1, 2], [1, 3]]],
        'O': [[[0, 0], [1, 0], [0, 1], [1, 1]]],
        'L': [[[0, 1], [1, 1], [2, 1], [2, 0]], [[1, 0], [1, 1], [1, 2], [2, 2]], [[0, 1], [1, 1], [2, 1], [0, 2]], [[0, 0], [1, 0], [1, 1], [1, 2]]],
        'J': [[[0, 0], [0, 1], [1, 1], [2, 1]], [[1, 0], [2, 0], [1, 1], [1, 2]], [[0, 1], [1, 1], [2, 1], [2, 2]], [[1, 0], [1, 1], [0, 2], [1, 2]]],
        'S': [[[1, 0], [2, 0], [0, 1], [1, 1]], [[1, 0], [1, 1], [2, 1], [2, 2]], [[1, 1], [2, 1], [0, 2], [1, 2]], [[0, 0], [0, 1], [1, 1], [1, 2]]],
        'Z': [[[0, 0], [1, 0], [1, 1], [2, 1]], [[2, 0], [1, 1], [2, 1], [1, 2]], [[0, 1], [1, 1], [1, 2], [2, 2]], [[1, 0], [0, 1], [1, 1], [0, 2]]],
    }

    COLORS = {
        'T': (178, 102, 255), 'I': (102, 204, 255), 'O': (255, 255, 102),
        'L': (255, 153, 51), 'J': (51, 102, 255), 'S': (102, 255, 102),
        'Z': (255, 102, 102)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 50)

        # Calculate grid rendering properties
        self.cell_size = 18
        self.grid_render_width = self.GRID_WIDTH * self.cell_size
        self.grid_render_height = self.GRID_HEIGHT * self.cell_size
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_render_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_render_height) // 2
        
        # This is here to allow the environment to be instantiated without calling reset,
        # as per the Gymnasium API. State will be properly initialized in reset().
        self.grid = []
        self.current_piece = None
        self.next_piece_shape = 'I' # Dummy value
        self.score = 0
        self.steps = 0
        self.stage = 1
        self.total_lines_cleared = 0
        self.game_over = False
        self.win = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = [[self.COLOR_BG for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.stage = 1
        self.lines_cleared_in_stage = 0
        self.total_lines_cleared = 0
        self._update_drop_speed()

        self.drop_counter = 0.0
        self.piece_bag = []
        self._refill_bag()

        self.current_piece = None
        self.next_piece_shape = self.piece_bag.pop()
        self._spawn_piece()

        self.last_shift_state = False
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self.steps += 1
        reward = 0
        piece_locked_this_frame = False

        # --- Action Handling ---
        if shift_held and not self.last_shift_state:
            # Hard drop on press, not hold
            reward += self._hard_drop()  # This locks the piece, current_piece becomes None
            piece_locked_this_frame = True
        self.last_shift_state = shift_held

        if not piece_locked_this_frame and self.current_piece:
            # Movement: 1=Up(Rot L), 2=Down(Rot R), 3=Left, 4=Right
            if movement == 1: self._move(0, 0, -1)  # Rotate Left
            elif movement == 2: self._move(0, 0, 1)  # Rotate Right
            elif movement == 3: self._move(-1, 0, 0)  # Move Left
            elif movement == 4: self._move(1, 0, 0)  # Move Right

            # Gravity
            current_drop_speed = self.drop_speed * 5.0 if space_held else self.drop_speed
            self.drop_counter += current_drop_speed
            if self.drop_counter >= 1.0:
                moves = int(self.drop_counter)
                moved_count = 0
                for _ in range(moves):
                    if self._move(0, 1, 0):
                        moved_count += 1
                    else:
                        break  # Stop if we hit something
                self.drop_counter -= moved_count

            # Check if piece has landed after movement/gravity
            if self._check_collision(self.current_piece, 0, 1):
                reward += self._lock_piece()  # Locks the piece, current_piece becomes None
                piece_locked_this_frame = True

        # --- Post-Lock Game Logic ---
        if piece_locked_this_frame:
            # Sound effect placeholder: # sfx_lock.play()
            lines_cleared = self._clear_lines()
            if lines_cleared > 0:
                # Sound effect placeholder: # sfx_clear.play()

                # Line clear rewards
                if lines_cleared == 1: reward += 1
                elif lines_cleared == 2: reward += 3
                elif lines_cleared == 3: reward += 5
                elif lines_cleared >= 4: reward += 10

                self._update_score_and_stage(lines_cleared)

                # Stage clear reward
                if self.lines_cleared_in_stage == 0 and self.stage > 1:
                    reward += 10

            self._spawn_piece()
            if self.game_over:
                reward -= 100

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1

        # Check for win/loss
        if self.stage > 3:
            self.win = True
            reward += 100

        terminated = self.game_over or self.win
        truncated = self.steps >= 2000
        
        if terminated or truncated:
            # Ensure a piece isn't mid-air on the final frame
            if self.current_piece:
                reward += self._lock_piece()
                self.current_piece = None

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Helper Methods ---

    def _update_drop_speed(self):
        base_speed = [1.0, 1.2, 1.4][min(self.stage - 1, 2)]
        self.drop_speed = (base_speed + self.lines_cleared_in_stage * 0.05) / 30.0  # units per frame

    def _refill_bag(self):
        self.piece_bag = list(self.SHAPES.keys())
        self.np_random.shuffle(self.piece_bag)

    def _spawn_piece(self):
        if not self.piece_bag:
            self._refill_bag()
        shape_key = self.next_piece_shape

        self.current_piece = {
            'shape': shape_key,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - 2,
            'y': 0,
            'color': self.COLORS[shape_key]
        }

        if self._check_collision(self.current_piece, 0, 0):
            self.game_over = True
            self.current_piece = None # Can't place the piece

        if not self.piece_bag:
            self._refill_bag()
        self.next_piece_shape = self.piece_bag.pop()

    def _check_collision(self, piece, dx, dy, rot=None):
        if piece is None:
            return True
        _rotation = piece['rotation'] if rot is None else rot
        shape_coords = self.SHAPES[piece['shape']][_rotation % len(self.SHAPES[piece['shape']])]

        for x, y in shape_coords:
            nx, ny = piece['x'] + x + dx, piece['y'] + y + dy
            if not (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT):
                return True  # Out of bounds
            if self.grid[ny][nx] != self.COLOR_BG:
                return True  # Collides with existing block
        return False

    def _move(self, dx, dy, drot):
        if self.current_piece is None:
            return False
        new_rot = (self.current_piece['rotation'] + drot) % len(self.SHAPES[self.current_piece['shape']])
        if not self._check_collision(self.current_piece, dx, dy, rot=new_rot):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            self.current_piece['rotation'] = new_rot
            return True
        return False

    def _hard_drop(self):
        # Sound effect placeholder: # sfx_hard_drop.play()
        if self.current_piece is None: return 0
        while self._move(0, 1, 0):
            pass  # Move down until collision
        return self._lock_piece()

    def _lock_piece(self):
        if self.current_piece is None: return 0
        shape_coords = self.SHAPES[self.current_piece['shape']][self.current_piece['rotation']]
        reward = 0.1  # Base reward for placement

        # Risky placement check & column height penalty
        total_filled_cells = 0
        for x, y in shape_coords:
            px, py = self.current_piece['x'] + x, self.current_piece['y'] + y
            if 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT:
                self.grid[py][px] = self.current_piece['color']
                # Check for hole creation (risky placement)
                if py + 1 < self.GRID_HEIGHT and self.grid[py + 1][px] == self.COLOR_BG:
                    is_hole = True
                    for ox, oy in shape_coords:  # Check if the hole is filled by another part of the same piece
                        if self.current_piece['x'] + ox == px and self.current_piece['y'] + oy == py + 1:
                            is_hole = False
                            break
                    if is_hole:
                        reward -= 2.0

        # Horizontal filled cells penalty
        for row in self.grid:
            total_filled_cells += sum(1 for cell in row if cell != self.COLOR_BG)
        reward -= total_filled_cells / (self.GRID_WIDTH * self.GRID_HEIGHT) * 0.1

        self.current_piece = None
        return reward

    def _clear_lines(self):
        lines_to_clear_indices = [i for i, row in enumerate(self.grid) if all(cell != self.COLOR_BG for cell in row)]

        if not lines_to_clear_indices:
            return 0
        
        # Sort indices in reverse to avoid messing up subsequent pops
        lines_to_clear_indices.sort(reverse=True)

        for y in lines_to_clear_indices:
            for x in range(self.GRID_WIDTH):
                # Create particles for visual effect
                for _ in range(3):
                    self.particles.append({
                        'x': self.grid_offset_x + x * self.cell_size + self.cell_size / 2,
                        'y': self.grid_offset_y + y * self.cell_size + self.cell_size / 2,
                        'vx': self.np_random.uniform(-2, 2),
                        'vy': self.np_random.uniform(-2, 2),
                        'life': 20,
                        'color': (255, 255, 255)
                    })
            self.grid.pop(y)
            self.grid.insert(0, [self.COLOR_BG for _ in range(self.GRID_WIDTH)])

        return len(lines_to_clear_indices)

    def _update_score_and_stage(self, num_cleared):
        score_map = {1: 100, 2: 300, 3: 500, 4: 800}
        self.score += score_map.get(num_cleared, 0) * self.stage
        self.lines_cleared_in_stage += num_cleared
        self.total_lines_cleared += num_cleared

        if self.lines_cleared_in_stage >= 10:
            self.stage += 1
            self.lines_cleared_in_stage = 0
            if self.stage <= 3:
                # Sound effect placeholder: # sfx_stage_up.play()
                self._update_drop_speed()

        self._update_drop_speed()

    def _get_ghost_piece_y(self):
        if self.current_piece is None: return 0
        y = self.current_piece['y']
        while not self._check_collision(self.current_piece, 0, y - self.current_piece['y'] + 1):
            y += 1
        return y

    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = (self.grid_offset_x + x * self.cell_size, self.grid_offset_y + y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.grid[y][x], rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw danger zone
        danger_surface = pygame.Surface((self.grid_render_width, 4 * self.cell_size), pygame.SRCALPHA)
        danger_surface.fill(self.COLOR_DANGER)
        self.screen.blit(danger_surface, (self.grid_offset_x, self.grid_offset_y))

        # Draw ghost piece
        if self.current_piece and not self.game_over:
            ghost_y = self._get_ghost_piece_y()
            ghost_piece = self.current_piece.copy()
            ghost_piece['y'] = ghost_y
            self._draw_piece(self.screen, ghost_piece, self.grid_offset_x, self.grid_offset_y, self.cell_size, ghost=True)

        # Draw current piece
        if self.current_piece and not self.game_over:
            self._draw_piece(self.screen, self.current_piece, self.grid_offset_x, self.grid_offset_y, self.cell_size)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), 3, (*p['color'], alpha))

    def _draw_piece(self, surface, piece, offset_x, offset_y, cell_size, ghost=False):
        shape_coords = self.SHAPES[piece['shape']][piece['rotation']]
        for x, y in shape_coords:
            px, py = piece['x'] + x, piece['y'] + y
            color = piece['color']
            if ghost:
                self._draw_block(surface, color, px, py, cell_size, offset_x, offset_y, ghost=True)
            else:
                self._draw_block(surface, color, px, py, cell_size, offset_x, offset_y)

    def _draw_block(self, surface, color, grid_x, grid_y, cell_size, offset_x, offset_y, ghost=False):
        rect = pygame.Rect(offset_x + grid_x * cell_size, offset_y + grid_y * cell_size, cell_size, cell_size)
        if ghost:
            pygame.draw.rect(surface, self.COLOR_GHOST, rect, 2, border_radius=3)
        else:
            light_color = tuple(min(255, c + 50) for c in color)
            dark_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.rect(surface, color, rect, border_radius=3)
            pygame.draw.rect(surface, light_color, rect.inflate(-cell_size * 0.6, -cell_size * 0.6), border_radius=2)
            pygame.draw.rect(surface, dark_color, rect, 2, border_radius=3)

    def _render_ui(self):
        # UI panel on the right
        ui_x = self.grid_offset_x + self.grid_render_width + 20

        # Score
        score_text = self.font_main.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_x, self.grid_offset_y))
        score_val = self.font_title.render(f"{self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_val, (ui_x, self.grid_offset_y + 30))

        # Lines
        lines_text = self.font_main.render("LINES", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (ui_x, self.grid_offset_y + 90))
        lines_val = self.font_small.render(f"{self.total_lines_cleared}", True, self.COLOR_TEXT)
        self.screen.blit(lines_val, (ui_x, self.grid_offset_y + 120))

        # Stage
        stage_text = self.font_main.render("STAGE", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (ui_x, self.grid_offset_y + 160))
        stage_val = self.font_small.render(f"{self.stage} / 3", True, self.COLOR_TEXT)
        self.screen.blit(stage_val, (ui_x, self.grid_offset_y + 190))

        # Next Piece
        next_text = self.font_main.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (ui_x, self.grid_offset_y + 230))
        next_piece_preview = {
            'shape': self.next_piece_shape, 'rotation': 0, 'x': 0, 'y': 0,
            'color': self.COLORS[self.next_piece_shape]
        }
        self._draw_piece(self.screen, next_piece_preview, ui_x, self.grid_offset_y + 260, self.cell_size)

        # Game Over / Win message
        if self.game_over:
            self._draw_overlay_message("GAME OVER")
        elif self.win:
            self._draw_overlay_message("YOU WIN!", self.COLOR_SCORE)

    def _draw_overlay_message(self, text, color=COLOR_TEXT):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        msg_render = self.font_title.render(text, True, color)
        msg_rect = msg_render.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
        self.screen.blit(msg_render, msg_rect)

    # --- Gymnasium Interface Methods ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "lines_cleared": self.total_lines_cleared
        }

    def close(self):
        pygame.quit()