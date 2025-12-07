
# Generated: 2025-08-27T12:45:23.064206
# Source Brief: brief_00150.md
# Brief Index: 150

        
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
        "Controls: ←→ to move, ↓ for soft drop, ↑ to do nothing. "
        "Space for hard drop. Shift to swap with next piece."
    )

    game_description = (
        "A fast-paced puzzle game. Manipulate falling colored squares to create "
        "and clear full rows of a single color. Clear 50 rows to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.BOARD_X_OFFSET = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.BOARD_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        self.MAX_STEPS = 2000
        self.WIN_CONDITION_ROWS = 50
        self.FPS = 30

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_BOARD_BG = (30, 35, 50)
        self.COLOR_TEXT = (220, 230, 255)
        self.COLOR_TEXT_SHADOW = (10, 15, 20)
        self.PIECE_COLORS = [
            (0, 0, 0),  # 0 is empty
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 80, 255),   # Blue
            (255, 255, 80),  # Yellow
            (80, 255, 255),  # Cyan
            (255, 80, 255),  # Magenta
            (255, 160, 80),  # Orange
            (160, 80, 255),  # Purple
            (200, 200, 200), # White
            (160, 255, 160)  # Light Green
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.board = None
        self.active_piece = None
        self.next_piece_color_idx = None
        self.score = None
        self.rows_cleared = None
        self.steps = None
        self.game_over = None
        self.fall_counter = None
        self.drop_speed = None
        self.particles = None
        self.row_flash_timers = None
        self.cleared_rows_this_step = 0
        
        # --- Action State ---
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.rows_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_counter = 0.0
        self.drop_speed = 1.0  # cells per second
        self.particles = []
        self.row_flash_timers = {}
        self.cleared_rows_this_step = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self._spawn_piece()  # Sets next piece
        self._spawn_piece()  # Sets active piece and new next piece

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.cleared_rows_this_step = 0
        
        is_soft_dropping = self._handle_input(action)
        self._update_physics(is_soft_dropping)

        reward = self._calculate_reward(action)
        terminated = self._check_termination()

        if terminated and not self.game_over: # Win condition met
            self.game_over = True
            reward = 100.0
        elif self.game_over: # Loss condition met
            reward = -100.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if self.active_piece:
            if movement == 3: # Left
                self._move(-1)
            elif movement == 4: # Right
                self._move(1)

            if space_held and not self.prev_space_held:
                # sfx: hard_drop_sound()
                self._hard_drop()
            
            if shift_held and not self.prev_shift_held:
                # sfx: swap_sound()
                self._swap_piece()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return movement == 2

    def _update_physics(self, is_soft_dropping):
        if not self.active_piece:
            return

        fall_increment = self.drop_speed / self.FPS
        if is_soft_dropping:
            fall_increment *= 10.0  # Soft drop is much faster

        self.fall_counter += fall_increment
        
        if self.fall_counter >= 1.0:
            moves_to_make = int(self.fall_counter)
            self.fall_counter -= moves_to_make
            for _ in range(moves_to_make):
                if not self.active_piece or not self._drop_piece_one_step():
                    break

    def _drop_piece_one_step(self):
        new_pos = (self.active_piece['pos'][0], self.active_piece['pos'][1] + 1)
        if self._is_valid_pos(new_pos):
            self.active_piece['pos'] = new_pos
            return True
        else:
            self._place_piece()
            return False

    def _calculate_reward(self, action):
        reward = 0
        movement = action[0]

        if self.cleared_rows_this_step > 0:
            reward += 10 * self.cleared_rows_this_step

        if movement in [3, 4] and self.active_piece: # Left or Right move
            px, py = self.active_piece['pos']
            color_idx = self.active_piece['color_idx']
            is_advantageous = False
            for y in range(py + 1, self.GRID_HEIGHT):
                if self.board[y, px] != 0:
                    if self.board[y, px] == color_idx:
                        is_advantageous = True
                    break
            if is_advantageous:
                reward += 0.1
            else:
                reward -= 0.2
        
        return reward

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS or self.rows_cleared >= self.WIN_CONDITION_ROWS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "rows_cleared": self.rows_cleared,
            "steps": self.steps,
        }

    # --- Game Logic Helpers ---

    def _spawn_piece(self):
        self.active_piece = {
            'pos': (self.GRID_WIDTH // 2 - 1, 0),
            'color_idx': self.next_piece_color_idx
        }
        self.next_piece_color_idx = self.np_random.integers(1, len(self.PIECE_COLORS))
        if not self._is_valid_pos(self.active_piece['pos']):
            self.game_over = True
            self.active_piece = None

    def _move(self, dx):
        if not self.active_piece: return
        new_pos = (self.active_piece['pos'][0] + dx, self.active_piece['pos'][1])
        if self._is_valid_pos(new_pos):
            self.active_piece['pos'] = new_pos

    def _hard_drop(self):
        if not self.active_piece: return
        while self._drop_piece_one_step():
            self.score += 1 # Small bonus for hard dropping
        self.fall_counter = 0

    def _swap_piece(self):
        if not self.active_piece: return
        original_pos = self.active_piece['pos']
        self.active_piece['color_idx'], self.next_piece_color_idx = \
            self.next_piece_color_idx, self.active_piece['color_idx']
        if not self._is_valid_pos(original_pos):
            # If swap results in invalid position, swap back
            self.active_piece['color_idx'], self.next_piece_color_idx = \
                self.next_piece_color_idx, self.active_piece['color_idx']

    def _is_valid_pos(self, pos):
        px, py = pos
        if not (0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT):
            return False
        if self.board[py, px] != 0:
            return False
        return True

    def _place_piece(self):
        if not self.active_piece: return
        px, py = self.active_piece['pos']
        color_idx = self.active_piece['color_idx']
        self.board[py, px] = color_idx
        self.active_piece = None
        # sfx: place_piece_sound()

        cleared_count = self._check_and_clear_rows()
        if cleared_count > 0:
            self.cleared_rows_this_step = cleared_count
            self.score += (100 * cleared_count) * cleared_count # Combo bonus
            self.rows_cleared += cleared_count
            self.drop_speed = 1.0 + (self.rows_cleared // 5) * 0.05
            # sfx: row_clear_sound()

        self._spawn_piece()

    def _check_and_clear_rows(self):
        rows_to_clear = []
        for r in range(self.GRID_HEIGHT):
            first_color = self.board[r, 0]
            if first_color == 0: continue
            
            is_uniform_and_full = all(self.board[r, c] == first_color for c in range(self.GRID_WIDTH))
            if is_uniform_and_full:
                rows_to_clear.append(r)

        if rows_to_clear:
            for r_idx in rows_to_clear:
                self.row_flash_timers[r_idx] = self.FPS // 3
                color = self.PIECE_COLORS[self.board[r_idx, 0]]
                for _ in range(40):
                    self._create_particle(r_idx, color)

            # Rebuild board by dropping rows above cleared ones
            new_board = np.zeros_like(self.board)
            new_row_idx = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if r not in rows_to_clear:
                    new_board[new_row_idx] = self.board[r]
                    new_row_idx -= 1
            self.board = new_board

        return len(rows_to_clear)

    # --- Rendering Helpers ---

    def _render_game(self):
        # Draw board background and border
        board_rect = pygame.Rect(self.BOARD_X_OFFSET, self.BOARD_Y_OFFSET,
                                 self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BOARD_BG, board_rect)
        
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.BOARD_X_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.BOARD_Y_OFFSET), (x, self.BOARD_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.BOARD_Y_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_X_OFFSET, y), (self.BOARD_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, y))

        # Draw placed pieces
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_idx = self.board[r, c]
                if color_idx != 0:
                    self._draw_cell(c, r, self.PIECE_COLORS[color_idx])

        # Draw row clear flash
        flashing_rows = list(self.row_flash_timers.keys())
        for r_idx in flashing_rows:
            self.row_flash_timers[r_idx] -= 1
            if self.row_flash_timers[r_idx] <= 0:
                del self.row_flash_timers[r_idx]
            else:
                flash_alpha = 150 * (self.row_flash_timers[r_idx] / (self.FPS // 3))
                flash_surface = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                flash_surface.fill((255, 255, 255, flash_alpha))
                self.screen.blit(flash_surface, (self.BOARD_X_OFFSET, self.BOARD_Y_OFFSET + r_idx * self.CELL_SIZE))

        # Draw active piece with glow
        if self.active_piece:
            px, py = self.active_piece['pos']
            color = self.PIECE_COLORS[self.active_piece['color_idx']]
            
            # Glow effect
            glow_color = (*color, 70) # color with alpha
            glow_size = int(self.CELL_SIZE * 1.5)
            glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, glow_color, (glow_size // 2, glow_size // 2), glow_size // 2)
            
            glow_x = self.BOARD_X_OFFSET + px * self.CELL_SIZE + (self.CELL_SIZE - glow_size) // 2
            glow_y = self.BOARD_Y_OFFSET + py * self.CELL_SIZE + (self.CELL_SIZE - glow_size) // 2
            self.screen.blit(glow_surface, (glow_x, glow_y))
            
            self._draw_cell(px, py, color)

        # Draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # Score display
        self._draw_text(f"SCORE: {self.score}", (20, 20), self.font_medium, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        self._draw_text(f"ROWS: {self.rows_cleared} / {self.WIN_CONDITION_ROWS}", (20, 50), self.font_medium, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Next piece preview
        self._draw_text("NEXT", (self.WIDTH - 100, 20), self.font_medium, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, align="center")
        preview_bg_rect = pygame.Rect(self.WIDTH - 120, 50, 80, 80)
        pygame.draw.rect(self.screen, self.COLOR_BOARD_BG, preview_bg_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_bg_rect, width=1, border_radius=5)
        
        if self.next_piece_color_idx:
            color = self.PIECE_COLORS[self.next_piece_color_idx]
            cell_size = self.CELL_SIZE * 2
            cell_x = preview_bg_rect.centerx - cell_size // 2
            cell_y = preview_bg_rect.centery - cell_size // 2
            
            rect = pygame.Rect(cell_x, cell_y, cell_size, cell_size)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in color), rect.inflate(-6, -6), border_radius=3)


        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            message = "YOU WIN!" if self.rows_cleared >= self.WIN_CONDITION_ROWS else "GAME OVER"
            self._draw_text(message, (self.WIDTH // 2, self.HEIGHT // 2 - 30), self.font_large, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, align="center")
            self._draw_text("Reset to play again", (self.WIDTH // 2, self.HEIGHT // 2 + 20), self.font_medium, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, align="center")

    def _draw_cell(self, grid_x, grid_y, color):
        x = self.BOARD_X_OFFSET + grid_x * self.CELL_SIZE
        y = self.BOARD_Y_OFFSET + grid_y * self.CELL_SIZE
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        # Main color
        pygame.draw.rect(self.screen, color, rect, border_radius=2)
        
        # 3D-ish highlight/shadow
        highlight_color = tuple(min(255, c + 40) for c in color)
        shadow_color = tuple(max(0, c - 40) for c in color)
        pygame.draw.line(self.screen, highlight_color, (x+1, y+1), (x + self.CELL_SIZE - 2, y+1))
        pygame.draw.line(self.screen, highlight_color, (x+1, y+1), (x+1, y + self.CELL_SIZE - 2))
        pygame.draw.line(self.screen, shadow_color, (x + self.CELL_SIZE - 2, y+2), (x + self.CELL_SIZE - 2, y + self.CELL_SIZE - 2))
        pygame.draw.line(self.screen, shadow_color, (x+2, y + self.CELL_SIZE - 2), (x + self.CELL_SIZE - 2, y + self.CELL_SIZE - 2))

    def _draw_text(self, text, pos, font, color, shadow_color, align="left"):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if align == "center":
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _create_particle(self, row_idx, color):
        px = self.BOARD_X_OFFSET + self.np_random.uniform(0, self.GRID_WIDTH * self.CELL_SIZE)
        py = self.BOARD_Y_OFFSET + row_idx * self.CELL_SIZE + self.np_random.uniform(0, self.CELL_SIZE)
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        lifespan = self.np_random.integers(self.FPS // 2, self.FPS)
        size = self.np_random.uniform(2, 5)
        self.particles.append({'pos': [px, py], 'vel': [vx, vy], 'life': lifespan, 'max_life': lifespan, 'color': color, 'size': size})

    def _update_and_draw_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1

            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            
            alpha = int(255 * (p['life'] / p['max_life']))
            color_with_alpha = (*p['color'], alpha)
            
            # Use a small surface for each particle to handle alpha correctly
            particle_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color_with_alpha, (p['size'], p['size']), p['size'])
            self.screen.blit(particle_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

    def validate_implementation(self):
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

    def close(self):
        pygame.quit()