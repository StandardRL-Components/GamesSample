
# Generated: 2025-08-28T05:33:31.887349
# Source Brief: brief_02657.md
# Brief Index: 2657

        
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
        "Controls: ←→ to move, ↑ to rotate, ↓ for soft drop. Space for hard drop, Shift to hold a piece."
    )

    game_description = (
        "A fast-paced, falling block puzzle game. Place pieces to clear lines. Clear 10 lines to win. "
        "Get bonus rewards for clearing multiple lines at once or for 'risky' placements with no gaps underneath."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.BLOCK_SIZE = 18
        self.SIDE_PANEL_WIDTH = 150

        # Centered playfield
        self.PLAYFIELD_W = self.GRID_WIDTH * self.BLOCK_SIZE
        self.PLAYFIELD_H = self.GRID_HEIGHT * self.BLOCK_SIZE
        self.PLAYFIELD_X = (self.WIDTH - self.PLAYFIELD_W) / 2
        self.PLAYFIELD_Y = (self.HEIGHT - self.PLAYFIELD_H) / 2

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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # --- Assets & Colors ---
        self._define_assets()

        # --- State Variables ---
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.held_piece = None
        self.can_hold = True
        self.piece_bag = []
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_counter = 0
        self.fall_speed = 20  # Lower is faster
        self.lock_delay = 15 # Frames to lock after landing
        self.lock_counter = 0
        self.particles = []

        self.reset()
        self.validate_implementation()

    def _define_assets(self):
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PANEL = (25, 25, 35)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_GHOST = (255, 255, 255, 50)
        
        self.PIECE_COLORS = [
            (0, 0, 0),          # 0: Empty
            (0, 240, 240),      # 1: I (Cyan)
            (240, 240, 0),      # 2: O (Yellow)
            (160, 0, 240),      # 3: T (Purple)
            (0, 240, 0),        # 4: S (Green)
            (240, 0, 0),        # 5: Z (Red)
            (0, 0, 240),        # 6: J (Blue)
            (240, 160, 0),      # 7: L (Orange)
        ]

        # Tetromino shapes (indices correspond to PIECE_COLORS)
        self.PIECE_SHAPES = [
            [], # Empty
            [[1, 1, 1, 1]], # I
            [[1, 1], [1, 1]], # O
            [[0, 1, 0], [1, 1, 1]], # T
            [[0, 1, 1], [1, 1, 0]], # S
            [[1, 1, 0], [0, 1, 1]], # Z
            [[1, 0, 0], [1, 1, 1]], # J
            [[0, 0, 1], [1, 1, 1]], # L
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.held_piece = None
        self.can_hold = True
        self.particles = []
        self._fill_piece_bag()
        self._new_piece()
        self._new_piece() # To populate current and next
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = self.game_over

        if not terminated:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            piece_placed = False
            placement_info = {}

            if shift_held and self.can_hold:
                self._hold_piece()
            elif space_held:
                placement_info = self._hard_drop()
                piece_placed = True
            else:
                if movement == 1: self._rotate_piece()
                elif movement == 3: self._move_piece(-1)
                elif movement == 4: self._move_piece(1)
                
                self._update_gravity(soft_drop=(movement == 2))
                
                locked, lock_info = self._check_and_lock_piece()
                if locked:
                    piece_placed = True
                    placement_info = lock_info

            if piece_placed:
                # sound_effect: 'lock_piece.wav'
                reward += self._calculate_placement_reward(placement_info)
                lines = self._clear_lines()
                if lines > 0:
                    # sound_effect: 'line_clear.wav'
                    self.score += [0, 100, 300, 500, 800][lines] * (self.lines_cleared // 10 + 1)
                    reward += lines # Add to the line clear reward from placement
                self.lines_cleared += lines
                self.can_hold = True
                self._new_piece() # This also handles game over check

        self._update_particles()

        if not terminated and self.game_over:
            reward = -100.0
            terminated = True
            # sound_effect: 'game_over.wav'
        elif not terminated and self.lines_cleared >= 10:
            reward = 100.0
            terminated = True
            self.game_over = True
            # sound_effect: 'win_game.wav'
        elif self.steps >= 1000:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _fill_piece_bag(self):
        self.piece_bag = list(range(1, len(self.PIECE_SHAPES)))
        self.np_random.shuffle(self.piece_bag)

    def _new_piece(self):
        if not self.piece_bag:
            self._fill_piece_bag()
        
        self.current_piece = self.next_piece
        self.next_piece = {
            "id": self.piece_bag.pop(0),
            "rotation": 0,
            "x": self.GRID_WIDTH // 2 - 1,
            "y": 0
        }

        if self.current_piece:
            self.current_piece["x"] = self.GRID_WIDTH // 2 - len(self._get_current_shape()[0]) // 2
            self.current_piece["y"] = 0
            self.fall_counter = 0
            self.lock_counter = 0
            if not self._is_valid_position():
                self.game_over = True

    def _get_current_shape(self, piece=None):
        if piece is None: piece = self.current_piece
        if not piece: return []
        
        shape = self.PIECE_SHAPES[piece["id"]]
        for _ in range(piece["rotation"] % 4):
            shape = list(zip(*shape[::-1]))
        return shape

    def _is_valid_position(self, piece_offset_x=0, piece_offset_y=0, rotation_offset=0):
        if not self.current_piece: return False
        
        temp_piece = self.current_piece.copy()
        temp_piece["x"] += piece_offset_x
        temp_piece["y"] += piece_offset_y
        temp_piece["rotation"] += rotation_offset
        
        shape = self._get_current_shape(temp_piece)
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = temp_piece["x"] + c, temp_piece["y"] + r
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT and self.grid[grid_y, grid_x] == 0):
                        return False
        return True

    def _move_piece(self, dx):
        if self._is_valid_position(piece_offset_x=dx):
            self.current_piece["x"] += dx
            self.lock_counter = 0 # Reset lock on successful movement

    def _rotate_piece(self):
        if not self.current_piece: return
        
        original_rotation = self.current_piece["rotation"]
        self.current_piece["rotation"] = (self.current_piece["rotation"] + 1) % 4
        
        # Wall kick checks
        offsets = [0, -1, 1, -2, 2] # 0: no kick, -1: kick left, etc.
        for offset in offsets:
            if self._is_valid_position(piece_offset_x=offset):
                self.current_piece["x"] += offset
                self.lock_counter = 0 # Reset lock on successful rotation
                return
        
        # If no rotation is valid, revert
        self.current_piece["rotation"] = original_rotation

    def _hold_piece(self):
        self.can_hold = False
        if self.held_piece is None:
            self.held_piece = {"id": self.current_piece["id"], "rotation": 0}
            self._new_piece()
        else:
            self.current_piece, self.held_piece = self.held_piece, {"id": self.current_piece["id"], "rotation": 0}
            self.current_piece["x"] = self.GRID_WIDTH // 2 - len(self._get_current_shape()[0]) // 2
            self.current_piece["y"] = 0
            if not self._is_valid_position():
                self.game_over = True
        self.fall_counter = 0
        self.lock_counter = 0

    def _hard_drop(self):
        if not self.current_piece: return {}
        
        dy = 0
        while self._is_valid_position(piece_offset_y=dy + 1):
            dy += 1
        self.current_piece["y"] += dy
        return self._lock_piece()

    def _update_gravity(self, soft_drop):
        if not self.current_piece: return
        
        speed_multiplier = 5 if soft_drop else 1
        self.fall_counter += speed_multiplier
        if self.fall_counter >= self.fall_speed:
            self.fall_counter = 0
            if self._is_valid_position(piece_offset_y=1):
                self.current_piece["y"] += 1
            
    def _check_and_lock_piece(self):
        if not self.current_piece: return False, {}
        
        if not self._is_valid_position(piece_offset_y=1):
            self.lock_counter += 1
            if self.lock_counter >= self.lock_delay:
                return True, self._lock_piece()
        else:
            self.lock_counter = 0
        return False, {}

    def _lock_piece(self):
        if not self.current_piece: return {}

        shape = self._get_current_shape()
        is_risky = self._check_risky_placement()
        
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = self.current_piece["x"] + c, self.current_piece["y"] + r
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y, grid_x] = self.current_piece["id"]
        
        self.current_piece = None
        return {"is_risky": is_risky}

    def _check_risky_placement(self):
        if not self.current_piece: return False

        shape = self._get_current_shape()
        piece_cols = set()
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    piece_cols.add(self.current_piece["x"] + c)

        for col in piece_cols:
            lowest_y_in_col = -1
            for r, row in enumerate(shape):
                if 0 <= col - self.current_piece["x"] < len(row) and row[col - self.current_piece["x"]]:
                    lowest_y_in_col = max(lowest_y_in_col, self.current_piece["y"] + r)
            
            if lowest_y_in_col < self.GRID_HEIGHT - 1:
                if self.grid[lowest_y_in_col + 1, col] == 0:
                    return False # Found a gap
        return True

    def _calculate_placement_reward(self, placement_info):
        if placement_info.get("is_risky", False):
            return 2.0
        # Line clear reward is handled separately after clearing
        return -0.2

    def _clear_lines(self):
        lines_to_clear = [r for r, row in enumerate(self.grid) if np.all(row > 0)]
        if not lines_to_clear:
            return 0
        
        for r in lines_to_clear:
            self._spawn_line_clear_particles(r)

        rows_to_keep = np.array([row for r, row in enumerate(self.grid) if r not in lines_to_clear])
        num_cleared = len(lines_to_clear)
        new_grid = np.zeros_like(self.grid)
        if rows_to_keep.shape[0] > 0:
            new_grid[num_cleared:] = rows_to_keep
        self.grid = new_grid
        return num_cleared

    def _spawn_line_clear_particles(self, row_index):
        y = self.PLAYFIELD_Y + row_index * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        for i in range(self.GRID_WIDTH * 3):
            x = self.PLAYFIELD_X + (i / 3) * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.np_random.integers(20, 40)
            color = self.PIECE_COLORS[self.np_random.integers(1, len(self.PIECE_COLORS))]
            self.particles.append([x, y, vx, vy, life, color])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[1] # x += vx
            p[1] += p[2] # y += vy
            p[2] += 0.1  # gravity on vy
            p[4] -= 1    # life -= 1
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw playfield border and background
        pygame.draw.rect(self.screen, self.COLOR_PANEL, (self.PLAYFIELD_X - 5, self.PLAYFIELD_Y - 5, self.PLAYFIELD_W + 10, self.PLAYFIELD_H + 10))
        pygame.draw.rect(self.screen, self.COLOR_BG, (self.PLAYFIELD_X, self.PLAYFIELD_Y, self.PLAYFIELD_W, self.PLAYFIELD_H))

        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.PLAYFIELD_X + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.PLAYFIELD_Y), (x, self.PLAYFIELD_Y + self.PLAYFIELD_H))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.PLAYFIELD_Y + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAYFIELD_X, y), (self.PLAYFIELD_X + self.PLAYFIELD_W, y))

        # Draw placed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] > 0:
                    self._draw_block(c, r, self.grid[r, c])

        # Draw ghost and current piece
        if self.current_piece and not self.game_over:
            # Ghost piece
            dy = 0
            while self._is_valid_position(piece_offset_y=dy + 1):
                dy += 1
            ghost_y = self.current_piece["y"] + dy
            shape = self._get_current_shape()
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.current_piece["x"] + c, ghost_y + r, self.current_piece["id"], is_ghost=True)
            
            # Current piece
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.current_piece["x"] + c, self.current_piece["y"] + r, self.current_piece["id"])

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), int(p[4] / 10 + 1))

    def _draw_block(self, grid_c, grid_r, piece_id, is_ghost=False):
        x = self.PLAYFIELD_X + grid_c * self.BLOCK_SIZE
        y = self.PLAYFIELD_Y + grid_r * self.BLOCK_SIZE
        color = self.PIECE_COLORS[piece_id]
        
        if is_ghost:
            pygame.gfxdraw.rectangle(self.screen, (x, y, self.BLOCK_SIZE, self.BLOCK_SIZE), (*color, 80))
        else:
            # Main block color
            pygame.draw.rect(self.screen, color, (x, y, self.BLOCK_SIZE, self.BLOCK_SIZE))
            # 3D effect
            highlight = tuple(min(255, c + 40) for c in color)
            shadow = tuple(max(0, c - 40) for c in color)
            pygame.draw.line(self.screen, highlight, (x, y), (x + self.BLOCK_SIZE - 1, y), 2)
            pygame.draw.line(self.screen, highlight, (x, y), (x, y + self.BLOCK_SIZE - 1), 2)
            pygame.draw.line(self.screen, shadow, (x + self.BLOCK_SIZE - 1, y), (x + self.BLOCK_SIZE - 1, y + self.BLOCK_SIZE - 1), 2)
            pygame.draw.line(self.screen, shadow, (x, y + self.BLOCK_SIZE - 1), (x + self.BLOCK_SIZE - 1, y + self.BLOCK_SIZE - 1), 2)

    def _render_ui(self):
        # --- Left Panel (Hold) ---
        pygame.draw.rect(self.screen, self.COLOR_PANEL, (20, 20, self.SIDE_PANEL_WIDTH, 120))
        text = self.font_small.render("HOLD", True, self.COLOR_TEXT)
        self.screen.blit(text, (20 + (self.SIDE_PANEL_WIDTH - text.get_width()) / 2, 30))
        if self.held_piece:
            self._draw_ui_piece(self.held_piece, 20 + self.SIDE_PANEL_WIDTH / 2, 85)

        # --- Right Panel (Next & Score) ---
        right_panel_x = self.WIDTH - self.SIDE_PANEL_WIDTH - 20
        pygame.draw.rect(self.screen, self.COLOR_PANEL, (right_panel_x, 20, self.SIDE_PANEL_WIDTH, self.HEIGHT - 40))
        
        # Next piece
        text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(text, (right_panel_x + (self.SIDE_PANEL_WIDTH - text.get_width()) / 2, 30))
        if self.next_piece:
            self._draw_ui_piece(self.next_piece, right_panel_x + self.SIDE_PANEL_WIDTH / 2, 85)
            
        # Score
        text = self.font_small.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(text, (right_panel_x + (self.SIDE_PANEL_WIDTH - text.get_width()) / 2, 160))
        score_text = self.font_large.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (right_panel_x + (self.SIDE_PANEL_WIDTH - score_text.get_width()) / 2, 180))

        # Lines
        text = self.font_small.render("LINES", True, self.COLOR_TEXT)
        self.screen.blit(text, (right_panel_x + (self.SIDE_PANEL_WIDTH - text.get_width()) / 2, 240))
        lines_text = self.font_large.render(f"{self.lines_cleared:02d} / 10", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (right_panel_x + (self.SIDE_PANEL_WIDTH - lines_text.get_width()) / 2, 260))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.lines_cleared >= 10 else "GAME OVER"
            text = self.font_large.render(msg, True, (255, 255, 255))
            self.screen.blit(text, ((self.WIDTH - text.get_width()) / 2, (self.HEIGHT - text.get_height()) / 2))

    def _draw_ui_piece(self, piece, center_x, center_y):
        shape = self._get_current_shape(piece)
        w = len(shape[0]) * self.BLOCK_SIZE
        h = len(shape) * self.BLOCK_SIZE
        start_x = center_x - w / 2
        start_y = center_y - h / 2
        
        color = self.PIECE_COLORS[piece["id"]]
        highlight = tuple(min(255, c + 40) for c in color)
        shadow = tuple(max(0, c - 40) for c in color)
        
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    x = start_x + c * self.BLOCK_SIZE
                    y = start_y + r * self.BLOCK_SIZE
                    pygame.draw.rect(self.screen, color, (x, y, self.BLOCK_SIZE, self.BLOCK_SIZE))
                    pygame.draw.line(self.screen, highlight, (x, y), (x + self.BLOCK_SIZE - 1, y), 2)
                    pygame.draw.line(self.screen, highlight, (x, y), (x, y + self.BLOCK_SIZE - 1), 2)
                    pygame.draw.line(self.screen, shadow, (x + self.BLOCK_SIZE - 1, y), (x + self.BLOCK_SIZE - 1, y + self.BLOCK_SIZE - 1), 2)
                    pygame.draw.line(self.screen, shadow, (x, y + self.BLOCK_SIZE - 1), (x + self.BLOCK_SIZE - 1, y + self.BLOCK_SIZE - 1), 2)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
        }

    def close(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for display
    pygame.display.set_caption("Gymnasium Tetris")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Action state
    movement = 0 # 0: none, 1: up, 2: down, 3: left, 4: right
    space_held = 0
    shift_held = 0
    
    # Key repeat handling for smooth movement
    key_last_press_time = {'left': 0, 'right': 0, 'down': 0}
    KEY_REPEAT_DELAY = 120 # ms
    KEY_REPEAT_INTERVAL = 40 # ms
    
    while not terminated:
        # --- Event Handling ---
        movement_this_frame = 0
        space_this_frame = 0
        shift_this_frame = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement_this_frame = 1 # Rotate
                if event.key == pygame.K_SPACE:
                    space_this_frame = 1 # Hard drop
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_this_frame = 1 # Hold
                if event.key == pygame.K_ESCAPE:
                    terminated = True
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    
        # --- Continuous Key Presses (for smooth movement) ---
        keys = pygame.key.get_pressed()
        current_time = pygame.time.get_ticks()

        if keys[pygame.K_LEFT]:
            if current_time - key_last_press_time['left'] > (KEY_REPEAT_DELAY if key_last_press_time['left'] == 0 else KEY_REPEAT_INTERVAL):
                movement_this_frame = 3
                key_last_press_time['left'] = current_time
        else:
            key_last_press_time['left'] = 0

        if keys[pygame.K_RIGHT]:
            if current_time - key_last_press_time['right'] > (KEY_REPEAT_DELAY if key_last_press_time['right'] == 0 else KEY_REPEAT_INTERVAL):
                movement_this_frame = 4
                key_last_press_time['right'] = current_time
        else:
            key_last_press_time['right'] = 0

        if keys[pygame.K_DOWN]:
            if current_time - key_last_press_time['down'] > KEY_REPEAT_INTERVAL:
                movement_this_frame = 2
                key_last_press_time['down'] = current_time
        else:
            key_last_press_time['down'] = 0

        # --- Step Environment ---
        action = [movement_this_frame, space_this_frame, shift_this_frame]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Lines: {info['lines_cleared']}")
            
        # --- Render to Screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()
    pygame.quit()