
# Generated: 2025-08-28T02:01:46.138477
# Source Brief: brief_04312.md
# Brief Index: 4312

        
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
        "Controls: ←→ to move, ↑↓ to rotate. Hold space to soft drop, press shift to hard drop."
    )

    game_description = (
        "Strategically place falling blocks to clear rows and achieve the target score before the board fills up."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    MAX_STEPS = 10000
    WIN_CONDITION_ROWS = 20
    
    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 60)
    COLOR_TEXT = (220, 220, 220)
    COLOR_GHOST = (255, 255, 255, 60)
    COLOR_UI_BG = (30, 30, 45, 200)
    COLOR_LINE_CLEAR = (255, 255, 255)

    PIECE_COLORS = [
        (0, 0, 0),  # 0: Empty
        (0, 240, 240),  # 1: I piece (Cyan)
        (0, 0, 240),    # 2: J piece (Blue)
        (240, 160, 0),  # 3: L piece (Orange)
        (240, 240, 0),  # 4: O piece (Yellow)
        (0, 240, 0),    # 5: S piece (Green)
        (160, 0, 240),  # 6: T piece (Purple)
        (240, 0, 0),    # 7: Z piece (Red)
    ]

    # --- Piece Shapes ---
    # 4 rotations for each piece, defined by (row, col) offsets from a pivot
    PIECE_SHAPES = {
        1: [[(0, -1), (0, 0), (0, 1), (0, 2)],  # I
            [(-1, 1), (0, 1), (1, 1), (2, 1)],
            [(-1, -1), (-1, 0), (-1, 1), (-1, 2)],
            [(-1, 0), (0, 0), (1, 0), (2, 0)]],
        2: [[(-1, -1), (0, -1), (0, 0), (0, 1)],  # J
            [(-1, 0), (-1, 1), (0, 0), (1, 0)],
            [(0, -1), (0, 0), (0, 1), (1, 1)],
            [(-1, 0), (0, 0), (1, 0), (1, -1)]],
        3: [[(-1, 1), (0, -1), (0, 0), (0, 1)],  # L
            [(-1, 0), (0, 0), (1, 0), (1, 1)],
            [(0, -1), (0, 0), (0, 1), (1, -1)],
            [(-1, -1), (-1, 0), (0, 0), (1, 0)]],
        4: [[(0, 0), (0, 1), (1, 0), (1, 1)]],  # O
        5: [[(-1, 1), (0, 0), (0, 1), (1, 0)],  # S
            [(-1, 0), (0, 0), (0, -1), (1, -1)],
            [(-1, 0), (0, 0), (0, -1), (1, -1)],
            [(0, 1), (1, 1), (1, 0), (-1, 0)]],
        6: [[(-1, 0), (0, -1), (0, 0), (0, 1)],  # T
            [(-1, 0), (0, 0), (0, 1), (1, 0)],
            [(0, -1), (0, 0), (0, 1), (1, 0)],
            [(-1, 0), (0, -1), (0, 0), (1, 0)]],
        7: [[(-1, -1), (0, -1), (0, 0), (1, 0)],  # Z
            [(-1, 1), (0, 0), (0, 1), (1, 0)],
            [(-1, 0), (0, 0), (0, -1), (1, -1)],
            [(0, -1), (1, -1), (1, 0), (-1, 0)]],
    }

    # Wall kick data (SRS)
    WALL_KICK_DATA = {
        'JLSTZ': {
            (0, 1): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
            (1, 0): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
            (1, 2): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
            (2, 1): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
            (2, 3): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
            (3, 2): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
            (3, 0): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
            (0, 3): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        },
        'I': {
            (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
            (1, 0): [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
            (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
            (2, 1): [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
            (2, 3): [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
            (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
            (3, 0): [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
            (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
        }
    }


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
        
        self.grid_offset_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT + 4, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.rows_cleared = 0
        self.game_over = False
        
        self.piece_bag = []
        self._fill_piece_bag()
        
        self.current_piece = self._get_new_piece()
        self.next_piece = self._get_new_piece()
        
        self.initial_fall_speed = 30 # frames per grid cell
        self.fall_speed = self.initial_fall_speed
        self.fall_counter = 0
        self.soft_drop_active = False

        self.last_shift_held = False
        self.move_cooldown = 0
        self.rotate_cooldown = 0

        self.line_clear_animation = []
        self.line_clear_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = self.game_over
        
        if terminated:
            return self._get_observation(), 0, terminated, False, self._get_info()

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Cooldowns to prevent single key press causing multiple actions in one frame
        if self.move_cooldown > 0: self.move_cooldown -= 1
        if self.rotate_cooldown > 0: self.rotate_cooldown -= 1

        # Hard drop on shift press (rising edge)
        hard_dropped = False
        if shift_held and not self.last_shift_held:
            # sfx: hard_drop.wav
            while self._is_valid_position(self.current_piece, dy=1):
                self.current_piece['y'] += 1
            hard_dropped = True
        self.last_shift_held = shift_held

        # Movement and Rotation
        if not hard_dropped:
            if movement == 3 and self.move_cooldown == 0: # Left
                if self._is_valid_position(self.current_piece, dx=-1):
                    self.current_piece['x'] -= 1
                    self.move_cooldown = 4
            elif movement == 4 and self.move_cooldown == 0: # Right
                if self._is_valid_position(self.current_piece, dx=1):
                    self.current_piece['x'] += 1
                    self.move_cooldown = 4
            elif movement == 1 and self.rotate_cooldown == 0: # Up -> Rotate Clockwise
                self._rotate_piece(1)
                self.rotate_cooldown = 6
            elif movement == 2 and self.rotate_cooldown == 0: # Down -> Rotate Counter-Clockwise
                self._rotate_piece(-1)
                self.rotate_cooldown = 6

        # Soft drop
        self.soft_drop_active = space_held

        # --- Game Logic Update ---
        self.fall_counter += 5 if self.soft_drop_active else 1
        
        piece_locked = False
        if self.fall_counter >= self.fall_speed or hard_dropped:
            self.fall_counter = 0
            if self._is_valid_position(self.current_piece, dy=1):
                self.current_piece['y'] += 1
            else:
                self._lock_piece()
                piece_locked = True
                reward += 0.1 # Reward for placing a block
                
                cleared_rows_count, cleared_rows_indices = self._clear_rows()
                if cleared_rows_count > 0:
                    # sfx: line_clear.wav
                    self.rows_cleared += cleared_rows_count
                    self.line_clear_animation = cleared_rows_indices
                    self.line_clear_timer = 10 # frames
                    
                    # Update difficulty
                    speed_increase_tiers = self.rows_cleared // 2
                    self.fall_speed = max(self.initial_fall_speed * 0.5, self.initial_fall_speed - speed_increase_tiers)

                    # Calculate reward for clearing lines
                    if cleared_rows_count == 1: reward += 1
                    elif cleared_rows_count == 2: reward += 3
                    elif cleared_rows_count == 3: reward += 6
                    elif cleared_rows_count >= 4: reward += 10 # Tetris!

                # Get new piece
                self.current_piece = self.next_piece
                self.next_piece = self._get_new_piece()

                # Check for game over
                if not self._is_valid_position(self.current_piece):
                    # sfx: game_over.wav
                    self.game_over = True
                    terminated = True
                    reward -= 100

        # --- Termination Checks ---
        if self.rows_cleared >= self.WIN_CONDITION_ROWS and not terminated:
            # sfx: win.wav
            self.game_over = True
            terminated = True
            reward += 100

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            
        if self.line_clear_timer > 0: self.line_clear_timer -= 1
        else: self.line_clear_animation = []

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "rows_cleared": self.rows_cleared}

    # --- Helper Methods ---

    def _fill_piece_bag(self):
        self.piece_bag = list(range(1, 8))
        self.np_random.shuffle(self.piece_bag)

    def _get_new_piece(self):
        if not self.piece_bag:
            self._fill_piece_bag()
        
        piece_id = self.piece_bag.pop()
        return {
            'id': piece_id,
            'x': self.GRID_WIDTH // 2 - 1,
            'y': 0,
            'rotation': 0
        }

    def _get_piece_coords(self, piece, dx=0, dy=0, rotation_offset=0):
        shape = self.PIECE_SHAPES[piece['id']]
        rotation = (piece['rotation'] + rotation_offset) % len(shape)
        coords = []
        for r_off, c_off in shape[rotation]:
            coords.append((piece['y'] + r_off + dy, piece['x'] + c_off + dx))
        return coords

    def _is_valid_position(self, piece, dx=0, dy=0, rotation_offset=0):
        for r, c in self._get_piece_coords(piece, dx, dy, rotation_offset):
            if not (0 <= c < self.GRID_WIDTH and r < self.GRID_HEIGHT + 4):
                return False
            if r >= 0 and self.grid[r, c] != 0:
                return False
        return True

    def _rotate_piece(self, direction):
        # sfx: rotate.wav
        old_rotation = self.current_piece['rotation']
        new_rotation = (old_rotation + direction + 4) % len(self.PIECE_SHAPES[self.current_piece['id']])
        
        kick_data_key = 'I' if self.current_piece['id'] == 1 else 'JLSTZ'
        rotation_key = (old_rotation, new_rotation)

        # Try base rotation
        if self._is_valid_position(self.current_piece, rotation_offset=direction):
            self.current_piece['rotation'] = new_rotation
            return

        # Try wall kicks
        if rotation_key in self.WALL_KICK_DATA[kick_data_key]:
            for dx, dy in self.WALL_KICK_DATA[kick_data_key][rotation_key]:
                # Note: SRS dy is inverted compared to our grid's y-axis
                if self._is_valid_position(self.current_piece, dx=dx, dy=-dy, rotation_offset=direction):
                    self.current_piece['x'] += dx
                    self.current_piece['y'] -= dy
                    self.current_piece['rotation'] = new_rotation
                    return

    def _lock_piece(self):
        # sfx: lock_piece.wav
        for r, c in self._get_piece_coords(self.current_piece):
            if 0 <= r < self.GRID_HEIGHT + 4 and 0 <= c < self.GRID_WIDTH:
                self.grid[r, c] = self.current_piece['id']
    
    def _clear_rows(self):
        full_rows = []
        for r in range(self.GRID_HEIGHT + 4):
            if np.all(self.grid[r, :] != 0):
                full_rows.append(r)
        
        if not full_rows:
            return 0, []

        score_map = {1: 100, 2: 300, 3: 500, 4: 800}
        self.score += score_map.get(len(full_rows), 0)

        # Shift rows down
        for r in sorted(full_rows, reverse=True):
            self.grid[1:r+1, :] = self.grid[:r, :]
            self.grid[0, :] = 0

        return len(full_rows), full_rows

    # --- Rendering Methods ---

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.grid_offset_x, self.grid_offset_y, 
            self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))

        # Draw locked pieces
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                cell_val = self.grid[r + 4, c]
                if cell_val != 0:
                    self._draw_cell(c, r, self.PIECE_COLORS[cell_val])

        # Draw ghost piece
        if not self.game_over:
            ghost_piece = self.current_piece.copy()
            while self._is_valid_position(ghost_piece, dy=1):
                ghost_piece['y'] += 1
            for r, c in self._get_piece_coords(ghost_piece):
                if r >= 4:
                    self._draw_cell(c, r - 4, self.COLOR_GHOST, is_ghost=True)

        # Draw current piece
        if not self.game_over:
            color = self.PIECE_COLORS[self.current_piece['id']]
            for r, c in self._get_piece_coords(self.current_piece):
                if r >= 4:
                    self._draw_cell(c, r - 4, color)
        
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_BG,
                (self.grid_offset_x + i * self.CELL_SIZE, self.grid_offset_y),
                (self.grid_offset_x + i * self.CELL_SIZE, self.grid_offset_y + self.GRID_HEIGHT * self.CELL_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_BG,
                (self.grid_offset_x, self.grid_offset_y + i * self.CELL_SIZE),
                (self.grid_offset_x + self.GRID_WIDTH * self.CELL_SIZE, self.grid_offset_y + i * self.CELL_SIZE))
        
        # Draw line clear animation
        if self.line_clear_timer > 0:
            alpha = 255 * (self.line_clear_timer / 10)
            flash_color = (*self.COLOR_LINE_CLEAR, alpha)
            for r in self.line_clear_animation:
                if r >= 4:
                    rect = pygame.Rect(self.grid_offset_x, self.grid_offset_y + (r - 4) * self.CELL_SIZE,
                                       self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
                    surf = pygame.Surface(rect.size, pygame.SRCALPHA)
                    surf.fill(flash_color)
                    self.screen.blit(surf, rect.topleft)

    def _draw_cell(self, grid_x, grid_y, color, is_ghost=False):
        px, py = self.grid_offset_x + grid_x * self.CELL_SIZE, self.grid_offset_y + grid_y * self.CELL_SIZE
        rect = (px, py, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            pygame.gfxdraw.box(self.screen, rect, color)
        else:
            pygame.draw.rect(self.screen, color, rect)
            # Add a subtle 3D effect
            highlight = tuple(min(255, c + 40) for c in color[:3])
            shadow = tuple(max(0, c - 40) for c in color[:3])
            pygame.draw.line(self.screen, highlight, (px, py), (px + self.CELL_SIZE - 1, py))
            pygame.draw.line(self.screen, highlight, (px, py), (px, py + self.CELL_SIZE - 1))
            pygame.draw.line(self.screen, shadow, (px + self.CELL_SIZE - 1, py), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1))
            pygame.draw.line(self.screen, shadow, (px, py + self.CELL_SIZE - 1), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1))
            
    def _render_ui(self):
        # --- Right Panel (Next Piece) ---
        next_box_x = self.grid_offset_x + self.GRID_WIDTH * self.CELL_SIZE + 20
        next_box_y = self.grid_offset_y
        
        s = pygame.Surface((120, 100), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (next_box_x, next_box_y))
        
        text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(text, (next_box_x + (120 - text.get_width())//2, next_box_y + 10))

        if not self.game_over:
            color = self.PIECE_COLORS[self.next_piece['id']]
            coords = self._get_piece_coords(self.next_piece)
            min_c = min(c[1] for c in coords)
            max_c = max(c[1] for c in coords)
            min_r = min(c[0] for c in coords)
            max_r = max(c[0] for c in coords)
            
            piece_width = (max_c - min_c + 1) * self.CELL_SIZE
            piece_height = (max_r - min_r + 1) * self.CELL_SIZE

            for r_off, c_off in self.PIECE_SHAPES[self.next_piece['id']][0]:
                 self._draw_cell_ui(next_box_x + (120 - piece_width) // 2 + (c_off-min_c) * self.CELL_SIZE,
                                    next_box_y + 55 + (r_off-min_r) * self.CELL_SIZE, color)

        # --- Left Panel (Score & Lines) ---
        info_box_x = self.grid_offset_x - 140
        info_box_y = self.grid_offset_y

        s = pygame.Surface((120, 140), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (info_box_x, info_box_y))

        # Score
        score_title = self.font_small.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_title, (info_box_x + (120 - score_title.get_width())//2, info_box_y + 10))
        score_val = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_val, (info_box_x + (120 - score_val.get_width())//2, info_box_y + 35))

        # Lines
        lines_title = self.font_small.render("LINES", True, self.COLOR_TEXT)
        self.screen.blit(lines_title, (info_box_x + (120 - lines_title.get_width())//2, info_box_y + 75))
        lines_val = self.font_large.render(f"{self.rows_cleared}", True, self.COLOR_TEXT)
        self.screen.blit(lines_val, (info_box_x + (120 - lines_val.get_width())//2, info_box_y + 100))

        # --- Game Over / Win Message ---
        if self.game_over:
            message = "YOU WIN!" if self.rows_cleared >= self.WIN_CONDITION_ROWS else "GAME OVER"
            text_surf = self.font_large.render(message, True, self.COLOR_LINE_CLEAR)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            
            s = pygame.Surface(text_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, text_rect.topleft)
            self.screen.blit(text_surf, text_rect)

    def _draw_cell_ui(self, px, py, color):
        rect = (px, py, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, color, rect)
        highlight = tuple(min(255, c + 40) for c in color[:3])
        shadow = tuple(max(0, c - 40) for c in color[:3])
        pygame.draw.line(self.screen, highlight, (px, py), (px + self.CELL_SIZE - 1, py))
        pygame.draw.line(self.screen, highlight, (px, py), (px, py + self.CELL_SIZE - 1))
        pygame.draw.line(self.screen, shadow, (px + self.CELL_SIZE - 1, py), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1))
        pygame.draw.line(self.screen, shadow, (px, py + self.CELL_SIZE - 1), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1))

    def validate_implementation(self):
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

# Example usage for visualization
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for display
    pygame.display.set_caption("Tetris Environment")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Key mapping
    key_to_action = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
    }

    while running:
        movement_action = 0
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- ENV RESET ---")

        keys = pygame.key.get_pressed()
        
        for key, move in key_to_action.items():
            if keys[key]:
                movement_action = move
                break # Prioritize one movement per frame

        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward}, Steps: {info['steps']}")
            # Optionally reset automatically
            # obs, info = env.reset()
            # total_reward = 0
        
        # --- Display the observation ---
        # The observation is (H, W, C), but pygame surfaces are (W, H)
        # So we need to transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()