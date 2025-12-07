
# Generated: 2025-08-28T04:06:28.620973
# Source Brief: brief_05142.md
# Brief Index: 5142

        
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
    A fast-paced, arcade puzzle game where the player maneuvers falling blocks to clear lines.
    This environment is designed for visual quality and engaging gameplay, featuring smooth animations,
    a "ghost piece" for better placement, and the classic "hold" mechanic.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑ to rotate, ↓ to soft drop. "
        "Space for hard drop, Shift to hold piece."
    )

    game_description = (
        "Strategically maneuver falling blocks to clear lines and achieve the highest score. "
        "Game speed increases as you clear more lines. Clear 10 lines to win!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_HEADER = (150, 150, 180)
    COLOR_WHITE = (255, 255, 255)

    # Tetromino shapes and colors
    TETROMINOES = {
        'I': ([[1, 1, 1, 1]], (66, 217, 245)),
        'O': ([[1, 1], [1, 1]], (245, 225, 66)),
        'T': ([[0, 1, 0], [1, 1, 1]], (188, 66, 245)),
        'J': ([[1, 0, 0], [1, 1, 1]], (66, 84, 245)),
        'L': ([[0, 0, 1], [1, 1, 1]], (245, 158, 66)),
        'S': ([[0, 1, 1], [1, 1, 0]], (111, 245, 66)),
        'Z': ([[1, 1, 0], [0, 1, 1]], (245, 66, 70))
    }
    
    # Super Rotation System (SRS) Wall Kick Data
    # (rotation_change) -> list of (dx, dy) kicks to test
    WALL_KICKS_JLSTZ = {
        (0, 1): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        (1, 0): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
        (1, 2): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
        (2, 1): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        (2, 3): [(0, 0), (1, 0), (1, -1), (0, 2), (1, -2)],
        (3, 2): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        (3, 0): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        (0, 3): [(0, 0), (1, 0), (1, -1), (0, 2), (1, -2)],
    }
    WALL_KICKS_I = {
        (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
        (1, 0): [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
        (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
        (2, 1): [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
        (2, 3): [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
        (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
        (3, 0): [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
        (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        self.render_mode = render_mode
        self.game_state_attributes = [
            'steps', 'score', 'game_over', 'board', 'lines_cleared',
            'current_piece', 'next_piece_shape', 'held_piece_shape',
            'can_hold', 'fall_timer', 'fall_speed', 'reward_this_step',
            'line_clear_animation', 'piece_bag'
        ]
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        # Board: 20 visible rows + 4 hidden rows at the top for spawning
        self.board = np.zeros((self.GRID_HEIGHT + 4, self.GRID_WIDTH), dtype=int)
        
        self.piece_bag = []
        self._refill_piece_bag()
        
        self.next_piece_shape = self._pop_from_bag()
        self._spawn_new_piece()
        
        self.held_piece_shape = None
        self.can_hold = True
        
        self.fall_speed = 1.0  # seconds per grid cell
        self.fall_timer = 0.0
        
        self.line_clear_animation = None # (timer, [row_indices])

        self.reward_this_step = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0.0
        
        if self.game_over:
            return self._get_observation(), self.reward_this_step, True, False, self._get_info()

        self.steps += 1
        
        # --- Handle line clear animation ---
        if self.line_clear_animation:
            self.line_clear_animation = (self.line_clear_animation[0] - 1, self.line_clear_animation[1])
            if self.line_clear_animation[0] <= 0:
                self._finish_line_clear(self.line_clear_animation[1])
                self.line_clear_animation = None
            # Freeze game during animation
            return self._get_observation(), self.reward_this_step, self.game_over, False, self._get_info()

        # --- Handle player actions ---
        self._handle_actions(action)

        # --- Apply gravity ---
        time_delta = self.clock.tick(30) / 1000.0  # seconds
        self.fall_timer += time_delta
        
        is_soft_dropping = action[0] == 2
        current_fall_speed = 0.05 if is_soft_dropping else self.fall_speed
        if is_soft_dropping:
            self.reward_this_step += 0.01

        if self.fall_timer >= current_fall_speed:
            self.fall_timer = 0
            self._move_piece(0, 1)

        # --- Check for termination ---
        terminated = self.game_over or self.lines_cleared >= 10 or self.steps >= 1000
        if terminated and not self.game_over:
             if self.lines_cleared >= 10: # Win condition
                self.score += 100
                self.reward_this_step += 100
             self.game_over = True

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_actions(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: # Rotate
            self._rotate_piece()
        elif movement == 3: # Left
            self._move_piece(-1, 0)
            self.reward_this_step -= 0.001
        elif movement == 4: # Right
            self._move_piece(1, 0)
            self.reward_this_step -= 0.001
        
        if space_pressed: # Hard drop
            self._hard_drop()
            
        if shift_pressed: # Hold
            self._hold_piece()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_board_pieces()
        if not self.game_over:
            self._render_ghost_piece()
            self._render_current_piece()
        self._render_ui()
        if self.line_clear_animation:
            self._render_line_clear_effect()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
        }
        
    # --- Piece Management ---
    def _refill_piece_bag(self):
        self.piece_bag = list(self.TETROMINOES.keys())
        self.np_random.shuffle(self.piece_bag)

    def _pop_from_bag(self):
        if not self.piece_bag:
            self._refill_piece_bag()
        return self.piece_bag.pop()
        
    def _spawn_new_piece(self):
        shape_key = self.next_piece_shape
        self.next_piece_shape = self._pop_from_bag()
        
        shape, color = self.TETROMINOES[shape_key]
        start_x = (self.GRID_WIDTH - len(shape[0])) // 2
        start_y = 0 if shape_key == 'I' else 1 # Spawn I piece one row higher
        
        self.current_piece = {
            'shape_key': shape_key,
            'shape': shape,
            'color': color,
            'x': start_x,
            'y': start_y, # Start in hidden area
            'rotation': 0
        }
        
        self.can_hold = True
        
        if self._check_collision(self.current_piece['x'], self.current_piece['y'], self.current_piece['shape']):
            self.game_over = True
            self.reward_this_step -= 100
            self.score -= 100

    def _check_collision(self, x, y, shape):
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    board_x, board_y = x + c, y + r
                    if not (0 <= board_x < self.GRID_WIDTH and 0 <= board_y < self.GRID_HEIGHT + 4):
                        return True
                    if self.board[board_y, board_x] != 0:
                        return True
        return False

    def _move_piece(self, dx, dy):
        if self.current_piece is None: return
        
        new_x, new_y = self.current_piece['x'] + dx, self.current_piece['y'] + dy
        if not self._check_collision(new_x, new_y, self.current_piece['shape']):
            self.current_piece['x'], self.current_piece['y'] = new_x, new_y
        elif dy > 0: # Collision while moving down
            self._lock_piece()

    def _rotate_piece(self):
        if self.current_piece is None or self.current_piece['shape_key'] == 'O': return
        
        original_shape = self.current_piece['shape']
        original_rot = self.current_piece['rotation']
        
        # Create rotated shape
        rotated_shape = list(zip(*self.current_piece['shape'][::-1]))
        new_rot = (original_rot + 1) % 4
        
        # Get wall kick data
        kick_data = self.WALL_KICKS_I if self.current_piece['shape_key'] == 'I' else self.WALL_KICKS_JLSTZ
        kicks = kick_data.get((original_rot, new_rot), [])

        for kick_x, kick_y in kicks:
            # Note: SRS y-axis is inverted compared to our grid y-axis
            new_x = self.current_piece['x'] + kick_x
            new_y = self.current_piece['y'] - kick_y
            if not self._check_collision(new_x, new_y, rotated_shape):
                self.current_piece['shape'] = rotated_shape
                self.current_piece['rotation'] = new_rot
                self.current_piece['x'], self.current_piece['y'] = new_x, new_y
                # sfx: rotate
                return

    def _hard_drop(self):
        if self.current_piece is None: return
        
        ghost_y = self._get_ghost_piece_y()
        self.current_piece['y'] = ghost_y
        self._lock_piece()
        # sfx: hard_drop

    def _hold_piece(self):
        if not self.can_hold: return
        
        if self.held_piece_shape is None:
            self.held_piece_shape = self.current_piece['shape_key']
            self._spawn_new_piece()
        else:
            self.held_piece_shape, self.next_piece_shape = self.current_piece['shape_key'], self.held_piece_shape
            self._spawn_new_piece()
        
        self.can_hold = False
        # sfx: hold

    def _lock_piece(self):
        if self.current_piece is None: return
        
        piece = self.current_piece
        color_index = list(self.TETROMINOES.keys()).index(piece['shape_key']) + 1
        
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    self.board[piece['y'] + r, piece['x'] + c] = color_index
        
        self.current_piece = None
        self._check_for_line_clears()
        # sfx: lock
        
    # --- Game Logic ---
    def _check_for_line_clears(self):
        full_rows = []
        for r in range(self.GRID_HEIGHT + 4):
            if np.all(self.board[r, :] != 0):
                full_rows.append(r)
        
        if full_rows:
            num_cleared = len(full_rows)
            # Rewards
            rewards = {1: 1, 2: 2, 3: 4, 4: 8}
            self.reward_this_step += rewards.get(num_cleared, 0)
            
            # Scoring
            scores = {1: 40, 2: 100, 3: 300, 4: 1200}
            self.score += scores.get(num_cleared, 0)
            
            self.lines_cleared += num_cleared
            self.line_clear_animation = (10, full_rows) # 10 frames animation
            # sfx: line_clear
        else:
            self._spawn_new_piece() # No lines cleared, spawn next piece immediately

    def _finish_line_clear(self, cleared_rows):
        # Remove cleared rows
        self.board = np.delete(self.board, cleared_rows, axis=0)
        
        # Add new empty rows at the top
        new_rows = np.zeros((len(cleared_rows), self.GRID_WIDTH), dtype=int)
        self.board = np.vstack((new_rows, self.board))
        
        # Increase game speed
        speed_increase = 0.05 * len(cleared_rows)
        self.fall_speed = max(0.1, self.fall_speed - speed_increase)

        self._spawn_new_piece()

    # --- Rendering ---
    def _draw_block(self, surface, x, y, color, alpha=255):
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        # Main block color
        pygame.gfxdraw.box(surface, rect, (*color, alpha))
        
        # Highlight/Bevel effect
        highlight_color = tuple(min(255, c + 40) for c in color)
        shadow_color = tuple(max(0, c - 40) for c in color)
        
        pygame.draw.line(surface, (*highlight_color, alpha), rect.topleft, rect.topright, 1)
        pygame.draw.line(surface, (*highlight_color, alpha), rect.topleft, rect.bottomleft, 1)
        pygame.draw.line(surface, (*shadow_color, alpha), rect.bottomright, rect.topright, 1)
        pygame.draw.line(surface, (*shadow_color, alpha), rect.bottomright, rect.bottomleft, 1)
        
        # Inner fill
        inner_rect = rect.inflate(-4, -4)
        pygame.gfxdraw.box(surface, inner_rect, (*color, alpha))

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_Y_OFFSET), (px, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, py), (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, py))

    def _render_board_pieces(self):
        keys = list(self.TETROMINOES.keys())
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                board_r = r + 4 # Account for hidden rows
                color_index = self.board[board_r, c]
                if color_index != 0:
                    color = self.TETROMINOES[keys[color_index - 1]][1]
                    px = self.GRID_X_OFFSET + c * self.CELL_SIZE
                    py = self.GRID_Y_OFFSET + r * self.CELL_SIZE
                    self._draw_block(self.screen, px, py, color)

    def _render_current_piece(self):
        if self.current_piece is None: return
        piece = self.current_piece
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    px = self.GRID_X_OFFSET + (piece['x'] + c) * self.CELL_SIZE
                    # Only draw if inside the visible grid area
                    if (piece['y'] + r) >= 4:
                        py = self.GRID_Y_OFFSET + (piece['y'] + r - 4) * self.CELL_SIZE
                        self._draw_block(self.screen, px, py, piece['color'])

    def _get_ghost_piece_y(self):
        if self.current_piece is None: return 0
        y = self.current_piece['y']
        while not self._check_collision(self.current_piece['x'], y + 1, self.current_piece['shape']):
            y += 1
        return y
        
    def _render_ghost_piece(self):
        if self.current_piece is None: return
        
        ghost_y = self._get_ghost_piece_y()
        piece = self.current_piece
        
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    px = self.GRID_X_OFFSET + (piece['x'] + c) * self.CELL_SIZE
                    if (ghost_y + r) >= 4:
                        py = self.GRID_Y_OFFSET + (ghost_y + r - 4) * self.CELL_SIZE
                        rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
                        pygame.gfxdraw.rectangle(self.screen, rect, (*piece['color'], 100))

    def _render_line_clear_effect(self):
        timer, rows = self.line_clear_animation
        alpha = int(255 * (math.sin(timer / 10 * math.pi * 2) * 0.5 + 0.5))
        for r in rows:
            if r >= 4:
                rect = pygame.Rect(
                    self.GRID_X_OFFSET,
                    self.GRID_Y_OFFSET + (r - 4) * self.CELL_SIZE,
                    self.GRID_WIDTH * self.CELL_SIZE,
                    self.CELL_SIZE
                )
                pygame.gfxdraw.box(self.screen, rect, (*self.COLOR_WHITE, alpha))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Lines
        lines_header = self.font_small.render("LINES", True, self.COLOR_UI_HEADER)
        lines_text = self.font_main.render(f"{self.lines_cleared}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lines_header, (20, 60))
        self.screen.blit(lines_text, (20, 80))

        # Next Piece
        self._render_side_panel(self.next_piece_shape, "NEXT", self.SCREEN_WIDTH - 120)
        
        # Held Piece
        self._render_side_panel(self.held_piece_shape, "HOLD", self.SCREEN_WIDTH - 120, 150)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "YOU WIN!" if self.lines_cleared >= 10 else "GAME OVER"
            text_surf = self.font_main.render(win_text, True, self.COLOR_WHITE)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _render_side_panel(self, shape_key, title, x_pos, y_pos=20):
        header_text = self.font_small.render(title, True, self.COLOR_UI_HEADER)
        self.screen.blit(header_text, (x_pos, y_pos))

        if shape_key is not None:
            shape, color = self.TETROMINOES[shape_key]
            w, h = len(shape[0]), len(shape)
            start_x = x_pos + (100 - w * self.CELL_SIZE) / 2
            start_y = y_pos + 40 + (60 - h * self.CELL_SIZE) / 2

            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.screen, start_x + c * self.CELL_SIZE, start_y + r * self.CELL_SIZE, color)

    def close(self):
        pygame.quit()
        
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Puzzle Blocks")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0

    while running:
        action = np.array([movement, space_held, shift_held])
        
        # Reset held keys after one frame
        movement = 0
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        # Soft drop is continuous
        elif keys[pygame.K_DOWN]:
            movement = 2
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for 'r' to reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False

    env.close()