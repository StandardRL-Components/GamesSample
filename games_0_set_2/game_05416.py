
# Generated: 2025-08-28T04:58:01.099725
# Source Brief: brief_05416.md
# Brief Index: 5416

        
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
    A fast-paced, top-down falling block puzzle game where strategic placement and risky maneuvers are rewarded.
    The goal is to clear horizontal lines of blocks by filling them completely. Clearing multiple lines at once
    yields a higher score. The game ends when the stack of blocks reaches the top of the playfield or a
    target score is achieved. The block fall speed increases over time, adding to the challenge.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←/→ to move, ↑ to rotate clockwise, ↓ to rotate counter-clockwise. "
        "Hold Space for soft drop, press Shift for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade puzzle game. Clear lines by filling them with falling blocks. "
        "Score big by clearing multiple lines at once. Don't let the stack reach the top!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAYFIELD_W = 10
    PLAYFIELD_H = 20
    CELL_SIZE = 18

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_BORDER = (140, 140, 150)
    COLOR_WHITE = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_OVERLAY = (0, 0, 0, 180)

    # --- Tetromino Shapes & Colors ---
    # Index 0 is empty
    TETROMINO_COLORS = [
        (0, 0, 0),        # 0: Empty
        (255, 80, 80),    # 1: S (Red)
        (80, 255, 80),    # 2: Z (Green)
        (80, 80, 255),    # 3: J (Blue)
        (255, 160, 0),    # 4: L (Orange)
        (255, 255, 80),   # 5: O (Yellow)
        (160, 80, 255),   # 6: T (Purple)
        (80, 255, 255),   # 7: I (Cyan)
    ]

    TETROMINOES = {
        'S': [[[0,1,1],[1,1,0],[0,0,0]], [[1,0,0],[1,1,0],[0,1,0]], [[0,0,0],[0,1,1],[1,1,0]], [[0,1,0],[0,1,1],[0,0,1]]],
        'Z': [[[1,1,0],[0,1,1],[0,0,0]], [[0,1,0],[1,1,0],[1,0,0]], [[0,0,0],[1,1,0],[0,1,1]], [[0,0,1],[0,1,1],[0,1,0]]],
        'J': [[[1,0,0],[1,1,1],[0,0,0]], [[0,1,1],[0,1,0],[0,1,0]], [[0,0,0],[1,1,1],[0,0,1]], [[0,1,0],[0,1,0],[1,1,0]]],
        'L': [[[0,0,1],[1,1,1],[0,0,0]], [[0,1,0],[0,1,0],[0,1,1]], [[0,0,0],[1,1,1],[1,0,0]], [[1,1,0],[0,1,0],[0,1,0]]],
        'O': [[[0,1,1,0],[0,1,1,0],[0,0,0,0]], [[0,1,1,0],[0,1,1,0],[0,0,0,0]], [[0,1,1,0],[0,1,1,0],[0,0,0,0]], [[0,1,1,0],[0,1,1,0],[0,0,0,0]]],
        'T': [[[0,1,0],[1,1,1],[0,0,0]], [[0,1,0],[1,1,0],[0,1,0]], [[0,0,0],[1,1,1],[0,1,0]], [[0,1,0],[0,1,1],[0,1,0]]],
        'I': [[[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]], [[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]], [[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0]], [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]]],
    }
    
    TETROMINO_COLOR_MAP = {'S': 1, 'Z': 2, 'J': 3, 'L': 4, 'O': 5, 'T': 6, 'I': 7}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_m = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.playfield_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.PLAYFIELD_W * self.CELL_SIZE) // 2 - 120,
            (self.SCREEN_HEIGHT - self.PLAYFIELD_H * self.CELL_SIZE) // 2,
            self.PLAYFIELD_W * self.CELL_SIZE,
            self.PLAYFIELD_H * self.CELL_SIZE
        )
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.grid = [[0 for _ in range(self.PLAYFIELD_W)] for _ in range(self.PLAYFIELD_H)]
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.reward_this_step = 0
        
        self.fall_frequency = 30  # Frames per one-cell drop
        self.fall_timer = 0
        
        self.line_clear_anim_timer = 0
        self.cleared_rows = []
        
        self.piece_queue = [random.choice(list(self.TETROMINOES.keys())) for _ in range(2)]
        self._new_piece()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over or self.win:
            terminated = True
            return self._get_observation(), 0, terminated, False, self._get_info()
            
        self.reward_this_step = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        is_turn_over = False
        if self.line_clear_anim_timer == 0:
            if shift_held:
                # sfx_hard_drop
                self._handle_hard_drop()
                is_turn_over = True
            else:
                self._handle_movement_input(movement)
                
                self.fall_timer += 1
                if space_held:
                    self.fall_timer += 4  # Soft drop speed bonus
                
                if self.fall_timer >= self.fall_frequency:
                    self.fall_timer = 0
                    if not self._move_piece(0, 1):
                        is_turn_over = True

        if is_turn_over:
            self._place_piece()

        if self.steps > 0 and self.steps % 100 == 0 and self.fall_frequency > 5:
            self.fall_frequency = max(5, self.fall_frequency - 1)

        if self.line_clear_anim_timer > 0:
            self.line_clear_anim_timer -= 1
            if self.line_clear_anim_timer == 0:
                self._perform_line_clear()

        self.win = self.score >= 500
        terminated = self.game_over or self.win or self.steps >= 10000
        
        if self.win and not self.game_over: self.reward_this_step += 100
        elif self.game_over: self.reward_this_step -= 100
             
        reward = self.reward_this_step
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _new_piece(self):
        self.current_piece_shape = self.piece_queue.pop(0)
        self.piece_queue.append(random.choice(list(self.TETROMINOES.keys())))
        
        self.current_piece_rot = 0
        piece_width = len(self.TETROMINOES[self.current_piece_shape][0][0])
        self.current_piece_x = self.PLAYFIELD_W // 2 - piece_width // 2
        self.current_piece_y = 0
        
        if self._check_collision(self.current_piece_rot, self.current_piece_x, self.current_piece_y):
            self.game_over = True

    def _handle_movement_input(self, movement):
        if movement == 3: self._move_piece(-1, 0) # Left
        elif movement == 4: self._move_piece(1, 0) # Right
        elif movement == 1: self._rotate_piece(clockwise=True) # Up
        elif movement == 2: self._rotate_piece(clockwise=False) # Down

    def _handle_hard_drop(self):
        dy = 0
        while not self._check_collision(self.current_piece_rot, self.current_piece_x, self.current_piece_y + dy + 1):
            dy += 1
        self.current_piece_y += dy
    
    def _move_piece(self, dx, dy):
        if not self._check_collision(self.current_piece_rot, self.current_piece_x + dx, self.current_piece_y + dy):
            self.current_piece_x += dx
            self.current_piece_y += dy
            return True
        return False

    def _rotate_piece(self, clockwise=True):
        new_rot = (self.current_piece_rot + (1 if clockwise else -1)) % len(self.TETROMINOES[self.current_piece_shape])
        for kick_dx in [0, 1, -1, 2, -2]:
             if not self._check_collision(new_rot, self.current_piece_x + kick_dx, self.current_piece_y):
                self.current_piece_rot = new_rot
                self.current_piece_x += kick_dx
                # sfx_rotate
                return

    def _place_piece(self):
        # sfx_place_block
        piece_coords = self._get_piece_coords()
        
        self.reward_this_step += 0.1 # Placement reward
        
        empty_cells = 0
        cols_occupied = {x for x, y in piece_coords}
        for col in cols_occupied:
            lowest_y_in_col = max(y for x, y in piece_coords if x == col)
            for row in range(lowest_y_in_col + 1, self.PLAYFIELD_H):
                if 0 <= col < self.PLAYFIELD_W and 0 <= row < self.PLAYFIELD_H and self.grid[row][col] == 0:
                    empty_cells += 1
        self.reward_this_step -= 0.02 * empty_cells

        color_idx = self.TETROMINO_COLOR_MAP[self.current_piece_shape]
        for x, y in piece_coords:
            if 0 <= y < self.PLAYFIELD_H and 0 <= x < self.PLAYFIELD_W:
                self.grid[y][x] = color_idx
        
        self._check_and_clear_lines()
        if not self.game_over:
            self._new_piece()

    def _check_and_clear_lines(self):
        full_rows = [r for r, row in enumerate(self.grid) if all(cell != 0 for cell in row)]
        
        if full_rows:
            # sfx_line_clear
            self.cleared_rows = full_rows
            self.line_clear_anim_timer = 8 # frames
            
            num_lines = len(full_rows)
            reward_map = {1: 1, 2: 3, 3: 6, 4: 10}
            self.reward_this_step += reward_map.get(num_lines, 0)
            
            score_map = {1: 10, 2: 30, 3: 60, 4: 100}
            self.score += score_map.get(num_lines, 0)
        else:
            self.cleared_rows = []

    def _perform_line_clear(self):
        if not self.cleared_rows: return
        new_grid = [row for i, row in enumerate(self.grid) if i not in self.cleared_rows]
        for _ in range(len(self.cleared_rows)):
            new_grid.insert(0, [0 for _ in range(self.PLAYFIELD_W)])
        self.grid = new_grid
        self.cleared_rows = []

    def _check_collision(self, rotation, px, py):
        piece_shape = self.TETROMINOES[self.current_piece_shape][rotation]
        for r, row in enumerate(piece_shape):
            for c, cell in enumerate(row):
                if cell:
                    x, y = px + c, py + r
                    if not (0 <= x < self.PLAYFIELD_W and y < self.PLAYFIELD_H):
                        return True
                    if y >= 0 and self.grid[y][x] != 0:
                        return True
        return False

    def _get_piece_coords(self, shape=None, rotation=None, px=None, py=None):
        shape = shape or self.current_piece_shape
        rotation = rotation if rotation is not None else self.current_piece_rot
        px = px if px is not None else self.current_piece_x
        py = py if py is not None else self.current_piece_y
            
        coords = []
        piece_shape = self.TETROMINOES[shape][rotation]
        for r, row in enumerate(piece_shape):
            for c, cell in enumerate(row):
                if cell:
                    coords.append((px + c, py + r))
        return coords

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_BORDER, self.playfield_rect.inflate(4, 4), 2, border_radius=3)
        
        for r in range(1, self.PLAYFIELD_H):
            y = self.playfield_rect.top + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.playfield_rect.left, y), (self.playfield_rect.right, y))
        for c in range(1, self.PLAYFIELD_W):
            x = self.playfield_rect.left + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.playfield_rect.top), (x, self.playfield_rect.bottom))
        
        for r, row in enumerate(self.grid):
            for c, color_idx in enumerate(row):
                if color_idx != 0:
                    self._draw_cell(self.screen, self.playfield_rect.left + c * self.CELL_SIZE, self.playfield_rect.top + r * self.CELL_SIZE, color_idx)

        if not (self.game_over or self.win) and self.line_clear_anim_timer == 0:
            ghost_y = self.current_piece_y
            while not self._check_collision(self.current_piece_rot, self.current_piece_x, ghost_y + 1):
                ghost_y += 1
            
            ghost_coords = self._get_piece_coords(px=self.current_piece_x, py=ghost_y)
            for x, y in ghost_coords:
                if y >= 0:
                     self._draw_cell(self.screen, self.playfield_rect.left + x * self.CELL_SIZE, self.playfield_rect.top + y * self.CELL_SIZE, self.TETROMINO_COLOR_MAP[self.current_piece_shape], is_ghost=True)

        if not (self.game_over or self.win) and self.line_clear_anim_timer == 0:
            piece_coords = self._get_piece_coords()
            color_idx = self.TETROMINO_COLOR_MAP[self.current_piece_shape]
            for x, y in piece_coords:
                if y >= 0:
                    self._draw_cell(self.screen, self.playfield_rect.left + x * self.CELL_SIZE, self.playfield_rect.top + y * self.CELL_SIZE, color_idx)
        
        if self.line_clear_anim_timer > 0:
            flash_alpha = 150 + 100 * math.sin(self.line_clear_anim_timer * math.pi / 8)
            flash_color = (255, 255, 255, flash_alpha)
            for r in self.cleared_rows:
                rect = pygame.Rect(self.playfield_rect.left, self.playfield_rect.top + r * self.CELL_SIZE, self.playfield_rect.width, self.CELL_SIZE)
                s = pygame.Surface(rect.size, pygame.SRCALPHA)
                s.fill(flash_color)
                self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        score_text = self.font_m.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.playfield_rect.right + 40, self.playfield_rect.top + 20))
        score_val = self.font_l.render(f"{self.score:06d}", True, self.COLOR_WHITE)
        self.screen.blit(score_val, (self.playfield_rect.right + 40, self.playfield_rect.top + 50))
        
        next_text = self.font_m.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.playfield_rect.right + 40, self.playfield_rect.top + 140))
        
        next_box_rect = pygame.Rect(self.playfield_rect.right + 40, self.playfield_rect.top + 170, 4 * self.CELL_SIZE, 4 * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, next_box_rect, border_radius=3)
        
        next_shape_key = self.piece_queue[0]
        next_piece_coords = self._get_piece_coords(shape=next_shape_key, rotation=0, px=0, py=0)
        color_idx = self.TETROMINO_COLOR_MAP[next_shape_key]
        
        w = max(c for c,r in next_piece_coords) + 1 if next_piece_coords else 0
        h = max(r for c,r in next_piece_coords) + 1 if next_piece_coords else 0
        offset_x = next_box_rect.centerx - (w * self.CELL_SIZE) / 2
        offset_y = next_box_rect.centery - (h * self.CELL_SIZE) / 2
        
        for x, y in next_piece_coords:
            self._draw_cell(self.screen, offset_x + x * self.CELL_SIZE, offset_y + y * self.CELL_SIZE, color_idx)
            
        if self.game_over or self.win:
            overlay = pygame.Surface(self.playfield_rect.size, pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, self.playfield_rect.topleft)
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            end_text = self.font_l.render(message, True, self.COLOR_WHITE)
            text_rect = end_text.get_rect(center=self.playfield_rect.center)
            self.screen.blit(end_text, text_rect)

    def _draw_cell(self, surface, x, y, color_idx, is_ghost=False):
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        color = self.TETROMINO_COLORS[color_idx]
        
        if is_ghost:
            pygame.draw.rect(surface, color, rect, 2, border_radius=3)
        else:
            light_color = tuple(min(255, c + 40) for c in color)
            dark_color = tuple(max(0, c - 40) for c in color)
            
            pygame.draw.rect(surface, dark_color, rect, border_radius=3)
            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(surface, color, inner_rect, border_radius=2)
            
            pygame.draw.line(surface, light_color, (rect.left + 2, rect.top + 2), (rect.right - 3, rect.top + 2), 1)
            pygame.draw.line(surface, light_color, (rect.left + 2, rect.top + 2), (rect.left + 2, rect.bottom - 3), 1)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
    
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Gymnasium Tetris")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # For handling single-press actions like rotation
    last_movement_key_state = {key: False for key in key_map}
    
    while not terminated:
        movement = 0
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                    shift_held = 1
        
        keys = pygame.key.get_pressed()

        # Handle rotation/movement with a simple debounce to feel better for human play
        # The agent would learn to pulse these actions itself
        movement_key_pressed = False
        for key, move_val in key_map.items():
            if keys[key]:
                if not last_movement_key_state[key]:
                    movement = move_val
                    movement_key_pressed = True
                last_movement_key_state[key] = True
                if movement_key_pressed: break
            else:
                last_movement_key_state[key] = False

        if keys[pygame.K_SPACE]:
            space_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)
        
    pygame.quit()
    print(f"Game Over! Final Score: {info['score']}")