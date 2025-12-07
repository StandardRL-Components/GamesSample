
# Generated: 2025-08-27T23:06:09.537479
# Source Brief: brief_03352.md
# Brief Index: 3352

        
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
        "Controls: ←→ to move, ↑ to rotate CW, Shift to rotate CCW. ↓ for soft drop, Space for hard drop."
    )

    game_description = (
        "A falling block puzzle game. Place pieces to clear horizontal lines. Clear 20 lines to win, but don't let the stack reach the top!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2 - 100
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2
    
    WIN_CONDITION_LINES = 20
    MAX_STEPS = 10000

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_UI_BG = (30, 30, 40)
    COLOR_UI_FRAME = (60, 60, 70)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_VALUE = (255, 255, 100)
    COLOR_GHOST = (255, 255, 255, 50)
    
    PIECE_COLORS = [
        (0, 0, 0),          # 0: Empty
        (0, 240, 240),      # 1: I (Cyan)
        (240, 240, 0),      # 2: O (Yellow)
        (160, 0, 240),      # 3: T (Purple)
        (240, 160, 0),      # 4: L (Orange)
    ]

    # --- Piece Shapes (Tetrominos) ---
    # Centered around a pivot point for rotation
    PIECE_SHAPES = {
        'I': [[(0, -1), (0, 0), (0, 1), (0, 2)], [( -1, 0), (0, 0), (1, 0), (2, 0)]],
        'O': [[(0, 0), (1, 0), (0, 1), (1, 1)]],
        'T': [[(-1, 0), (0, 0), (1, 0), (0, -1)],
              [(0, -1), (0, 0), (0, 1), (1, 0)],
              [(-1, 0), (0, 0), (1, 0), (0, 1)],
              [(0, -1), (0, 0), (0, 1), (-1, 0)]],
        'L': [[(-1, -1), (-1, 0), (0, 0), (1, 0)],
              [(0, -1), (0, 0), (0, 1), (1, -1)],
              [(-1, 0), (0, 0), (1, 0), (1, 1)],
              [(-1, 1), (0, 1), (0, 0), (0, -1)]],
    }
    PIECE_KEYS = list(PIECE_SHAPES.keys())

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
        
        self.font_m = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 32, bold=True)

        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        self.next_piece_shape = self.np_random.choice(self.PIECE_KEYS)
        self._spawn_new_piece()
        
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.reward = 0
        
        self.fall_progress = 0.0
        self.base_fall_speed = 1.0  # cells per second
        
        self.prev_action_state = {'move': 0, 'space': False, 'shift': False}
        self.move_timer = 0
        self.move_delay = 5  # frames before auto-repeat
        
        self.line_clear_animation_timer = 0
        self.lines_to_clear = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        self.reward = -0.01  # Small penalty per step to encourage efficiency

        if not self.game_over:
            if self.line_clear_animation_timer > 0:
                self.line_clear_animation_timer -= 1
                if self.line_clear_animation_timer == 0:
                    self._perform_line_clear()
            else:
                self._handle_input(action)
                self._update_physics()

        terminated = self.game_over or self.lines_cleared >= self.WIN_CONDITION_LINES or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over and self.lines_cleared >= self.WIN_CONDITION_LINES:
            self.reward += 100 # Win bonus
        elif terminated and self.game_over:
            self.reward -= 50 # Lose penalty
            
        return (
            self._get_observation(),
            self.reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle one-shot actions (on press) ---
        up_pressed = movement == 1
        up_on_press = up_pressed and self.prev_action_state['move'] != 1
        shift_on_press = shift_held and not self.prev_action_state['shift']
        space_on_press = space_held and not self.prev_action_state['space']

        if up_on_press:
            self._rotate_piece(1) # SFX: Rotate
        elif shift_on_press:
            self._rotate_piece(-1) # SFX: Rotate

        if space_on_press:
            self._hard_drop() # SFX: HardDrop
            return # Hard drop ends the turn immediately

        # --- Handle continuous actions (held) ---
        down_held = movement == 2
        soft_drop_multiplier = 5.0 if down_held else 1.0

        # Horizontal movement with Delayed Auto-Shift (DAS)
        left_held, right_held = movement == 3, movement == 4
        if not left_held and not right_held:
            self.move_timer = 0
        else:
            self.move_timer += 1
            if self.move_timer == 1 or self.move_timer > self.move_delay:
                if left_held:
                    self._move_piece(-1, 0) # SFX: Move
                if right_held:
                    self._move_piece(1, 0) # SFX: Move
        
        self.prev_action_state = {'move': movement, 'space': space_held, 'shift': shift_held}
        return soft_drop_multiplier

    def _update_physics(self):
        # Determine current fall speed
        level = self.lines_cleared // 5
        current_fall_speed = self.base_fall_speed + level * 0.5
        
        # Soft drop acceleration
        down_held = self.prev_action_state['move'] == 2
        effective_fall_speed = current_fall_speed * 10.0 if down_held else current_fall_speed

        # Update fall progress
        self.fall_progress += effective_fall_speed / 30.0 # Assuming 30 FPS
        
        if self.fall_progress >= 1.0:
            moves = int(self.fall_progress)
            self.fall_progress -= moves
            for _ in range(moves):
                if not self._move_piece(0, 1): # If move down fails
                    self._place_piece() # SFX: Place
                    break

    def _spawn_new_piece(self):
        self.current_piece = {
            'shape': self.next_piece_shape,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2,
            'y': 0,
            'color': self.PIECE_KEYS.index(self.next_piece_shape) + 1
        }
        self.next_piece_shape = self.np_random.choice(self.PIECE_KEYS)
        
        if not self._is_valid_position(self.current_piece):
            self.game_over = True # SFX: GameOver
            
    def _get_piece_coords(self, piece):
        shape_coords = self.PIECE_SHAPES[piece['shape']][piece['rotation'] % len(self.PIECE_SHAPES[piece['shape']])]
        return [(piece['x'] + dx, piece['y'] + dy) for dx, dy in shape_coords]

    def _is_valid_position(self, piece, offset_x=0, offset_y=0):
        test_piece = piece.copy()
        test_piece['x'] += offset_x
        test_piece['y'] += offset_y
        
        coords = self._get_piece_coords(test_piece)
        for x, y in coords:
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return False # Out of bounds
            if self.grid[y, x] != 0:
                return False # Collision with existing block
        return True

    def _move_piece(self, dx, dy):
        if self._is_valid_position(self.current_piece, dx, dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            return True
        return False

    def _rotate_piece(self, direction):
        original_rotation = self.current_piece['rotation']
        self.current_piece['rotation'] = (original_rotation + direction) % len(self.PIECE_SHAPES[self.current_piece['shape']])
        
        # Simple wall kick implementation
        if not self._is_valid_position(self.current_piece):
            # Try shifting left or right
            if self._is_valid_position(self.current_piece, -1, 0):
                self.current_piece['x'] -= 1
            elif self._is_valid_position(self.current_piece, 1, 0):
                self.current_piece['x'] += 1
            else: # Rotation failed
                self.current_piece['rotation'] = original_rotation
                return False
        return True

    def _place_piece(self):
        coords = self._get_piece_coords(self.current_piece)
        for x, y in coords:
            if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                self.grid[y, x] = self.current_piece['color']
        
        self.reward += 0.1 # Small reward for placing a piece
        self._check_for_line_clears()
        if not self.lines_to_clear:
             self._spawn_new_piece()

    def _hard_drop(self):
        while self._move_piece(0, 1):
            pass
        self._place_piece()

    def _check_for_line_clears(self):
        self.lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                self.lines_to_clear.append(r)
        
        if self.lines_to_clear:
            self.line_clear_animation_timer = 10 # frames for animation
            # SFX: LineClear
            
    def _perform_line_clear(self):
        num_cleared = len(self.lines_to_clear)
        if num_cleared > 0:
            line_rewards = {1: 1, 2: 2, 3: 4, 4: 8}
            self.reward += line_rewards.get(num_cleared, 0)
            self.score += [0, 100, 300, 500, 800][num_cleared] * (self.lines_cleared // 5 + 1)
            self.lines_cleared += num_cleared
            
            # Create new grid by dropping rows
            new_grid = np.zeros_like(self.grid)
            new_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if r not in self.lines_to_clear:
                    new_grid[new_row, :] = self.grid[r, :]
                    new_row -= 1
            self.grid = new_grid
            self.lines_to_clear = []

        self._spawn_new_piece()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
        }

    def _render_game(self):
        # Draw grid background and border
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_FRAME, grid_rect, 2)

        # Draw settled pieces
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    self._draw_cell(c, r, self.grid[r, c])
        
        if self.game_over:
            return
            
        # Draw ghost piece
        if not self.line_clear_animation_timer > 0:
            ghost_piece = self.current_piece.copy()
            while self._is_valid_position(ghost_piece, 0, 1):
                ghost_piece['y'] += 1
            self._draw_piece(ghost_piece, is_ghost=True)

        # Draw current piece
        if not self.line_clear_animation_timer > 0:
            self._draw_piece(self.current_piece)
            
        # Draw line clear animation
        if self.line_clear_animation_timer > 0:
            alpha = 255 * (self.line_clear_animation_timer / 10)
            flash_color = (255, 255, 255, alpha)
            for r in self.lines_to_clear:
                flash_rect = pygame.Rect(self.GRID_X, self.GRID_Y + r * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
                s = pygame.Surface(flash_rect.size, pygame.SRCALPHA)
                s.fill(flash_color)
                self.screen.blit(s, flash_rect.topleft)

    def _draw_piece(self, piece, is_ghost=False):
        coords = self._get_piece_coords(piece)
        for x, y in coords:
            self._draw_cell(x, y, piece['color'], is_ghost)
            
    def _draw_cell(self, grid_c, grid_r, color_index, is_ghost=False, offset_x=0, offset_y=0):
        x = self.GRID_X + grid_c * self.CELL_SIZE + offset_x
        y = self.GRID_Y + grid_r * self.CELL_SIZE + offset_y
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        color = self.PIECE_COLORS[color_index]
        
        if is_ghost:
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((color[0], color[1], color[2], 50))
            self.screen.blit(s, rect.topleft)
            pygame.draw.rect(s, (255,255,255,70), s.get_rect(), 1)
            self.screen.blit(s, rect.topleft)
        else:
            # Main block color with a subtle 3D effect
            light_color = tuple(min(255, c + 40) for c in color)
            dark_color = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(self.screen, dark_color, rect)
            pygame.draw.rect(self.screen, color, rect.inflate(-2, -2))
            pygame.draw.rect(self.screen, light_color, (x+1, y+1, self.CELL_SIZE-2, self.CELL_SIZE-4))
            pygame.draw.rect(self.screen, color, (x+1, y+1, self.CELL_SIZE-3, self.CELL_SIZE-5))


    def _render_ui(self):
        ui_x = self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20
        
        # --- Next Piece Box ---
        self._draw_ui_box(ui_x, self.GRID_Y, 140, 120, "NEXT")
        next_piece_obj = {
            'shape': self.next_piece_shape,
            'rotation': 0,
            'color': self.PIECE_KEYS.index(self.next_piece_shape) + 1
        }
        shape_coords = self.PIECE_SHAPES[next_piece_obj['shape']][0]
        
        # Center the piece in the preview box
        min_x = min(c[0] for c in shape_coords)
        max_x = max(c[0] for c in shape_coords)
        min_y = min(c[1] for c in shape_coords)
        max_y = max(c[1] for c in shape_coords)
        piece_width = (max_x - min_x + 1) * self.CELL_SIZE
        piece_height = (max_y - min_y + 1) * self.CELL_SIZE
        
        base_x = ui_x + (140 - piece_width) / 2
        base_y = self.GRID_Y + 40 + (80 - piece_height) / 2
        
        for dx, dy in shape_coords:
            self._draw_cell(0, 0, next_piece_obj['color'], False, base_x + (dx-min_x)*self.CELL_SIZE, base_y + (dy-min_y)*self.CELL_SIZE)

        # --- Score Box ---
        self._draw_ui_box(ui_x, self.GRID_Y + 140, 140, 80, "SCORE")
        score_text = self.font_l.render(f"{self.score}", True, self.COLOR_TEXT_VALUE)
        self.screen.blit(score_text, (ui_x + (140 - score_text.get_width()) / 2, self.GRID_Y + 170))

        # --- Lines Box ---
        self._draw_ui_box(ui_x, self.GRID_Y + 240, 140, 80, "LINES")
        lines_text = self.font_l.render(f"{self.lines_cleared}", True, self.COLOR_TEXT_VALUE)
        self.screen.blit(lines_text, (ui_x + (140 - lines_text.get_width()) / 2, self.GRID_Y + 270))

    def _draw_ui_box(self, x, y, w, h, title):
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (x, y, w, h))
        pygame.draw.rect(self.screen, self.COLOR_UI_FRAME, (x, y, w, h), 2)
        title_text = self.font_m.render(title, True, self.COLOR_TEXT)
        self.screen.blit(title_text, (x + (w - title_text.get_width()) / 2, y + 8))

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

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Key mapping for human play ---
    key_map = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
    }

    # --- Pygame setup for rendering ---
    pygame.display.set_caption("Puzzle Game")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    while not done:
        # --- Action generation from keyboard ---
        movement = 0 # No-op
        space = 0
        shift = 0
        
        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement = move_val
                break
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]

        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling & FPS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Match the environment's intended FPS

    print(f"Game Over! Final Score: {info['score']}, Lines: {info['lines_cleared']}")
    pygame.quit()