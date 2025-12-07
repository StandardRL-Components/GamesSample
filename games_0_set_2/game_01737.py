
# Generated: 2025-08-28T02:34:20.874710
# Source Brief: brief_01737.md
# Brief Index: 1737

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    """
    A fast-paced, grid-based puzzle game where the player manipulates falling 
    tetrominoes to clear lines and achieve a target score. This environment is
    designed for visual quality and engaging gameplay, suitable for both human
    players and reinforcement learning agents.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to rotate, ↓ for soft drop. Space for hard drop, Shift to hold piece."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based puzzle game. Manipulate falling tetrominoes to clear lines and achieve a high score before the stack reaches the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        self.MAX_STEPS = 30 * 120 # 120 seconds at 30fps
        self.WIN_CONDITION_LINES = 10

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_DANGER = (60, 30, 30)
        self.COLORS = [
            (0, 0, 0),         # 0: Empty
            (0, 240, 240),     # 1: I (Cyan)
            (240, 240, 0),     # 2: O (Yellow)
            (160, 0, 240),     # 3: T (Purple)
            (0, 0, 240),       # 4: J (Blue)
            (240, 160, 0),     # 5: L (Orange)
            (0, 240, 0),       # 6: S (Green)
            (240, 0, 0),       # 7: Z (Red)
            (128, 128, 128)    # 8: Ghost
        ]
        self.COLOR_UI_BG = (10, 10, 20)
        self.COLOR_UI_BORDER = (80, 80, 100)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_FLASH = (255, 255, 255)

        # Tetromino shapes
        self.TETROMINOES = {
            'I': {'shape': np.array([[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]], dtype=int), 'color': 1},
            'O': {'shape': np.array([[1,1], [1,1]], dtype=int), 'color': 2},
            'T': {'shape': np.array([[0,1,0], [1,1,1], [0,0,0]], dtype=int), 'color': 3},
            'J': {'shape': np.array([[1,0,0], [1,1,1], [0,0,0]], dtype=int), 'color': 4},
            'L': {'shape': np.array([[0,0,1], [1,1,1], [0,0,0]], dtype=int), 'color': 5},
            'S': {'shape': np.array([[0,1,1], [1,1,0], [0,0,0]], dtype=int), 'color': 6},
            'Z': {'shape': np.array([[1,1,0], [0,1,1], [0,0,0]], dtype=int), 'color': 7},
        }
        self.PIECE_TYPES = list(self.TETROMINOES.keys())

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # Initialize state variables (these are reset in `reset`)
        self.grid = None
        self.current_piece = None
        self.next_piece_type = None
        self.held_piece_type = None
        self.can_hold = None
        self.score = None
        self.lines_cleared = None
        self.level = None
        self.steps = None
        self.game_over = None
        self.fall_speed = None
        self.fall_timer = None
        self.prev_up_pressed = None
        self.prev_space_pressed = None
        self.prev_shift_pressed = None
        self.move_cooldown = None
        self.line_clear_animation = None
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        self.score = 0
        self.lines_cleared = 0
        self.level = 0
        self.steps = 0
        self.game_over = False
        
        self.fall_speed = 1.0
        self.fall_timer = 0
        
        self.held_piece_type = None
        self.can_hold = True
        
        # Input state
        self.prev_up_pressed = False
        self.prev_space_pressed = False
        self.prev_shift_pressed = False
        self.move_cooldown = 0
        
        # Animation state
        self.line_clear_animation = [] # list of (y_index, timer)

        # Initialize pieces using a 7-bag system for fair distribution
        piece_bag = self.PIECE_TYPES[:]
        self.np_random.shuffle(piece_bag)
        self.piece_queue = piece_bag
        
        self.current_piece = None
        self._spawn_piece() # Spawns next piece into current
        self._spawn_piece() # Spawns a new next piece
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        reward = 0
        terminated = self.game_over

        if terminated:
            return self._get_observation(), 0, terminated, False, self._get_info()

        self.steps += 1
        
        # Handle player input
        reward += self._handle_input(action)
        
        # Update game physics (falling)
        fall_reward, piece_locked = self._update_fall()
        reward += fall_reward
        
        if piece_locked:
            # play sound: piece_lock
            lock_reward, lines_cleared = self._lock_piece()
            reward += lock_reward
            
            if lines_cleared > 0:
                # play sound: line_clear
                self.score += [0, 100, 300, 500, 800][lines_cleared] * (self.level + 1)
                self.lines_cleared += lines_cleared
                self.level = self.lines_cleared // 10
                self.fall_speed = max(0.05, 1.0 - self.lines_cleared * 0.05)
            
            if np.any(self.grid[0:2, :]): # Check for topping out
                self.game_over = True
                reward -= 100
            else:
                self._spawn_piece()
                self.can_hold = True
                if self._check_collision(self.current_piece, (0, 0)): # Check for game over on spawn
                    self.game_over = True
                    reward -= 100
        
        # Check termination conditions
        if self.lines_cleared >= self.WIN_CONDITION_LINES:
            self.game_over = True
            reward += 100
            # play sound: win
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            
        terminated = self.game_over
        if terminated and self.lines_cleared < self.WIN_CONDITION_LINES:
            # play sound: game_over
            pass
        
        # MUST return exactly this 5-tuple
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        if self.game_over: return 0
        
        movement, space_action, shift_action = action[0], action[1], action[2]
        reward = -0.01 # Small time penalty per step

        self.move_cooldown = max(0, self.move_cooldown - 1)
        if self.move_cooldown == 0:
            if movement == 3: self._move(-1); self.move_cooldown = 2
            elif movement == 4: self._move(1); self.move_cooldown = 2

        up_pressed = (movement == 1)
        if up_pressed and not self.prev_up_pressed: self._rotate()
        self.prev_up_pressed = up_pressed

        if movement == 2: self.fall_timer += 4; reward += 0.001

        shift_pressed = (shift_action == 1)
        if shift_pressed and not self.prev_shift_pressed and self.can_hold: self._hold_piece()
        self.prev_shift_pressed = shift_pressed

        space_pressed = (space_action == 1)
        if space_pressed and not self.prev_space_pressed:
            drop_dist = 0
            while not self._check_collision(self.current_piece, (0, drop_dist + 1)):
                drop_dist += 1
            if drop_dist > 0: self.current_piece['y'] += drop_dist; reward += 0.1 # play sound: hard_drop
            self.fall_timer = int(self.fall_speed * 30)
        self.prev_space_pressed = space_pressed
        return reward

    def _update_fall(self):
        if self.game_over: return 0, False
        self.fall_timer += 1
        fall_frequency = int(self.fall_speed * 30)
        if self.fall_timer >= fall_frequency:
            self.fall_timer = 0
            if not self._check_collision(self.current_piece, (0, 1)):
                self.current_piece['y'] += 1; return 0, False
            else: return 0, True
        return 0, False

    def _lock_piece(self):
        piece = self.current_piece
        for y, row in enumerate(piece['shape']):
            for x, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = piece['x'] + x, piece['y'] + y
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y, grid_x] = piece['color']
        
        lines_cleared, reward = self._clear_lines()
        max_height = 0
        for r in range(self.GRID_HEIGHT):
            if np.any(self.grid[r,:]): max_height = self.GRID_HEIGHT - r; break
        reward -= (max_height / self.GRID_HEIGHT) ** 2
        return reward, lines_cleared

    def _clear_lines(self):
        lines_to_clear = [r for r in range(self.GRID_HEIGHT) if np.all(self.grid[r, :])]
        if not lines_to_clear: return 0, 0
        for y in lines_to_clear: self.line_clear_animation.append([y, 5])
        
        new_grid = np.zeros_like(self.grid)
        new_row = self.GRID_HEIGHT - 1
        for r in range(self.GRID_HEIGHT - 1, -1, -1):
            if r not in lines_to_clear: new_grid[new_row, :], new_row = self.grid[r, :], new_row - 1
        self.grid = new_grid
        
        num_cleared = len(lines_to_clear)
        reward = [0, 1, 3, 5, 8][num_cleared]
        return num_cleared, reward

    def _spawn_piece(self):
        if len(self.piece_queue) < 2:
            new_bag = self.PIECE_TYPES[:]; self.np_random.shuffle(new_bag); self.piece_queue.extend(new_bag)
        
        self.current_piece_type = self.next_piece_type if self.current_piece else self.piece_queue.pop(0)
        self.next_piece_type = self.piece_queue.pop(0)

        p_data = self.TETROMINOES[self.current_piece_type]
        shape = p_data['shape']
        self.current_piece = {'type': self.current_piece_type, 'shape': shape, 'color': p_data['color'],
                              'x': (self.GRID_WIDTH - shape.shape[1]) // 2, 'y': 0}

    def _hold_piece(self):
        # play sound: hold
        if self.held_piece_type is None:
            self.held_piece_type = self.current_piece_type; self._spawn_piece()
        else:
            self.held_piece_type, self.current_piece_type = self.current_piece_type, self.held_piece_type
            p_data = self.TETROMINOES[self.current_piece_type]
            shape = p_data['shape']
            self.current_piece = {'type': self.current_piece_type, 'shape': shape, 'color': p_data['color'],
                                  'x': (self.GRID_WIDTH - shape.shape[1]) // 2, 'y': 0}
        self.can_hold = False

    def _move(self, dx):
        if not self._check_collision(self.current_piece, (dx, 0)):
            self.current_piece['x'] += dx # play sound: move

    def _rotate(self):
        p = self.current_piece
        if p['type'] == 'O': return
        original_shape = p['shape']; rotated_shape = np.rot90(original_shape, -1)
        for ox, oy in [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1)]:
            p['shape'] = rotated_shape
            if not self._check_collision(p, (ox, oy)):
                p['x'] += ox; p['y'] += oy; return # play sound: rotate
        p['shape'] = original_shape

    def _check_collision(self, piece, offset):
        px, py = offset
        for y, row in enumerate(piece['shape']):
            for x, cell in enumerate(row):
                if cell:
                    gx, gy = piece['x'] + x + px, piece['y'] + y + py
                    if not (0 <= gx < self.GRID_WIDTH and 0 <= gy < self.GRID_HEIGHT) or self.grid[gy, gx] != 0:
                        return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        danger_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, 2 * self.CELL_SIZE)
        s = pygame.Surface(danger_rect.size, pygame.SRCALPHA); s.fill((*self.COLOR_DANGER, 50)); self.screen.blit(s, danger_rect.topleft)

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] != 0: self._draw_block(x, y, self.grid[y, x])

        if not self.game_over and self.current_piece:
            drop_dist = 0
            while not self._check_collision(self.current_piece, (0, drop_dist + 1)): drop_dist += 1
            ghost_p = self.current_piece.copy(); ghost_p['y'] += drop_dist
            self._draw_piece(ghost_p, ghost=True)
            self._draw_piece(self.current_piece)
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, grid_rect, 2)
        
        active_anims = []
        for anim in self.line_clear_animation:
            y, timer = anim; rect = pygame.Rect(self.GRID_X, self.GRID_Y + y * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
            pygame.gfxdraw.box(self.screen, rect, (*self.COLOR_FLASH, int(255 * (timer / 5.0))))
            anim[1] -= 1
            if anim[1] > 0: active_anims.append(anim)
        self.line_clear_animation = active_anims
    
    def _draw_piece(self, piece, ghost=False):
        for y, row in enumerate(piece['shape']):
            for x, cell in enumerate(row):
                if cell: self._draw_block(piece['x'] + x, piece['y'] + y, 8 if ghost else piece['color'], ghost)
    
    def _draw_block(self, grid_x, grid_y, color_idx, ghost=False):
        sx, sy = self.GRID_X + grid_x * self.CELL_SIZE, self.GRID_Y + grid_y * self.CELL_SIZE
        color = self.COLORS[color_idx]
        if ghost: pygame.draw.rect(self.screen, color, (sx, sy, self.CELL_SIZE, self.CELL_SIZE), 2)
        else:
            light = tuple(min(255, c + 40) for c in color); dark = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(self.screen, dark, (sx, sy, self.CELL_SIZE, self.CELL_SIZE))
            pygame.draw.rect(self.screen, color, (sx + 2, sy + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4))
            pygame.draw.line(self.screen, light, (sx, sy), (sx + self.CELL_SIZE - 1, sy), 2)
            pygame.draw.line(self.screen, light, (sx, sy), (sx, sy + self.CELL_SIZE - 1), 2)

    def _render_ui(self):
        next_box = pygame.Rect(self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 10, self.GRID_Y, 100, 80)
        self._draw_ui_panel(next_box, "NEXT")
        if self.next_piece_type: self._draw_ui_piece(self.next_piece_type, next_box)

        hold_box = pygame.Rect(self.GRID_X - 110, self.GRID_Y, 100, 80)
        self._draw_ui_panel(hold_box, "HOLD")
        if self.held_piece_type:
            self._draw_ui_piece(self.held_piece_type, hold_box)
            if not self.can_hold: s = pygame.Surface(hold_box.size, pygame.SRCALPHA); s.fill((0,0,0,128)); self.screen.blit(s, hold_box.topleft)

        score_box = pygame.Rect(next_box.left, next_box.bottom + 10, 100, 120)
        self._draw_ui_panel(score_box)
        self._draw_text("SCORE", score_box.centerx, score_box.top + 15, self.font_small)
        self._draw_text(f"{self.score}", score_box.centerx, score_box.top + 40, self.font_main)
        self._draw_text("LINES", score_box.centerx, score_box.top + 70, self.font_small)
        self._draw_text(f"{self.lines_cleared}", score_box.centerx, score_box.top + 95, self.font_main)

        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA); s.fill((0, 0, 0, 180)); self.screen.blit(s, (0, 0))
            msg = "YOU WIN!" if self.lines_cleared >= self.WIN_CONDITION_LINES else "GAME OVER"
            self._draw_text(msg, self.WIDTH // 2, self.HEIGHT // 2 - 20, pygame.font.SysFont("Consolas", 48, bold=True))

    def _draw_ui_panel(self, rect, title=None):
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, rect); pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, rect, 2)
        if title: self._draw_text(title, rect.centerx, rect.top + 15, self.font_small)

    def _draw_ui_piece(self, piece_type, rect):
        p_data = self.TETROMINOES[piece_type]; shape = p_data['shape']; color_idx = p_data['color']
        bs = 10; w, h = shape.shape[1] * bs, shape.shape[0] * bs
        px, py = rect.centerx - w // 2, rect.centery - h // 2 + 10
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell: block_r = pygame.Rect(px + x * bs, py + y * bs, bs, bs); pygame.draw.rect(self.screen, self.COLORS[color_idx], block_r); pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, block_r, 1)

    def _draw_text(self, text, x, y, font, color=None):
        text_surface = font.render(text, True, color if color else self.COLOR_TEXT)
        self.screen.blit(text_surface, text_surface.get_rect(center=(x, y)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lines_cleared": self.lines_cleared, "level": self.level}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Call this at the end of __init__ to verify implementation:
        print("Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "dummy" to run headless
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    obs, info = env.reset()
    
    # To display the game, you'll need a different setup
    # This example just runs a few random steps
    print("Running 10 random steps...")
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print("Episode terminated.")
            break
    print("Example run finished.")
    env.close()