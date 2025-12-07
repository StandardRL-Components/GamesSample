
# Generated: 2025-08-27T22:35:02.562631
# Source Brief: brief_03174.md
# Brief Index: 3174

        
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
        "Controls: ←→ to move, ↑↓ to rotate. Hold shift for soft drop, press space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, falling block puzzle game. Clear lines to score points and prevent the stack from reaching the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT = 10, 20
    GRID_SIZE = 18
    PLAYFIELD_OFFSET_X = (SCREEN_WIDTH - PLAYFIELD_WIDTH * GRID_SIZE) // 2
    PLAYFIELD_OFFSET_Y = (SCREEN_HEIGHT - PLAYFIELD_HEIGHT * GRID_SIZE) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_WHITE = (255, 255, 255)

    # Tetromino shapes and colors
    PIECE_SHAPES = [
        [[1, 1, 1, 1]],  # I
        [[1, 1, 1], [0, 0, 1]],  # J
        [[1, 1, 1], [1, 0, 0]],  # L
        [[1, 1], [1, 1]],  # O
        [[0, 1, 1], [1, 1, 0]],  # S
        [[1, 1, 1], [0, 1, 0]],  # T
        [[1, 1, 0], [0, 1, 1]],  # Z
    ]
    PIECE_COLORS = [
        (66, 173, 244),   # I - Cyan
        (0, 0, 230),      # J - Blue
        (244, 160, 65),   # L - Orange
        (244, 232, 65),   # O - Yellow
        (100, 244, 65),   # S - Green
        (173, 65, 244),   # T - Purple
        (244, 65, 65),    # Z - Red
    ]

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

        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared_total = 0
        self.steps = 0
        self.game_over = False
        self.drop_timer = 0
        self.base_drop_speed = 30 # in frames (30fps = 1 sec)
        self.current_drop_speed = self.base_drop_speed
        
        self.line_clear_animation = [] # list of (row_index, timer)
        self.last_reward = 0

        self.reset()
        
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.PLAYFIELD_HEIGHT, self.PLAYFIELD_WIDTH), dtype=int)
        self.current_piece = self._create_new_piece()
        self.next_piece = self._create_new_piece()
        self.score = 0
        self.lines_cleared_total = 0
        self.steps = 0
        self.game_over = False
        self.current_drop_speed = self.base_drop_speed
        self.drop_timer = 0
        self.line_clear_animation = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action: 0:none, 1:up/rotCW, 2:down/rotCCW, 3:left, 4:right
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        # Handle player actions
        if not self.line_clear_animation: # Don't allow moves during line clear
            if movement == 1: self._rotate_piece(clockwise=True)
            elif movement == 2: self._rotate_piece(clockwise=False)
            elif movement == 3: self._move_piece(-1)
            elif movement == 4: self._move_piece(1)
            
            if space_held:
                reward += self._hard_drop()
            else:
                self.drop_timer += 1
                if shift_held:
                    # Soft drop accelerates fall
                    self.drop_timer += 4 

        # Automatic drop
        if not space_held and self.drop_timer >= self.current_drop_speed:
            self.drop_timer = 0
            if not self._move_piece(0, 1):
                reward += self._lock_piece()

        # Update line clear animation
        if self.line_clear_animation:
            new_animation = []
            for row, timer in self.line_clear_animation:
                if timer > 1:
                    new_animation.append((row, timer - 1))
            self.line_clear_animation = new_animation
            if not self.line_clear_animation:
                self._finish_line_clear()

        self.steps += 1
        win_condition = self.lines_cleared_total >= 10
        terminated = self.game_over or self.steps >= 1000 or win_condition
        
        if self.game_over:
            reward = -50.0
        elif win_condition:
            reward += 50.0
            
        self.last_reward = reward
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_new_piece(self):
        piece_idx = self.np_random.integers(0, len(self.PIECE_SHAPES))
        shape = self.PIECE_SHAPES[piece_idx]
        return {
            'shape': shape,
            'color_idx': piece_idx + 1,
            'x': self.PLAYFIELD_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0,
            'rotation': 0
        }

    def _is_valid_position(self, piece_shape, x, y):
        for r, row in enumerate(piece_shape):
            for c, cell in enumerate(row):
                if cell:
                    px, py = x + c, y + r
                    if not (0 <= px < self.PLAYFIELD_WIDTH and 0 <= py < self.PLAYFIELD_HEIGHT and self.grid[py, px] == 0):
                        return False
        return True

    def _rotate_piece(self, clockwise=True):
        shape = self.current_piece['shape']
        if clockwise:
            new_shape = [list(row) for row in zip(*shape[::-1])]
        else:
            new_shape = [list(row) for row in zip(*shape)][::-1]

        # Wall kick
        for offset in [0, 1, -1, 2, -2]:
            if self._is_valid_position(new_shape, self.current_piece['x'] + offset, self.current_piece['y']):
                self.current_piece['shape'] = new_shape
                self.current_piece['x'] += offset
                # sfx: rotate
                break

    def _move_piece(self, dx, dy=0):
        if self._is_valid_position(self.current_piece['shape'], self.current_piece['x'] + dx, self.current_piece['y'] + dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            # sfx: move
            return True
        return False

    def _lock_piece(self):
        # sfx: lock
        shape = self.current_piece['shape']
        x, y = self.current_piece['x'], self.current_piece['y']
        
        # Check for safe placement
        max_y_of_piece = 0
        for r_idx, row in enumerate(shape):
             if any(cell for cell in row):
                 max_y_of_piece = y + r_idx
        
        is_safe_placement = max_y_of_piece < self.PLAYFIELD_HEIGHT - 3
        reward = 0.1 if not is_safe_placement else -0.02
        
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self.grid[y + r, x + c] = self.current_piece['color_idx']

        reward += self._check_and_clear_lines()
        
        self.current_piece = self.next_piece
        self.next_piece = self._create_new_piece()

        if not self._is_valid_position(self.current_piece['shape'], self.current_piece['x'], self.current_piece['y']):
            self.game_over = True
            # sfx: game_over
            
        return reward

    def _hard_drop(self):
        # sfx: hard_drop
        while self._move_piece(0, 1):
            pass
        return self._lock_piece()

    def _check_and_clear_lines(self):
        lines_to_clear = []
        for r in range(self.PLAYFIELD_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                lines_to_clear.append(r)

        if lines_to_clear:
            # sfx: line_clear
            for r in lines_to_clear:
                self.line_clear_animation.append((r, 10)) # Animate for 10 frames
            
            num_cleared = len(lines_to_clear)
            self.lines_cleared_total += num_cleared
            
            # Update difficulty: 0.05s faster per 2 lines = 1.5 frames at 30fps
            speed_increase = int(self.lines_cleared_total / 2) * 1.5
            self.current_drop_speed = max(5, self.base_drop_speed - speed_increase)

            if num_cleared == 1: return 1.0
            if num_cleared == 2: return 3.0
            if num_cleared == 3: return 7.0
            if num_cleared >= 4: return 15.0
        return 0.0
    
    def _finish_line_clear(self):
        cleared_rows = sorted([row for row, timer in self.line_clear_animation if timer <=1], reverse=True)
        if not cleared_rows: return

        for r in cleared_rows:
            self.grid = np.delete(self.grid, r, axis=0)
            
        new_rows = np.zeros((len(cleared_rows), self.PLAYFIELD_WIDTH), dtype=int)
        self.grid = np.vstack((new_rows, self.grid))

    def _get_ghost_position(self):
        ghost_y = self.current_piece['y']
        while self._is_valid_position(self.current_piece['shape'], self.current_piece['x'], ghost_y + 1):
            ghost_y += 1
        return ghost_y

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
            "lines_cleared": self.lines_cleared_total,
        }
        
    def _draw_block(self, surface, x, y, color_idx, is_ghost=False):
        if color_idx == 0: return
        base_color = self.PIECE_COLORS[color_idx - 1]
        
        rect = pygame.Rect(x, y, self.GRID_SIZE, self.GRID_SIZE)
        
        if is_ghost:
            pygame.draw.rect(surface, base_color, rect, 2, border_radius=3)
        else:
            light_color = tuple(min(255, c + 40) for c in base_color)
            dark_color = tuple(max(0, c - 40) for c in base_color)
            
            pygame.draw.rect(surface, light_color, rect, border_radius=3)
            pygame.draw.rect(surface, base_color, rect.inflate(-4, -4), border_radius=2)
            
            # Simple 3D effect
            pygame.draw.line(surface, dark_color, rect.bottomleft, rect.bottomright, 2)
            pygame.draw.line(surface, dark_color, rect.topright, rect.bottomright, 2)

    def _render_game(self):
        # Draw playfield border
        border_rect = pygame.Rect(self.PLAYFIELD_OFFSET_X - 2, self.PLAYFIELD_OFFSET_Y - 2,
                                  self.PLAYFIELD_WIDTH * self.GRID_SIZE + 4, self.PLAYFIELD_HEIGHT * self.GRID_SIZE + 4)
        pygame.draw.rect(self.screen, self.COLOR_GRID, border_rect, 2, border_radius=5)

        # Draw grid and locked pieces
        for r in range(self.PLAYFIELD_HEIGHT):
            is_clearing = any(anim_r == r for anim_r, timer in self.line_clear_animation)
            if is_clearing:
                flash_rect = pygame.Rect(self.PLAYFIELD_OFFSET_X, self.PLAYFIELD_OFFSET_Y + r * self.GRID_SIZE,
                                         self.PLAYFIELD_WIDTH * self.GRID_SIZE, self.GRID_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_WHITE, flash_rect)
            else:
                for c in range(self.PLAYFIELD_WIDTH):
                    color_idx = self.grid[r, c]
                    if color_idx != 0:
                        self._draw_block(self.screen, 
                                       self.PLAYFIELD_OFFSET_X + c * self.GRID_SIZE,
                                       self.PLAYFIELD_OFFSET_Y + r * self.GRID_SIZE,
                                       color_idx)

        # Draw ghost piece
        if not self.game_over and self.current_piece:
            ghost_y = self._get_ghost_position()
            shape = self.current_piece['shape']
            for r_idx, row in enumerate(shape):
                for c_idx, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.screen,
                                       self.PLAYFIELD_OFFSET_X + (self.current_piece['x'] + c_idx) * self.GRID_SIZE,
                                       self.PLAYFIELD_OFFSET_Y + (ghost_y + r_idx) * self.GRID_SIZE,
                                       self.current_piece['color_idx'], is_ghost=True)

        # Draw current piece
        if not self.game_over and self.current_piece:
            shape = self.current_piece['shape']
            for r_idx, row in enumerate(shape):
                for c_idx, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.screen,
                                       self.PLAYFIELD_OFFSET_X + (self.current_piece['x'] + c_idx) * self.GRID_SIZE,
                                       self.PLAYFIELD_OFFSET_Y + (self.current_piece['y'] + r_idx) * self.GRID_SIZE,
                                       self.current_piece['color_idx'])

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"SCORE", True, self.COLOR_UI_TEXT)
        score_val = self.font_main.render(f"{self.score:.2f}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (40, 40))
        self.screen.blit(score_val, (40, 70))
        
        # Lines display
        lines_text = self.font_main.render(f"LINES", True, self.COLOR_UI_TEXT)
        lines_val = self.font_main.render(f"{self.lines_cleared_total}", True, self.COLOR_WHITE)
        self.screen.blit(lines_text, (40, 120))
        self.screen.blit(lines_val, (40, 150))

        # Next piece preview
        next_text = self.font_main.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (self.SCREEN_WIDTH - 150, 40))
        preview_box = pygame.Rect(self.SCREEN_WIDTH - 160, 70, 120, 80)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_box, 2, border_radius=5)
        
        if self.next_piece:
            shape = self.next_piece['shape']
            color_idx = self.next_piece['color_idx']
            start_x = preview_box.centerx - (len(shape[0]) * self.GRID_SIZE) / 2
            start_y = preview_box.centery - (len(shape) * self.GRID_SIZE) / 2
            for r_idx, row in enumerate(shape):
                for c_idx, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.screen, start_x + c_idx * self.GRID_SIZE, start_y + r_idx * self.GRID_SIZE, color_idx)

        # Game Over text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            go_text = self.font_main.render("GAME OVER", True, (255, 80, 80))
            text_rect = go_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(go_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    pygame.display.set_caption("Gymnasium Block Puzzle")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action mapping for human keyboard ---
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: 
                    obs, info = env.reset()
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Score: {info['score']:.2f}, Lines: {info['lines_cleared']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    env.close()