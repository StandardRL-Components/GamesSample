
# Generated: 2025-08-28T05:44:04.212720
# Source Brief: brief_05676.md
# Brief Index: 5676

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑ to rotate, ↓ for soft drop. Space for hard drop, Shift to hold a piece."
    )

    game_description = (
        "A grid-based puzzle game. Manipulate falling tetrominoes to clear rows and clear 10 lines before time runs out or the stack reaches the top."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    BLOCK_SIZE = 18
    MAX_STEPS = 3600 # 60 seconds at 60 FPS
    VICTORY_LINES = 10

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_UI_BG = (30, 30, 40)
    COLOR_UI_BORDER = (60, 60, 70)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_ACCENT = (255, 255, 100)
    COLOR_GAMEOVER = (200, 0, 0, 150)

    TETROMINO_COLORS = [
        (0, 0, 0),          # 0: Empty
        (3, 211, 252),      # I-piece (Cyan)
        (252, 186, 3),      # O-piece (Yellow)
        (177, 3, 252),      # T-piece (Purple)
        (3, 61, 252),       # J-piece (Blue)
        (252, 128, 3),      # L-piece (Orange)
        (252, 3, 3),        # S-piece (Red)
        (3, 252, 74),       # Z-piece (Green)
    ]

    TETROMINO_SHAPES = [
        [], # 0: Empty
        [[1, 0], [1, 1], [1, 2], [1, 3]],  # I
        [[0, 1], [0, 2], [1, 1], [1, 2]],  # O
        [[0, 1], [1, 0], [1, 1], [1, 2]],  # T
        [[0, 0], [1, 0], [1, 1], [1, 2]],  # J
        [[0, 2], [1, 0], [1, 1], [1, 2]],  # L
        [[0, 1], [0, 2], [1, 0], [1, 1]],  # S
        [[0, 0], [0, 1], [1, 1], [1, 2]],  # Z
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_m = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 36, bold=True)

        self.grid_pixel_width = self.GRID_WIDTH * self.BLOCK_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.BLOCK_SIZE
        self.grid_x_offset = (self.WIDTH - self.grid_pixel_width) // 2
        self.grid_y_offset = (self.HEIGHT - self.grid_pixel_height) // 2

        self.reset()
        
        # This check is not part of the standard gym API but is required by the prompt
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        self.piece_bag = list(range(1, len(self.TETROMINO_SHAPES)))
        self.np_random.shuffle(self.piece_bag)
        
        self.current_piece = self._new_piece()
        self.next_piece = self._new_piece()
        self.held_piece_shape_idx = 0
        self.can_swap_hold = True

        self.score = 0
        self.total_lines_cleared = 0
        self.steps = 0
        self.time_remaining = self.MAX_STEPS
        
        self.game_over = False
        self.victory = False

        self.fall_timer = 0
        self.fall_speed = 30 # Ticks per fall (at 60fps, this is 0.5s)

        self.move_timer = 0
        self.move_delay = 5 # Ticks before repeated move
        self.last_movement = 0

        self.last_up_pressed = False
        self.last_space_pressed = False
        self.last_shift_pressed = False
        
        self.line_clear_animation = None # Will be a tuple (list_of_rows, timer)

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.001 # Small penalty for time passing
        self.steps += 1
        self.time_remaining -= 1

        if self.line_clear_animation:
            self.line_clear_animation = (self.line_clear_animation[0], self.line_clear_animation[1] - 1)
            if self.line_clear_animation[1] <= 0:
                self._finish_line_clear()
                self.line_clear_animation = None
            # Pause game during animation
        elif not self.game_over and not self.victory:
            # --- Handle Input ---
            self._handle_input(movement, space_held, shift_held)
            
            # --- Apply Gravity & Lock Piece ---
            soft_dropping = movement == 2
            if soft_dropping:
                self.fall_timer += self.fall_speed // 2 + 1 # Faster drop
                reward += 0.01

            self.fall_timer += 1
            if self.fall_timer >= self.fall_speed:
                self.fall_timer = 0
                self.current_piece['y'] += 1
                if self._check_collision(self.current_piece):
                    self.current_piece['y'] -= 1
                    lock_reward = self._lock_piece()
                    reward += lock_reward
                    
                    lines_cleared = self._check_and_start_line_clear()
                    if lines_cleared > 0:
                        # sfx: line clear
                        reward += lines_cleared ** 2 # Bonus for multi-line clears

        # --- Check Termination Conditions ---
        if self.total_lines_cleared >= self.VICTORY_LINES and not self.victory:
            self.victory = True
            reward += 10 # Victory reward
        
        terminated = self.game_over or self.victory or self.time_remaining <= 0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # --- Rotation (Up Arrow) ---
        if movement == 1 and not self.last_up_pressed:
            # sfx: rotate
            rotated_piece = self.current_piece.copy()
            rotated_piece['shape'] = self._rotate_shape(rotated_piece['shape'])
            if not self._check_collision(rotated_piece):
                self.current_piece = rotated_piece
            else: # Wall kick
                for dx in [-1, 1, -2, 2]:
                    kicked_piece = rotated_piece.copy()
                    kicked_piece['x'] += dx
                    if not self._check_collision(kicked_piece):
                        self.current_piece = kicked_piece
                        break
        self.last_up_pressed = (movement == 1)

        # --- Horizontal Movement (Left/Right) ---
        if movement in [3, 4]:
            if movement != self.last_movement:
                self.move_timer = 0
            
            self.move_timer += 1
            if self.move_timer == 1 or self.move_timer > self.move_delay:
                # sfx: move
                dx = -1 if movement == 3 else 1
                self.current_piece['x'] += dx
                if self._check_collision(self.current_piece):
                    self.current_piece['x'] -= dx
        else:
            self.move_timer = 0
        self.last_movement = movement

        # --- Hard Drop (Space) ---
        if space_held and not self.last_space_pressed:
            # sfx: hard drop
            while not self._check_collision(self.current_piece):
                self.current_piece['y'] += 1
                self.score += 2 # Small bonus for hard drop
            self.current_piece['y'] -= 1
            self.fall_timer = self.fall_speed # Force immediate lock
        self.last_space_pressed = space_held
        
        # --- Hold (Shift) ---
        if shift_held and not self.last_shift_pressed and self.can_swap_hold:
            # sfx: hold
            self.can_swap_hold = False
            if self.held_piece_shape_idx == 0:
                self.held_piece_shape_idx = self.current_piece['shape_idx']
                self.current_piece = self._new_piece()
            else:
                current_idx = self.current_piece['shape_idx']
                self.current_piece = self._new_piece(force_shape_idx=self.held_piece_shape_idx)
                self.held_piece_shape_idx = current_idx
        self.last_shift_pressed = shift_held


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
            "lines_cleared": self.total_lines_cleared,
            "time_remaining": self.time_remaining,
        }

    def close(self):
        pygame.quit()

    # --- Helper Functions: Game Logic ---

    def _new_piece(self, force_shape_idx=None):
        if force_shape_idx:
             shape_idx = force_shape_idx
        else:
            if not self.piece_bag:
                self.piece_bag = list(range(1, len(self.TETROMINO_SHAPES)))
                self.np_random.shuffle(self.piece_bag)
            shape_idx = self.piece_bag.pop()
            
        shape = self.TETROMINO_SHAPES[shape_idx]
        piece = {
            'shape_idx': shape_idx,
            'shape': shape,
            'x': self.GRID_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0,
            'color': self.TETROMINO_COLORS[shape_idx]
        }
        if self._check_collision(piece):
            self.game_over = True
        return piece

    def _check_collision(self, piece):
        for r_offset, row in enumerate(piece['shape']):
            for c_offset, cell in enumerate(row):
                if cell:
                    grid_y = piece['y'] + r_offset
                    grid_x = piece['x'] + c_offset
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return True  # Wall collision
                    if self.grid[grid_y, grid_x] != 0:
                        return True  # Other piece collision
        return False

    def _lock_piece(self):
        # sfx: lock
        avg_height = 0
        stack_rows = 0
        for r in range(self.GRID_HEIGHT):
            if np.any(self.grid[r, :]):
                avg_height += (self.GRID_HEIGHT - r)
                stack_rows += 1
        avg_height = avg_height / stack_rows if stack_rows > 0 else 0

        placement_height = 0
        piece_y_max = 0
        for r_offset, row in enumerate(self.current_piece['shape']):
            if np.any(row):
                 piece_y_max = max(piece_y_max, self.current_piece['y'] + r_offset)
        
        placement_height = self.GRID_HEIGHT - piece_y_max
        
        reward = 0
        if placement_height > avg_height:
             reward += 0.2 # Risky placement
        else:
             reward -= 0.1 # Safe placement

        for r_offset, row in enumerate(self.current_piece['shape']):
            for c_offset, cell in enumerate(row):
                if cell:
                    self.grid[self.current_piece['y'] + r_offset, self.current_piece['x'] + c_offset] = self.current_piece['shape_idx']
        
        self.current_piece = self.next_piece
        self.next_piece = self._new_piece()
        self.can_swap_hold = True
        return reward

    def _check_and_start_line_clear(self):
        full_rows = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                full_rows.append(r)
        
        if full_rows:
            self.line_clear_animation = (full_rows, 10) # Animate for 10 frames
        return len(full_rows)

    def _finish_line_clear(self):
        rows_to_clear, _ = self.line_clear_animation
        
        # Update score and line count
        lines_cleared = len(rows_to_clear)
        score_map = {1: 100, 2: 300, 3: 500, 4: 800}
        self.score += score_map.get(lines_cleared, 0)
        self.total_lines_cleared += lines_cleared
        
        # Create a new grid without the cleared lines
        new_grid = np.zeros_like(self.grid)
        new_row_idx = self.GRID_HEIGHT - 1
        for r in range(self.GRID_HEIGHT - 1, -1, -1):
            if r not in rows_to_clear:
                new_grid[new_row_idx, :] = self.grid[r, :]
                new_row_idx -= 1
        self.grid = new_grid

    def _rotate_shape(self, shape):
        return [[shape[y][x] for y in range(len(shape))] for x in range(len(shape[0]) - 1, -1, -1)]

    # --- Helper Functions: Rendering ---
    
    def _draw_block(self, surface, x, y, color):
        rect = pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        # Main color
        pygame.draw.rect(surface, color, rect)
        
        # 3D effect
        highlight = tuple(min(c + 50, 255) for c in color)
        shadow = tuple(max(c - 50, 0) for c in color)
        pygame.draw.line(surface, highlight, rect.topleft, rect.topright, 2)
        pygame.draw.line(surface, highlight, rect.topleft, rect.bottomleft, 2)
        pygame.draw.line(surface, shadow, rect.bottomright, rect.topright, 2)
        pygame.draw.line(surface, shadow, rect.bottomright, rect.bottomleft, 2)

    def _draw_piece(self, surface, piece, offset_x, offset_y, is_ghost=False):
        color = piece['color']
        if is_ghost:
            color = (color[0] // 4, color[1] // 4, color[2] // 4)
            
        for r_offset, row in enumerate(piece['shape']):
            for c_offset, cell in enumerate(row):
                if cell:
                    x = offset_x + (piece['x'] + c_offset) * self.BLOCK_SIZE
                    y = offset_y + (piece['y'] + r_offset) * self.BLOCK_SIZE
                    if is_ghost:
                        pygame.draw.rect(surface, (100, 100, 100), (x, y, self.BLOCK_SIZE, self.BLOCK_SIZE), 1)
                    else:
                        self._draw_block(surface, x, y, color)

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.grid_x_offset, self.grid_y_offset, self.grid_pixel_width, self.grid_pixel_height))

        # Draw locked pieces
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    color = self.TETROMINO_COLORS[self.grid[r, c]]
                    x = self.grid_x_offset + c * self.BLOCK_SIZE
                    y = self.grid_y_offset + r * self.BLOCK_SIZE
                    self._draw_block(self.screen, x, y, color)
        
        # Draw line clear animation
        if self.line_clear_animation:
            rows, timer = self.line_clear_animation
            flash_color = (255, 255, 255) if (timer // 2) % 2 == 0 else self.COLOR_GRID
            for r in rows:
                pygame.draw.rect(self.screen, flash_color, (self.grid_x_offset, self.grid_y_offset + r * self.BLOCK_SIZE, self.grid_pixel_width, self.BLOCK_SIZE))

        if not self.game_over and not self.victory:
            # Draw ghost piece
            ghost_piece = self.current_piece.copy()
            while not self._check_collision(ghost_piece):
                ghost_piece['y'] += 1
            ghost_piece['y'] -= 1
            self._draw_piece(self.screen, ghost_piece, self.grid_x_offset, self.grid_y_offset, is_ghost=True)

            # Draw current piece
            self._draw_piece(self.screen, self.current_piece, self.grid_x_offset, self.grid_y_offset)

        # Draw grid border
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, (self.grid_x_offset, self.grid_y_offset, self.grid_pixel_width, self.grid_pixel_height), 2)
    
    def _render_ui(self):
        # --- Info Panels ---
        panel_width = 120
        panel_height = 80
        
        # Next Piece Panel
        next_x = self.grid_x_offset + self.grid_pixel_width + 20
        next_y = self.grid_y_offset
        self._draw_info_panel(next_x, next_y, panel_width, panel_height, "NEXT", self.next_piece)
        
        # Held Piece Panel
        held_x = self.grid_x_offset - panel_width - 20
        held_y = self.grid_y_offset
        held_piece = self._get_held_piece_as_dict()
        self._draw_info_panel(held_x, held_y, panel_width, panel_height, "HOLD", held_piece)
        
        # --- Top Bar Info ---
        score_text = self.font_m.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))
        
        time_str = f"{self.time_remaining // 60:02}:{self.time_remaining % 60:02}"
        time_text = self.font_m.render(f"TIME: {time_str}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(centerx=self.WIDTH / 2, y=15)
        self.screen.blit(time_text, time_rect)
        
        lines_text = self.font_m.render(f"LINES: {self.total_lines_cleared}/{self.VICTORY_LINES}", True, self.COLOR_TEXT)
        lines_rect = lines_text.get_rect(right=self.WIDTH - 20, y=15)
        self.screen.blit(lines_text, lines_rect)

        # --- Game Over / Victory Screen ---
        if self.game_over or self.victory:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_GAMEOVER)
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY!" if self.victory else "GAME OVER"
            text = self.font_l.render(message, True, self.COLOR_TEXT_ACCENT)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)

    def _draw_info_panel(self, x, y, w, h, title, piece):
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (x, y, w, h))
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, (x, y, w, h), 2)
        
        title_text = self.font_m.render(title, True, self.COLOR_TEXT)
        title_rect = title_text.get_rect(centerx=x + w/2, y=y + 5)
        self.screen.blit(title_text, title_rect)
        
        if piece:
            shape = piece['shape']
            shape_w = (max(c for r,c in shape) + 1) * self.BLOCK_SIZE
            shape_h = (max(r for r,c in shape) + 1) * self.BLOCK_SIZE
            
            piece_draw_x = x + (w - shape_w) / 2
            piece_draw_y = y + (h - shape_h) / 2 + 10

            for r_offset, row in enumerate(shape):
                for c_offset, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.screen, piece_draw_x + c_offset * self.BLOCK_SIZE, piece_draw_y + r_offset * self.BLOCK_SIZE, piece['color'])

    def _get_held_piece_as_dict(self):
        if self.held_piece_shape_idx == 0:
            return None
        shape = self.TETROMINO_SHAPES[self.held_piece_shape_idx]
        return {
            'shape_idx': self.held_piece_shape_idx,
            'shape': shape,
            'x': 0, 'y': 0,
            'color': self.TETROMINO_COLORS[self.held_piece_shape_idx]
        }
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's not part of the required Gymnasium interface but is useful for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tetris Environment")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Input to Action Mapping ---
        movement = 0 # 0: none
        space_held = 0 # 0: released
        shift_held = 0 # 0: released
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        # Display reward for debugging
        reward_text = env.font_m.render(f"Total Reward: {total_reward:.2f}", True, (255, 255, 0))
        screen.blit(reward_text, (10, env.HEIGHT - 30))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward}")
            # In a real scenario, you might wait for a reset command
            # Here, we'll just keep showing the final screen until 'r' or quit
            pass

        clock.tick(60) # Run at 60 FPS
        
    env.close()