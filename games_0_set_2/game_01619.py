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
        "Controls: ←→ to move, ↑ to rotate, ↓ for soft drop. Space for hard drop, Shift to hold a piece."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic falling block puzzle game. Place pieces to clear lines, score points, and try to survive as the speed increases. Clear 20 lines to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.rng = np.random.default_rng()

        # Game constants
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 20
        self.CELL_SIZE = 18
        self.GRID_X = (self.screen_width - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y = (self.screen_height - self.GRID_HEIGHT * self.CELL_SIZE) // 2

        self.MAX_STEPS = 2000
        self.WIN_CONDITION = 20

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_WHITE = (255, 255, 255)
        self.PIECE_COLORS = [
            (0, 0, 0),  # 0: Empty
            (255, 80, 80),   # 1: S (Red)
            (80, 255, 80),   # 2: Z (Green)
            (80, 80, 255),   # 3: J (Blue)
            (255, 165, 0),  # 4: L (Orange)
            (80, 220, 220),  # 5: I (Cyan)
            (255, 255, 80),  # 6: O (Yellow)
            (160, 80, 240),  # 7: T (Purple)
        ]
        self.PIECE_SHADOW_COLORS = [tuple(max(0, c - 60) for c in color) for color in self.PIECE_COLORS]

        # Fonts
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Piece shapes
        self.PIECES = {
            1: [[[0,1,1],[1,1,0],[0,0,0]]], # S
            2: [[[1,1,0],[0,1,1],[0,0,0]]], # Z
            3: [[[1,0,0],[1,1,1],[0,0,0]]], # J
            4: [[[0,0,1],[1,1,1],[0,0,0]]], # L
            5: [[[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]]], # I
            6: [[[1,1],[1,1]]], # O
            7: [[[0,1,0],[1,1,1],[0,0,0]]], # T
        }
        self._generate_rotations()

        # Initialize state variables
        self.grid = None
        self.current_piece = None
        self.next_piece_type = None
        self.held_piece_type = None
        self.can_hold = True
        self.fall_timer = 0
        self.base_fall_speed = 15 # frames per grid cell
        self.fall_speed_frames = self.base_fall_speed
        self.previous_space_held = False
        self.previous_shift_held = False
        self.line_clear_animation = [] # list of (y_coord, timer)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        # This call was causing the error during __init__, it's good practice to
        # let the user call reset() for the first time.
        # However, to maintain the original structure, we'll ensure it works.
        self.reset()

    def _generate_rotations(self):
        for piece_type in [1, 2, 3, 4, 7]: # S, Z, J, L, T
            base_shape = np.array(self.PIECES[piece_type][0])
            rotations = [base_shape.tolist()]
            for _ in range(3):
                base_shape = np.rot90(base_shape)
                rotations.append(base_shape.tolist())
            self.PIECES[piece_type] = rotations
        # I piece has special rotations
        i_shape = np.array(self.PIECES[5][0])
        self.PIECES[5] = [i_shape.tolist(), np.rot90(i_shape).tolist()]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.fall_timer = 0
        self.fall_speed_frames = self.base_fall_speed
        self.can_hold = True
        self.held_piece_type = None
        self.line_clear_animation = []
        self.previous_space_held = False
        self.previous_shift_held = False

        self.next_piece_type = self.rng.integers(1, len(self.PIECES) + 1)
        self._spawn_piece()
        
        return self._get_observation(), self._get_info()

    def _spawn_piece(self):
        self.current_piece = {
            "type": self.next_piece_type,
            "rotation": 0,
            "x": self.GRID_WIDTH // 2 - 2,
            "y": 0,
        }
        self.next_piece_type = self.rng.integers(1, len(self.PIECES) + 1)
        self.can_hold = True
        if self._check_collision(0, 0):
            self.game_over = True

    def _get_piece_shape(self, piece):
        piece_shapes = self.PIECES[piece["type"]]
        return piece_shapes[piece["rotation"] % len(piece_shapes)]

    def _check_collision(self, dx, dy, piece=None):
        if piece is None:
            piece = self.current_piece
        
        shape = self._get_piece_shape(piece)
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x = piece["x"] + c + dx
                    grid_y = piece["y"] + r + dy
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return True # Wall collision
                    if self.grid[grid_y, grid_x] != 0:
                        return True # Piece collision
        return False

    def _rotate_piece(self):
        original_rotation = self.current_piece["rotation"]
        self.current_piece["rotation"] = (self.current_piece["rotation"] + 1) % len(self.PIECES[self.current_piece["type"]])
        
        # Wall kick logic
        if self._check_collision(0, 0):
            # Try moving right
            if not self._check_collision(1, 0):
                self.current_piece["x"] += 1
            # Try moving left
            elif not self._check_collision(-1, 0):
                self.current_piece["x"] -= 1
            # Try moving further right (for I-piece)
            elif not self._check_collision(2, 0):
                self.current_piece["x"] += 2
            # Try moving further left (for I-piece)
            elif not self._check_collision(-2, 0):
                self.current_piece["x"] -= 2
            # If all kicks fail, revert rotation
            else:
                self.current_piece["rotation"] = original_rotation
                return False
        return True

    def _place_piece(self):
        shape = self._get_piece_shape(self.current_piece)
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x = self.current_piece["x"] + c
                    grid_y = self.current_piece["y"] + r
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y, grid_x] = self.current_piece["type"]
        # sfx: piece_lock.wav

    def _clear_lines(self):
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r] > 0):
                lines_to_clear.append(r)
        
        if lines_to_clear:
            # sfx: line_clear.wav
            for r in lines_to_clear:
                self.grid[r] = 0
                self.line_clear_animation.append({"y": r, "timer": 5})
            
            # Shift rows down
            cleared_count = len(lines_to_clear)
            rows_to_keep = [r for r in range(self.GRID_HEIGHT) if r not in lines_to_clear]
            new_grid = np.zeros_like(self.grid)
            new_grid[cleared_count:] = self.grid[rows_to_keep]
            self.grid = new_grid
            
            self.lines_cleared += cleared_count
            
            # Update fall speed
            speed_increase_tiers = self.lines_cleared // 5
            self.fall_speed_frames = max(3, self.base_fall_speed - speed_increase_tiers)
            
            return cleared_count
        return 0

    def _hold_piece(self):
        if not self.can_hold:
            return
        
        # sfx: hold.wav
        if self.held_piece_type is None:
            self.held_piece_type = self.current_piece["type"]
            self._spawn_piece()
        else:
            self.held_piece_type, self.current_piece["type"] = self.current_piece["type"], self.held_piece_type
            self.current_piece["rotation"] = 0
            self.current_piece["x"] = self.GRID_WIDTH // 2 - 2
            self.current_piece["y"] = 0
            if self._check_collision(0, 0):
                self.game_over = True
        
        self.can_hold = False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_action, shift_action = action
        space_pressed = space_action == 1 and not self.previous_space_held
        shift_pressed = shift_action == 1 and not self.previous_shift_held

        reward = -0.01  # Small penalty per step to encourage speed
        lines_cleared_this_step = 0
        action_taken = False

        # 1. Handle player input
        if shift_pressed:
            self._hold_piece()
            reward -= 0.2 # Safe action penalty
            action_taken = True
        elif space_pressed:
            # Hard drop
            while not self._check_collision(0, 1):
                self.current_piece["y"] += 1
                reward += 0.02 # Small reward for dropping
            self.fall_timer = self.fall_speed_frames # Force lock
            action_taken = True
        else:
            # Movement
            if movement == 1: # Up -> Rotate
                if self._rotate_piece():
                    reward -= 0.2
                    action_taken = True
            elif movement == 2: # Down -> Soft drop
                if not self._check_collision(0, 1):
                    self.current_piece["y"] += 1
                    self.fall_timer = 0
                    reward += 0.01
                    action_taken = True
            elif movement == 3: # Left
                if not self._check_collision(-1, 0):
                    self.current_piece["x"] -= 1
                    reward -= 0.2
                    action_taken = True
            elif movement == 4: # Right
                if not self._check_collision(1, 0):
                    self.current_piece["x"] += 1
                    reward -= 0.2
                    action_taken = True

        # 2. Update game physics (auto-fall)
        self.fall_timer += 1
        if self.fall_timer >= self.fall_speed_frames:
            self.fall_timer = 0
            if not self._check_collision(0, 1):
                self.current_piece["y"] += 1
            else:
                # Lock piece
                self._place_piece()
                lines_cleared_this_step = self._clear_lines()
                self._spawn_piece() # This also checks for game over
        
        # 3. Calculate rewards
        if lines_cleared_this_step > 0:
            rewards = {1: 1, 2: 2, 3: 4, 4: 8}
            reward += rewards.get(lines_cleared_this_step, 0)
            self.score += rewards.get(lines_cleared_this_step, 0) * 100

        # 4. Check for termination
        terminated = False
        truncated = False
        if self.game_over:
            terminated = True
            reward = -100
        elif self.lines_cleared >= self.WIN_CONDITION:
            terminated = True
            reward = 100
            self.score += 1000 # Win bonus
        elif self.steps >= self.MAX_STEPS:
            truncated = True

        # 5. Update state
        self.steps += 1
        self.previous_space_held = (space_action == 1)
        self.previous_shift_held = (shift_action == 1)
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            start = (self.GRID_X + x * self.CELL_SIZE, self.GRID_Y)
            end = (self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_BG, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.GRID_X, self.GRID_Y + y * self.CELL_SIZE)
            end = (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_BG, start, end)

        # Draw placed pieces
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    self._draw_cell(c, r, self.grid[r, c], self.GRID_X, self.GRID_Y)

        # Draw ghost piece
        if not self.game_over:
            ghost_piece = self.current_piece.copy()
            while not self._check_collision(0, 1, ghost_piece):
                ghost_piece["y"] += 1
            self._draw_piece(ghost_piece, self.GRID_X, self.GRID_Y, is_ghost=True)

        # Draw current piece
        if not self.game_over:
            self._draw_piece(self.current_piece, self.GRID_X, self.GRID_Y)

        # Draw line clear animation
        for anim in self.line_clear_animation[:]:
            y = self.GRID_Y + anim["y"] * self.CELL_SIZE
            alpha = int(255 * (anim["timer"] / 5.0))
            flash_surface = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (self.GRID_X, y))
            anim["timer"] -= 1
            if anim["timer"] <= 0:
                self.line_clear_animation.remove(anim)

    def _draw_piece(self, piece, grid_x, grid_y, is_ghost=False):
        shape = self._get_piece_shape(piece)
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_cell(piece["x"] + c, piece["y"] + r, piece["type"], grid_x, grid_y, is_ghost)
    
    def _draw_cell(self, c, r, piece_type, grid_x, grid_y, is_ghost=False):
        x = grid_x + c * self.CELL_SIZE
        y = grid_y + r * self.CELL_SIZE
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            color = self.PIECE_SHADOW_COLORS[piece_type]
            pygame.draw.rect(self.screen, color, rect, width=2, border_radius=3)
        else:
            color = self.PIECE_COLORS[piece_type]
            shadow_color = self.PIECE_SHADOW_COLORS[piece_type]
            
            # Draw main block
            pygame.gfxdraw.box(self.screen, rect.inflate(-2, -2), color)
            # Draw outline
            pygame.gfxdraw.rectangle(self.screen, rect.inflate(-2,-2), shadow_color)
            
            # 3D effect
            pygame.draw.line(self.screen, self.COLOR_WHITE, (x + 2, y + 2), (x + self.CELL_SIZE - 3, y + 2), 1)
            pygame.draw.line(self.screen, self.COLOR_WHITE, (x + 2, y + 2), (x + 2, y + self.CELL_SIZE - 3), 1)

    def _render_ui(self):
        # Next Piece
        next_text = self.font_small.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_Y))
        next_box_rect = pygame.Rect(self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_Y + 30, 4 * self.CELL_SIZE, 4 * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, next_box_rect, border_radius=5)
        if self.next_piece_type is not None:
            piece_to_draw = {"type": self.next_piece_type, "rotation": 0, "x": 0, "y": 0}
            shape = self._get_piece_shape(piece_to_draw)
            shape_w = len(shape[0])
            shape_h = len(shape)
            draw_x = next_box_rect.centerx - (shape_w * self.CELL_SIZE) / 2
            draw_y = next_box_rect.centery - (shape_h * self.CELL_SIZE) / 2
            self._draw_piece(piece_to_draw, int(draw_x), int(draw_y))

        # Held Piece
        hold_text = self.font_small.render("HOLD", True, self.COLOR_UI_TEXT)
        self.screen.blit(hold_text, (self.GRID_X - 100, self.GRID_Y))
        hold_box_rect = pygame.Rect(self.GRID_X - 100, self.GRID_Y + 30, 4 * self.CELL_SIZE, 4 * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, hold_box_rect, border_radius=5)
        if self.held_piece_type is not None:
            piece_to_draw = {"type": self.held_piece_type, "rotation": 0, "x": 0, "y": 0}
            shape = self._get_piece_shape(piece_to_draw)
            shape_w = len(shape[0])
            shape_h = len(shape)
            draw_x = hold_box_rect.centerx - (shape_w * self.CELL_SIZE) / 2
            draw_y = hold_box_rect.centery - (shape_h * self.CELL_SIZE) / 2
            self._draw_piece(piece_to_draw, int(draw_x), int(draw_y))

        # Score and Lines
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        lines_text = self.font_large.render(f"LINES: {self.lines_cleared}/{self.WIN_CONDITION}", True, self.COLOR_UI_TEXT)
        lines_rect = lines_text.get_rect(topright=(self.screen_width - 20, 20))
        self.screen.blit(lines_text, lines_rect)

        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            end_text_str = "YOU WIN!" if self.lines_cleared >= self.WIN_CONDITION else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_WHITE)
            end_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, end_rect)

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

    def close(self):
        pygame.quit()