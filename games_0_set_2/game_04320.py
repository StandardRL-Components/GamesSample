
# Generated: 2025-08-28T02:02:26.319844
# Source Brief: brief_04320.md
# Brief Index: 4320

        
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
        "Controls: ↑ to rotate clockwise, ↓ to speed up, ←→ to move. Space to hard drop, Shift to rotate counter-clockwise."
    )

    game_description = (
        "A fast-paced, grid-based falling block puzzle game where strategic rotations and risky placements are rewarded."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 60)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_DANGER = (100, 20, 20)
    
    PIECE_COLORS = [
        (0, 240, 240),   # I (Cyan)
        (240, 240, 0),   # O (Yellow)
        (160, 0, 240),   # T (Purple)
        (0, 0, 240),     # J (Blue)
        (240, 160, 0),   # L (Orange)
        (0, 240, 0),     # S (Green)
        (240, 0, 0),     # Z (Red)
    ]

    # Tetromino shapes
    PIECE_SHAPES = [
        [[1, 1, 1, 1]], # I
        [[1, 1], [1, 1]], # O
        [[0, 1, 0], [1, 1, 1]], # T
        [[1, 0, 0], [1, 1, 1]], # J
        [[0, 0, 1], [1, 1, 1]], # L
        [[0, 1, 1], [1, 1, 0]], # S
        [[1, 1, 0], [0, 1, 1]], # Z
    ]

    MAX_STEPS = 1000
    WIN_CONDITION_LINES = 10

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        self.game_state = "playing" # "playing", "win", "lose"
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Use python's random for piece generation
            random.seed(seed)
        
        self.grid = [[0 for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.game_state = "playing"

        self.fall_time = 0
        self.fall_speed = 1.0 # seconds per grid cell
        self.initial_fall_speed = 1.0
        self.fall_speed_reduction_per_line = 0.05
        
        self.particles = []
        self.line_clear_animation = []

        # Action state tracking for single-press events
        self.prev_up_held = False
        self.prev_space_held = False
        self.prev_shift_held = False

        self._new_piece()
        self._new_piece() # Call twice to populate current and next piece
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        up_held = movement == 1

        # --- Handle single-press actions ---
        up_pressed = up_held and not self.prev_up_held
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        self.prev_up_held = up_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        if self.game_state == "playing":
            # --- Player Actions ---
            if movement == 3: # Move Left
                if self._move_piece(-1, 0): reward -= 0.02
            elif movement == 4: # Move Right
                if self._move_piece(1, 0): reward -= 0.02
            
            if up_pressed: # Rotate Clockwise
                self._rotate_piece()
            
            if shift_pressed: # Rotate Counter-Clockwise
                self._rotate_piece(clockwise=False)
            
            # --- Gravity and Drops ---
            soft_drop = movement == 2
            self.fall_time += 1 / 30.0 * (5.0 if soft_drop else 1.0) # 30fps, soft drop is 5x faster

            if space_pressed: # Hard Drop
                # Sfx: Hard drop sound
                dropped_rows = 0
                while self._move_piece(0, 1):
                    dropped_rows += 1
                self._lock_piece()
                reward += self._check_lines()
                reward += 0.1 # Placement reward
            elif self.fall_time >= self.fall_speed:
                self.fall_time = 0
                if not self._move_piece(0, 1):
                    # Sfx: Piece lock sound
                    self._lock_piece()
                    reward += self._check_lines()
                    reward += 0.1 # Placement reward

        # --- Update Game State ---
        self._update_particles()
        self._update_line_clear_animation()

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and self.game_state == "playing": # First frame of termination
            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                self.game_state = "win"
                reward += 100
            else:
                self.game_state = "lose"
                reward -= 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lines": self.lines_cleared}

    # --- Game Logic Helpers ---

    def _new_piece(self):
        if not hasattr(self, 'next_piece_shape'):
            self.current_piece_shape = random.choice(self.PIECE_SHAPES)
            self.current_piece_color_idx = self.PIECE_SHAPES.index(self.current_piece_shape)
        else:
            self.current_piece_shape = self.next_piece_shape
            self.current_piece_color_idx = self.next_piece_color_idx

        self.next_piece_shape = random.choice(self.PIECE_SHAPES)
        self.next_piece_color_idx = self.PIECE_SHAPES.index(self.next_piece_shape)
        
        self.current_piece_rot = 0
        self.current_piece_x = self.GRID_WIDTH // 2 - len(self.current_piece_shape[0]) // 2
        self.current_piece_y = 0

        if self._check_collision(self.current_piece_shape, (self.current_piece_x, self.current_piece_y)):
            self.game_over = True
            # Sfx: Game over sound

    def _get_rotated_piece(self, shape, clockwise=True):
        if clockwise:
            return [list(row) for row in zip(*shape[::-1])]
        else:
            return [list(row) for row in zip(*shape)][::-1]

    def _rotate_piece(self, clockwise=True):
        # Sfx: Rotate sound
        rotated_shape = self._get_rotated_piece(self.current_piece_shape, clockwise)
        # Basic wall kick
        if not self._check_collision(rotated_shape, (self.current_piece_x, self.current_piece_y)):
            self.current_piece_shape = rotated_shape
            return True
        # Try kicking left
        if not self._check_collision(rotated_shape, (self.current_piece_x - 1, self.current_piece_y)):
            self.current_piece_x -= 1
            self.current_piece_shape = rotated_shape
            return True
        # Try kicking right
        if not self._check_collision(rotated_shape, (self.current_piece_x + 1, self.current_piece_y)):
            self.current_piece_x += 1
            self.current_piece_shape = rotated_shape
            return True
        return False

    def _move_piece(self, dx, dy):
        if not self._check_collision(self.current_piece_shape, (self.current_piece_x + dx, self.current_piece_y + dy)):
            self.current_piece_x += dx
            self.current_piece_y += dy
            return True
        return False

    def _check_collision(self, shape, offset):
        off_x, off_y = offset
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = off_x + x, off_y + y
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return True # Out of bounds
                    if self.grid[grid_y][grid_x]:
                        return True # Collides with existing block
        return False

    def _lock_piece(self):
        for y, row in enumerate(self.current_piece_shape):
            for x, cell in enumerate(row):
                if cell:
                    self.grid[self.current_piece_y + y][self.current_piece_x + x] = self.current_piece_color_idx + 1
        self._new_piece()

    def _check_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if all(self.grid[y]):
                lines_to_clear.append(y)

        if lines_to_clear:
            # Sfx: Line clear sound
            for y in lines_to_clear:
                self.line_clear_animation.append({"y": y, "timer": 10}) # 10 frames animation
                for x in range(self.GRID_WIDTH):
                    self._create_particles(x, y, self.PIECE_COLORS[self.grid[y][x]-1])
                self.grid.pop(y)
                self.grid.insert(0, [0 for _ in range(self.GRID_WIDTH)])
            
            self.lines_cleared += len(lines_to_clear)
            self.fall_speed = max(0.1, self.initial_fall_speed - self.lines_cleared * self.fall_speed_reduction_per_line)
            
            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                self.game_over = True
            
            # Calculate reward
            num_cleared = len(lines_to_clear)
            rewards = {1: 1, 2: 2, 3: 4, 4: 8}
            reward = rewards.get(num_cleared, 0)
            self.score += reward * 10 # Score is just for display
            return reward
        return 0

    def _create_particles(self, grid_x, grid_y, color):
        for _ in range(5):
            px = self.GRID_X + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
            py = self.GRID_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.particles.append({"x": px, "y": py, "vx": vx, "vy": vy, "life": 20, "color": color})
    
    def _update_particles(self):
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _update_line_clear_animation(self):
        for anim in self.line_clear_animation:
            anim["timer"] -= 1
        self.line_clear_animation = [anim for anim in self.line_clear_animation if anim["timer"] > 0]

    # --- Rendering Helpers ---

    def _render_text(self, text, font, color, pos, shadow=True):
        x, y = pos
        if shadow:
            text_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf, (x + 2, y + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _draw_block(self, x, y, color_idx, alpha=255):
        color = self.PIECE_COLORS[color_idx]
        
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        # Main block color
        main_color = tuple(c * 0.9 for c in color)
        pygame.gfxdraw.box(self.screen, rect, (*main_color, alpha))
        
        # Lighter top/left edge for 3D effect
        light_color = color
        pygame.draw.line(self.screen, light_color, (x, y), (x + self.CELL_SIZE - 1, y), 2)
        pygame.draw.line(self.screen, light_color, (x, y), (x, y + self.CELL_SIZE - 1), 2)

        # Darker bottom/right edge
        dark_color = tuple(c * 0.6 for c in color)
        pygame.draw.line(self.screen, dark_color, (x + 1, y + self.CELL_SIZE - 1), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1))
        pygame.draw.line(self.screen, dark_color, (x + self.CELL_SIZE - 1, y + 1), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1))

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw danger zone
        if any(any(row) for row in self.grid[:4]):
            danger_alpha = 100 + math.sin(pygame.time.get_ticks() * 0.01) * 50
            danger_surf = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, 4 * self.CELL_SIZE), pygame.SRCALPHA)
            danger_surf.fill((*self.COLOR_DANGER, danger_alpha))
            self.screen.blit(danger_surf, (self.GRID_X, self.GRID_Y))

        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (self.GRID_X + x * self.CELL_SIZE, self.GRID_Y), (self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT * self.CELL_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (self.GRID_X, self.GRID_Y + y * self.CELL_SIZE), (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE))
        
        # Draw locked blocks
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell:
                    self._draw_block(self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE, cell - 1)

        # Draw ghost piece
        if self.game_state == "playing":
            ghost_y = self.current_piece_y
            while not self._check_collision(self.current_piece_shape, (self.current_piece_x, ghost_y + 1)):
                ghost_y += 1
            for y, row in enumerate(self.current_piece_shape):
                for x, cell in enumerate(row):
                    if cell:
                        self._draw_block(
                            self.GRID_X + (self.current_piece_x + x) * self.CELL_SIZE,
                            self.GRID_Y + (ghost_y + y) * self.CELL_SIZE,
                            self.current_piece_color_idx,
                            alpha=80
                        )

        # Draw current piece
        if self.game_state == "playing":
            for y, row in enumerate(self.current_piece_shape):
                for x, cell in enumerate(row):
                    if cell:
                        self._draw_block(
                            self.GRID_X + (self.current_piece_x + x) * self.CELL_SIZE,
                            self.GRID_Y + (self.current_piece_y + y) * self.CELL_SIZE,
                            self.current_piece_color_idx
                        )
        
        # Draw line clear animation
        for anim in self.line_clear_animation:
            y = anim["y"]
            alpha = int(255 * (anim["timer"] / 10))
            flash_surf = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surf.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surf, (self.GRID_X, self.GRID_Y + y * self.CELL_SIZE))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20.0))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p["x"]), int(p["y"]), 3, color)

    def _render_ui(self):
        # Score
        self._render_text("SCORE", self.font_small, self.COLOR_TEXT, (40, 40))
        self._render_text(f"{self.score:06d}", self.font_medium, self.COLOR_TEXT, (40, 65))
        
        # Lines
        self._render_text("LINES", self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - 140, 40))
        self._render_text(f"{self.lines_cleared}/{self.WIN_CONDITION_LINES}", self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH - 140, 65))

        # Next Piece
        self._render_text("NEXT", self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - 140, self.SCREEN_HEIGHT - 150))
        next_box_rect = pygame.Rect(self.SCREEN_WIDTH - 150, self.SCREEN_HEIGHT - 125, 100, 100)
        pygame.draw.rect(self.screen, self.COLOR_GRID, next_box_rect, 0, 5)
        pygame.draw.rect(self.screen, self.COLOR_BG, next_box_rect, 2, 5)

        if hasattr(self, 'next_piece_shape'):
            shape = self.next_piece_shape
            w, h = len(shape[0]), len(shape)
            off_x = next_box_rect.centerx - (w * self.CELL_SIZE) / 2
            off_y = next_box_rect.centery - (h * self.CELL_SIZE) / 2
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell:
                        self._draw_block(int(off_x + x * self.CELL_SIZE), int(off_y + y * self.CELL_SIZE), self.next_piece_color_idx)

        # Game Over / Win message
        if self.game_state != "playing":
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.game_state == "win" else "GAME OVER"
            color = (100, 255, 100) if self.game_state == "win" else (255, 100, 100)
            
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2 + 3, self.SCREEN_HEIGHT/2 + 3))
            self.screen.blit(text_surf, text_rect)
            
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

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