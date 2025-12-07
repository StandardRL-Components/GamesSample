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


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑↓ to rotate. Press space to drop the shape instantly."
    )

    game_description = (
        "A fast-paced puzzle game. Place falling shapes to fill the grid against the clock."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        self.FPS = 30
        self.MAX_TIME_SECONDS = 120
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_ACCENT = (100, 100, 255)
        self.COLOR_GHOST = (255, 255, 255, 50)
        self.SHAPE_COLORS = [
            (255, 80, 80),    # Red
            (80, 255, 80),    # Green
            (80, 80, 255),    # Blue
            (255, 255, 80),   # Yellow
            (80, 255, 255),   # Cyan
            (255, 80, 255),   # Magenta
            (255, 160, 80)    # Orange
        ]

        # Shape definitions (tetrominoes)
        self.SHAPES = [
            [[1, 1, 1, 1]],         # I
            [[1, 1, 0], [0, 1, 1]], # Z
            [[0, 1, 1], [1, 1, 0]], # S
            [[1, 1, 1], [0, 1, 0]], # T
            [[1, 1], [1, 1]],       # O
            [[1, 0, 0], [1, 1, 1]], # L
            [[0, 0, 1], [1, 1, 1]]  # J
        ]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game state variables
        self.grid = None
        self.current_shape_matrix = None
        self.current_shape_pos = None
        self.current_shape_color = None
        self.next_shape_matrix = None
        self.next_shape_color = None
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        
        self.fall_counter = 0
        self.fall_speed = self.FPS // 2  # Fall one step every 0.5 seconds
        
        self.move_cooldown = 0
        self.rotate_cooldown = 0
        self.move_cooldown_rate = 4
        self.rotate_cooldown_rate = 6

        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.particles = []
        
        self.move_cooldown = 0
        self.rotate_cooldown = 0

        self._new_shape(is_first=True)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        reward = 0
        self.steps += 1
        self.time_remaining -= 1
        
        if self.move_cooldown > 0: self.move_cooldown -= 1
        if self.rotate_cooldown > 0: self.rotate_cooldown -= 1

        if not self.game_over:
            movement = action[0]
            rotation = action[1] # 0=none, 1=rotate
            space_held = action[2] == 1 # Use action[2] for hard drop

            # Map movement and rotation from discrete actions
            # movement: 0=none, 1=up, 2=down, 3=left, 4=right
            # rotation: 0=none, 1=rotate
            
            # Rotation
            if rotation == 1:
                movement = 1 # Prioritize rotation if both are triggered

            reward += self._handle_input(movement, space_held)
            
            if not self.game_over: # Hard drop might end the game
                update_reward = self._update_game_state()
                reward += update_reward

        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.game_over = self.game_over or terminated
        
        # truncated is always false as termination is handled by game logic
        truncated = False

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info


    def _handle_input(self, movement, space_held):
        reward = 0
        
        # Hard drop (Space)
        if space_held:
            # Move down until collision
            while self._is_valid_position(self.current_shape_matrix, (self.current_shape_pos[0], self.current_shape_pos[1] + 1)):
                self.current_shape_pos = (self.current_shape_pos[0], self.current_shape_pos[1] + 1)
            reward += self._lock_shape()
            self.fall_counter = 0 # Reset fall timer
            return reward

        # Movement (Left/Right)
        if movement in [3, 4] and self.move_cooldown == 0:
            dx = -1 if movement == 3 else 1
            if self._is_valid_position(self.current_shape_matrix, (self.current_shape_pos[0] + dx, self.current_shape_pos[1])):
                self.current_shape_pos = (self.current_shape_pos[0] + dx, self.current_shape_pos[1])
                self.move_cooldown = self.move_cooldown_rate

        # Rotation (Up/Down)
        if movement in [1, 2] and self.rotate_cooldown == 0:
            rotated_shape = np.rot90(self.current_shape_matrix, k=1 if movement == 1 else -1)
            if self._is_valid_position(rotated_shape, self.current_shape_pos):
                self.current_shape_matrix = rotated_shape
                self.rotate_cooldown = self.rotate_cooldown_rate
            # Wall kick
            elif self._is_valid_position(rotated_shape, (self.current_shape_pos[0] + 1, self.current_shape_pos[1])):
                self.current_shape_matrix = rotated_shape
                self.current_shape_pos = (self.current_shape_pos[0] + 1, self.current_shape_pos[1])
                self.rotate_cooldown = self.rotate_cooldown_rate
            elif self._is_valid_position(rotated_shape, (self.current_shape_pos[0] - 1, self.current_shape_pos[1])):
                self.current_shape_matrix = rotated_shape
                self.current_shape_pos = (self.current_shape_pos[0] - 1, self.current_shape_pos[1])
                self.rotate_cooldown = self.rotate_cooldown_rate
        return reward

    def _update_game_state(self):
        self.fall_counter += 1
        if self.fall_counter >= self.fall_speed:
            self.fall_counter = 0
            if self._is_valid_position(self.current_shape_matrix, (self.current_shape_pos[0], self.current_shape_pos[1] + 1)):
                self.current_shape_pos = (self.current_shape_pos[0], self.current_shape_pos[1] + 1)
            else:
                return self._lock_shape()
        return 0

    def _lock_shape(self):
        reward = 0
        shape_h, shape_w = self.current_shape_matrix.shape
        x_offset, y_offset = self.current_shape_pos
        
        filled_cells = 0
        for y in range(shape_h):
            for x in range(shape_w):
                if self.current_shape_matrix[y, x] != 0:
                    grid_y, grid_x = y_offset + y, x_offset + x
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y, grid_x] = self.current_shape_color
                        filled_cells += 1
                        # Spawn particle on lock
                        px = self.GRID_X + grid_x * self.CELL_SIZE + self.CELL_SIZE // 2
                        py = self.GRID_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE // 2
                        self._spawn_particles(px, py, self.SHAPE_COLORS[self.current_shape_color - 1])

        reward += filled_cells # +1 per cell filled
        
        # Check for completed lines for bonus reward (lines are not cleared)
        lines_completed = 0
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] > 0):
                lines_completed += 1
        reward += lines_completed * 10

        self.score += reward
        self._new_shape()
        return reward
        
    def _new_shape(self, is_first=False):
        if is_first:
            shape_idx = self.np_random.integers(len(self.SHAPES))
            self.next_shape_matrix = np.array(self.SHAPES[shape_idx], dtype=int)
            self.next_shape_color = self.np_random.integers(1, len(self.SHAPE_COLORS) + 1)

        self.current_shape_matrix = self.next_shape_matrix
        self.current_shape_color = self.next_shape_color
        
        shape_w = self.current_shape_matrix.shape[1]
        self.current_shape_pos = (self.GRID_WIDTH // 2 - shape_w // 2, 0)

        shape_idx = self.np_random.integers(len(self.SHAPES))
        self.next_shape_matrix = np.array(self.SHAPES[shape_idx], dtype=int)
        self.next_shape_color = self.np_random.integers(1, len(self.SHAPE_COLORS) + 1)
        
        if not self._is_valid_position(self.current_shape_matrix, self.current_shape_pos):
            self.game_over = True

    def _is_valid_position(self, shape_matrix, offset):
        shape_h, shape_w = shape_matrix.shape
        x_offset, y_offset = offset
        for y in range(shape_h):
            for x in range(shape_w):
                if shape_matrix[y, x] != 0:
                    grid_y, grid_x = y_offset + y, x_offset + x
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return False
                    if self.grid[grid_y, grid_x] != 0:
                        return False
        return True

    def _check_termination(self):
        if self.game_over: # Game over from invalid placement
            return True, -100
        if self.time_remaining <= 0:
            return True, 0 # Neutral reward on time out
        if np.all(self.grid > 0): # Grid completely filled
            return True, 100
        return False, 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame and numpy have different coordinate systems, so we need to transpose.
        # Pygame: (width, height), Numpy: (height, width)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        # Draw grid background
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw locked blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] != 0:
                    color_index = int(self.grid[y, x]) - 1
                    self._draw_cell(x, y, self.SHAPE_COLORS[color_index])
        
        if not self.game_over:
            # Draw ghost piece
            ghost_pos = self._get_ghost_position()
            self._draw_shape(self.current_shape_matrix, ghost_pos, self.COLOR_GHOST, is_ghost=True)

            # Draw current shape
            color = self.SHAPE_COLORS[self.current_shape_color - 1]
            self._draw_shape(self.current_shape_matrix, self.current_shape_pos, color)
        
        # Update and draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # Draw game frame and UI panels
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_X - 6, self.GRID_Y - 6, self.GRID_WIDTH * self.CELL_SIZE + 12, self.GRID_HEIGHT * self.CELL_SIZE + 12), 3, border_radius=5)
        
        # Right panel
        right_panel_x = self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20
        
        # Score
        self._draw_text("SCORE", self.font_small, right_panel_x, self.GRID_Y)
        self._draw_text(f"{self.score:06d}", self.font_large, right_panel_x, self.GRID_Y + 15)
        
        # Time
        self._draw_text("TIME", self.font_small, right_panel_x, self.GRID_Y + 90)
        time_ratio = max(0, self.time_remaining / self.MAX_STEPS)
        time_bar_w = 150
        pygame.draw.rect(self.screen, self.COLOR_GRID, (right_panel_x, self.GRID_Y + 105, time_bar_w, 20))
        pygame.draw.rect(self.screen, self.COLOR_UI_ACCENT, (right_panel_x, self.GRID_Y + 105, time_bar_w * time_ratio, 20))

        # Next Shape
        self._draw_text("NEXT", self.font_small, right_panel_x, self.GRID_Y + 150)
        if self.next_shape_matrix is not None:
            next_shape_w = self.next_shape_matrix.shape[1] * self.CELL_SIZE
            next_shape_h = self.next_shape_matrix.shape[0] * self.CELL_SIZE
            next_offset_x = right_panel_x + (time_bar_w - next_shape_w) // 2
            next_offset_y = self.GRID_Y + 170 + (4 * self.CELL_SIZE - next_shape_h) // 2
            self._draw_shape(self.next_shape_matrix, (0, 0), self.SHAPE_COLORS[self.next_shape_color - 1], custom_offset=(next_offset_x, next_offset_y))
        
        # Fill %
        total_cells = self.GRID_WIDTH * self.GRID_HEIGHT
        filled_cells = np.count_nonzero(self.grid)
        fill_percent = (filled_cells / total_cells) * 100
        self._draw_text("GRID FILL", self.font_small, right_panel_x, self.GRID_Y + 260)
        self._draw_text(f"{fill_percent:.1f}%", self.font_large, right_panel_x, self.GRID_Y + 275)

        # Game Over / Win message
        if self.game_over:
            is_win = np.all(self.grid > 0)
            message = "YOU WIN!" if is_win else "GAME OVER"
            color = (100, 255, 100) if is_win else (255, 100, 100)
            self._draw_text_centered(message, self.font_large, self.WIDTH // 2, self.HEIGHT // 2, color)


    def _draw_cell(self, grid_x, grid_y, color, custom_offset=None):
        if custom_offset:
            px, py = custom_offset
            rect = pygame.Rect(px + grid_x * self.CELL_SIZE, py + grid_y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        else:
            rect = pygame.Rect(self.GRID_X + grid_x * self.CELL_SIZE, self.GRID_Y + grid_y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        highlight = tuple(min(255, c + 40) for c in color)
        shadow = tuple(max(0, c - 40) for c in color)
        
        pygame.draw.rect(self.screen, shadow, rect)
        pygame.draw.rect(self.screen, color, rect.inflate(-4, -4))
        pygame.draw.line(self.screen, highlight, rect.topleft, rect.topright)
        pygame.draw.line(self.screen, highlight, rect.topleft, rect.bottomleft)

    def _draw_shape(self, shape_matrix, offset, color, is_ghost=False, custom_offset=None):
        shape_h, shape_w = shape_matrix.shape
        x_offset, y_offset = offset
        for y in range(shape_h):
            for x in range(shape_w):
                if shape_matrix[y, x] != 0:
                    if custom_offset:
                        self._draw_cell(x, y, color, custom_offset=custom_offset)
                    elif is_ghost:
                        rect = pygame.Rect(self.GRID_X + (x_offset + x) * self.CELL_SIZE, self.GRID_Y + (y_offset + y) * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                        pygame.draw.rect(s, color, s.get_rect(), 2, border_radius=2)
                        self.screen.blit(s, rect.topleft)
                    else:
                        self._draw_cell(x_offset + x, y_offset + y, color)
    
    def _get_ghost_position(self):
        pos = self.current_shape_pos
        while self._is_valid_position(self.current_shape_matrix, (pos[0], pos[1] + 1)):
            pos = (pos[0], pos[1] + 1)
        return pos

    def _spawn_particles(self, x, y, color):
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(10, 21)
            self.particles.append({'pos': [x, y], 'vel': vel, 'life': lifespan, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 20))
                color = p['color'] + (alpha,)
                radius = int(p['life'] / 4)
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def _draw_text(self, text, font, x, y, color=None):
        if color is None: color = self.COLOR_UI_TEXT
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))
        
    def _draw_text_centered(self, text, font, cx, cy, color=None):
        if color is None: color = self.COLOR_UI_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(cx, cy))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "grid_fill_percent": (np.count_nonzero(self.grid) / (self.GRID_WIDTH * self.GRID_HEIGHT)) * 100,
        }

    def close(self):
        pygame.quit()