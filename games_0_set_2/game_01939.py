
# Generated: 2025-08-28T03:09:56.656998
# Source Brief: brief_01939.md
# Brief Index: 1939

        
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
    user_guide = "Controls: Use arrow keys to move the selector. Press space to clear a block group."

    # Must be a short, user-facing description of the game:
    game_description = "Clear the grid by selecting groups of two or more same-colored blocks. Clear the whole board to win!"

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_LINES = (40, 50, 70)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_SHADOW = (10, 10, 20)
        self.COLOR_CURSOR = (255, 255, 100)
        self.BLOCK_COLORS = [
            (0, 0, 0),          # 0: Empty
            (255, 80, 80),      # 1: Red
            (80, 255, 80),      # 2: Green
            (80, 150, 255),     # 3: Blue
            (255, 255, 80),     # 4: Yellow
            (200, 80, 255),     # 5: Purple
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # --- Game Grid Layout ---
        self.board_pixel_size = 360
        self.cell_size = self.board_pixel_size // self.GRID_SIZE
        self.board_offset_x = (self.WIDTH - self.board_pixel_size) // 2
        self.board_offset_y = (self.HEIGHT - self.board_pixel_size) // 2

        # --- State Variables (initialized in reset) ---
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = None
        
        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.particles = []
        
        # Generate a valid starting board
        while True:
            self.grid = self.np_random.integers(1, len(self.BLOCK_COLORS), size=(self.GRID_SIZE, self.GRID_SIZE))
            if self._check_for_valid_moves():
                break

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        # shift_held is ignored as per brief

        reward = 0
        blocks_cleared_this_step = 0

        # --- Action Handling ---
        
        # 1. Handle Movement
        moved = self._handle_movement(movement)
        
        # 2. Handle Block Clearing
        if space_pressed:
            blocks_cleared_this_step = self._clear_selected_blocks()
            if blocks_cleared_this_step > 0:
                # Sound effect placeholder: # pygame.mixer.Sound('clear.wav').play()
                self._apply_gravity()
                reward += blocks_cleared_this_step
                self.score += blocks_cleared_this_step
        
        # 3. Handle No-Op
        if not moved and not space_pressed:
             reward -= 0.1 # Penalty for doing nothing

        # --- Update Game State ---
        self.steps += 1
        self._update_particles()
        
        # --- Check Termination Conditions ---
        board_cleared = np.sum(self.grid) == 0
        no_moves_left = not self._check_for_valid_moves() if blocks_cleared_this_step > 0 else False
        max_steps_reached = self.steps >= self.MAX_STEPS

        terminated = False
        if board_cleared:
            reward += 100 # Win bonus
            self.game_over = True
            terminated = True
        elif no_moves_left or max_steps_reached:
            # No explicit penalty, the inability to earn more points is the penalty
            self.game_over = True
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # truncated always False
            self._get_info()
        )
    
    # --- Helper Functions for step() ---
    
    def _handle_movement(self, movement):
        moved = False
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            moved = True
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
            moved = True
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            moved = True
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
            moved = True
        return moved
        
    def _find_connected_blocks(self, x, y):
        if self.grid[y, x] == 0:
            return []

        target_color = self.grid[y, x]
        q = [(x, y)]
        visited = set([(x, y)])
        connected_blocks = []

        while q:
            cx, cy = q.pop(0)
            connected_blocks.append((cx, cy))

            # Check 8 neighbors (orthogonally and diagonally)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = cx + dx, cy + dy
                    
                    if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                        if (nx, ny) not in visited and self.grid[ny, nx] == target_color:
                            visited.add((nx, ny))
                            q.append((nx, ny))
        
        return connected_blocks
        
    def _clear_selected_blocks(self):
        cx, cy = self.cursor_pos
        connected = self._find_connected_blocks(cx, cy)
        
        if len(connected) > 1:
            color_index = self.grid[cy, cx]
            for x, y in connected:
                self.grid[y, x] = 0
                self._create_particles(x, y, color_index)
            return len(connected)
        return 0
        
    def _apply_gravity(self):
        for x in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[y, x] != 0:
                    self.grid[empty_row, x], self.grid[y, x] = self.grid[y, x], self.grid[empty_row, x]
                    empty_row -= 1
                    
    def _check_for_valid_moves(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y, x] != 0:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0: continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                                if self.grid[ny, nx] == self.grid[y, x]:
                                    return True
        return False
        
    # --- Rendering and Observation ---
    
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
            "cursor_pos": list(self.cursor_pos),
            "blocks_remaining": int(np.count_nonzero(self.grid))
        }
        
    def _render_text(self, text, font, color, pos, shadow_color=None, shadow_offset=(2, 2)):
        if shadow_color:
            text_surf_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_surf_shadow, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)
        
    def _render_ui(self):
        self._render_text(f"Score: {self.score}", self.font_small, self.COLOR_UI_TEXT, (15, 15), self.COLOR_UI_SHADOW)
        
        steps_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        text_width = self.font_small.size(steps_text)[0]
        self._render_text(steps_text, self.font_small, self.COLOR_UI_TEXT, (self.WIDTH - text_width - 15, 15), self.COLOR_UI_SHADOW)

        if self.game_over:
            win_text = "BOARD CLEARED!" if np.sum(self.grid) == 0 else "NO MORE MOVES"
            text_w, text_h = self.font_large.size(win_text)
            pos = ((self.WIDTH - text_w) // 2, (self.HEIGHT - text_h) // 2)
            self._render_text(win_text, self.font_large, self.COLOR_CURSOR, pos, self.COLOR_UI_SHADOW)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            start_pos_v = (self.board_offset_x + i * self.cell_size, self.board_offset_y)
            end_pos_v = (self.board_offset_x + i * self.cell_size, self.board_offset_y + self.board_pixel_size)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos_v, end_pos_v, 1)
            start_pos_h = (self.board_offset_x, self.board_offset_y + i * self.cell_size)
            end_pos_h = (self.board_offset_x + self.board_pixel_size, self.board_offset_y + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos_h, end_pos_h, 1)
            
        # Draw blocks
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_index = self.grid[y, x]
                if color_index > 0:
                    color = self.BLOCK_COLORS[color_index]
                    darker_color = tuple(c * 0.7 for c in color)
                    
                    px = self.board_offset_x + x * self.cell_size
                    py = self.board_offset_y + y * self.cell_size
                    
                    block_rect = pygame.Rect(px + 1, py + 1, self.cell_size - 2, self.cell_size - 2)
                    inner_rect = pygame.Rect(px + 4, py + 4, self.cell_size - 8, self.cell_size - 8)
                    
                    pygame.draw.rect(self.screen, darker_color, block_rect, border_radius=4)
                    pygame.draw.rect(self.screen, color, inner_rect, border_radius=3)
        
        # Draw particles
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            p_color = tuple(c * life_ratio for c in p['color'])
            p_size = int(p['size'] * life_ratio)
            if p_size > 0:
                 pygame.draw.circle(self.screen, p_color, p['pos'], p_size)

        # Draw cursor
        if not self.game_over:
            cx, cy = self.cursor_pos
            cursor_rect = pygame.Rect(
                self.board_offset_x + cx * self.cell_size,
                self.board_offset_y + cy * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            alpha = 128 + int(math.sin(pygame.time.get_ticks() * 0.01) * 64)
            cursor_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, alpha), (0, 0, self.cell_size, self.cell_size), border_radius=6)
            pygame.draw.rect(cursor_surface, self.COLOR_CURSOR, (0, 0, self.cell_size, self.cell_size), 3, border_radius=6)
            self.screen.blit(cursor_surface, cursor_rect.topleft)
            
    # --- Particle System ---
    
    def _create_particles(self, grid_x, grid_y, color_index):
        px = self.board_offset_x + grid_x * self.cell_size + self.cell_size // 2
        py = self.board_offset_y + grid_y * self.cell_size + self.cell_size // 2
        color = self.BLOCK_COLORS[color_index]
        
        for _ in range(10): # Create 10 particles per block
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(15, 30) # frames
            size = random.randint(3, 7)
            self.particles.append({
                'pos': [px, py], 'vel': vel, 'color': color,
                'life': life, 'max_life': life, 'size': size
            })
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def close(self):
        pygame.quit()