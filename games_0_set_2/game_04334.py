
# Generated: 2025-08-28T02:06:11.267632
# Source Brief: brief_04334.md
# Brief Index: 4334

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to match a group of 3 or more same-colored tiles. Press Shift to reset the cursor to the top-left."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the grid of colored tiles by matching groups of 3 or more in a race against time. Create large combos for bonus points!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    NUM_TILE_TYPES = 5
    MATCH_THRESHOLD = 3
    
    # Timing
    FPS = 30
    MAX_TIME_SECONDS = 60
    MAX_STEPS = MAX_TIME_SECONDS * FPS

    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (45, 55, 65)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_TIMER_BAR = (50, 205, 50)
    COLOR_TIMER_BAR_WARN = (255, 165, 0)
    COLOR_TIMER_BAR_DANGER = (220, 20, 60)
    
    TILE_COLORS = [
        (0, 0, 0), # 0: Empty
        (255, 87, 87),   # 1: Red
        (87, 189, 255),  # 2: Blue
        (87, 255, 153),  # 3: Green
        (255, 245, 87),  # 4: Yellow
        (214, 87, 255),  # 5: Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.grid_rect = self._calculate_grid_rect()
        
        # Initialize state variables
        self.reset()
    
    def _calculate_grid_rect(self):
        # Center the grid
        max_dim = min(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        grid_pixel_size = int(max_dim * 0.85)
        self.tile_size = grid_pixel_size // self.GRID_SIZE
        grid_pixel_size = self.tile_size * self.GRID_SIZE # Recalculate to avoid gaps
        
        offset_x = (self.SCREEN_WIDTH - grid_pixel_size) // 2
        offset_y = (self.SCREEN_HEIGHT - grid_pixel_size) // 2
        return pygame.Rect(offset_x, offset_y, grid_pixel_size, grid_pixel_size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_STEPS
        
        self.board = self._generate_board()
        self.tile_fall_offsets = np.zeros_like(self.board, dtype=float)

        self.cursor_pos = [0, 0]
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.move_cooldown = 0
        self.MOVE_COOLDOWN_FRAMES = 5 # Prevent flying cursor
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _generate_board(self):
        while True:
            board = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            if self._has_valid_moves(board):
                return board

    def _has_valid_moves(self, board):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self._find_connected_tiles(r, c, board, check_only=True) >= self.MATCH_THRESHOLD:
                    return True
        return False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        reward = 0
        
        self.steps += 1
        self.time_left -= 1
        
        # --- Handle Actions ---
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        
        if movement > 0 and self.move_cooldown == 0:
            if movement == 1: # Up
                self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 2: # Down
                self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
            elif movement == 3: # Left
                self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 4: # Right
                self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
            self.move_cooldown = self.MOVE_COOLDOWN_FRAMES
        
        if shift_press:
            self.cursor_pos = [0, 0]
            # sound: cursor_reset.wav
            
        if space_press:
            reward += self._attempt_match()
            # sound: match.wav or no_match.wav
            
        if movement == 0 and not space_press and not shift_press:
            reward -= 0.01 # Small penalty for inactivity
        
        self._update_particles()
        self._update_tile_fall()

        # --- Check Termination ---
        terminated = False
        if self.time_left <= 0:
            reward -= 100
            terminated = True
        elif np.all(self.board == 0): # Board cleared
            reward += 100
            terminated = True
        elif not self._has_valid_moves(self.board):
            reward -= 100
            terminated = True
        
        self.game_over = terminated
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _attempt_match(self):
        r, c = self.cursor_pos
        if self.board[r, c] == 0:
            return 0

        connected_tiles = self._find_connected_tiles(r, c, self.board)
        
        if len(connected_tiles) >= self.MATCH_THRESHOLD:
            num_cleared = len(connected_tiles)
            for tr, tc in connected_tiles:
                self._create_particles(tr, tc, self.board[tr, tc])
                self.board[tr, tc] = 0
            
            self._apply_gravity_and_refill()
            
            # Calculate reward
            reward = num_cleared * 1.0 # Base reward for each tile
            if num_cleared >= 4:
                reward += 5 # Bonus for larger match
            
            self.score += num_cleared
            return reward
        return 0

    def _find_connected_tiles(self, r, c, board, check_only=False):
        target_color = board[r, c]
        if target_color == 0:
            return [] if not check_only else 0

        q = deque([(r, c)])
        visited = set([(r, c)])
        
        while q:
            curr_r, curr_c = q.popleft()
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = curr_r + dr, curr_c + dc
                
                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and \
                   (nr, nc) not in visited and board[nr, nc] == target_color:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        
        return len(visited) if check_only else list(visited)

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_SIZE):
            write_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.board[r, c] != 0:
                    if r != write_row:
                        self.board[write_row, c] = self.board[r, c]
                        self.tile_fall_offsets[write_row, c] = (write_row - r) * self.tile_size
                        self.board[r, c] = 0
                    write_row -= 1
            
            # Refill top
            for r in range(write_row, -1, -1):
                self.board[r, c] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)
                self.tile_fall_offsets[r, c] = (write_row + 1) * self.tile_size
    
    def _create_particles(self, r, c, color_index):
        tile_center_x = self.grid_rect.left + c * self.tile_size + self.tile_size / 2
        tile_center_y = self.grid_rect.top + r * self.tile_size + self.tile_size / 2
        color = self.TILE_COLORS[color_index]

        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifespan = self.np_random.integers(15, 30)
            size = self.np_random.uniform(2, 5)
            self.particles.append([tile_center_x, tile_center_y, vx, vy, lifespan, color, size])
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[1] # x += vx
            p[1] += p[2] # y += vy
            p[4] -= 1    # lifespan--

    def _update_tile_fall(self):
        self.tile_fall_offsets = np.maximum(0, self.tile_fall_offsets - (self.tile_size / 3.0))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_tiles()
        self._render_cursor()
        self._render_particles()

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.grid_rect.left + i * self.tile_size, self.grid_rect.top)
            end_pos = (self.grid_rect.left + i * self.tile_size, self.grid_rect.bottom)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.grid_rect.left, self.grid_rect.top + i * self.tile_size)
            end_pos = (self.grid_rect.right, self.grid_rect.top + i * self.tile_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_tiles(self):
        padding = int(self.tile_size * 0.1)
        radius = int(self.tile_size * 0.2)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = self.board[r, c]
                if color_index != 0:
                    y_offset = self.tile_fall_offsets[r, c]
                    tile_rect = pygame.Rect(
                        self.grid_rect.left + c * self.tile_size + padding,
                        int(self.grid_rect.top + r * self.tile_size + padding - y_offset),
                        self.tile_size - 2 * padding,
                        self.tile_size - 2 * padding
                    )
                    pygame.draw.rect(self.screen, self.TILE_COLORS[color_index], tile_rect, border_radius=radius)

    def _render_cursor(self):
        r, c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_rect.left + c * self.tile_size,
            self.grid_rect.top + r * self.tile_size,
            self.tile_size,
            self.tile_size
        )
        pulse = abs(math.sin(self.steps * 0.2))
        color = (255, 255, 255, 150 + 105 * pulse)
        
        temp_surf = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        pygame.draw.rect(temp_surf, color, temp_surf.get_rect(), width=4, border_radius=int(self.tile_size * 0.25))
        self.screen.blit(temp_surf, cursor_rect.topleft)

    def _render_particles(self):
        for x, y, vx, vy, lifespan, color, size in self.particles:
            alpha = max(0, min(255, int(255 * (lifespan / 30.0))))
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(size), (*color, alpha))
    
    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Timer bar
        timer_ratio = max(0, self.time_left / self.MAX_STEPS)
        bar_width = self.SCREEN_WIDTH - 40
        bar_height = 20
        
        bar_color = self.COLOR_TIMER_BAR
        if timer_ratio < 0.5: bar_color = self.COLOR_TIMER_BAR_WARN
        if timer_ratio < 0.2: bar_color = self.COLOR_TIMER_BAR_DANGER

        pygame.draw.rect(self.screen, self.COLOR_GRID, (20, self.SCREEN_HEIGHT - 40, bar_width, bar_height))
        pygame.draw.rect(self.screen, bar_color, (20, self.SCREEN_HEIGHT - 40, bar_width * timer_ratio, bar_height))
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            win_condition = np.all(self.board == 0)
            
            if win_condition:
                end_text_str = "BOARD CLEARED!"
                end_color = (100, 255, 100)
            else:
                end_text_str = "GAME OVER"
                end_color = (255, 100, 100)
                
            end_text = self.font_large.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(end_text, text_rect)
            self.screen.blit(final_score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Interactive Play ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    
    # Game loop
    while not terminated:
        movement = 0 # No-op
        space = 0
        shift = 0
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    print("Game Over!")
    print(f"Final Info: {info}")
    
    # Keep the final screen visible for a moment
    pygame.time.wait(3000)
    
    env.close()