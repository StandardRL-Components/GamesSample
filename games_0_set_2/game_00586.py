import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem, "
        "then move the cursor to an adjacent gem and press Space again to swap. "
        "Press Shift to deselect."
    )

    game_description = (
        "A gem-matching puzzle game. Swap adjacent gems to create lines of 3 or more "
        "of the same color. Collect 100 gems within 20 moves to win. "
        "Creating combos of 4 or 5 gems yields bonus rewards!"
    )

    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    GEM_TYPES = 6
    CELL_SIZE = 48
    GRID_OFFSET_X = (640 - GRID_WIDTH * CELL_SIZE) // 2
    GRID_OFFSET_Y = (400 - GRID_HEIGHT * CELL_SIZE) + 20

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECTED = (255, 255, 255)
    
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
        (255, 160, 80),  # Orange
    ]

    # Game Mechanics
    MAX_MOVES = 20
    WIN_SCORE = 100
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 16)
        
        self.game_state = "IDLE" # IDLE, SWAPPING, MATCHING, FALLING
        self.animation_timer = 0
        self.animation_duration = 0.2 # seconds
        self.swap_info = {}
        self.matched_gems = set()
        self.fall_info = {}
        self.particles = []

        # self.reset() is called by the wrapper, no need to call it here.
        # However, to make the class instantiable on its own, we need to initialize
        # attributes that are normally set in reset.
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.cursor_pos = [0,0]
        self.selected_gem = None
        self.last_action = self.action_space.sample()

    def _generate_board(self):
        board = self.np_random.integers(0, self.GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while self._find_matches(board) or not self._find_possible_moves(board):
            board = self.np_random.integers(0, self.GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        return board

    def _find_matches(self, board):
        matches = set()
        # Horizontal matches
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if board[r, c] == board[r, c + 1] == board[r, c + 2] and board[r, c] != -1:
                    matches.update([(r, c), (r, c + 1), (r, c + 2)])
        # Vertical matches
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if board[r, c] == board[r + 1, c] == board[r + 2, c] and board[r, c] != -1:
                    matches.update([(r, c), (r + 1, c), (r + 2, c)])
        return matches

    def _find_possible_moves(self, board):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Try swapping right
                if c < self.GRID_WIDTH - 1:
                    board[r, c], board[r, c + 1] = board[r, c + 1], board[r, c]
                    if self._find_matches(board):
                        board[r, c], board[r, c + 1] = board[r, c + 1], board[r, c]
                        return True
                    board[r, c], board[r, c + 1] = board[r, c + 1], board[r, c]
                # Try swapping down
                if r < self.GRID_HEIGHT - 1:
                    board[r, c], board[r + 1, c] = board[r + 1, c], board[r, c]
                    if self._find_matches(board):
                        board[r, c], board[r + 1, c] = board[r + 1, c], board[r, c]
                        return True
                    board[r, c], board[r + 1, c] = board[r + 1, c], board[r, c]
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        
        self.grid = self._generate_board()
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_gem = None
        
        self.game_state = "IDLE"
        self.animation_timer = 0
        self.swap_info = {}
        self.matched_gems = set()
        self.fall_info = {}
        self.particles = []
        
        self.last_action = self.action_space.sample()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_state != "IDLE":
            self.animation_timer += 1 / 30.0 
            
            if self.animation_timer >= self.animation_duration:
                self.animation_timer = 0
                
                if self.game_state == "SWAPPING":
                    p1, p2 = self.swap_info['p1'], self.swap_info['p2']
                    self.grid[p1], self.grid[p2] = self.grid[p2], self.grid[p1]
                    
                    matches = self._find_matches(self.grid)
                    if matches:
                        self.matched_gems = matches
                        self.game_state = "MATCHING"
                        self.animation_duration = 0.3 
                        
                        match_count = len(self.matched_gems)
                        reward += match_count
                        if match_count == 4: reward += 5
                        if match_count >= 5: reward += 10
                        self.score += match_count
                        
                        for r, c in self.matched_gems:
                            self._create_particles(r, c, self.grid[r, c])
                    else:
                        self.grid[p1], self.grid[p2] = self.grid[p2], self.grid[p1]
                        reward += -0.1
                        self.moves_left += 1
                        self.game_state = "IDLE"
                    self.swap_info = {}

                elif self.game_state == "MATCHING":
                    for r, c in self.matched_gems:
                        self.grid[r, c] = -1
                    self.matched_gems = set()
                    
                    self._prepare_fall()
                    if self.fall_info:
                        self.game_state = "FALLING"
                        self.animation_duration = 0.2
                    else:
                        self._check_cascade_or_end_turn(reward)

                elif self.game_state == "FALLING":
                    self._apply_gravity_and_refill()
                    self.fall_info = {}
                    reward = self._check_cascade_or_end_turn(reward)

        if self.game_state == "IDLE":
            movement, space_press, shift_press = self._process_action_presses(action)

            if movement != 0:
                if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
                elif movement == 2: self.cursor_pos[0] = min(self.GRID_HEIGHT - 1, self.cursor_pos[0] + 1)
                elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
                elif movement == 4: self.cursor_pos[1] = min(self.GRID_WIDTH - 1, self.cursor_pos[1] + 1)
            
            if shift_press:
                self.selected_gem = None
            
            if space_press:
                r, c = self.cursor_pos
                if self.selected_gem is None:
                    self.selected_gem = (r, c)
                else:
                    sr, sc = self.selected_gem
                    is_adjacent = abs(r - sr) + abs(c - sc) == 1
                    if is_adjacent:
                        self.moves_left -= 1
                        self.game_state = "SWAPPING"
                        self.animation_duration = 0.15
                        self.swap_info = {'p1': (sr, sc), 'p2': (r, c)}
                        self.selected_gem = None
                    else:
                        self.selected_gem = (r, c)

        self.steps += 1
        terminated = self._check_termination()
        if terminated and not self.game_over:
             self.game_over = True
             if self.score >= self.WIN_SCORE:
                 reward += 100
             else:
                 reward += -50

        self.last_action = action
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _process_action_presses(self, action):
        movement = action[0]
        space_pressed = action[1] == 1 and self.last_action[1] == 0
        shift_pressed = action[2] == 1 and self.last_action[2] == 0
        return movement, space_pressed, shift_pressed

    def _prepare_fall(self):
        self.fall_info = {}
        for c in range(self.GRID_WIDTH):
            empty_count = 0
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    self.fall_info[(r, c)] = (r + empty_count, c)

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_WIDTH):
            write_idx = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if write_idx != r:
                        self.grid[write_idx, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    write_idx -= 1
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.GEM_TYPES)

    def _check_cascade_or_end_turn(self, reward):
        matches = self._find_matches(self.grid)
        if matches:
            self.matched_gems = matches
            self.game_state = "MATCHING"
            self.animation_duration = 0.3
            match_count = len(self.matched_gems)
            reward += match_count
            self.score += match_count
            for r, c in self.matched_gems:
                self._create_particles(r, c, self.grid[r, c])
        else:
            self.game_state = "IDLE"
            if not self._find_possible_moves(self.grid):
                self.grid = self._generate_board()
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.score >= self.WIN_SCORE:
            return True
        if self.moves_left <= 0 and self.game_state == "IDLE":
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "game_state": self.game_state,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_bg()
        self._render_gems()
        self._render_cursor()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_bg(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

    def _render_gems(self):
        prog = self.animation_timer / self.animation_duration if self.animation_duration > 0 else 1.0
        
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type == -1:
                    continue

                pos_x = self.GRID_OFFSET_X + c * self.CELL_SIZE
                pos_y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
                
                if self.game_state == "SWAPPING" and self.swap_info:
                    if (r, c) == self.swap_info['p1']:
                        r2, c2 = self.swap_info['p2']
                        pos_x = int(pygame.math.lerp(pos_x, self.GRID_OFFSET_X + c2 * self.CELL_SIZE, prog))
                        pos_y = int(pygame.math.lerp(pos_y, self.GRID_OFFSET_Y + r2 * self.CELL_SIZE, prog))
                    elif (r, c) == self.swap_info['p2']:
                        r1, c1 = self.swap_info['p1']
                        pos_x = int(pygame.math.lerp(pos_x, self.GRID_OFFSET_X + c1 * self.CELL_SIZE, prog))
                        pos_y = int(pygame.math.lerp(pos_y, self.GRID_OFFSET_Y + r1 * self.CELL_SIZE, prog))
                
                elif self.game_state == "FALLING" and (r,c) in self.fall_info:
                    r_new, _ = self.fall_info[(r,c)]
                    start_y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
                    end_y = self.GRID_OFFSET_Y + r_new * self.CELL_SIZE
                    pos_y = int(pygame.math.lerp(start_y, end_y, prog))

                scale = 1.0
                if self.game_state == "MATCHING" and (r,c) in self.matched_gems:
                    scale = 1.0 - prog
                
                self._draw_gem(pos_x, pos_y, gem_type, scale)

    def _draw_gem(self, x, y, gem_type, scale=1.0):
        color = self.GEM_COLORS[gem_type]
        radius = int(self.CELL_SIZE * 0.4 * scale)
        center_x = x + self.CELL_SIZE // 2
        center_y = y + self.CELL_SIZE // 2

        if radius <= 0: return

        # Use gfxdraw for anti-aliasing
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        
        # Create a brighter color for the outline and convert it to a tuple
        outline_color = tuple(min(255, c + 50) for c in color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, outline_color)
        
        # Highlight
        highlight_color = (255, 255, 255, 90)
        highlight_radius = int(radius * 0.5)
        highlight_offset_x = int(radius * 0.3)
        highlight_offset_y = int(radius * -0.4)
        
        s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, highlight_color, (radius - highlight_offset_x, radius - highlight_offset_y), highlight_radius)
        self.screen.blit(s, (center_x - radius, center_y - radius))


    def _render_cursor(self):
        r, c = self.cursor_pos
        rect = pygame.Rect(
            self.GRID_OFFSET_X + c * self.CELL_SIZE,
            self.GRID_OFFSET_Y + r * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE,
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3)
        
        if self.selected_gem:
            sr, sc = self.selected_gem
            selected_rect = pygame.Rect(
                self.GRID_OFFSET_X + sc * self.CELL_SIZE,
                self.GRID_OFFSET_Y + sr * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE,
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, selected_rect, 3)

    def _create_particles(self, r, c, gem_type):
        center_x = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        color = self.GEM_COLORS[gem_type]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.uniform(0.3, 0.7)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color})

    def _render_particles(self):
        dt = 1/30.0
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 20 * dt
            p['life'] -= dt
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / p['max_life']))
                color = (*p['color'], alpha)
                size = int(5 * (p['life'] / p['max_life']))
                if size > 0:
                    rect = pygame.Rect(int(p['pos'][0]), int(p['pos'][1]), size, size)
                    
                    s = pygame.Surface((size, size), pygame.SRCALPHA)
                    s.fill(color)
                    self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        score_text = self.font_large.render(f"Gems: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.screen.get_width() - moves_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            result_text_str = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            result_text = self.font_large.render(result_text_str, True, (255, 255, 255))
            text_rect = result_text.get_rect(center=(320, 200))
            self.screen.blit(result_text, text_rect)

    def close(self):
        pygame.quit()