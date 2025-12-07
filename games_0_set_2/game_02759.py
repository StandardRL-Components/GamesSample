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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to select a gem. "
        "With a gem selected, use arrow keys to swap it with an adjacent gem. Shift deselects."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. "
        "Reach the target score before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    BOARD_WIDTH = 8
    BOARD_HEIGHT = 8
    NUM_GEM_TYPES = 6
    WIN_SCORE = 100
    MAX_MOVES = 20

    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    GRID_SIZE = 40
    GRID_OFFSET_X = (SCREEN_WIDTH - BOARD_WIDTH * GRID_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - BOARD_HEIGHT * GRID_SIZE) // 2 + 20

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_SELECTOR = (255, 255, 255, 100)
    COLOR_SELECTED_GEM = (255, 80, 80, 150)
    
    GEM_COLORS = [
        (255, 50, 50),   # Red
        (50, 255, 50),   # Green
        (80, 80, 255),   # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
        (255, 150, 50),  # Orange
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
        
        try:
            self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
            self.font_small = pygame.font.SysFont("Arial", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 24)

        
        # Initialize state variables to None, they will be set in reset()
        self.board = None
        self.selector_pos = None
        self.selected_gem_pos = None
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.steps = 0
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.selector_pos = [self.BOARD_WIDTH // 2, self.BOARD_HEIGHT // 2]
        self.selected_gem_pos = None
        
        self.board = self._generate_initial_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward_this_step = 0
        
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Handle Deselection ---
        # Deselect if shift is pressed, or if space is pressed on the currently selected gem
        if self.selected_gem_pos is not None and (shift_pressed or (space_pressed and self.selector_pos == self.selected_gem_pos)):
            # Sound effect: Deselect
            self.selected_gem_pos = None
            space_pressed = False # Consume the space press to prevent re-selection in the same step

        # --- Handle Input and Game Logic ---
        if self.selected_gem_pos is None:
            # NO GEM SELECTED: Move selector or select a gem
            if movement != 0: # Move selector
                dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
                self.selector_pos[0] = max(0, min(self.BOARD_WIDTH - 1, self.selector_pos[0] + dx))
                self.selector_pos[1] = max(0, min(self.BOARD_HEIGHT - 1, self.selector_pos[1] + dy))
            
            if space_pressed: # Select gem
                self.selected_gem_pos = list(self.selector_pos)
                # Sound effect: Select gem

        else:
            # GEM IS SELECTED: Attempt a swap
            if movement != 0:
                dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
                target_pos = [self.selected_gem_pos[0] + dx, self.selected_gem_pos[1] + dy]

                if 0 <= target_pos[0] < self.BOARD_WIDTH and 0 <= target_pos[1] < self.BOARD_HEIGHT:
                    # Sound effect: Swap attempt
                    self.moves_left -= 1
                    
                    self._swap_gems(self.selected_gem_pos, target_pos)
                    
                    all_matches = self._find_all_matches()
                    
                    if not all_matches:
                        # No match, swap back
                        # Sound effect: Invalid swap
                        self._swap_gems(self.selected_gem_pos, target_pos)
                    else:
                        # Match found, process it
                        # Sound effect: Match success
                        total_gems_cleared_this_turn = 0
                        
                        # Cascade loop
                        while all_matches:
                            num_cleared = self._process_matches(all_matches)
                            total_gems_cleared_this_turn += num_cleared
                            self._gems_fall()
                            self._refill_board()
                            all_matches = self._find_all_matches()
                        
                        reward_this_step += total_gems_cleared_this_turn
                        self.score += total_gems_cleared_this_turn

                    self.selected_gem_pos = None # Deselect after action
                    self.selector_pos = target_pos # Move selector to target for fluid control
            
            # If space is pressed on a different tile while another is selected, just deselect
            elif space_pressed:
                self.selected_gem_pos = None

        terminated = self._check_termination()
        if terminated and self.score >= self.WIN_SCORE:
            reward_this_step += 100

        return self._get_observation(), reward_this_step, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _render_game(self):
        # Draw grid lines
        for i in range(self.BOARD_WIDTH + 1):
            x = self.GRID_OFFSET_X + i * self.GRID_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.BOARD_HEIGHT * self.GRID_SIZE))
        for i in range(self.BOARD_HEIGHT + 1):
            y = self.GRID_OFFSET_Y + i * self.GRID_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.BOARD_WIDTH * self.GRID_SIZE, y))

        # Draw gems
        if self.board is not None:
            for y in range(self.BOARD_HEIGHT):
                for x in range(self.BOARD_WIDTH):
                    gem_type = self.board[y][x]
                    if gem_type != -1:
                        self._draw_gem(x, y, gem_type)
        
        # Draw selector
        sx, sy = self.selector_pos
        rect = pygame.Rect(self.GRID_OFFSET_X + sx * self.GRID_SIZE, self.GRID_OFFSET_Y + sy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        
        s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_SELECTOR, s.get_rect(), border_radius=4)
        pygame.draw.rect(s, (255,255,255), s.get_rect(), width=2, border_radius=4)
        self.screen.blit(s, rect.topleft)

        # Draw selected gem highlight
        if self.selected_gem_pos:
            gx, gy = self.selected_gem_pos
            rect = pygame.Rect(self.GRID_OFFSET_X + gx * self.GRID_SIZE, self.GRID_OFFSET_Y + gy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
            s.fill(self.COLOR_SELECTED_GEM)
            self.screen.blit(s, rect.topleft)

    def _draw_gem(self, x, y, gem_type):
        center_x = self.GRID_OFFSET_X + x * self.GRID_SIZE + self.GRID_SIZE // 2
        center_y = self.GRID_OFFSET_Y + y * self.GRID_SIZE + self.GRID_SIZE // 2
        radius = self.GRID_SIZE // 2 - 6
        color = self.GEM_COLORS[gem_type]
        
        if gem_type == 0: # Circle
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        elif gem_type == 1: # Square
            rect = pygame.Rect(center_x - radius, center_y - radius, radius * 2, radius * 2)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
        elif gem_type == 2: # Triangle
            points = [(center_x, center_y - radius), (center_x - radius, center_y + radius), (center_x + radius, center_y + radius)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 3: # Diamond
            points = [(center_x, center_y - radius), (center_x - radius, center_y), (center_x, center_y + radius), (center_x + radius, center_y)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 4: # Hexagon
            points = [(center_x + int(radius * math.cos(math.pi / 3 * i)), center_y + int(radius * math.sin(math.pi / 3 * i))) for i in range(6)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif gem_type == 5: # Star
            points = []
            for i in range(10):
                r = radius if i % 2 == 0 else radius / 2
                angle = math.pi / 5 * i - math.pi / 2
                points.append((center_x + int(r * math.cos(angle)), center_y + int(r * math.sin(angle))))
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        highlight_color = tuple(min(255, c + 60) for c in color)
        pygame.draw.circle(self.screen, highlight_color, (center_x - radius//3, center_y - radius//3), 3)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 10))

        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            message = "You Win!" if self.score >= self.WIN_SCORE else "Game Over"
            end_text = self.font_large.render(message, True, (255, 255, 100))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _generate_initial_board(self):
        board = [[-1 for _ in range(self.BOARD_WIDTH)] for _ in range(self.BOARD_HEIGHT)]
        
        while True:
            for y in range(self.BOARD_HEIGHT):
                for x in range(self.BOARD_WIDTH):
                    board[y][x] = self.np_random.integers(0, self.NUM_GEM_TYPES)
            
            # FIX: Temporarily assign the generated board to self.board so _find_all_matches can work.
            self.board = board
            
            if not self._find_all_matches():
                break
        return board

    def _find_all_matches(self):
        matches = set()
        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                gem = self.board[y][x]
                if gem == -1: continue

                # Horizontal check
                if x < self.BOARD_WIDTH - 2 and self.board[y][x+1] == gem and self.board[y][x+2] == gem:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
                # Vertical check
                if y < self.BOARD_HEIGHT - 2 and self.board[y+1][x] == gem and self.board[y+2][x] == gem:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return list(matches)

    def _process_matches(self, matches):
        for x, y in matches:
            self.board[y][x] = -1 # Mark as empty
        return len(matches)

    def _gems_fall(self):
        for x in range(self.BOARD_WIDTH):
            write_idx = self.BOARD_HEIGHT - 1
            for read_idx in range(self.BOARD_HEIGHT - 1, -1, -1):
                if self.board[read_idx][x] != -1:
                    if write_idx != read_idx:
                        self.board[write_idx][x] = self.board[read_idx][x]
                        self.board[read_idx][x] = -1
                    write_idx -= 1

    def _refill_board(self):
        for y in range(self.BOARD_HEIGHT):
            for x in range(self.BOARD_WIDTH):
                if self.board[y][x] == -1:
                    self.board[y][x] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _swap_gems(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        self.board[y1][x1], self.board[y2][x2] = self.board[y2][x2], self.board[y1][x1]

    def _check_termination(self):
        if self.moves_left <= 0 or self.score >= self.WIN_SCORE:
            self.game_over = True
            return True
        return False
        
    def close(self):
        pygame.font.quit()
        pygame.quit()