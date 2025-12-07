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
        "Controls: Arrow keys to move cursor. Space to select/swap tiles. Shift to deselect."
    )

    game_description = (
        "Swap adjacent gems to match 3 or more. Clear the entire board within 25 moves to win!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 8
        self.TILE_SIZE = 40
        self.NUM_COLORS = 6
        self.MAX_MOVES = 25
        self.MAX_STEPS = 1000

        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.TILE_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.TILE_SIZE) // 2 + 20

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 60, 80)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_TEXT_SHADOW = (10, 15, 20)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTED = (255, 255, 0)
        self.TILE_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Initialize state variables
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_tile_pos = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []
        self.rng = None
        self.shuffling = False

        self.reset()
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.cursor_pos = [0, 0]
        self.selected_tile_pos = None
        self.particles = []
        self.shuffling = False
        
        self._generate_board()
        while not self._find_all_possible_moves():
            self._generate_board()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self.shuffling = False

        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1

        # --- Handle player input ---
        if shift_press and self.selected_tile_pos:
            self.selected_tile_pos = None
        
        if movement > 0:
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        if space_press:
            c, r = self.cursor_pos
            if self.grid[r][c] is None: # Can't select an empty space
                self.selected_tile_pos = None
            elif not self.selected_tile_pos:
                self.selected_tile_pos = [c, r]
            else:
                sel_c, sel_r = self.selected_tile_pos
                if abs(c - sel_c) + abs(r - sel_r) == 1: # Is adjacent
                    reward += self._attempt_swap(sel_r, sel_c, r, c)
                    self.selected_tile_pos = None
                elif c == sel_c and r == sel_r: # Clicked same tile
                    self.selected_tile_pos = None
                else: # Selected a non-adjacent tile
                    self.selected_tile_pos = [c, r]

        # --- Post-move game logic ---
        # Check for no-move state and reshuffle if necessary
        if not self._is_board_clear() and not self._find_all_possible_moves():
            self._shuffle_board()
            self.shuffling = True # For visual feedback

        terminated = self._check_termination()
        if terminated:
            if self._is_board_clear():
                reward += 100
                self.win_message = "YOU WIN!"
            else:
                reward -= 50
                self.win_message = "GAME OVER"
            self.game_over = True

        truncated = False
        if self.steps >= self.MAX_STEPS and not terminated:
            truncated = True
            reward -= 50
            self.win_message = "TIME UP!"
            self.game_over = True
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _attempt_swap(self, r1, c1, r2, c2):
        self.moves_left -= 1
        
        # Perform swap
        self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c2], self.grid[r1][c1]

        total_reward = 0
        total_cleared_tiles = 0
        
        # Cascade loop
        while True:
            match_groups = self._find_matches()
            if not match_groups:
                break
            
            all_matched_tiles = set()
            for group in match_groups:
                all_matched_tiles.update(group)
                
                # Bonus for large matches
                if len(group) == 4: total_reward += 5
                elif len(group) >= 5: total_reward += 10
            
            num_cleared = len(all_matched_tiles)
            total_cleared_tiles += num_cleared
            total_reward += num_cleared # +1 per tile
            
            self._clear_matches(all_matched_tiles) # This also creates particles
            self._apply_gravity()
        
        if total_cleared_tiles == 0:
            # Invalid move, swap back
            self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c2], self.grid[r1][c1]
            return -0.1
        
        self.score += total_reward
        return total_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_tiles()
        self._update_and_render_particles()
        self._render_cursor_and_selection()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "tiles_left": sum(1 for r in range(self.GRID_HEIGHT) for c in range(self.GRID_WIDTH) if self.grid[r][c] is not None),
        }

    # --- Game Logic Helpers ---
    def _generate_board(self):
        self.grid = [[self.rng.integers(0, self.NUM_COLORS) for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        # Ensure no matches on start
        while self._find_matches():
            matches = self._find_matches()
            all_matched_tiles = set().union(*matches)
            for r, c in all_matched_tiles:
                self.grid[r][c] = self.rng.integers(0, self.NUM_COLORS)

    def _find_matches(self):
        match_groups = []
        all_matched_tiles = set()
        
        # Horizontal
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if self.grid[r][c] is not None and self.grid[r][c] == self.grid[r][c+1] == self.grid[r][c+2]:
                    match = {(r, c), (r, c+1), (r, c+2)}
                    # Extend match
                    i = c + 3
                    while i < self.GRID_WIDTH and self.grid[r][i] == self.grid[r][c]:
                        match.add((r, i))
                        i += 1
                    if not match.issubset(all_matched_tiles):
                        match_groups.append(match)
                        all_matched_tiles.update(match)

        # Vertical
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if self.grid[r][c] is not None and self.grid[r][c] == self.grid[r+1][c] == self.grid[r+2][c]:
                    match = {(r, c), (r+1, c), (r+2, c)}
                    # Extend match
                    i = r + 3
                    while i < self.GRID_HEIGHT and self.grid[i][c] == self.grid[r][c]:
                        match.add((i, c))
                        i += 1
                    if not match.issubset(all_matched_tiles):
                        match_groups.append(match)
                        all_matched_tiles.update(match)
        return match_groups

    def _find_all_possible_moves(self):
        moves = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] is None: continue
                # Swap right
                if c < self.GRID_WIDTH - 1 and self.grid[r][c+1] is not None:
                    self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c]
                    if self._find_matches(): moves.append(((r, c), (r, c+1)))
                    self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c] # Swap back
                # Swap down
                if r < self.GRID_HEIGHT - 1 and self.grid[r+1][c] is not None:
                    self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c]
                    if self._find_matches(): moves.append(((r, c), (r+1, c)))
                    self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c] # Swap back
        return moves

    def _clear_matches(self, matches):
        for r, c in matches:
            if self.grid[r][c] is not None:
                # Spawn particles
                tile_color = self.TILE_COLORS[self.grid[r][c]]
                for _ in range(10): # Number of particles per tile
                    self.particles.append({
                        'pos': [
                            self.GRID_OFFSET_X + c * self.TILE_SIZE + self.TILE_SIZE / 2,
                            self.GRID_OFFSET_Y + r * self.TILE_SIZE + self.TILE_SIZE / 2
                        ],
                        'vel': [self.rng.uniform(-2, 2), self.rng.uniform(-3, -1)],
                        'color': tile_color,
                        'life': self.rng.integers(15, 30)
                    })
                self.grid[r][c] = None

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r][c] is not None:
                    if r != empty_row:
                        self.grid[empty_row][c] = self.grid[r][c]
                        self.grid[r][c] = None
                    empty_row -= 1
    
    def _is_board_clear(self):
        return all(self.grid[r][c] is None for r in range(self.GRID_HEIGHT) for c in range(self.GRID_WIDTH))

    def _shuffle_board(self):
        tiles = [self.grid[r][c] for r in range(self.GRID_HEIGHT) for c in range(self.GRID_WIDTH) if self.grid[r][c] is not None]
        self.rng.shuffle(tiles)
        
        tile_idx = 0
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] is not None:
                    if tile_idx < len(tiles):
                        self.grid[r][c] = tiles[tile_idx]
                        tile_idx += 1
        
        # Ensure no new matches are formed and a move is possible
        if self._find_matches() or not self._find_all_possible_moves():
            self._shuffle_board() # Recurse until valid state

    def _check_termination(self):
        return self.moves_left <= 0 or self._is_board_clear()

    # --- Rendering ---
    def _render_grid(self):
        for r in range(self.GRID_HEIGHT + 1):
            y = self.GRID_OFFSET_Y + r * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH * self.TILE_SIZE, y), 1)
        for c in range(self.GRID_WIDTH + 1):
            x = self.GRID_OFFSET_X + c * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.TILE_SIZE), 1)

    def _render_tiles(self):
        gem_size = self.TILE_SIZE - 8
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_idx = self.grid[r][c]
                if color_idx is not None:
                    color = self.TILE_COLORS[color_idx]
                    x = self.GRID_OFFSET_X + c * self.TILE_SIZE + self.TILE_SIZE // 2
                    y = self.GRID_OFFSET_Y + r * self.TILE_SIZE + self.TILE_SIZE // 2
                    
                    # Draw gem polygon
                    points = [
                        (x, y - gem_size/2),
                        (x + gem_size/2, y),
                        (x, y + gem_size/2),
                        (x - gem_size/2, y)
                    ]
                    pygame.gfxdraw.filled_polygon(self.screen, points, color)
                    pygame.gfxdraw.aapolygon(self.screen, points, (255,255,255, 80))

                    # Highlight
                    highlight_color = tuple(min(255, val + 60) for val in color)
                    pygame.draw.line(self.screen, highlight_color, (x-gem_size/2+2, y-2), (x-2, y-gem_size/2+2), 2)


    def _render_cursor_and_selection(self):
        # Draw selected tile highlight
        if self.selected_tile_pos:
            c, r = self.selected_tile_pos
            rect = pygame.Rect(
                self.GRID_OFFSET_X + c * self.TILE_SIZE,
                self.GRID_OFFSET_Y + r * self.TILE_SIZE,
                self.TILE_SIZE, self.TILE_SIZE
            )
            pulse = (math.sin(self.steps * 0.5) + 1) / 2 * 100 + 155
            pygame.draw.rect(self.screen, (pulse, pulse, 0), rect, 3)

        # Draw cursor
        c, r = self.cursor_pos
        rect = pygame.Rect(
            self.GRID_OFFSET_X + c * self.TILE_SIZE,
            self.GRID_OFFSET_Y + r * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(p['life'] * 15)))
                size = max(1, p['life'] // 6)
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

    def _render_text(self, text, font, x, y, color=None, shadow_color=None, center=False):
        if color is None:
            color = self.COLOR_TEXT
        if shadow_color is None:
            shadow_color = self.COLOR_TEXT_SHADOW
            
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Moves Left
        self._render_text(f"Moves: {self.moves_left}", self.font_medium, 20, 10)
        # Score
        self._render_text(f"Score: {int(self.score)}", self.font_medium, self.SCREEN_WIDTH // 2, 10, center=True)
        # Tiles Left
        tiles_left = sum(1 for r in range(self.GRID_HEIGHT) for c in range(self.GRID_WIDTH) if self.grid[r][c] is not None)
        self._render_text(f"Tiles: {tiles_left}", self.font_medium, self.SCREEN_WIDTH - 120, 10)

        # Shuffle message
        if self.shuffling:
            self._render_text("No moves! Shuffling...", self.font_small, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 15, center=True, color=self.COLOR_SELECTED)

        # Game over message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            self._render_text(self.win_message, self.font_large, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, center=True)

    def validate_implementation(self):
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
        assert isinstance(trunc, bool) # Changed to bool
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")