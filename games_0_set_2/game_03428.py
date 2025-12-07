
# Generated: 2025-08-27T23:19:42.411023
# Source Brief: brief_03428.md
# Brief Index: 3428

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ↑↓←→ to move selector. Space to select/swap. Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent gems to create matches of 3 or more. Collect 20 red gems to win before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Game Constants ---
    GRID_ROWS = 8
    GRID_COLS = 8
    NUM_GEM_TYPES = 4  # Not including the special red one
    GEM_TYPE_RED = 1
    
    GOAL_RED_GEMS = 20
    STARTING_MOVES = 30
    
    # --- Visual Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (50, 60, 80)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SELECTOR = (255, 255, 0)
    COLOR_SELECTED_GEM = (255, 255, 255, 100) # RGBA for transparency
    
    GEM_COLORS = {
        1: (255, 50, 50),   # Red (Target)
        2: (50, 200, 50),   # Green
        3: (80, 80, 255),   # Blue
        4: (255, 200, 0),   # Yellow
        5: (200, 0, 255),   # Purple
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_main = pygame.font.Font(pygame.font.match_font('consolas', 'arial'), 24)
            self.font_title = pygame.font.Font(pygame.font.match_font('consolas', 'arial'), 32)
            self.font_small = pygame.font.Font(pygame.font.match_font('consolas', 'arial'), 16)
        except:
            self.font_main = pygame.font.Font(None, 28)
            self.font_title = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 20)

        # Board layout
        self.gem_size = 40
        self.grid_line_width = 2
        self.board_width = self.GRID_COLS * self.gem_size
        self.board_height = self.GRID_ROWS * self.gem_size
        self.grid_offset_x = (self.SCREEN_WIDTH - self.board_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.board_height) // 2 + 20

        # State variables are initialized in reset()
        self.grid = None
        self.selector_pos = None
        self.selected_gem_pos = None
        self.moves_left = 0
        self.red_gems_collected = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        
        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Comment out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.moves_left = self.STARTING_MOVES
        self.red_gems_collected = 0
        self.score = 0
        self.game_over = False
        self.selector_pos = (0, 0)
        self.selected_gem_pos = None
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        
        self._generate_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._update_particles()
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        reward = 0
        
        # --- Handle Action ---
        if shift_pressed:
            self.selected_gem_pos = None
            # sfx: deselect
        
        self._move_selector(movement)
        
        if space_pressed:
            reward = self._handle_selection()
            
        # --- Check Termination ---
        terminated = False
        if self.red_gems_collected >= self.GOAL_RED_GEMS:
            reward += 50
            self.game_over = True
            terminated = True
            # sfx: win_game
        elif self.moves_left <= 0:
            reward -= 50
            self.game_over = True
            terminated = True
            # sfx: lose_game
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_gems()
        self._render_selector()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "red_gems_collected": self.red_gems_collected,
            "moves_left": self.moves_left,
        }

    # --- Game Logic ---

    def _generate_board(self):
        self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 2, size=(self.GRID_ROWS, self.GRID_COLS))
        # Ensure no initial matches and at least one possible move
        while self._find_all_matches() or not self._has_possible_moves():
             self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 2, size=(self.GRID_ROWS, self.GRID_COLS))

    def _move_selector(self, movement):
        r, c = self.selector_pos
        if movement == 1: r = (r - 1 + self.GRID_ROWS) % self.GRID_ROWS  # Up
        elif movement == 2: r = (r + 1) % self.GRID_ROWS  # Down
        elif movement == 3: c = (c - 1 + self.GRID_COLS) % self.GRID_COLS  # Left
        elif movement == 4: c = (c + 1) % self.GRID_COLS  # Right
        self.selector_pos = (r, c)

    def _handle_selection(self):
        if self.selected_gem_pos is None:
            self.selected_gem_pos = self.selector_pos
            # sfx: select_gem
            return 0
        else:
            if self._are_adjacent(self.selected_gem_pos, self.selector_pos):
                return self._attempt_swap()
            else:
                self.selected_gem_pos = self.selector_pos
                # sfx: select_gem
                return 0

    def _attempt_swap(self):
        pos1 = self.selected_gem_pos
        pos2 = self.selector_pos
        self.selected_gem_pos = None

        self._swap_gems(pos1, pos2)
        
        all_matches = self._find_all_matches()
        if not all_matches:
            self._swap_gems(pos1, pos2) # Swap back
            # sfx: invalid_swap
            return -0.1
        
        # A valid move was made
        self.moves_left -= 1
        
        total_reward = 0
        
        while all_matches:
            # sfx: match_found
            
            # --- Calculate rewards for current matches ---
            num_matched_gems = len(all_matches)
            if 3 <= num_matched_gems <= 3: total_reward += 1
            elif num_matched_gems == 4: total_reward += 2
            elif num_matched_gems >= 5: total_reward += 3

            # --- Remove gems and update score/goal ---
            for r, c in all_matches:
                gem_type = self.grid[r, c]
                if gem_type == self.GEM_TYPE_RED:
                    self.red_gems_collected = min(self.GOAL_RED_GEMS, self.red_gems_collected + 1)
                    total_reward += 5
                
                self.score += 10
                self._create_particles(r, c, self.GEM_COLORS[gem_type])
                self.grid[r, c] = 0 # Mark as empty
            
            # --- Apply gravity and refill ---
            self._apply_gravity()
            self._fill_top_rows()
            
            # --- Check for new matches (cascade) ---
            all_matches = self._find_all_matches()
            
        return total_reward

    def _swap_gems(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

    def _find_all_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        # Vertical
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _has_possible_moves(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Try swapping right
                if c < self.GRID_COLS - 1:
                    self._swap_gems((r, c), (r, c + 1))
                    if self._find_all_matches():
                        self._swap_gems((r, c), (r, c + 1)) # Swap back
                        return True
                    self._swap_gems((r, c), (r, c + 1)) # Swap back
                # Try swapping down
                if r < self.GRID_ROWS - 1:
                    self._swap_gems((r, c), (r + 1, c))
                    if self._find_all_matches():
                        self. _swap_gems((r, c), (r + 1, c)) # Swap back
                        return True
                    self._swap_gems((r, c), (r + 1, c)) # Swap back
        return False

    def _apply_gravity(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1

    def _fill_top_rows(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 2)

    def _are_adjacent(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2) == 1

    # --- Particle System ---
    def _create_particles(self, r, c, color):
        px, py = self._grid_to_pixel(r, c)
        px += self.gem_size // 2
        py += self.gem_size // 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(20, 40)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    # --- Rendering ---
    
    def _grid_to_pixel(self, r, c):
        x = self.grid_offset_x + c * self.gem_size
        y = self.grid_offset_y + r * self.gem_size
        return x, y

    def _render_grid(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.grid_offset_y + r * self.gem_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, y), (self.grid_offset_x + self.board_width, y), self.grid_line_width)
        for c in range(self.GRID_COLS + 1):
            x = self.grid_offset_x + c * self.gem_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_offset_y), (x, self.grid_offset_y + self.board_height), self.grid_line_width)

    def _render_gems(self):
        radius = self.gem_size // 2 - 4
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                gem_type = self.grid[r, c]
                if gem_type != 0:
                    px, py = self._grid_to_pixel(r, c)
                    center_x = px + self.gem_size // 2
                    center_y = py + self.gem_size // 2
                    
                    color = self.GEM_COLORS[gem_type]
                    highlight = tuple(min(255, val + 60) for val in color)
                    
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                    
                    # Shine effect
                    pygame.gfxdraw.arc(self.screen, center_x, center_y, radius - 2, 225, 270, highlight)
                    
                    # Highlight selected gem
                    if self.selected_gem_pos == (r, c):
                        s = pygame.Surface((self.gem_size, self.gem_size), pygame.SRCALPHA)
                        pygame.draw.rect(s, self.COLOR_SELECTED_GEM, (0, 0, self.gem_size, self.gem_size), border_radius=8)
                        self.screen.blit(s, (px, py))

    def _render_selector(self):
        r, c = self.selector_pos
        px, py = self._grid_to_pixel(r, c)
        rect = (px, py, self.gem_size, self.gem_size)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, 3, border_radius=8)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, (p['life'] / 40) * 255)
            color = (*p['color'], alpha)
            s = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (2, 2), 2)
            self.screen.blit(s, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

    def _render_ui(self):
        # --- Top UI Bar ---
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        red_gems_text = self.font_main.render(f"Red Gems: {self.red_gems_collected}/{self.GOAL_RED_GEMS}", True, self.GEM_COLORS[self.GEM_TYPE_RED])
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        
        self.screen.blit(moves_text, (20, 20))
        self.screen.blit(red_gems_text, (self.SCREEN_WIDTH // 2 - red_gems_text.get_width() // 2, 20))
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 20))

        # --- Game Over Message ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.red_gems_collected >= self.GOAL_RED_GEMS:
                msg = "You Win!"
                color = (100, 255, 100)
            else:
                msg = "Game Over"
                color = (255, 100, 100)
                
            title_text = self.font_title.render(msg, True, color)
            title_rect = title_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20))
            
            reset_text = self.font_small.render("Call reset() to play again", True, self.COLOR_TEXT)
            reset_rect = reset_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20))

            overlay.blit(title_text, title_rect)
            overlay.blit(reset_text, reset_rect)
            self.screen.blit(overlay, (0, 0))

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