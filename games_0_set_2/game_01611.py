
# Generated: 2025-08-28T02:09:18.173036
# Source Brief: brief_01611.md
# Brief Index: 1611

        
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

    # User-facing strings
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a tile, then move to an adjacent tile and press Space again to swap. Shift to deselect."
    )
    game_description = (
        "Swap adjacent gems to create matches of 3 or more. Clear the entire board before the timer runs out to win. Create combos for extra points!"
    )

    # Frame advance setting
    auto_advance = True

    # --- Constants ---
    # Game settings
    GRID_WIDTH, GRID_HEIGHT = 10, 10
    NUM_TILE_TYPES = 6
    MAX_STEPS = 3600  # 2 minutes at 30fps
    TIME_LIMIT_SECONDS = 120

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (50, 60, 70)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BAR_BG = (70, 80, 90)
    COLOR_UI_BAR_FG = (100, 200, 255)
    TILE_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 120, 255),   # Blue
        (255, 255, 80),   # Yellow
        (200, 80, 255),   # Purple
        (255, 160, 80),   # Orange
    ]

    # Sizing
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_AREA_WIDTH, GRID_AREA_HEIGHT = 360, 360
    TILE_SIZE = GRID_AREA_WIDTH // GRID_WIDTH
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_AREA_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_AREA_HEIGHT) // 2

    # Animation timings (in frames)
    ANIM_SWAP_DURATION = 6
    ANIM_MATCH_DURATION = 10
    ANIM_FALL_DURATION = 8

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
        self.font_main = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # State variables initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_tile = None
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.game_state = 'IDLE'
        self.animation_progress = 0
        self.swapping_tiles = []
        self.matching_tiles = []
        self.falling_tiles = {} # { (r, c): { 'from': (r, c), 'color': int } }
        self.particles = []
        self.step_reward = 0
        self.combo_multiplier = 1
        self.last_action_buttons = [0, 0] # For edge detection of space/shift

        self.reset()
        
        # self.validate_implementation() # Uncomment for debugging

    def _get_grid_pos(self, screen_x, screen_y):
        """Converts screen coordinates to grid coordinates."""
        grid_x = (screen_x - self.GRID_OFFSET_X) // self.TILE_SIZE
        grid_y = (screen_y - self.GRID_OFFSET_Y) // self.TILE_SIZE
        if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT:
            return (grid_x, grid_y)
        return None

    def _get_screen_pos(self, grid_x, grid_y):
        """Converts grid coordinates to screen coordinates (top-left of the tile)."""
        return (
            self.GRID_OFFSET_X + grid_x * self.TILE_SIZE,
            self.GRID_OFFSET_Y + grid_y * self.TILE_SIZE
        )

    def _generate_board(self):
        """Generates a new board, ensuring no initial matches and at least one possible move."""
        while True:
            self.grid = self.np_random.integers(0, self.NUM_TILE_TYPES, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
            
            # Remove initial matches
            while True:
                matches = self._find_matches()
                if not matches:
                    break
                for r, c in matches:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_TILE_TYPES)
            
            # Check for possible moves
            if self._check_for_possible_moves():
                break

    def _find_matches(self, grid=None):
        """Finds all tiles that are part of a match of 3 or more."""
        if grid is None:
            grid = self.grid
            
        matches = set()
        # Horizontal matches
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if grid[r, c] != -1 and grid[r, c] == grid[r, c+1] and grid[r, c+1] == grid[r, c+2]:
                    # Find full extent of match
                    start, end = c, c + 2
                    while start > 0 and grid[r, start-1] == grid[r, c]: start -= 1
                    while end < self.GRID_WIDTH - 1 and grid[r, end+1] == grid[r, c]: end += 1
                    for i in range(start, end + 1): matches.add((r, i))
        # Vertical matches
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if grid[r, c] != -1 and grid[r, c] == grid[r+1, c] and grid[r+1, c] == grid[r+2, c]:
                    # Find full extent of match
                    start, end = r, r + 2
                    while start > 0 and grid[start-1, c] == grid[r, c]: start -= 1
                    while end < self.GRID_HEIGHT - 1 and grid[end+1, c] == grid[r, c]: end += 1
                    for i in range(start, end + 1): matches.add((i, c))
        return list(matches)

    def _check_for_possible_moves(self):
        """Checks if any valid moves exist on the current board."""
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Check swap with right neighbor
                if c < self.GRID_WIDTH - 1:
                    temp_grid = self.grid.copy()
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                    if self._find_matches(temp_grid):
                        return True
                # Check swap with bottom neighbor
                if r < self.GRID_HEIGHT - 1:
                    temp_grid = self.grid.copy()
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
                    if self._find_matches(temp_grid):
                        return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_board()
        
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.selected_tile = None
        self.steps = 0
        self.score = 0
        self.time_remaining = self.TIME_LIMIT_SECONDS * 30
        self.game_over = False
        self.game_state = 'IDLE'
        self.animation_progress = 0
        self.swapping_tiles = []
        self.matching_tiles = []
        self.falling_tiles = {}
        self.particles = []
        self.step_reward = 0
        self.combo_multiplier = 1
        self.last_action_buttons = [0, 0]

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.step_reward = 0
        self.game_over = self._check_termination()
        if self.game_over:
            return self._get_observation(), self.step_reward, True, False, self._get_info()

        self._update_game_state(action)
        self._update_animations()

        self.steps += 1
        self.time_remaining -= 1

        terminated = self._check_termination()
        if terminated and not self.game_over: # Termination event happened this step
            self.game_over = True
            if np.all(self.grid == -1): # Win condition
                self.step_reward += 100
            elif self.time_remaining <= 0: # Loss condition
                self.step_reward -= 100

        return self._get_observation(), self.step_reward, terminated, False, self._get_info()
    
    def _update_game_state(self, action):
        if self.game_state != 'IDLE':
            return # Ignore input during animations

        movement, space_action, shift_action = action
        space_pressed = space_action == 1 and self.last_action_buttons[0] == 0
        shift_pressed = shift_action == 1 and self.last_action_buttons[1] == 0
        self.last_action_buttons = [space_action, shift_action]
        
        # --- Handle Input ---
        # Movement
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.cursor_pos = (
                (self.cursor_pos[0] + dx) % self.GRID_WIDTH,
                (self.cursor_pos[1] + dy) % self.GRID_HEIGHT
            )

        # Cancel selection
        if shift_pressed and self.selected_tile:
            self.selected_tile = None
            # sfx: cancel_selection

        # Select/Swap
        if space_pressed:
            r, c = self.cursor_pos[1], self.cursor_pos[0]
            if self.selected_tile is None:
                self.selected_tile = (r, c)
                # sfx: select_tile
            else:
                sr, sc = self.selected_tile
                is_adjacent = abs(r - sr) + abs(c - sc) == 1
                if (r, c) == (sr, sc): # Deselect if same tile
                    self.selected_tile = None
                    # sfx: cancel_selection
                elif is_adjacent:
                    self.swapping_tiles = [(sr, sc), (r, c)]
                    self.game_state = 'SWAP_ANIM'
                    self.animation_progress = 0
                    self.selected_tile = None
                    # sfx: swap_attempt
                else: # Select new tile if not adjacent
                    self.selected_tile = (r, c)
                    # sfx: select_tile

    def _update_animations(self):
        if self.game_state == 'IDLE':
            return
        
        self.animation_progress += 1
        
        if self.game_state == 'SWAP_ANIM':
            if self.animation_progress >= self.ANIM_SWAP_DURATION:
                (r1, c1), (r2, c2) = self.swapping_tiles
                self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                matches1 = self._find_matches()
                if matches1:
                    self._start_match_phase(matches1)
                else: # Invalid swap, swap back
                    self.game_state = 'INVALID_SWAP_ANIM'
                    self.animation_progress = 0
                    # sfx: invalid_swap
        
        elif self.game_state == 'INVALID_SWAP_ANIM':
            if self.animation_progress >= self.ANIM_SWAP_DURATION:
                (r1, c1), (r2, c2) = self.swapping_tiles
                self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1] # Swap back
                self.game_state = 'IDLE'
                self.swapping_tiles = []

        elif self.game_state == 'MATCH_ANIM':
            if self.animation_progress >= self.ANIM_MATCH_DURATION:
                for r, c in self.matching_tiles:
                    self.grid[r, c] = -1 # Mark as empty
                self._start_fall_phase()

        elif self.game_state == 'FALL_ANIM':
            if self.animation_progress >= self.ANIM_FALL_DURATION:
                # Apply the fall logic to the grid
                new_grid = self.grid.copy()
                for (r,c), data in self.falling_tiles.items():
                    fr, fc = data['from']
                    new_grid[r, c] = data['color']
                    if new_grid[fr, fc] == data['color']:
                        new_grid[fr, fc] = -1
                self.grid = new_grid
                
                # Refill top rows
                for c in range(self.GRID_WIDTH):
                    empty_count = 0
                    for r in range(self.GRID_HEIGHT):
                        if self.grid[r, c] == -1:
                            empty_count += 1
                    for i in range(empty_count):
                        self.grid[i, c] = self.np_random.integers(0, self.NUM_TILE_TYPES)

                self.falling_tiles = {}
                new_matches = self._find_matches()
                if new_matches:
                    self.combo_multiplier += 1
                    self._start_match_phase(new_matches)
                    # sfx: combo_match
                else:
                    self.game_state = 'IDLE'
                    self.combo_multiplier = 1
                    if not self._check_for_possible_moves():
                        self._generate_board() # Reshuffle
                        # sfx: reshuffle_board
    
    def _start_match_phase(self, matches):
        self.game_state = 'MATCH_ANIM'
        self.animation_progress = 0
        self.matching_tiles = matches
        
        # Calculate reward
        base_reward = 0
        if len(matches) == 3: base_reward = 1
        elif len(matches) == 4: base_reward = 2
        else: base_reward = 3
        
        self.step_reward += (base_reward + len(matches) * 0.1) * self.combo_multiplier
        self.score += int((base_reward * 10) * self.combo_multiplier)
        
        # Create particles
        for r, c in self.matching_tiles:
            # Check grid bounds and tile validity
            if 0 <= r < self.GRID_HEIGHT and 0 <= c < self.GRID_WIDTH and self.grid[r, c] != -1:
                color = self.TILE_COLORS[self.grid[r, c]]
                screen_pos = self._get_screen_pos(c, r)
                center_x = screen_pos[0] + self.TILE_SIZE // 2
                center_y = screen_pos[1] + self.TILE_SIZE // 2
                for _ in range(15):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 4)
                    vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
                    self.particles.append({'pos': pygame.Vector2(center_x, center_y), 'vel': vel, 'life': 20, 'color': color})
        # sfx: match_explosion

    def _start_fall_phase(self):
        self.game_state = 'FALL_ANIM'
        self.animation_progress = 0
        self.falling_tiles = {}
        # sfx: tiles_fall
        
        for c in range(self.GRID_WIDTH):
            fall_dist = 0
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] == -1:
                    fall_dist += 1
                elif fall_dist > 0 and self.grid[r, c] != -1:
                    self.falling_tiles[(r + fall_dist, c)] = {
                        'from': (r, c),
                        'color': self.grid[r, c]
                    }

    def _check_termination(self):
        is_board_clear = np.all(self.grid == -1)
        is_time_up = self.time_remaining <= 0
        is_max_steps = self.steps >= self.MAX_STEPS
        return is_board_clear or is_time_up or is_max_steps

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": max(0, self.time_remaining // 30)}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_OFFSET_X + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_AREA_HEIGHT))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_OFFSET_Y + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_AREA_WIDTH, y))
        
        # Draw tiles
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                tile_color_idx = self.grid[r, c]
                if tile_color_idx == -1:
                    continue

                screen_x, screen_y = self._get_screen_pos(c, r)
                tile_rect = pygame.Rect(screen_x, screen_y, self.TILE_SIZE, self.TILE_SIZE)
                color = self.TILE_COLORS[tile_color_idx]
                
                # Handle animations
                is_swapping = any((r,c) == t for t in self.swapping_tiles)
                is_matching = (r,c) in self.matching_tiles
                is_falling = any((r,c) == data['from'] for data in self.falling_tiles.values())

                if self.game_state == 'SWAP_ANIM' and is_swapping:
                    p = self.animation_progress / self.ANIM_SWAP_DURATION
                    (r1, c1), (r2, c2) = self.swapping_tiles
                    pos1_start = pygame.Vector2(self._get_screen_pos(c1, r1))
                    pos2_start = pygame.Vector2(self._get_screen_pos(c2, r2))
                    if (r,c) == (r1,c1):
                        pos = pos1_start.lerp(pos2_start, p)
                    else:
                        pos = pos2_start.lerp(pos1_start, p)
                    tile_rect.topleft = (int(pos.x), int(pos.y))

                elif self.game_state == 'INVALID_SWAP_ANIM' and is_swapping:
                    p = self.animation_progress / self.ANIM_SWAP_DURATION
                    (r1, c1), (r2, c2) = self.swapping_tiles
                    pos1_start = pygame.Vector2(self._get_screen_pos(c1, r1))
                    pos2_start = pygame.Vector2(self._get_screen_pos(c2, r2))
                    if (r,c) == (r1,c1):
                        pos = pos2_start.lerp(pos1_start, p)
                    else:
                        pos = pos1_start.lerp(pos2_start, p)
                    tile_rect.topleft = (int(pos.x), int(pos.y))

                elif self.game_state == 'MATCH_ANIM' and is_matching:
                    p = self.animation_progress / self.ANIM_MATCH_DURATION
                    scale = 1.0 - p
                    new_size = int(self.TILE_SIZE * scale)
                    offset = (self.TILE_SIZE - new_size) // 2
                    tile_rect = pygame.Rect(screen_x + offset, screen_y + offset, max(0, new_size), max(0, new_size))
                
                elif self.game_state == 'FALL_ANIM' and is_falling:
                    continue

                self._draw_tile(tile_rect, color)

        # Draw falling tiles separately to be on top
        if self.game_state == 'FALL_ANIM':
            p = self.animation_progress / self.ANIM_FALL_DURATION
            for (tr, tc), data in self.falling_tiles.items():
                fr, fc = data['from']
                start_pos = pygame.Vector2(self._get_screen_pos(fc, fr))
                end_pos = pygame.Vector2(self._get_screen_pos(tc, tr))
                pos = start_pos.lerp(end_pos, p)
                tile_rect = pygame.Rect(int(pos.x), int(pos.y), self.TILE_SIZE, self.TILE_SIZE)
                self._draw_tile(tile_rect, self.TILE_COLORS[data['color']])

        # Draw selected tile highlight
        if self.selected_tile and self.game_state == 'IDLE':
            r, c = self.selected_tile
            x, y = self._get_screen_pos(c, r)
            pulse = (math.sin(self.steps * 0.3) + 1) / 2 * 100 + 155
            pygame.draw.rect(self.screen, (pulse, pulse, pulse), (x, y, self.TILE_SIZE, self.TILE_SIZE), 4, border_radius=5)
        
        # Draw cursor
        cur_x, cur_y = self._get_screen_pos(self.cursor_pos[0], self.cursor_pos[1])
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cur_x, cur_y, self.TILE_SIZE, self.TILE_SIZE), 3, border_radius=5)

        # Update and draw particles
        self._update_and_draw_particles()

    def _draw_tile(self, rect, color):
        if rect.width <= 0 or rect.height <= 0: return
        inset = 3
        inner_rect = rect.inflate(-inset*2, -inset*2)
        if inner_rect.width <= 0 or inner_rect.height <= 0: return

        pygame.draw.rect(self.screen, color, inner_rect, border_radius=5)
        
        highlight_color = tuple(min(255, c + 60) for c in color)
        shadow_color = tuple(max(0, c - 60) for c in color)
        
        # Simple 3D effect
        pygame.draw.line(self.screen, highlight_color, inner_rect.topleft, inner_rect.topright, 2)
        pygame.draw.line(self.screen, highlight_color, inner_rect.topleft, inner_rect.bottomleft, 2)
        pygame.draw.line(self.screen, shadow_color, inner_rect.bottomleft, inner_rect.bottomright, 2)
        pygame.draw.line(self.screen, shadow_color, inner_rect.topright, inner_rect.bottomright, 2)
        
    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Drag
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(p['life'] / 20 * 255)))
                size = max(1, int(p['life'] / 20 * 5))
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, p['color'] + (alpha,), (size, size), size)
                self.screen.blit(temp_surf, p['pos'] - pygame.Vector2(size, size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))
        
        # Time Bar
        bar_width = 200
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 20
        bar_y = 20
        
        time_ratio = max(0, self.time_remaining / (self.TIME_LIMIT_SECONDS * 30))
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FG, (bar_x, bar_y, int(bar_width * time_ratio), bar_height), border_radius=5)
    
    def close(self):
        pygame.quit()

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