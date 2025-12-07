
# Generated: 2025-08-28T04:49:47.221893
# Source Brief: brief_05374.md
# Brief Index: 5374

        
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
        "Controls: ↑↓←→ to move cursor. Space to select/swap. Shift to deselect."
    )

    game_description = (
        "Fast-paced match-3 puzzle. Clear the board by swapping gems before time runs out."
    )

    auto_advance = True

    # --- Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_GEM_TYPES = 3
    MAX_STEPS = 2700  # 90 seconds * 30 FPS
    MAX_TIME = 90.0

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 70)
    COLOR_SCORE = (220, 220, 240)
    COLOR_TIMER = (220, 220, 240)
    COLOR_TIMER_WARN = (255, 100, 100)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECT = (255, 255, 0)
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 80, 255),   # Blue
    ]
    
    # Animation timings (in frames)
    SWAP_DURATION = 8
    CLEAR_DURATION = 10
    FALL_DURATION = 8

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width = 640
        self.screen_height = 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Isometric projection setup
        self.tile_width = 40
        self.tile_height = self.tile_width * 0.5
        self.origin_x = self.screen_width // 2
        self.origin_y = 120

        self.np_random = None
        self.grid = None
        self.cursor_pos = None
        self.selected_gem = None
        self.game_state = "INPUT" # INPUT, SWAP, MATCH, CLEAR, FALL
        self.animation_timer = 0
        self.swap_info = None
        self.matched_gems = set()
        self.falling_gems = []
        self.particles = []
        self.combo_count = 0
        self.non_scoring_moves = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        
        self.reset()
        # self.validate_implementation() # Optional self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Fallback if super().reset() doesn't set it
            if self.np_random is None:
                 self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_TIME
        self.game_over = False
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_gem = None
        self.game_state = "INPUT"
        self.animation_timer = 0
        self.swap_info = None
        self.matched_gems = set()
        self.falling_gems = []
        self.particles = []
        self.combo_count = 0
        self.non_scoring_moves = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self._create_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Update Time and Step Count ---
        self.steps += 1
        self.time_remaining -= 1.0 / 30.0 # Assuming 30 FPS

        # --- Unpack Actions and Handle Input State ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Game State Machine ---
        if self.game_state == "INPUT":
            reward += self._handle_input(movement, space_pressed, shift_pressed)
        elif self.game_state == "SWAP":
            self.animation_timer += 1
            if self.animation_timer >= self.SWAP_DURATION:
                self._finalize_swap()
                self.game_state = "MATCH"
        elif self.game_state == "MATCH":
            match_result = self._find_and_process_matches()
            if match_result:
                num_cleared, is_combo = match_result
                reward += num_cleared * 1.0
                if is_combo:
                    reward += 5.0 # Combo bonus
                # sfx: match_found
                self.game_state = "CLEAR"
                self.animation_timer = 0
            else:
                if self.swap_info and self.swap_info['is_invalid']:
                    # sfx: invalid_swap
                    reward += -0.1
                    self.non_scoring_moves += 1
                    self._revert_swap()
                else:
                    self.swap_info = None
                    self.combo_count = 0
                    self.game_state = "INPUT"
                    if self._check_no_moves():
                        self._shuffle_board()

        elif self.game_state == "CLEAR":
            self.animation_timer += 1
            if self.animation_timer >= self.CLEAR_DURATION:
                self._apply_gravity()
                self.game_state = "FALL"
                self.animation_timer = 0
        elif self.game_state == "FALL":
            self.animation_timer += 1
            if self.animation_timer >= self.FALL_DURATION:
                self.falling_gems = []
                self.game_state = "MATCH" # Check for cascades

        self._update_particles()
        
        # --- Check Termination Conditions ---
        if not np.any(self.grid):
            reward += 100.0
            terminated = True
            self.game_over = True
        elif self.time_remaining <= 0:
            reward += -50.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward += -10.0
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Game Logic Helpers ---

    def _handle_input(self, movement, space_pressed, shift_pressed):
        if shift_pressed and self.selected_gem:
            # sfx: deselect
            self.selected_gem = None
            return 0
        
        # Move cursor
        if movement != 0:
            # sfx: cursor_move
            r, c = self.cursor_pos
            if movement == 1: r -= 1
            elif movement == 2: r += 1
            elif movement == 3: c -= 1
            elif movement == 4: c += 1
            self.cursor_pos = [r % self.GRID_HEIGHT, c % self.GRID_WIDTH]

        if space_pressed:
            r, c = self.cursor_pos
            if self.selected_gem is None:
                # sfx: select
                self.selected_gem = [r, c]
            else:
                sr, sc = self.selected_gem
                is_adjacent = abs(sr - r) + abs(sc - c) == 1
                if is_adjacent:
                    # sfx: swap_start
                    self._start_swap(self.selected_gem, self.cursor_pos)
                    self.selected_gem = None
                else:
                    # sfx: select
                    self.selected_gem = [r, c]
        return 0

    def _start_swap(self, pos1, pos2):
        self.game_state = "SWAP"
        self.animation_timer = 0
        self.swap_info = {'pos1': pos1, 'pos2': pos2, 'is_invalid': False}
        
        r1, c1 = pos1
        r2, c2 = pos2
        val1 = self.grid[r1, c1]
        val2 = self.grid[r2, c2]
        self.grid[r1, c1] = val2
        self.grid[r2, c2] = val1

    def _finalize_swap(self):
        # The swap is already done in _start_swap, this just transitions state
        pass

    def _revert_swap(self):
        # Swap back if it was invalid
        r1, c1 = self.swap_info['pos1']
        r2, c2 = self.swap_info['pos2']
        val1 = self.grid[r1, c1]
        val2 = self.grid[r2, c2]
        self.grid[r1, c1] = val2
        self.grid[r2, c2] = val1
        self.swap_info = None
        self.game_state = "INPUT"

    def _find_and_process_matches(self):
        all_matches = self._find_all_matches()
        if not all_matches:
            if self.swap_info:
                self.swap_info['is_invalid'] = True
            return None

        self.matched_gems = all_matches
        is_combo = self.combo_count > 0
        self.combo_count += 1
        self.non_scoring_moves = 0
        
        for r, c in all_matches:
            self.score += 1
            # Spawn particles
            sx, sy = self._iso_to_screen(r, c)
            gem_color = self.GEM_COLORS[self.grid[r, c] - 1]
            for _ in range(10):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                self.particles.append({
                    'pos': [sx, sy],
                    'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                    'life': self.np_random.integers(15, 25),
                    'color': gem_color,
                })
        
        for r, c in all_matches:
            self.grid[r, c] = 0

        return len(all_matches), is_combo

    def _apply_gravity(self):
        self.matched_gems = set()
        self.falling_gems = []
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                        self.falling_gems.append({'from': [r,c], 'to': [empty_row, c], 'val': self.grid[empty_row, c]})
                    empty_row -= 1
            
            # Refill from top
            for r in range(empty_row, -1, -1):
                new_gem = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
                self.grid[r, c] = new_gem
                self.falling_gems.append({'from': [-1, c], 'to': [r, c], 'val': new_gem})

    def _check_no_moves(self):
        if self.non_scoring_moves >= 10:
            return True
        return len(self._find_possible_moves()) == 0

    def _shuffle_board(self):
        # sfx: shuffle
        flat_gems = [gem for gem in self.grid.flatten() if gem > 0]
        self.np_random.shuffle(flat_gems)
        
        new_grid = np.zeros_like(self.grid)
        k = 0
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r,c] > 0 and k < len(flat_gems):
                    new_grid[r,c] = flat_gems[k]
                    k += 1
        self.grid = new_grid
        
        # Ensure no matches and at least one move
        while self._find_all_matches() or not self._find_possible_moves():
            self._remove_matches(self._find_all_matches())
            self._apply_gravity()
            if not self._find_possible_moves():
                 self.np_random.shuffle(self.grid) # Simple brute force shuffle
        self.non_scoring_moves = 0

    # --- Board Creation ---
    
    def _create_board(self):
        self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while True:
            matches = self._find_all_matches()
            if not matches:
                if self._find_possible_moves():
                    break
                else: # No moves possible, reshuffle
                    self.np_random.shuffle(self.grid.flat)
            else:
                self._remove_matches(matches)
                self._apply_gravity() # This also refills

    def _remove_matches(self, matches):
        for r, c in matches:
            self.grid[r, c] = 0

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == 0: continue
                
                # Horizontal
                if c < self.GRID_WIDTH - 2 and self.grid[r, c] == self.grid[r, c + 1] == self.grid[r, c + 2]:
                    matches.update([(r, c), (r, c + 1), (r, c + 2)])
                
                # Vertical
                if r < self.GRID_HEIGHT - 2 and self.grid[r, c] == self.grid[r + 1, c] == self.grid[r + 2, c]:
                    matches.update([(r, c), (r + 1, c), (r + 2, c)])
        return matches

    def _find_possible_moves(self):
        moves = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                val = self.grid[r, c]
                # Swap right
                if c < self.GRID_WIDTH - 1:
                    self.grid[r, c], self.grid[r, c + 1] = self.grid[r, c + 1], self.grid[r, c]
                    if self._find_all_matches(): moves.append(((r, c), (r, c + 1)))
                    self.grid[r, c], self.grid[r, c + 1] = self.grid[r, c + 1], self.grid[r, c] # Swap back
                # Swap down
                if r < self.GRID_HEIGHT - 1:
                    self.grid[r, c], self.grid[r + 1, c] = self.grid[r + 1, c], self.grid[r, c]
                    if self._find_all_matches(): moves.append(((r, c), (r + 1, c)))
                    self.grid[r, c], self.grid[r + 1, c] = self.grid[r + 1, c], self.grid[r, c] # Swap back
        return moves

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_bg()
        self._render_gems()
        self._render_cursor_and_selection()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_bg(self):
        for r in range(self.GRID_HEIGHT + 1):
            p1 = self._iso_to_screen(r, 0)
            p2 = self._iso_to_screen(r, self.GRID_WIDTH)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        for c in range(self.GRID_WIDTH + 1):
            p1 = self._iso_to_screen(0, c)
            p2 = self._iso_to_screen(self.GRID_HEIGHT, c)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)

    def _render_gems(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_val = self.grid[r, c]
                if gem_val == 0: continue
                
                # Check if gem is part of an active animation
                is_swapping = False
                if self.game_state == "SWAP" and self.swap_info:
                    if (r, c) == tuple(self.swap_info['pos1']) or (r, c) == tuple(self.swap_info['pos2']):
                        is_swapping = True

                is_falling = False
                for fall in self.falling_gems:
                    if (r,c) == tuple(fall['to']):
                        is_falling = True
                        break

                if not is_swapping and not is_falling:
                    self._draw_gem(r, c, gem_val, 1.0)
        
        # Draw animated gems on top
        self._render_swap_animation()
        self._render_fall_animation()
        self._render_clear_animation()

    def _render_swap_animation(self):
        if self.game_state != "SWAP" or not self.swap_info: return
        
        progress = self.animation_timer / self.SWAP_DURATION
        
        r1, c1 = self.swap_info['pos1']
        r2, c2 = self.swap_info['pos2']
        
        # Note: grid is already swapped, so we get the values from their new positions
        val1 = self.grid[r1, c1]
        val2 = self.grid[r2, c2]

        # Draw gem 2 moving from pos2 to pos1
        self._draw_gem_interpolated(r2, c2, r1, c1, val1, progress)
        # Draw gem 1 moving from pos1 to pos2
        self._draw_gem_interpolated(r1, c1, r2, c2, val2, progress)

    def _render_fall_animation(self):
        if self.game_state != "FALL": return
        progress = self.animation_timer / self.FALL_DURATION
        for fall in self.falling_gems:
            r_to, c_to = fall['to']
            r_from, c_from = fall['from']
            self._draw_gem_interpolated(r_from, c_from, r_to, c_to, fall['val'], progress)

    def _render_clear_animation(self):
        if self.game_state != "CLEAR" or not self.matched_gems: return
        
        progress = self.animation_timer / self.CLEAR_DURATION
        scale = 1.0 - progress
        alpha = 255 * (1.0 - progress)
        
        for r, c in self.matched_gems:
            # We can't get gem value as it's set to 0, so we just draw a white flash
            self._draw_gem(r, c, -1, scale, int(alpha))

    def _render_cursor_and_selection(self):
        # Draw selection
        if self.selected_gem:
            r, c = self.selected_gem
            pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
            color = (
                self.COLOR_SELECT[0], 
                self.COLOR_SELECT[1], 
                self.COLOR_SELECT[2], 
                100 + pulse * 100
            )
            self._draw_iso_rect(r, c, color)
            
        # Draw cursor
        if self.game_state == "INPUT":
            r, c = self.cursor_pos
            self._draw_iso_rect(r, c, self.COLOR_CURSOR + (150,))

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], max(0, int(p['life'] / 5)))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_str = f"{max(0, self.time_remaining):.1f}"
        timer_color = self.COLOR_TIMER_WARN if self.time_remaining < 10 else self.COLOR_TIMER
        timer_text = self.font_large.render(time_str, True, timer_color)
        self.screen.blit(timer_text, (self.screen_width - timer_text.get_width() - 10, 10))

    # --- Drawing Primitives ---
    
    def _iso_to_screen(self, r, c):
        x = self.origin_x + (c - r) * self.tile_width / 2
        y = self.origin_y + (c + r) * self.tile_height / 2
        return int(x), int(y)

    def _draw_gem(self, r, c, gem_val, scale=1.0, alpha=255):
        x, y = self._iso_to_screen(r, c)
        w = self.tile_width * scale
        h = self.tile_height * scale
        
        points = [
            (x, y - h / 2),
            (x + w / 2, y),
            (x, y + h / 2),
            (x - w / 2, y),
        ]
        
        if gem_val == -1: # White flash for clearing
            color = (255, 255, 255)
        else:
            color = self.GEM_COLORS[gem_val - 1]

        # Use gfxdraw for antialiasing
        pygame.gfxdraw.filled_polygon(self.screen, points, color + (alpha,))
        pygame.gfxdraw.aapolygon(self.screen, points, color + (alpha,))

    def _draw_gem_interpolated(self, r1, c1, r2, c2, gem_val, progress):
        x1, y1 = self._iso_to_screen(r1, c1)
        x2, y2 = self._iso_to_screen(r2, c2)
        
        ix = x1 + (x2 - x1) * progress
        iy = y1 + (y2 - y1) * progress
        
        w = self.tile_width
        h = self.tile_height
        points = [
            (int(ix), int(iy - h / 2)),
            (int(ix + w / 2), int(iy)),
            (int(ix), int(iy + h / 2)),
            (int(ix - w / 2), int(iy)),
        ]
        color = self.GEM_COLORS[gem_val - 1]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _draw_iso_rect(self, r, c, color):
        x, y = self._iso_to_screen(r, c)
        w = self.tile_width
        h = self.tile_height
        points = [
            (x, y - h / 2),
            (x + w / 2, y),
            (x, y + h / 2),
            (x - w / 2, y),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles

    # --- Gymnasium Interface Helpers ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "combo": self.combo_count,
            "game_state": self.game_state
        }
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")