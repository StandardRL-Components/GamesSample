
# Generated: 2025-08-28T04:30:59.561995
# Source Brief: brief_02347.md
# Brief Index: 2347

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrows to move the cursor. Press space to select a gem. "
        "Move to an adjacent gem and press space again to swap. Press shift to deselect."
    )

    # Short, user-facing description of the game
    game_description = (
        "Match 3 or more gems to clear them from the board. Create chain reactions for big "
        "scores! Clear the board to win, but watch your move count."
    )

    # The game state is static until an action is received.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 5, 5
        self.NUM_GEM_TYPES = 4
        self.MOVES_LIMIT = 15
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_UI_TEXT = (220, 220, 230)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECTED = (255, 255, 0)
        self.GEM_COLORS = [
            (255, 80, 80),    # Red
            (80, 255, 80),    # Green
            (80, 150, 255),   # Blue
            (255, 150, 50),   # Orange
        ]
        
        # --- Layout ---
        self.GEM_SIZE = 50
        self.GEM_PADDING = 8
        self.GRID_WIDTH_PX = self.GRID_COLS * (self.GEM_SIZE + self.GEM_PADDING) - self.GEM_PADDING
        self.GRID_HEIGHT_PX = self.GRID_ROWS * (self.GEM_SIZE + self.GEM_PADDING) - self.GEM_PADDING
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH_PX) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT_PX) // 2 + 20

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        
        # --- Game State (initialized in reset) ---
        self.grid = None
        self.cursor_pos = None
        self.selected_gem = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.cleared_colors = None
        
        # --- Animation State ---
        self.animations = []
        self.particles = []

        # Initialize state variables
        self.reset()

        # Validate implementation after full initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MOVES_LIMIT
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_gem = None
        self.animations.clear()
        self.particles.clear()
        self.cleared_colors = [False] * self.NUM_GEM_TYPES

        # Generate a grid with at least one possible move
        while True:
            self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_ROWS, self.GRID_COLS))
            if self._check_for_possible_moves():
                break

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        # 1. Process ongoing animations first
        if self.animations:
            self.animations[0]['progress'] += 0.15  # Animation speed
            if self.animations[0]['progress'] >= 1.0:
                reward += self._finish_animation(self.animations.pop(0))
        # 2. If no animations, process player input
        else:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            # Handle cursor movement
            if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
            elif movement == 2 and self.cursor_pos[1] < self.GRID_ROWS - 1: self.cursor_pos[1] += 1
            elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
            elif movement == 4 and self.cursor_pos[0] < self.GRID_COLS - 1: self.cursor_pos[0] += 1
            
            # Handle deselection
            if shift_held:
                self.selected_gem = None
            
            # Handle selection/swap
            if space_held:
                if self.selected_gem is None:
                    # Select a gem
                    self.selected_gem = tuple(self.cursor_pos)
                else:
                    # Attempt to swap with selected gem
                    dist = abs(self.selected_gem[0] - self.cursor_pos[0]) + abs(self.selected_gem[1] - self.cursor_pos[1])
                    if dist == 1:
                        # Valid adjacent swap
                        self.moves_left -= 1
                        reward -= 0.1 # Small penalty for using a move
                        self._start_swap_animation(self.selected_gem, tuple(self.cursor_pos))
                        self.selected_gem = None
                    else:
                        # Invalid swap, just re-select the new cursor position
                        self.selected_gem = tuple(self.cursor_pos)

        # 3. Check for termination conditions
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            # Check for win condition (all gems cleared)
            if np.sum(self.grid) == 0:
                reward += 100 # Win bonus
            else:
                reward -= 50 # Loss penalty

        if self.steps >= self.MAX_STEPS:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _finish_animation(self, anim):
        """Called when an animation completes. Applies game logic and may trigger new animations."""
        reward = 0
        if anim['type'] == 'swap':
            # Apply the swap to the grid
            p1, p2 = anim['pos1'], anim['pos2']
            self.grid[p1[1], p1[0]], self.grid[p2[1], p2[0]] = self.grid[p2[1], p2[0]], self.grid[p1[1], p1[0]]
            
            # Check for matches
            matches1 = self._find_matches_at(p1)
            matches2 = self._find_matches_at(p2)
            all_matches = matches1.union(matches2)

            if all_matches:
                reward += self._handle_matches(all_matches)
            elif not anim.get('is_swap_back', False):
                # No match, swap back
                self._start_swap_animation(p1, p2, is_swap_back=True)
        
        elif anim['type'] == 'match':
            # Gems have finished their "pop" animation, now remove them and apply gravity
            empty_cols = self._apply_gravity(anim['gems'])
            self._start_fall_animation(empty_cols)
        
        elif anim['type'] == 'fall':
            # Falling gems have settled, now check for chain reactions
            all_matches = self._find_all_matches()
            if all_matches:
                reward += self._handle_matches(all_matches)
        return reward

    def _handle_matches(self, matched_gems):
        """Processes a set of matched gems, calculates reward, and starts match animation."""
        reward = 0
        num_matched = len(matched_gems)
        
        if num_matched == 3: reward += 1
        elif num_matched == 4: reward += 2
        elif num_matched >= 5: reward += 3
        
        self.score += reward

        for r, c in matched_gems:
            self._create_particles(r, c, self.grid[r, c])
            
        self._start_match_animation(matched_gems)
        # sfx: gem_match_sound()
        
        # Check for clearing a color
        gem_types_in_match = {self.grid[r, c] for r, c in matched_gems}
        for gem_type in gem_types_in_match:
            if not self.cleared_colors[gem_type - 1]:
                if np.count_nonzero(self.grid == gem_type) == len([pos for pos in matched_gems if self.grid[pos] == gem_type]):
                    reward += 10
                    self.score += 10
                    self.cleared_colors[gem_type - 1] = True
                    # sfx: color_clear_fanfare()

        return reward

    def _apply_gravity(self, cleared_gems):
        """Makes gems fall down into empty spaces."""
        cols_with_gaps = {c for r, c in cleared_gems}
        empty_in_col = {c: 0 for c in cols_with_gaps}

        for r, c in cleared_gems:
            self.grid[r, c] = 0 # Mark as empty
        
        for c in sorted(list(cols_with_gaps)):
            write_row = self.GRID_ROWS - 1
            for read_row in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[read_row, c] != 0:
                    if read_row != write_row:
                        self.grid[write_row, c] = self.grid[read_row, c]
                        self.grid[read_row, c] = 0
                    write_row -= 1
            empty_in_col[c] = write_row + 1 # Number of empty cells from the top

        # Fill top rows with new gems
        for c, num_empty in empty_in_col.items():
            for r in range(num_empty):
                self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
        
        return empty_in_col
    
    # --- Animation Starters ---
    def _start_swap_animation(self, pos1, pos2, is_swap_back=False):
        self.animations.append({
            'type': 'swap', 'progress': 0.0,
            'pos1': pos1, 'pos2': pos2, 'is_swap_back': is_swap_back
        })
        # sfx: gem_swap_swoosh()

    def _start_match_animation(self, gems):
        self.animations.append({'type': 'match', 'progress': 0.0, 'gems': gems})

    def _start_fall_animation(self, empty_cols):
        self.animations.append({'type': 'fall', 'progress': 0.0, 'cols': empty_cols})

    # --- Match Finding Logic ---
    def _find_matches_at(self, pos):
        """Finds matches involving a specific grid position."""
        r, c = pos
        if self.grid[r, c] == 0: return set()
        
        color = self.grid[r, c]
        
        # Horizontal
        h_match = {(r, c)}
        # Left
        for i in range(c - 1, -1, -1):
            if self.grid[r, i] == color: h_match.add((r, i))
            else: break
        # Right
        for i in range(c + 1, self.GRID_COLS):
            if self.grid[r, i] == color: h_match.add((r, i))
            else: break
            
        # Vertical
        v_match = {(r, c)}
        # Up
        for i in range(r - 1, -1, -1):
            if self.grid[i, c] == color: v_match.add((i, c))
            else: break
        # Down
        for i in range(r + 1, self.GRID_ROWS):
            if self.grid[i, c] == color: v_match.add((i, c))
            else: break
        
        matches = set()
        if len(h_match) >= 3: matches.update(h_match)
        if len(v_match) >= 3: matches.update(v_match)
        return matches

    def _find_all_matches(self):
        """Finds all matches on the entire board."""
        all_matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                all_matches.update(self._find_matches_at((r, c)))
        return all_matches

    def _check_for_possible_moves(self):
        """Checks if any valid swap exists on the board."""
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Check swap right
                if c < self.GRID_COLS - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_matches_at((r,c)) or self._find_matches_at((r,c+1)):
                        self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                # Check swap down
                if r < self.GRID_ROWS - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_matches_at((r,c)) or self._find_matches_at((r+1,c)):
                        self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
        return False

    def _check_termination(self):
        if np.sum(self.grid) == 0: return True # Win
        if self.moves_left <= 0: return True # Loss
        if not self.animations and not self._check_for_possible_moves(): return True # No more moves
        return False

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid_lines()
        
        # Get animation data
        swap_anim = next((a for a in self.animations if a['type'] == 'swap'), None)
        match_anim = next((a for a in self.animations if a['type'] == 'match'), None)
        fall_anim = next((a for a in self.animations if a['type'] == 'fall'), None)
        
        # Draw all gems
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                gem_type = self.grid[r, c]
                if gem_type == 0: continue

                pos_x, pos_y = self._grid_to_screen(r, c)
                size = self.GEM_SIZE
                alpha = 255
                
                # Handle animations affecting this gem
                if swap_anim:
                    p1, p2, prog = swap_anim['pos1'], swap_anim['pos2'], swap_anim['progress']
                    if (r, c) == p1: pos_x, pos_y = self._interpolate_pos(p1, p2, prog)
                    elif (r, c) == p2: pos_x, pos_y = self._interpolate_pos(p2, p1, prog)
                
                if match_anim and (r, c) in match_anim['gems']:
                    prog = match_anim['progress']
                    size = int(self.GEM_SIZE * (1.0 - prog))
                    alpha = int(255 * (1.0 - prog))
                
                if fall_anim:
                    if c in fall_anim['cols']:
                        # This gem might be falling
                        is_falling = True
                        for read_row in range(self.GRID_ROWS - 1, -1, -1):
                            if self.grid[read_row, c] == gem_type:
                                # This is complex, so we simplify: just draw at final pos
                                is_falling = False; break
                        if is_falling:
                            num_empty = fall_anim['cols'][c]
                            start_y = pos_y - num_empty * (self.GEM_SIZE + self.GEM_PADDING)
                            pos_y = int(start_y + (pos_y - start_y) * fall_anim['progress'])

                self._draw_gem(gem_type, pos_x, pos_y, size, alpha)

        self._update_and_draw_particles()
        self._draw_cursor_and_selection()

    def _draw_grid_lines(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * (self.GEM_SIZE + self.GEM_PADDING) - self.GEM_PADDING // 2
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X - self.GEM_PADDING//2, y), (self.GRID_OFFSET_X + self.GRID_WIDTH_PX - self.GEM_PADDING//2, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * (self.GEM_SIZE + self.GEM_PADDING) - self.GEM_PADDING // 2
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y - self.GEM_PADDING//2), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT_PX - self.GEM_PADDING//2), 1)

    def _draw_gem(self, gem_type, x, y, size, alpha=255):
        if size <= 0: return
        color = self.GEM_COLORS[gem_type - 1]
        rect = pygame.Rect(x - size // 2, y - size // 2, size, size)
        
        # Use a temporary surface for alpha blending
        temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Draw a rounded rectangle for a softer look
        pygame.draw.rect(temp_surf, color + (alpha,), (0, 0, size, size), border_radius=int(size*0.3))
        
        # Add a subtle highlight
        highlight_color = (255, 255, 255, int(80 * (alpha/255.0)))
        pygame.gfxdraw.arc(temp_surf, size//2, size//2, int(size*0.35), 120, 300, highlight_color)
        
        self.screen.blit(temp_surf, rect.topleft)

    def _draw_cursor_and_selection(self):
        # Draw selection highlight
        if self.selected_gem:
            r, c = self.selected_gem
            x, y = self._grid_to_screen(r, c)
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
            size = self.GEM_SIZE + int(pulse * 8)
            pygame.gfxdraw.aacircle(self.screen, x, y, size // 2, self.COLOR_SELECTED)
            pygame.gfxdraw.aacircle(self.screen, x, y, size // 2 - 1, self.COLOR_SELECTED)

        # Draw cursor
        r, c = self.cursor_pos
        x, y = self._grid_to_screen(r, c)
        rect = pygame.Rect(x - self.GEM_SIZE//2 - 4, y - self.GEM_SIZE//2 - 4, self.GEM_SIZE + 8, self.GEM_SIZE + 8)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=10)

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Moves display
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(moves_text, moves_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = np.sum(self.grid) == 0
            end_text_str = "YOU WIN!" if win_condition else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, (255, 255, 100))
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    # --- Particle System ---
    def _create_particles(self, r, c, gem_type):
        x, y = self._grid_to_screen(r, c)
        color = self.GEM_COLORS[gem_type - 1]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'life': random.uniform(15, 30)
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 30))
                pygame.draw.circle(self.screen, p['color'] + (alpha,), p['pos'], max(0, int(p['life'] / 5)))

    # --- Helpers ---
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _grid_to_screen(self, r, c):
        x = self.GRID_OFFSET_X + c * (self.GEM_SIZE + self.GEM_PADDING) + self.GEM_SIZE // 2
        y = self.GRID_OFFSET_Y + r * (self.GEM_SIZE + self.GEM_PADDING) + self.GEM_SIZE // 2
        return x, y
    
    def _interpolate_pos(self, pos1_grid, pos2_grid, progress):
        x1, y1 = self._grid_to_screen(pos1_grid[0], pos1_grid[1])
        x2, y2 = self._grid_to_screen(pos2_grid[0], pos2_grid[1])
        interp_x = int(x1 + (x2 - x1) * progress)
        interp_y = int(y1 + (y2 - y1) * progress)
        return interp_x, interp_y

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()