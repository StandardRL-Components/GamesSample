
# Generated: 2025-08-28T05:03:28.340546
# Source Brief: brief_05440.md
# Brief Index: 5440

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrows to move the cursor. Press Space to select a gem, "
        "then move the cursor to an adjacent gem and press Space again to swap. "
        "Press Shift to deselect."
    )

    # Short, user-facing description of the game
    game_description = (
        "Swap adjacent gems to create matches of 3 or more in a race against "
        "time to clear the board and achieve a high score."
    )

    # Frames auto-advance for time limits and smooth graphics
    auto_advance = True

    # --- Constants ---
    # Game parameters
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_GEM_TYPES = 6
    BOARD_CLEAR_REWARD = 100
    TIME_OUT_PENALTY = -50
    MATCH_REWARD_PER_GEM = 1
    COMBO_REWARD = 5
    INVALID_SWAP_PENALTY = -0.1
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS * 2 # Generous buffer

    # Visuals
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GEM_SIZE = 40
    GEM_SPACING = 4
    ANIMATION_SPEED = 0.2  # Progress per frame, so 1/0.2 = 5 frames
    PARTICLE_LIFESPAN = 20

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_CURSOR = (255, 255, 0, 150)
    COLOR_SELECTION = (255, 255, 255, 200)
    GEM_COLORS = [
        (255, 50, 50),    # Red
        (50, 255, 50),    # Green
        (50, 150, 255),   # Blue
        (255, 255, 50),   # Yellow
        (255, 50, 255),   # Magenta
        (50, 255, 255),   # Cyan
    ]
    PARTICLE_COLORS = GEM_COLORS

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        self.grid_pixel_width = self.GRID_WIDTH * (self.GEM_SIZE + self.GEM_SPACING) - self.GEM_SPACING
        self.grid_pixel_height = self.GRID_HEIGHT * (self.GEM_SIZE + self.GEM_SPACING) - self.GEM_SPACING
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_pixel_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_pixel_height) // 2

        self.grid = None
        self.cursor_pos = None
        self.selected_gem = None
        self.game_state = None
        self.animations = []
        self.particles = []
        self.time_remaining = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.reward_this_step = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.gems_on_board = 0
        self.cursor_move_cooldown = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.GAME_DURATION_SECONDS
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_gem = None
        self.game_state = 'IDLE'
        self.animations = []
        self.particles = []
        self.reward_this_step = 0
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True # Prevent action on first frame
        self.cursor_move_cooldown = 0

        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        self.reward_this_step = 0
        self.steps += 1
        self.time_remaining -= 1 / self.FPS

        self._update_animations()
        self._update_particles()
        
        if self.game_state == 'IDLE':
            self._handle_input(action)
        
        if self.game_state == 'CHECK_MATCHES':
            self._process_matches()

        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.gems_on_board == 0:
                self.reward_this_step += self.BOARD_CLEAR_REWARD
            else: # Time out
                self.reward_this_step += self.TIME_OUT_PENALTY
            self.game_over = True

        reward = self.reward_this_step
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_board(self):
        self.grid = self.np_random.integers(1, self.NUM_GEM_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        
        while True:
            matches = self._find_matches()
            if not matches:
                break
            for r, c in matches:
                self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)

        while not self._find_possible_moves():
            # If no moves, shuffle the board and repeat match removal
            self.np_random.shuffle(self.grid.flat)
            while True:
                matches = self._find_matches()
                if not matches:
                    break
                for r, c in matches:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)

        self.gems_on_board = self.GRID_WIDTH * self.GRID_HEIGHT

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # Cursor movement
        if self.cursor_move_cooldown > 0:
            self.cursor_move_cooldown -= 1
        elif movement > 0:
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Up
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1) # Down
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Left
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1) # Right
            self.cursor_move_cooldown = 4 # Cooldown in frames

        # Deselect
        if shift_press and self.selected_gem:
            self.selected_gem = None
        
        # Select / Swap
        if space_press:
            r, c = self.cursor_pos[1], self.cursor_pos[0]
            if self.grid[r,c] == 0: return # Can't select empty space

            if not self.selected_gem:
                self.selected_gem = (r, c)
            else:
                sel_r, sel_c = self.selected_gem
                # Check for adjacency
                if abs(r - sel_r) + abs(c - sel_c) == 1:
                    self._attempt_swap((sel_r, sel_c), (r, c))
                else: # Invalid selection, treat as new selection
                    self.selected_gem = (r, c)

    def _attempt_swap(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        
        # Tentative swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
        matches1 = self._find_matches_at(r1, c1)
        matches2 = self._find_matches_at(r2, c2)
        
        if not matches1 and not matches2:
            # Invalid swap, revert
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            self.reward_this_step += self.INVALID_SWAP_PENALTY
            self.selected_gem = None
            self.animations.append({'type': 'invalid_swap', 'pos': pos2, 'progress': 0})
        else:
            # Valid swap, start animation
            self.game_state = 'SWAPPING'
            self.animations.append({'type': 'swap', 'pos1': pos1, 'pos2': pos2, 'progress': 0})
            self.selected_gem = None

    def _process_matches(self):
        matches = self._find_matches()
        if not matches:
            self.game_state = 'IDLE'
            return

        # Score rewards
        num_cleared = len(matches)
        self.reward_this_step += num_cleared * self.MATCH_REWARD_PER_GEM
        if num_cleared > 3:
            self.reward_this_step += self.COMBO_REWARD
        
        # Create particles and mark for removal
        for r, c in matches:
            self._create_particles(r, c)
            self.grid[r, c] = 0 # Mark for removal
        
        self.gems_on_board -= len(matches)
        self.animations.append({'type': 'match', 'gems': matches, 'progress': 0})
        self.game_state = 'REMOVING_GEMS'
        # Sound placeholder: # sfx_match.play()

    def _handle_gravity_and_refill(self):
        cols_to_refill = set()
        moved_gems = {} # { (from_r, c) : (to_r, c) }

        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                        moved_gems[(r, c)] = (empty_row, c)
                    empty_row -= 1
            
            if empty_row >= 0:
                cols_to_refill.add(c)
                for r_new in range(empty_row, -1, -1):
                    self.grid[r_new, c] = self.np_random.integers(1, self.NUM_GEM_TYPES + 1)
        
        if moved_gems or cols_to_refill:
            self.animations.append({'type': 'drop', 'moved': moved_gems, 'refill_cols': cols_to_refill, 'progress': 0})
            self.game_state = 'DROPPING_GEMS'
            # Sound placeholder: # sfx_drop.play()
        else:
            self.game_state = 'CHECK_MATCHES'

    def _update_animations(self):
        if not self.animations:
            return

        finished_animations = []
        for anim in self.animations:
            anim['progress'] = min(1.0, anim['progress'] + self.ANIMATION_SPEED)
            if anim['progress'] >= 1.0:
                finished_animations.append(anim)
        
        for anim in finished_animations:
            self.animations.remove(anim)
            if anim['type'] == 'swap':
                self.game_state = 'CHECK_MATCHES'
            elif anim['type'] == 'match':
                self._handle_gravity_and_refill()
            elif anim['type'] == 'drop':
                self.game_state = 'CHECK_MATCHES'
            elif anim['type'] == 'invalid_swap':
                self.game_state = 'IDLE'

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == 0: continue
                # Horizontal
                if c < self.GRID_WIDTH - 2 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.GRID_HEIGHT - 2 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _find_matches_at(self, r, c):
        gem_type = self.grid[r, c]
        if gem_type == 0: return set()
        
        h_matches, v_matches = { (r, c) }, { (r, c) }

        # Horizontal check
        for i in [-1, 1]:
            for j in range(1, self.GRID_WIDTH):
                nc = c + i*j
                if 0 <= nc < self.GRID_WIDTH and self.grid[r, nc] == gem_type:
                    h_matches.add((r, nc))
                else: break
        
        # Vertical check
        for i in [-1, 1]:
            for j in range(1, self.GRID_HEIGHT):
                nr = r + i*j
                if 0 <= nr < self.GRID_HEIGHT and self.grid[nr, c] == gem_type:
                    v_matches.add((nr, c))
                else: break
        
        found = set()
        if len(h_matches) >= 3: found.update(h_matches)
        if len(v_matches) >= 3: found.update(v_matches)
        return found
    
    def _find_possible_moves(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Try swapping right
                if c < self.GRID_WIDTH - 1:
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                    if self._find_matches_at(r,c) or self._find_matches_at(r,c+1):
                        self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                        return True
                    self.grid[r,c], self.grid[r,c+1] = self.grid[r,c+1], self.grid[r,c]
                # Try swapping down
                if r < self.GRID_HEIGHT - 1:
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
                    if self._find_matches_at(r,c) or self._find_matches_at(r+1,c):
                        self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
                        return True
                    self.grid[r,c], self.grid[r+1,c] = self.grid[r+1,c], self.grid[r,c]
        return False

    def _check_termination(self):
        return self.time_remaining <= 0 or self.gems_on_board == 0 or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "gems_left": self.gems_on_board, "time_left": self.time_remaining}
    
    def _grid_to_pixel(self, r, c):
        x = self.grid_offset_x + c * (self.GEM_SIZE + self.GEM_SPACING) + self.GEM_SIZE // 2
        y = self.grid_offset_y + r * (self.GEM_SIZE + self.GEM_SPACING) + self.GEM_SIZE // 2
        return int(x), int(y)

    def _create_particles(self, r, c):
        px, py = self._grid_to_pixel(r, c)
        gem_type = self.grid[r, c]
        if gem_type == 0: return
        color = self.GEM_COLORS[gem_type - 1]
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': self.PARTICLE_LIFESPAN, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_bg()
        self._render_gems()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_bg(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                px, py = self._grid_to_pixel(r, c)
                rect = pygame.Rect(px - self.GEM_SIZE // 2, py - self.GEM_SIZE // 2, self.GEM_SIZE, self.GEM_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, border_radius=4)

    def _render_gems(self):
        gem_positions = {}
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_positions[(r, c)] = self._grid_to_pixel(r, c)
        
        # Apply animations to positions
        for anim in self.animations:
            prog = anim['progress']
            if anim['type'] == 'swap':
                r1, c1 = anim['pos1']
                r2, c2 = anim['pos2']
                p1 = self._grid_to_pixel(r1, c1)
                p2 = self._grid_to_pixel(r2, c2)
                gem_positions[(r1, c1)] = (p1[0] + (p2[0]-p1[0])*prog, p1[1] + (p2[1]-p1[1])*prog)
                gem_positions[(r2, c2)] = (p2[0] + (p1[0]-p2[0])*prog, p2[1] + (p1[1]-p2[1])*prog)
            elif anim['type'] == 'drop':
                # Animate falling gems
                for (fr, c), (tr, c) in anim['moved'].items():
                    start_y = self._grid_to_pixel(fr, c)[1]
                    end_y = self._grid_to_pixel(tr, c)[1]
                    x = self._grid_to_pixel(fr, c)[0]
                    gem_positions[(fr, c)] = (x, start_y + (end_y-start_y)*prog)
                # Animate new gems
                for c in anim['refill_cols']:
                    for r in range(self.GRID_HEIGHT):
                        if self.grid[r, c] != 0 and (r,c) not in [v for k,v in anim['moved'].items()]:
                            start_y = self.grid_offset_y - self.GEM_SIZE
                            end_y = self._grid_to_pixel(r, c)[1]
                            x = self._grid_to_pixel(r,c)[0]
                            gem_positions[(r,c)] = (x, start_y + (end_y-start_y)*prog)

        # Draw gems
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type == 0: continue

                px, py = gem_positions.get((r,c), self._grid_to_pixel(r,c))
                color = self.GEM_COLORS[gem_type - 1]
                size = self.GEM_SIZE * 0.8
                alpha = 255

                for anim in self.animations:
                    if anim['type'] == 'match' and (r, c) in anim['gems']:
                        size *= (1 + anim['progress']) # Scale up
                        alpha = 255 * (1 - anim['progress']) # Fade out
                    if anim['type'] == 'invalid_swap' and (r,c) == anim['pos']:
                        offset = math.sin(anim['progress'] * math.pi * 4) * 5
                        px += offset

                pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), int(size/2), (*color, int(alpha)))
                pygame.gfxdraw.aacircle(self.screen, int(px), int(py), int(size/2), (*color, int(alpha)))

    def _render_cursor(self):
        if self.game_state != 'IDLE': return

        # Draw selection highlight
        if self.selected_gem:
            r, c = self.selected_gem
            px, py = self._grid_to_pixel(r, c)
            size = self.GEM_SIZE + 4
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            alpha = 100 + pulse * 100
            pygame.gfxdraw.box(self.screen, (px-size//2, py-size//2, size, size), (*self.COLOR_SELECTION[:3], int(alpha)))

        # Draw cursor
        r, c = self.cursor_pos
        px, py = self._grid_to_pixel(r, c)
        size = self.GEM_SIZE + 8
        pygame.gfxdraw.rectangle(self.screen, (px-size//2, py-size//2, size, size), self.COLOR_CURSOR)
        
    def _render_particles(self):
        for p in self.particles:
            size = int(max(0, (p['life'] / self.PARTICLE_LIFESPAN) * 5))
            pygame.draw.circle(self.screen, p['color'], p['pos'], size)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 10))

        # Time
        time_int = max(0, int(self.time_remaining))
        time_color = (255, 255, 255) if time_int > 10 else (255, 100, 100)
        time_text = self.font_large.render(f"TIME: {time_int}", True, time_color)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)

        # Gems left
        gems_text = self.font_small.render(f"GEMS LEFT: {self.gems_on_board}", True, (200, 200, 200))
        gems_rect = gems_text.get_rect(midbottom=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 10))
        self.screen.blit(gems_text, gems_rect)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")