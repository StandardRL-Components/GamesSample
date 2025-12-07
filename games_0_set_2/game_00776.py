import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem. "
        "Move to an adjacent gem and press Space again to swap."
    )

    game_description = (
        "A strategic gem-matching puzzle game. Swap adjacent gems on the isometric grid to "
        "form lines of three or more. Create combos and clear the board to reach the target score before you run out of moves."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    GEM_TYPES = 6
    WIN_SCORE = 50
    MAX_MOVES = 20
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECTED = (255, 255, 255, 150)

    GEM_COLORS = [
        (255, 50, 50),   # Red
        (50, 255, 50),   # Green
        (80, 80, 255),   # Blue
        (255, 255, 50),  # Yellow
        (255, 50, 255),  # Purple
        (50, 255, 255),  # Cyan
    ]
    GEM_HIGHLIGHT_COLORS = [tuple(min(255, int(c * 1.4)) for c in color) for color in GEM_COLORS]
    GEM_SHADOW_COLORS = [tuple(int(c * 0.6) for c in color) for color in GEM_COLORS]

    # --- Isometric Projection ---
    TILE_WIDTH = 48
    TILE_HEIGHT = TILE_WIDTH // 2
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 100

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.steps = 0
        self.game_state = 'AWAITING_INPUT' # AWAITING_INPUT, ANIMATING
        self.animations = []
        self.pending_reward = 0
        self.prev_space_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_pos = None
        self.game_state = 'AWAITING_INPUT'
        self.animations = []
        self.pending_reward = 0
        self.prev_space_held = False

        self._initialize_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        terminated = False
        truncated = False

        if self.game_state == 'AWAITING_INPUT':
            self._handle_player_action(movement, space_held)
        
        self._update_animations()

        if self.game_state == 'AWAITING_INPUT' and not self.animations:
             # Check for termination only when board is stable
            if self.moves_left <= 0 or self.score >= self.WIN_SCORE:
                self.game_over = True
                terminated = True
                if self.score >= self.WIN_SCORE:
                    self.pending_reward += 50 # Win bonus

        if self.steps >= self.MAX_STEPS -1:
            self.game_over = True
            truncated = True
            
        reward += self.pending_reward
        self.pending_reward = 0
        
        self.steps += 1
        self.prev_space_held = space_held

        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

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
            "moves_left": self.moves_left,
            "game_state": self.game_state
        }
    
    # --- Game Logic ---

    def _initialize_board(self):
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                self.grid[r, c] = self.np_random.integers(0, self.GEM_TYPES)
        
        # Ensure no initial matches
        while self._find_all_matches():
            matches = self._find_all_matches()
            for r, c in matches:
                # Avoid creating new matches by checking neighbors
                excluded_colors = set()
                if r > 0: excluded_colors.add(self.grid[r-1, c])
                if c > 0: excluded_colors.add(self.grid[r, c-1])
                
                possible_colors = [i for i in range(self.GEM_TYPES) if i not in excluded_colors]
                if not possible_colors: possible_colors = list(range(self.GEM_TYPES))
                
                self.grid[r, c] = self.np_random.choice(possible_colors)

    def _handle_player_action(self, movement, space_held):
        # Handle cursor movement
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1 # Up
        if movement == 2 and self.cursor_pos[1] < self.GRID_HEIGHT - 1: self.cursor_pos[1] += 1 # Down
        if movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1 # Left
        if movement == 4 and self.cursor_pos[0] < self.GRID_WIDTH - 1: self.cursor_pos[0] += 1 # Right

        # Handle selection/swap on space press
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            c, r = self.cursor_pos
            if self.selected_pos is None:
                # Select a gem
                self.selected_pos = [c, r]
            else:
                # A gem is already selected, attempt a swap
                sel_c, sel_r = self.selected_pos
                is_adjacent = abs(c - sel_c) + abs(r - sel_r) == 1
                
                if is_adjacent:
                    self._attempt_swap((sel_r, sel_c), (r, c))
                
                # Deselect after any second click
                self.selected_pos = None

    def _attempt_swap(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2

        # Simulate swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
        matches1 = self._find_matches_at(r1, c1)
        matches2 = self._find_matches_at(r2, c2)
        
        if not matches1 and not matches2:
            # Invalid move, swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            self.animations.append(SwapAnimation(pos1, pos2, valid=False, on_finish=self._on_animation_complete))
        else:
            # Valid move
            self.moves_left -= 1
            self.animations.append(SwapAnimation(pos1, pos2, valid=True, on_finish=self._process_matches))
        
        self.game_state = 'ANIMATING'

    def _process_matches(self):
        all_matches = self._find_all_matches()
        if not all_matches:
            self._on_animation_complete()
            return
        
        # Calculate reward
        num_matched = len(all_matches)
        self.pending_reward += num_matched
        self.score += num_matched
        
        # Bonus for larger matches
        if num_matched >= 4:
            self.pending_reward += 5

        # Add flash animations and mark for removal
        for r, c in all_matches:
            self.animations.append(FlashAnimation((r, c), on_finish=None))

        # Set a single callback for when all flashes are done
        self.animations[-1].on_finish = lambda: self._handle_gravity(all_matches)
        
    def _handle_gravity(self, removed_gems):
        cols_to_update = {c for r, c in removed_gems}
        
        fall_animations = []
        
        for c in cols_to_update:
            empty_r = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if (r, c) not in removed_gems:
                    if r != empty_r:
                        # This gem needs to fall
                        fall_animations.append(FallAnimation((r, c), (empty_r, c)))
                        self.grid[empty_r, c] = self.grid[r, c]
                    empty_r -= 1

            # Mark old positions as empty
            for r in range(empty_r, -1, -1):
                self.grid[r, c] = -1 # Sentinel for empty

        if not fall_animations:
            self._refill_board()
        else:
            for anim in fall_animations:
                self.animations.append(anim)
            self.animations[-1].on_finish = self._refill_board

    def _refill_board(self):
        refill_animations = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.GEM_TYPES)
                    refill_animations.append(RefillAnimation((r, c)))

        if not refill_animations:
             self._process_matches() # Check for chain reactions
        else:
            for anim in refill_animations:
                self.animations.append(anim)
            self.animations[-1].on_finish = self._process_matches # Check for chains

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                found = self._find_matches_at(r, c)
                if found:
                    matches.update(found)
        return list(matches)

    def _find_matches_at(self, r, c):
        gem_type = self.grid[r, c]
        if gem_type == -1: return []

        # Horizontal
        h_match = [(r, i) for i in range(self.GRID_WIDTH) if self.grid[r, i] == gem_type]
        h_groups = self._find_contiguous_groups([p[1] for p in h_match])
        
        # Vertical
        v_match = [(i, c) for i in range(self.GRID_HEIGHT) if self.grid[i, c] == gem_type]
        v_groups = self._find_contiguous_groups([p[0] for p in v_match])

        final_matches = set()
        for group in h_groups:
            if len(group) >= 3 and c in group:
                for col_idx in group:
                    final_matches.add((r, col_idx))
        
        for group in v_groups:
            if len(group) >= 3 and r in group:
                for row_idx in group:
                    final_matches.add((row_idx, c))

        return list(final_matches)

    def _find_contiguous_groups(self, indices):
        if not indices: return []
        indices.sort()
        groups = []
        current_group = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_group.append(indices[i])
            else:
                groups.append(current_group)
                current_group = [indices[i]]
        groups.append(current_group)
        return groups

    def _on_animation_complete(self):
        if not self.animations:
            self.game_state = 'AWAITING_INPUT'

    def _update_animations(self):
        if not self.animations:
            if self.game_state == 'ANIMATING':
                 self.game_state = 'AWAITING_INPUT'
            return

        finished_anims = [anim for anim in self.animations if anim.is_finished()]
        self.animations = [anim for anim in self.animations if not anim.is_finished()]

        for anim in self.animations:
            anim.update()
        
        for anim in finished_anims:
            if anim.on_finish:
                anim.on_finish()
        
        if not self.animations:
            self.game_state = 'AWAITING_INPUT'

    # --- Rendering ---

    def _grid_to_screen(self, r, c):
        x = self.ORIGIN_X + (c - r) * self.TILE_WIDTH // 2
        y = self.ORIGIN_Y + (c + r) * self.TILE_HEIGHT // 2
        return int(x), int(y)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            p1 = self._grid_to_screen(r, 0)
            p2 = self._grid_to_screen(r, self.GRID_WIDTH)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        for c in range(self.GRID_WIDTH + 1):
            p1 = self._grid_to_screen(0, c)
            p2 = self._grid_to_screen(self.GRID_HEIGHT, c)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)

        # Draw gems
        animation_subjects = set()
        for anim in self.animations:
            subject = anim.get_subject()
            if subject:
                if isinstance(subject, set):
                    animation_subjects.update(subject)
                else:
                    animation_subjects.add(subject)

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != -1 and (r,c) not in animation_subjects:
                    self._draw_gem(r, c, self.grid[r, c])
        
        # Draw animations
        for anim in self.animations:
            anim.draw(self)

        # Draw cursor and selection
        if self.selected_pos:
            sel_c, sel_r = self.selected_pos
            self._draw_highlight(sel_r, sel_c, self.COLOR_SELECTED)

        cur_c, cur_r = self.cursor_pos
        self._draw_highlight(cur_r, cur_c, self.COLOR_CURSOR, width=3)
        
    def _draw_gem(self, r, c, gem_type, offset_x=0, offset_y=0, scale=1.0):
        if gem_type < 0 or gem_type >= self.GEM_TYPES: return
        
        center_x, center_y = self._grid_to_screen(r, c)
        center_x += offset_x
        center_y += offset_y
        
        w = (self.TILE_WIDTH * 0.8 * scale) / 2
        h = (self.TILE_HEIGHT * 0.8 * scale) / 2

        points = [
            (center_x, center_y - h * 2), # Top
            (center_x + w, center_y),     # Right
            (center_x, center_y + h),     # Bottom
            (center_x - w, center_y),     # Left
        ]
        
        top_face = [points[0], points[3], (center_x, center_y), points[1]]
        bottom_face = [points[2], points[3], (center_x, center_y), points[1]]
        
        pygame.gfxdraw.filled_polygon(self.screen, top_face, self.GEM_HIGHLIGHT_COLORS[gem_type])
        pygame.gfxdraw.filled_polygon(self.screen, bottom_face, self.GEM_SHADOW_COLORS[gem_type])
        
        # Main color fill
        main_poly = [
            (center_x, center_y - h),
            (center_x + w * 0.8, center_y),
            (center_x, center_y + h * 0.8),
            (center_x - w * 0.8, center_y),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, main_poly, self.GEM_COLORS[gem_type])
        pygame.gfxdraw.aapolygon(self.screen, main_poly, self.GEM_COLORS[gem_type])


    def _draw_highlight(self, r, c, color, width=0):
        center_x, center_y = self._grid_to_screen(r, c)
        w, h = self.TILE_WIDTH // 2, self.TILE_HEIGHT // 2
        points = [
            (center_x, center_y - h),
            (center_x + w, center_y),
            (center_x, center_y + h),
            (center_x - w, center_y)
        ]
        if width == 0: # Fill
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        else: # Outline
            pygame.draw.lines(self.screen, color, True, points, width)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        moves_text = self.font_small.render(f"MOVES: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

# --- Animation Classes ---

class Animation:
    def __init__(self, duration, on_finish):
        self.duration = duration
        self.on_finish = on_finish
        self.progress = 0

    def update(self):
        if self.duration > 0:
            self.progress = min(1.0, self.progress + 1.0 / self.duration)
        else:
            self.progress = 1.0

    def is_finished(self):
        return self.progress >= 1.0

    def draw(self, env):
        pass

    def get_subject(self):
        return None

class SwapAnimation(Animation):
    def __init__(self, pos1, pos2, valid, on_finish):
        super().__init__(duration=6, on_finish=on_finish)
        self.r1, self.c1 = pos1
        self.r2, self.c2 = pos2
        self.valid = valid

    def draw(self, env):
        p1_x, p1_y = env._grid_to_screen(self.r1, self.c1)
        p2_x, p2_y = env._grid_to_screen(self.r2, self.c2)
        
        interp = self.progress if self.valid else math.sin(self.progress * math.pi)

        offset1_x = (p2_x - p1_x) * interp
        offset1_y = (p2_y - p1_y) * interp
        offset2_x = (p1_x - p2_x) * interp
        offset2_y = (p1_y - p2_y) * interp
        
        gem_type1 = env.grid[self.r2, self.c2] if self.valid else env.grid[self.r1, self.c1]
        gem_type2 = env.grid[self.r1, self.c1] if self.valid else env.grid[self.r2, self.c2]
        
        env._draw_gem(self.r1, self.c1, gem_type1, offset1_x, offset1_y)
        env._draw_gem(self.r2, self.c2, gem_type2, offset2_x, offset2_y)

    def get_subject(self):
        return {(self.r1, self.c1), (self.r2, self.c2)}

class FlashAnimation(Animation):
    def __init__(self, pos, on_finish):
        super().__init__(duration=10, on_finish=on_finish)
        self.r, self.c = pos
    
    def draw(self, env):
        scale = 1.0 + 0.3 * math.sin(self.progress * math.pi)
        env._draw_gem(self.r, self.c, env.grid[self.r, self.c], scale=scale)
    
    def get_subject(self):
        return (self.r, self.c)

class FallAnimation(Animation):
    def __init__(self, from_pos, to_pos):
        super().__init__(duration=8, on_finish=None)
        self.from_r, self.from_c = from_pos
        self.to_r, self.to_c = to_pos

    def draw(self, env):
        from_x, from_y = env._grid_to_screen(self.from_r, self.from_c)
        to_x, to_y = env._grid_to_screen(self.to_r, self.to_c)
        
        interp_y = from_y + (to_y - from_y) * self.progress
        
        # We draw the gem at its final grid location with a Y offset
        env._draw_gem(self.to_r, self.to_c, env.grid[self.to_r, self.to_c], offset_y=interp_y - to_y)

    def get_subject(self):
        return (self.to_r, self.to_c)

class RefillAnimation(Animation):
    def __init__(self, pos):
        super().__init__(duration=10, on_finish=None)
        self.r, self.c = pos
        
    def draw(self, env):
        scale = self.progress # Gem grows into place
        env._draw_gem(self.r, self.c, env.grid[self.r, self.c], scale=scale)

    def get_subject(self):
        return (self.r, self.c)