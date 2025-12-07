
# Generated: 2025-08-28T06:33:32.295137
# Source Brief: brief_02961.md
# Brief Index: 2961

        
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

    user_guide = (
        "Controls: Use arrows to move the cursor. Press space to select a monster. "
        "Move to an adjacent monster and press shift to swap."
    )

    game_description = (
        "Match 3 or more identical monsters in a grid to clear them. "
        "Clear the board across 3 stages before time runs out to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_SIZE = 8
        self.NUM_STAGES = 3
        self.TIME_PER_STAGE = 60
        self.MAX_STEPS = self.FPS * self.TIME_PER_STAGE * self.NUM_STAGES

        self.MONSTER_TYPES = 5
        self.COLORS = {
            "bg": (25, 25, 40),
            "grid": (50, 50, 70),
            "cursor": (255, 255, 0),
            "selection": (255, 255, 255),
            "text": (220, 220, 240),
            "win": (100, 255, 100),
            "lose": (255, 100, 100),
            "monsters": [
                (255, 80, 80),   # Red
                (80, 255, 80),   # Green
                (80, 150, 255),  # Blue
                (255, 255, 80),  # Yellow
                (200, 80, 255),  # Purple
            ]
        }

        # --- Gym Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.SysFont("monospace", 18, bold=True)
        self.end_font = pygame.font.SysFont("sans-serif", 50, bold=True)
        self.popup_font = pygame.font.SysFont("sans-serif", 20, bold=True)

        # --- Grid & Game Layout ---
        self.tile_size = 40
        self.grid_width = self.GRID_SIZE * self.tile_size
        self.grid_height = self.GRID_SIZE * self.tile_size
        self.grid_offset_x = (self.WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.HEIGHT - self.grid_height) // 2 + 20

        # --- State Variables ---
        # These are initialized in reset()
        self.grid = None
        self.score = None
        self.steps = None
        self.stage = None
        self.time_remaining = None
        self.game_over = None
        self.win_status = None
        self.cursor_pos = None
        self.selected_tile = None
        self.board_is_stable = None
        self.animations = None
        self.popups = None
        self.reward_this_step = None
        self.prev_action = None
        self.random_seed = None

        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.random_seed = seed
            random.seed(seed)
            np.random.seed(seed)
        
        self.score = 0
        self.steps = 0
        self.stage = 1
        self.time_remaining = self.TIME_PER_STAGE
        self.game_over = False
        self.win_status = None # "WIN" or "LOSE"

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_tile = None
        self.board_is_stable = False # Will be set to True after initial fill and check
        self.animations = []
        self.popups = []
        self.reward_this_step = 0
        self.prev_action = [0, 0, 0]

        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        
        if not self.game_over:
            self._update_timer()
            self._update_animations()

            if not self.animations:
                if not self.board_is_stable:
                    self._resolve_board()
                else:
                    self._handle_player_input(action)

        self.prev_action = action
        self.steps += 1
        
        terminated = self._check_termination()
        reward = self.reward_this_step

        if terminated and self.win_status == "WIN":
            reward += 100
        elif terminated and self.win_status == "LOSE":
            reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Core Game Logic ---

    def _generate_board(self):
        self.grid = np.random.randint(1, self.MONSTER_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        while self._find_all_matches() or not self._find_possible_moves():
            self.grid = np.random.randint(1, self.MONSTER_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        self.board_is_stable = True

    def _reshuffle_board(self):
        # Create a flat list of all monsters, shuffle, and reshape
        flat_grid = self.grid.flatten()
        random.shuffle(flat_grid)
        self.grid = flat_grid.reshape((self.GRID_SIZE, self.GRID_SIZE))
        
        # Add a visual effect for the shuffle
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                self.animations.append({
                    'type': 'shuffle', 'pos': (c, r), 'progress': 0, 
                    'duration': 15 + random.randint(0, 10)
                })
        
        self.board_is_stable = False
        self.selected_tile = None
        # After shuffle, re-check for matches/moves. If still bad, regenerate.
        if self._find_all_matches() or not self._find_possible_moves():
            self._generate_board()


    def _handle_player_input(self, action):
        movement, space, shift = action
        space_press = space == 1 and self.prev_action[1] == 0
        shift_press = shift == 1 and self.prev_action[2] == 0

        # Movement
        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_SIZE - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_SIZE - 1)

        # Select
        if space_press:
            self.selected_tile = tuple(self.cursor_pos)
            # Sound: select_tile.wav

        # Swap
        if shift_press and self.selected_tile:
            target_tile = tuple(self.cursor_pos)
            
            # Check for adjacency
            if abs(self.selected_tile[0] - target_tile[0]) + abs(self.selected_tile[1] - target_tile[1]) == 1:
                # Check if swap creates a match
                if self._check_swap_for_match(self.selected_tile, target_tile):
                    self._initiate_swap(self.selected_tile, target_tile)
                    # Sound: swap_success.wav
                else:
                    self._initiate_swap(self.selected_tile, target_tile, is_fail=True)
                    self.reward_this_step -= 0.1
                    # Sound: swap_fail.wav
            else: # Not adjacent, treat as re-selection
                 self.selected_tile = tuple(self.cursor_pos)
                 # Sound: select_tile.wav


    def _resolve_board(self):
        matches = self._find_all_matches()
        if matches:
            num_matched = len(matches)
            self.score += num_matched
            self.reward_this_step += num_matched

            # Add score popup
            centroid = np.mean(list(matches), axis=0)
            popup_pos = self._grid_to_pixel(centroid[0], centroid[1])
            self.popups.append({
                'text': f"+{num_matched}", 'pos': popup_pos, 'progress': 0, 'duration': 30
            })
            # Sound: match_clear.wav

            for r, c in matches:
                self.grid[r, c] = 0 # Mark for removal
                self.animations.append({
                    'type': 'explode', 'pos': (c, r), 'progress': 0, 'duration': 15,
                    'particles': self._create_particles(c, r)
                })
            
            self._initiate_falls()
            self.board_is_stable = False
        else:
            self.board_is_stable = True
            if np.all(self.grid == 0):
                self._advance_stage()
            elif not self._find_possible_moves():
                self._reshuffle_board()

    def _advance_stage(self):
        self.reward_this_step += 5
        self.stage += 1
        if self.stage > self.NUM_STAGES:
            self.game_over = True
            self.win_status = "WIN"
            # Sound: game_win.wav
        else:
            self.time_remaining = self.TIME_PER_STAGE
            self._generate_board()
            # Sound: stage_clear.wav

    def _update_timer(self):
        self.time_remaining -= 1 / self.FPS
        if self.time_remaining <= 0:
            self.time_remaining = 0
            self.game_over = True
            self.win_status = "LOSE"
            # Sound: game_lose.wav
    
    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage}

    # --- Match & Swap Helpers ---

    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 0: continue
                # Horizontal
                if c < self.GRID_SIZE - 2 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.GRID_SIZE - 2 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _check_swap_for_match(self, pos1, pos2):
        temp_grid = np.copy(self.grid)
        r1, c1 = pos1
        r2, c2 = pos2
        temp_grid[r1, c1], temp_grid[r2, c2] = temp_grid[r2, c2], temp_grid[r1, c1]
        
        for r, c in [pos1, pos2]:
            val = temp_grid[r,c]
            if val == 0: continue
            # Horizontal check
            h_count = 1
            for i in range(1, 3):
                if c - i >= 0 and temp_grid[r, c-i] == val: h_count += 1
                else: break
            for i in range(1, 3):
                if c + i < self.GRID_SIZE and temp_grid[r, c+i] == val: h_count += 1
                else: break
            if h_count >= 3: return True
            
            # Vertical check
            v_count = 1
            for i in range(1, 3):
                if r - i >= 0 and temp_grid[r-i, c] == val: v_count += 1
                else: break
            for i in range(1, 3):
                if r + i < self.GRID_SIZE and temp_grid[r+i, c] == val: v_count += 1
                else: break
            if v_count >= 3: return True
            
        return False

    def _find_possible_moves(self):
        moves = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Check swap with right neighbor
                if c < self.GRID_SIZE - 1:
                    if self._check_swap_for_match((r, c), (r, c+1)):
                        moves.append(((r, c), (r, c+1)))
                # Check swap with down neighbor
                if r < self.GRID_SIZE - 1:
                    if self._check_swap_for_match((r, c), (r+1, c)):
                        moves.append(((r, c), (r+1, c)))
        return moves
    
    def _initiate_falls(self):
        for c in range(self.GRID_SIZE):
            empty_count = 0
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    # Animate this tile falling
                    self.animations.append({
                        'type': 'fall', 'from_pos': (c, r), 'to_pos': (c, r + empty_count),
                        'progress': 0, 'duration': 10 + empty_count * 2
                    })
                    # Move in grid data structure
                    self.grid[r + empty_count, c] = self.grid[r, c]
                    self.grid[r, c] = 0

            # Refill top rows
            for i in range(empty_count):
                self.grid[i, c] = np.random.randint(1, self.MONSTER_TYPES + 1)
                self.animations.append({
                    'type': 'fall', 'from_pos': (c, i - empty_count), 'to_pos': (c, i),
                    'progress': 0, 'duration': 10 + (empty_count - i) * 2
                })

    def _initiate_swap(self, pos1, pos2, is_fail=False):
        duration = 8 if is_fail else 15
        self.animations.append({
            'type': 'swap', 'pos1': pos1, 'pos2': pos2, 
            'progress': 0, 'duration': duration, 'is_fail': is_fail
        })
        self.board_is_stable = False
        self.selected_tile = None


    # --- Animation & Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLORS["bg"])
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _update_animations(self):
        active_anims = []
        for anim in self.animations:
            anim['progress'] += 1
            if anim['progress'] < anim['duration']:
                active_anims.append(anim)
            else: # Animation finished
                if anim['type'] == 'swap' and not anim['is_fail']:
                    r1, c1 = anim['pos1']
                    r2, c2 = anim['pos2']
                    self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        self.animations = active_anims
        
        active_popups = []
        for popup in self.popups:
            popup['progress'] += 1
            if popup['progress'] < popup['duration']:
                active_popups.append(popup)
        self.popups = active_popups

    def _render_game(self):
        self._draw_grid()
        
        # Get a list of tiles involved in a swap animation
        swapping_tiles = set()
        for anim in self.animations:
            if anim['type'] == 'swap':
                swapping_tiles.add(anim['pos1'])
                swapping_tiles.add(anim['pos2'])
        
        # Draw static monsters
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] != 0 and (r, c) not in swapping_tiles and not any(a['type'] == 'fall' and a['to_pos'] == (c, r) for a in self.animations):
                    is_exploding = any(a['type'] == 'explode' and a['pos'] == (c, r) for a in self.animations)
                    if not is_exploding:
                        self._draw_monster((c, r), self.grid[r, c])

        # Draw animated elements
        for anim in self.animations:
            if anim['type'] == 'swap': self._draw_swap_anim(anim)
            if anim['type'] == 'fall': self._draw_fall_anim(anim)
            if anim['type'] == 'explode': self._draw_explode_anim(anim)
            if anim['type'] == 'shuffle': self._draw_shuffle_anim(anim)

        # Draw cursor and selection
        if not self.game_over:
            self._draw_cursor()
            if self.selected_tile:
                self._draw_selection()
    
    def _render_ui(self):
        # Top bar text
        score_text = self.ui_font.render(f"Score: {self.score}", True, self.COLORS["text"])
        time_text = self.ui_font.render(f"Time: {int(self.time_remaining)}", True, self.COLORS["text"])
        stage_text = self.ui_font.render(f"Stage: {self.stage}/{self.NUM_STAGES}", True, self.COLORS["text"])
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (self.WIDTH // 2 - time_text.get_width() // 2, 10))
        self.screen.blit(stage_text, (self.WIDTH - stage_text.get_width() - 10, 10))

        # Popups
        for popup in self.popups:
            self._draw_popup(popup)

        # Game Over / Win screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.win_status == "WIN":
                msg = "YOU WIN!"
                color = self.COLORS["win"]
            else:
                msg = "GAME OVER"
                color = self.COLORS["lose"]
            
            end_text = self.end_font.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _draw_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_x = self.grid_offset_x + i * self.tile_size
            pygame.draw.line(self.screen, self.COLORS["grid"], (start_x, self.grid_offset_y), (start_x, self.grid_offset_y + self.grid_height))
            # Horizontal
            start_y = self.grid_offset_y + i * self.tile_size
            pygame.draw.line(self.screen, self.COLORS["grid"], (self.grid_offset_x, start_y), (self.grid_offset_x + self.grid_width, start_y))

    def _draw_monster(self, grid_pos, monster_type, size_mod=0):
        c, r = grid_pos
        px, py = self._grid_to_pixel(c, r)
        
        color = self.COLORS["monsters"][monster_type - 1]
        size = self.tile_size * 0.7 + size_mod
        radius = int(size / 2)
        
        # Idle bobbing animation
        bob = math.sin(self.steps * 0.1 + c + r) * 2
        py += bob
        
        center = (int(px), int(py))
        
        if monster_type == 1: # Circle
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, color)
        elif monster_type == 2: # Square
            rect = pygame.Rect(center[0] - radius, center[1] - radius, size, size)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
        elif monster_type == 3: # Triangle
            points = [
                (center[0], center[1] - radius),
                (center[0] - radius, center[1] + radius * 0.7),
                (center[0] + radius, center[1] + radius * 0.7),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif monster_type == 4: # Diamond
            points = [
                (center[0], center[1] - radius),
                (center[0] - radius, center[1]),
                (center[0], center[1] + radius),
                (center[0] + radius, center[1]),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif monster_type == 5: # Star
            self._draw_star(center, radius, color)

    def _draw_star(self, center, radius, color):
        points = []
        for i in range(10):
            angle = math.pi / 5 * i - math.pi / 2
            r = radius if i % 2 == 0 else radius * 0.5
            points.append((center[0] + r * math.cos(angle), center[1] + r * math.sin(angle)))
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_cursor(self):
        c, r = self.cursor_pos
        px = self.grid_offset_x + c * self.tile_size
        py = self.grid_offset_y + r * self.tile_size
        rect = pygame.Rect(px, py, self.tile_size, self.tile_size)
        pygame.draw.rect(self.screen, self.COLORS["cursor"], rect, 3)

    def _draw_selection(self):
        c, r = self.selected_tile
        px = self.grid_offset_x + c * self.tile_size
        py = self.grid_offset_y + r * self.tile_size
        
        # Pulsing effect
        alpha = 128 + 127 * math.sin(self.steps * 0.2)
        color = (*self.COLORS["selection"], alpha)
        
        overlay = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        overlay.fill(color)
        self.screen.blit(overlay, (px, py))

    def _draw_swap_anim(self, anim):
        p = anim['progress'] / anim['duration']
        if anim['is_fail']:
            p = 1.0 - abs(1.0 - 2.0 * p) # Parabolic path: 0 -> 1 -> 0
        
        r1, c1 = anim['pos1']
        r2, c2 = anim['pos2']

        # Get monster types from grid (they haven't been swapped in data yet)
        type1 = self.grid[r1, c1]
        type2 = self.grid[r2, c2]

        # Interpolate positions
        pos1_interp = (c1 * (1 - p) + c2 * p, r1 * (1 - p) + r2 * p)
        pos2_interp = (c2 * (1 - p) + c1 * p, r2 * (1 - p) + r1 * p)
        
        if type1 != 0: self._draw_monster(pos1_interp, type1)
        if type2 != 0: self._draw_monster(pos2_interp, type2)

    def _draw_fall_anim(self, anim):
        p = anim['progress'] / anim['duration']
        
        from_c, from_r = anim['from_pos']
        to_c, to_r = anim['to_pos']
        
        monster_type = self.grid[to_r, to_c]
        if monster_type == 0: return

        interp_r = from_r * (1 - p) + to_r * p
        self._draw_monster((to_c, interp_r), monster_type)
        
    def _draw_explode_anim(self, anim):
        for particle in anim['particles']:
            particle['pos'][0] += particle['vel'][0]
            particle['pos'][1] += particle['vel'][1]
            particle['life'] -= 1
            
            p = particle['life'] / particle['max_life']
            radius = int(particle['size'] * p)
            if radius > 0:
                color = (*particle['color'], int(255 * p))
                center = (int(particle['pos'][0]), int(particle['pos'][1]))
                pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, color)
                pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, color)
    
    def _draw_shuffle_anim(self, anim):
        c, r = anim['pos']
        p = anim['progress'] / anim['duration']
        size_mod = math.sin(p * math.pi) * -self.tile_size * 0.5
        self._draw_monster((c,r), self.grid[r,c], size_mod)

    def _draw_popup(self, popup):
        p = popup['progress'] / popup['duration']
        
        text_surf = self.popup_font.render(popup['text'], True, self.COLORS['win'])
        alpha = 255 if p < 0.5 else 255 * (1 - (p-0.5)*2)
        text_surf.set_alpha(alpha)
        
        pos_y = popup['pos'][1] - 20 * p
        pos_x = popup['pos'][0]
        
        text_rect = text_surf.get_rect(center=(pos_x, pos_y))
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, c, r):
        px, py = self._grid_to_pixel(c, r)
        monster_type = self.grid[r, c]
        color = self.COLORS['monsters'][monster_type - 1]
        particles = []
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(10, 20)
            particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life, 'max_life': life,
                'size': random.randint(3, 7),
                'color': color
            })
        return particles

    def _grid_to_pixel(self, c, r):
        px = self.grid_offset_x + (c + 0.5) * self.tile_size
        py = self.grid_offset_y + (r + 0.5) * self.tile_size
        return px, py
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=42)
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