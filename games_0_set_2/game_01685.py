
# Generated: 2025-08-28T02:23:06.357554
# Source Brief: brief_01685.md
# Brief Index: 1685

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. The last direction moved "
        "selects the swap direction. Press space to perform the swap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent gems to create matches of 3 or more and reach a target "
        "score before time runs out in this vibrant grid-based puzzle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_GEM_TYPES = 6
        self.TARGET_SCORE = 500
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 60
        self.TOTAL_TIME = self.GAME_DURATION_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 80)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.GEM_COLORS = [
            (255, 80, 80),    # Red
            (80, 255, 80),    # Green
            (80, 150, 255),   # Blue
            (255, 255, 80),   # Yellow
            (255, 80, 255),   # Magenta
            (80, 255, 255),   # Cyan
        ]
        self.PARTICLE_COLORS = [pygame.Color(c) for c in self.GEM_COLORS]

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Grid and rendering setup
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # State variables (will be initialized in reset)
        self.grid = None
        self.cursor_pos = None
        self.last_move_dir = None
        self.score = None
        self.time_remaining = None
        self.game_over = None
        self.game_state = None
        self.steps = None
        self.space_was_pressed = None
        
        self.animation_state = {}
        self.particles = []

        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.steps = 0
        self.time_remaining = self.TOTAL_TIME
        self.game_over = False
        self.game_state = 'IDLE' # States: IDLE, SWAP_ANIM, MATCH_CHECK, FALL_ANIM
        self.space_was_pressed = False
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.last_move_dir = [1, 0] # Default swap direction right
        
        self.animation_state = {}
        self.particles = []
        
        self._initialize_grid()
        while not self._check_for_possible_moves():
            self._initialize_grid()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if not self.game_over:
            self.time_remaining -= 1
            
            # --- State Machine Logic ---
            if self.game_state == 'IDLE':
                reward += self._handle_input(action)
            elif self.game_state == 'SWAP_ANIM':
                self._update_swap_animation()
            elif self.game_state == 'MATCH_CHECK':
                match_reward = self._handle_matches()
                if match_reward > 0:
                    reward += match_reward
                    self.game_state = 'FALL_ANIM'
                    self._start_fall_animation()
                else: # No new matches from falling gems
                    if not self._check_for_possible_moves():
                        reward -= 10 # Penalty for needing a reshuffle
                        self._reshuffle_board()
                    self.game_state = 'IDLE'
            elif self.game_state == 'FALL_ANIM':
                self._update_fall_animation()
        
        self.steps += 1
        
        # --- Termination Check ---
        terminated = False
        if self.score >= self.TARGET_SCORE and not self.game_over:
            reward += 100
            terminated = True
            self.game_over = True
            self.game_state = 'WIN'
        elif self.time_remaining <= 0 and not self.game_over:
            reward -= 100
            terminated = True
            self.game_over = True
            self.game_state = 'LOSE'
        
        # --- Particle Update ---
        self._update_particles()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    # --- Game Logic Sub-functions ---

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # --- Cursor Movement ---
        if movement == 1 and self.cursor_pos[1] > 0: # Up
            self.cursor_pos[1] -= 1; self.last_move_dir = [0, -1]
        elif movement == 2 and self.cursor_pos[1] < self.GRID_SIZE - 1: # Down
            self.cursor_pos[1] += 1; self.last_move_dir = [0, 1]
        elif movement == 3 and self.cursor_pos[0] > 0: # Left
            self.cursor_pos[0] -= 1; self.last_move_dir = [-1, 0]
        elif movement == 4 and self.cursor_pos[0] < self.GRID_SIZE - 1: # Right
            self.cursor_pos[0] += 1; self.last_move_dir = [1, 0]
        
        # --- Swap Action ---
        if space_held and not self.space_was_pressed:
            self.space_was_pressed = True
            
            p1 = self.cursor_pos
            p2 = [p1[0] + self.last_move_dir[0], p1[1] + self.last_move_dir[1]]
            
            if 0 <= p2[0] < self.GRID_SIZE and 0 <= p2[1] < self.GRID_SIZE:
                self._start_swap_animation(p1, p2)
                # sfx: swap
                return -0.1 # Small cost for performing an action
        
        if not space_held:
            self.space_was_pressed = False
            
        return 0

    def _start_swap_animation(self, p1, p2):
        self.game_state = 'SWAP_ANIM'
        self.animation_state = {
            'type': 'swap',
            'p1': list(p1),
            'p2': list(p2),
            'duration': 5, # frames
            'timer': 0
        }
        # Immediately swap in the grid data
        g1_type = self.grid[p1[1], p1[0]]
        g2_type = self.grid[p2[1], p2[0]]
        self.grid[p1[1], p1[0]] = g2_type
        self.grid[p2[1], p2[0]] = g1_type

    def _update_swap_animation(self):
        self.animation_state['timer'] += 1
        if self.animation_state['timer'] >= self.animation_state['duration']:
            if self.animation_state.get('type') == 'reverse_swap':
                self.animation_state = {}
                self.game_state = 'IDLE'
                return

            matches = self._find_matches()
            if not matches:
                # No match, reverse the swap
                p1 = self.animation_state['p1']
                p2 = self.animation_state['p2']
                g1_type = self.grid[p1[1], p1[0]]
                g2_type = self.grid[p2[1], p2[0]]
                self.grid[p1[1], p1[0]] = g2_type
                self.grid[p2[1], p2[0]] = g1_type
                # sfx: invalid swap
                self.animation_state['type'] = 'reverse_swap'
                self.animation_state['timer'] = 0
            else:
                self.animation_state = {}
                self.game_state = 'MATCH_CHECK'

    def _handle_matches(self):
        matches = self._find_matches()
        if not matches:
            return 0
            
        reward = 0
        reward += len(matches) # +1 per gem cleared
        
        match_groups = self._get_match_groups(matches)
        for group in match_groups:
            size = len(group)
            if size == 3: reward += 10
            elif size == 4: reward += 20
            else: reward += 30
        
        self.score += reward
        
        for r, c in matches:
            # sfx: Match sound
            self._create_particles(c, r, self.grid[r, c])
            self.grid[r, c] = -1 # Mark as empty
        
        return reward
        
    def _start_fall_animation(self):
        moves = self._apply_gravity()
        if not moves:
            self.game_state = 'MATCH_CHECK'
            return

        self.animation_state = {
            'type': 'fall',
            'moves': moves,
            'duration': 6,
            'timer': 0
        }
        # sfx: gems falling

    def _update_fall_animation(self):
        self.animation_state['timer'] += 1
        if self.animation_state['timer'] >= self.animation_state['duration']:
            self.animation_state = {}
            self.game_state = 'MATCH_CHECK'

    # --- Grid & Board Management ---

    def _initialize_grid(self):
        self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_SIZE, self.GRID_SIZE))
        while True:
            matches = self._find_matches()
            if not matches:
                break
            for r, c in matches:
                self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _get_match_groups(self, matches):
        if not matches: return []
        groups, visited = [], set()
        for r_start, c_start in matches:
            if (r_start, c_start) in visited: continue
            group, q = set(), [(r_start, c_start)]
            visited.add((r_start, c_start))
            while q:
                r, c = q.pop(0)
                group.add((r, c))
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in matches and (nr, nc) not in visited:
                        visited.add((nr, nc)); q.append((nr, nc))
            groups.append(group)
        return groups

    def _apply_gravity(self):
        moves = []
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        gem_type = self.grid[r, c]
                        self.grid[empty_row, c] = gem_type
                        self.grid[r, c] = -1
                        moves.append({'from': (r, c), 'to': (empty_row, c), 'type': gem_type})
                    empty_row -= 1
            for r in range(empty_row, -1, -1):
                gem_type = self.np_random.integers(0, self.NUM_GEM_TYPES)
                self.grid[r, c] = gem_type
                moves.append({'from': (-1 - (empty_row - r), c), 'to': (r, c), 'type': gem_type})
        return moves

    def _check_for_possible_moves(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                original_gem = self.grid[r,c]
                # Check swap right
                if c < self.GRID_SIZE - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_matches(): self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]; return True
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                # Check swap down
                if r < self.GRID_SIZE - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_matches(): self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]; return True
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
        return False

    def _reshuffle_board(self):
        # sfx: Reshuffle sound
        flat_gems = self.grid.flatten().tolist()
        self.np_random.shuffle(flat_gems)
        self.grid = np.array(flat_gems).reshape((self.GRID_SIZE, self.GRID_SIZE))
        
        while self._find_matches() or not self._check_for_possible_moves():
            self._initialize_grid() # Re-create a valid board from scratch
        
        self.game_state = 'IDLE'

    # --- Particle System ---
    def _create_particles(self, c, r, gem_type):
        px = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        color = self.PARTICLE_COLORS[gem_type]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_gems()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT))
            y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH, y))

    def _render_gems(self):
        drawn_gems = set()
        
        if self.game_state == 'SWAP_ANIM' and 'p1' in self.animation_state:
            self._render_animated_swap(drawn_gems)
        elif self.game_state == 'FALL_ANIM' and 'moves' in self.animation_state:
            self._render_animated_fall(drawn_gems)
        
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (r, c) in drawn_gems or self.grid[r, c] == -1:
                    continue
                self._draw_gem(self.screen, self.grid[r, c], c, r)

    def _render_animated_swap(self, drawn_gems):
        state = self.animation_state
        progress = state['timer'] / state['duration']
        
        r1, c1 = state['p1']
        r2, c2 = state['p2']
        drawn_gems.add(tuple(state['p1'])); drawn_gems.add(tuple(state['p2']))
        
        gem1_type = self.grid[r2, c2]
        pos1_x = c1 * (1 - progress) + c2 * progress
        pos1_y = r1 * (1 - progress) + r2 * progress
        
        gem2_type = self.grid[r1, c1]
        pos2_x = c2 * (1 - progress) + c1 * progress
        pos2_y = r2 * (1 - progress) + r1 * progress
        
        self._draw_gem(self.screen, gem1_type, pos2_x, pos2_y)
        self._draw_gem(self.screen, gem2_type, pos1_x, pos1_y)

    def _render_animated_fall(self, drawn_gems):
        state = self.animation_state
        progress = state['timer'] / state['duration']
        
        for move in state['moves']:
            r_from, c_from = move['from']
            r_to, c_to = move['to']
            drawn_gems.add((r_to, c_to))
            y_pos = r_from * (1 - progress) + r_to * progress
            self._draw_gem(self.screen, move['type'], c_to, y_pos)

    def _draw_gem(self, surface, gem_type, c, r, scale=1.0):
        if gem_type < 0 or gem_type >= self.NUM_GEM_TYPES: return
        
        x = int(self.GRID_OFFSET_X + (c + 0.5) * self.CELL_SIZE)
        y = int(self.GRID_OFFSET_Y + (r + 0.5) * self.CELL_SIZE)
        radius = int(self.CELL_SIZE * 0.38 * scale)
        color = self.GEM_COLORS[gem_type]
        
        if gem_type == 0: # Circle (Red)
            pygame.gfxdraw.aacircle(surface, x, y, radius, color); pygame.gfxdraw.filled_circle(surface, x, y, radius, color)
        elif gem_type == 1: # Square (Green)
            pygame.draw.rect(surface, color, pygame.Rect(x - radius, y - radius, radius * 2, radius * 2), border_radius=3)
        elif gem_type == 2: # Diamond (Blue)
            points = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]; pygame.gfxdraw.aapolygon(surface, points, color); pygame.gfxdraw.filled_polygon(surface, points, color)
        elif gem_type == 3: # Triangle Up (Yellow)
            points = [(x, y - radius), (x + radius, y + radius//2), (x - radius, y + radius//2)]; pygame.gfxdraw.aapolygon(surface, points, color); pygame.gfxdraw.filled_polygon(surface, points, color)
        elif gem_type == 4: # Hexagon (Magenta)
            points = [(x + math.cos(math.radians(a)) * radius, y + math.sin(math.radians(a)) * radius) for a in range(0, 360, 60)]; pygame.gfxdraw.aapolygon(surface, points, color); pygame.gfxdraw.filled_polygon(surface, points, color)
        elif gem_type == 5: # Star (Cyan)
            points = []; rads = [radius, radius*0.5]*5
            for i in range(10): points.append((x + math.cos(math.radians(i*36)) * rads[i], y + math.sin(math.radians(i*36)) * rads[i]));
            pygame.gfxdraw.aapolygon(surface, points, color); pygame.gfxdraw.filled_polygon(surface, points, color)

    def _render_cursor(self):
        if self.game_state != 'IDLE': return
        
        cy, cx = self.cursor_pos
        rect = pygame.Rect(self.GRID_OFFSET_X + cx * self.CELL_SIZE, self.GRID_OFFSET_Y + cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 3, border_radius=4)
        
        dy, dx = self.last_move_dir
        scy, scx = cy + dy, cx + dx
        if 0 <= scx < self.GRID_SIZE and 0 <= scy < self.GRID_SIZE:
            s_rect = pygame.Rect(self.GRID_OFFSET_X + scx * self.CELL_SIZE, self.GRID_OFFSET_Y + scy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, (255, 255, 255, 100), s_rect, 2, border_radius=4)
            
            start = (rect.centerx + dx*self.CELL_SIZE*0.3, rect.centery + dy*self.CELL_SIZE*0.3)
            end = (s_rect.centerx - dx*self.CELL_SIZE*0.3, s_rect.centery - dy*self.CELL_SIZE*0.3)
            pygame.draw.line(self.screen, (255, 255, 255), start, end, 2)
            
            angle = math.atan2(dy, dx)
            p1 = (end[0] - 8 * math.cos(angle - math.pi/6), end[1] - 8 * math.sin(angle - math.pi/6))
            p2 = (end[0] - 8 * math.cos(angle + math.pi/6), end[1] - 8 * math.sin(angle + math.pi/6))
            pygame.draw.line(self.screen, (255, 255, 255), end, p1, 2); pygame.draw.line(self.screen, (255, 255, 255), end, p2, 2)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 15))))
            color = p['color']; color.a = alpha
            size = max(1, int(p['life']/5))
            pygame.draw.circle(self.screen, color, p['pos'], size)
            
    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        bar_w, bar_h, bar_x, bar_y = 200, 20, self.SCREEN_WIDTH - 220, 15
        ratio = max(0, self.time_remaining / self.TOTAL_TIME)
        bar_color = (255,0,0) if ratio < 0.2 else (255,255,0) if ratio < 0.5 else (0,255,0)
            
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, bar_w * ratio, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_w, bar_h), 1)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180)); self.screen.blit(overlay, (0, 0))
        
        text, color = ("YOU WIN!", (100, 255, 100)) if self.game_state == 'WIN' else ("TIME UP!", (255, 100, 100))
        surf = self.font_game_over.render(text, True, color)
        self.screen.blit(surf, surf.get_rect(center=(self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2)))
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_frames": self.time_remaining,
            "game_state": self.game_state,
        }

    def close(self):
        pygame.quit()