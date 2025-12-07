import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
import pygame.gfxdraw
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:30:55.493830
# Source Brief: brief_00462.md
# Brief Index: 462
# """import gymnasium as gym
class GameEnv(gym.Env):
    """
    Dreamfall Defenders: A Gymnasium environment where the agent matches dream tiles
    to power defensive constructs and survive a nightmare onslaught.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Match dream tiles to gain energy, then spend it to summon and upgrade constructs. "
        "Your constructs will automatically defend against waves of nightmares."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to select/swap tiles. "
        "Move the cursor over a portal and press space to build/upgrade. Press shift to cycle build options."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 8, 8
    TILE_SIZE = 40
    GRID_WIDTH, GRID_HEIGHT = GRID_COLS * TILE_SIZE, GRID_ROWS * TILE_SIZE
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH) // 4
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT) // 2
    NUM_TILE_TYPES = 5
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_GRID_BG = (20, 15, 40, 150)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_ENERGY = (255, 255, 100)
    COLOR_NIGHTMARE = [(255, 50, 50), (255, 120, 50)]
    TILE_COLORS = [
        (60, 180, 255),  # Blue
        (100, 255, 100), # Green
        (255, 100, 200), # Pink
        (180, 100, 255), # Purple
        (255, 200, 80),  # Orange
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        self.render_mode = render_mode
        # No need to call _initialize_state here, reset() will do it.
        # self.validate_implementation() is for dev, not needed in production __init__

    def _initialize_state(self):
        """Initializes all state variables to empty/default values."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.energy = 0.0
        self.max_energy = 100.0
        
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.cursor_pos = [0, 0]
        self.selected_tile = None
        
        self.portals = []
        self.constructs = []
        self.nightmares = []
        self.particles = []
        
        self.unlocked_constructs = [0]
        self.selected_construct_idx = 0
        
        self.nightmare_spawn_rate = 0.05
        self.nightmare_base_hp = 5
        self.nightmare_base_damage = 1
        
        self.no_match_steps = 0
        self.has_built_construct = False
        
        # Action state (for single press detection)
        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        
        self.energy = 20.0 # Starting energy
        self._create_grid()
        while self._find_matches():
            self._create_grid() # Ensure no matches on start
            
        self._setup_portals()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        step_reward = 0

        # --- Handle Input and Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Detect button presses (rising edge)
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        self._move_cursor(movement)
        
        if shift_press:
            # Cycle through available construct types
            self.selected_construct_idx = (self.selected_construct_idx + 1) % len(self.unlocked_constructs)
            
        if space_press:
            step_reward += self._handle_interaction()

        # --- Game Logic Updates ---
        match_reward = self._update_matches()
        step_reward += match_reward

        damage_reward = self._update_constructs()
        step_reward += damage_reward
        
        self._update_nightmares()
        self._update_particles()
        
        self._update_difficulty()
        self._check_softlock()

        # --- Termination and Final Reward ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if truncated and not terminated: # Victory by timeout
                step_reward += 100
            elif terminated: # Loss
                step_reward -= 100
        
        self.score += step_reward
        
        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    # --- Helper Methods: Game Logic ---
    
    def _create_grid(self):
        self.grid = self.np_random.integers(0, self.NUM_TILE_TYPES, size=(self.GRID_ROWS, self.GRID_COLS))

    def _setup_portals(self):
        portal_y_offset = self.GRID_Y_OFFSET + self.GRID_HEIGHT // 2 - 60
        portal_x = self.GRID_X_OFFSET + self.GRID_WIDTH + 60
        self.portals = [
            {'pos': (portal_x, portal_y_offset), 'construct_id': None},
            {'pos': (portal_x, portal_y_offset + 120), 'construct_id': None}
        ]

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Up
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1) # Down
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Left
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1) # Right
    
    def _handle_interaction(self):
        """Handle the 'space' press based on cursor location."""
        # Check if cursor is over a portal
        for i, portal in enumerate(self.portals):
            px, py = portal['pos']
            # Create a small rect around the portal for cursor collision
            portal_rect = pygame.Rect(px - 25, py - 25, 50, 50)
            
            # Check if the tile under the cursor is "in" the portal area
            # A bit of a fudge, but makes it playable
            cursor_rect = pygame.Rect(
                self.GRID_X_OFFSET + self.cursor_pos[1] * self.TILE_SIZE,
                self.GRID_Y_OFFSET + self.cursor_pos[0] * self.TILE_SIZE,
                self.TILE_SIZE, self.TILE_SIZE
            )

            if portal_rect.clipline(
                (self.SCREEN_WIDTH, cursor_rect.centery),
                (0, cursor_rect.centery)
            ):
                 # Simplistic check: if cursor is on the rightmost column, assume portal interaction
                if self.cursor_pos[1] == self.GRID_COLS - 1:
                    return self._handle_portal_interaction(i)

        # Otherwise, handle tile interaction
        return self._handle_tile_interaction()

    def _handle_tile_interaction(self):
        r, c = self.cursor_pos
        if self.selected_tile is None:
            self.selected_tile = (r, c)
            return 0
        else:
            sr, sc = self.selected_tile
            # Check for adjacency
            if abs(r - sr) + abs(c - sc) == 1:
                # Swap tiles
                self.grid[r, c], self.grid[sr, sc] = self.grid[sr, sc], self.grid[r, c]
                
                # Check if this swap creates a match
                matches = self._find_matches()
                if not matches:
                    # If no match, swap back
                    self.grid[r, c], self.grid[sr, sc] = self.grid[sr, sc], self.grid[r, c]
                else:
                    self.no_match_steps = 0 # Successful action resets counter
                
                self.selected_tile = None
                return 0
            else:
                # If not adjacent, select the new tile instead
                self.selected_tile = (r, c)
                return 0

    def _handle_portal_interaction(self, portal_idx):
        portal = self.portals[portal_idx]
        construct_type = self.unlocked_constructs[self.selected_construct_idx]
        cost = 15 + (construct_type * 10)
        
        if portal['construct_id'] is None: # Summon
            if self.energy >= cost:
                self.energy -= cost
                new_construct = self._create_construct(portal['pos'], construct_type)
                self.constructs.append(new_construct)
                portal['construct_id'] = id(new_construct)
                self.has_built_construct = True
                return 5 # Summon reward
        else: # Upgrade
            cost = 10 # Upgrade cost
            if self.energy >= cost:
                for con in self.constructs:
                    if id(con) == portal['construct_id']:
                        self.energy -= cost
                        con['level'] += 1
                        con['max_hp'] += 15
                        con['hp'] = con['max_hp']
                        con['damage'] += 2
                        con['range'] += 5
                        self._create_particles(con['pos'], (255,255,255), 20, 3)
                        return 2 # Upgrade reward
        return 0

    def _create_construct(self, pos, type_idx):
        base_stats = [
            {'hp': 50, 'damage': 2, 'range': 100, 'cooldown': 60, 'color': (0, 150, 255)}, # Ranged
            {'hp': 80, 'damage': 5, 'range': 40,  'cooldown': 90, 'color': (0, 255, 150)}, # Melee
            {'hp': 40, 'damage': 1, 'range': 150, 'cooldown': 30, 'color': (200, 100, 255)}, # Fast
        ]
        stats = base_stats[type_idx]
        return {
            'pos': pos, 'type': type_idx, 'level': 1,
            'hp': stats['hp'], 'max_hp': stats['hp'],
            'damage': stats['damage'], 'range': stats['range'],
            'color': stats['color'], 'fire_cooldown': 0,
            'max_cooldown': stats['cooldown']
        }

    def _find_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2] != -1:
                    matches.add((r, c)); matches.add((r, c+1)); matches.add((r, c+2))
        # Vertical
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c] != -1:
                    matches.add((r, c)); matches.add((r+1, c)); matches.add((r+2, c))
        return list(matches)

    def _update_matches(self):
        matches = self._find_matches()
        if not matches:
            return 0
        
        reward = 0
        for r, c in matches:
            reward += 1
            start_pos = (self.GRID_X_OFFSET + c * self.TILE_SIZE + self.TILE_SIZE // 2,
                         self.GRID_Y_OFFSET + r * self.TILE_SIZE + self.TILE_SIZE // 2)
            self._create_particles(start_pos, self.COLOR_ENERGY, 1, 5, is_energy=True)
            self.grid[r, c] = -1 # Mark for removal
        
        # Gravity
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1
            for r in range(empty_row, -1, -1):
                self.grid[r, c] = self.np_random.integers(0, self.NUM_TILE_TYPES)
        
        chain_reward = self._update_matches()
        return reward + chain_reward

    def _update_constructs(self):
        damage_reward = 0
        for con in self.constructs:
            if con['fire_cooldown'] > 0:
                con['fire_cooldown'] -= 1
                continue
            
            target = None
            min_dist = con['range']
            for n in self.nightmares:
                dist = math.hypot(n['pos'][0] - con['pos'][0], n['pos'][1] - con['pos'][1])
                if dist < min_dist:
                    min_dist = dist
                    target = n
            
            if target:
                con['fire_cooldown'] = con['max_cooldown']
                damage = con['damage']
                target['hp'] -= damage
                damage_reward += 0.1 * damage
                self._create_particles(con['pos'], con['color'], 1, 4, is_projectile=True, target=target['pos'])
        return damage_reward

    def _update_nightmares(self):
        if self.np_random.random() < self.nightmare_spawn_rate and self.has_built_construct:
            self._spawn_nightmare()
            
        for n in self.nightmares[:]:
            target_construct = None
            min_dist = float('inf')
            if not self.constructs:
                break
                
            for con in self.constructs:
                dist = math.hypot(n['pos'][0] - con['pos'][0], n['pos'][1] - con['pos'][1])
                if dist < min_dist:
                    min_dist = dist
                    target_construct = con
            
            if target_construct:
                if min_dist < 20:
                    target_construct['hp'] -= self.nightmare_base_damage
                    self._create_particles(target_construct['pos'], self.COLOR_NIGHTMARE[0], 10, 2)
                    if target_construct['hp'] <= 0:
                        self._create_particles(target_construct['pos'], (200,200,200), 30, 4)
                        for p in self.portals:
                            if p['construct_id'] == id(target_construct):
                                p['construct_id'] = None
                        self.constructs.remove(target_construct)
                else:
                    angle = math.atan2(target_construct['pos'][1] - n['pos'][1], target_construct['pos'][0] - n['pos'][0])
                    n['pos'] = (n['pos'][0] + math.cos(angle), n['pos'][1] + math.sin(angle))
        
        for n in self.nightmares[:]:
            if n['hp'] <= 0:
                self._create_particles(n['pos'], self.COLOR_NIGHTMARE[1], 15, 3)
                self.nightmares.remove(n)

    def _spawn_nightmare(self):
        side = self.np_random.integers(4)
        if side == 0: x, y = -10, self.np_random.uniform(0, self.SCREEN_HEIGHT)
        elif side == 1: x, y = self.SCREEN_WIDTH+10, self.np_random.uniform(0, self.SCREEN_HEIGHT)
        elif side == 2: x, y = self.np_random.uniform(0, self.SCREEN_WIDTH), -10
        else: x, y = self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT+10
        
        self.nightmares.append({
            'pos': (x, y), 'hp': self.nightmare_base_hp, 'max_hp': self.nightmare_base_hp,
            'color': random.choice(self.COLOR_NIGHTMARE)
        })

    def _update_particles(self):
        energy_bar_pos = (self.GRID_X_OFFSET + self.GRID_WIDTH + 15, self.SCREEN_HEIGHT - 50)
        for p in self.particles[:]:
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
                continue
            
            if p.get('is_energy'):
                target_pos = energy_bar_pos
                dist = math.hypot(target_pos[0] - p['pos'][0], target_pos[1] - p['pos'][1])
                if dist < 10:
                    self.energy = min(self.max_energy, self.energy + 0.2)
                    p['lifespan'] = 0
                else:
                    angle = math.atan2(target_pos[1] - p['pos'][1], target_pos[0] - p['pos'][0])
                    p['vel'] = (math.cos(angle) * 4, math.sin(angle) * 4)
            elif p.get('is_projectile'):
                target_pos = p['target']
                dist = math.hypot(target_pos[0] - p['pos'][0], target_pos[1] - p['pos'][1])
                if dist < 10:
                    p['lifespan'] = 0
                else:
                    angle = math.atan2(target_pos[1] - p['pos'][1], target_pos[0] - p['pos'][0])
                    p['vel'] = (math.cos(angle) * 8, math.sin(angle) * 8)
            
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])

    def _create_particles(self, pos, color, count, speed, is_energy=False, is_projectile=False, target=None):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = (math.cos(angle) * speed * self.np_random.uniform(0.5, 1.5),
                   math.sin(angle) * speed * self.np_random.uniform(0.5, 1.5))
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'lifespan': self.np_random.integers(20, 40),
                'color': color, 'is_energy': is_energy, 'is_projectile': is_projectile, 'target': target
            })

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.nightmare_spawn_rate = min(0.5, self.nightmare_spawn_rate + 0.01)
            self.nightmare_base_hp += 1
            self.nightmare_base_damage += 0.5
        
        if self.steps > 0 and self.steps % 400 == 0:
            if len(self.unlocked_constructs) < 3:
                self.unlocked_constructs.append(len(self.unlocked_constructs))

    def _check_softlock(self):
        if self._find_matches():
            self.no_match_steps = 0
            return
            
        temp_grid = self.grid.copy()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if c < self.GRID_COLS - 1:
                    temp_grid[r,c], temp_grid[r,c+1] = temp_grid[r,c+1], temp_grid[r,c]
                    if self._find_matches_on_grid(temp_grid):
                        self.no_match_steps = 0
                        return
                    temp_grid[r,c], temp_grid[r,c+1] = temp_grid[r,c+1], temp_grid[r,c]
                if r < self.GRID_ROWS - 1:
                    temp_grid[r,c], temp_grid[r+1,c] = temp_grid[r+1,c], temp_grid[r,c]
                    if self._find_matches_on_grid(temp_grid):
                        self.no_match_steps = 0
                        return
                    temp_grid[r,c], temp_grid[r+1,c] = temp_grid[r+1,c], temp_grid[r,c]
        
        self.no_match_steps += 1
        if self.no_match_steps >= 10:
            self.selected_tile = None
            self._create_grid()
            self.no_match_steps = 0

    def _find_matches_on_grid(self, grid):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if grid[r, c] == grid[r, c+1] == grid[r, c+2] != -1:
                    return True
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if grid[r, c] == grid[r+1, c] == grid[r+2, c] != -1:
                    return True
        return False

    def _check_termination(self):
        if self.has_built_construct and not self.constructs:
            return True
        return False
        
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "energy": self.energy}

    # --- Helper Methods: Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def render(self):
        return self._get_observation()

    def _render_game(self):
        self._render_grid_bg()
        self._render_portals()
        self._render_nightmares()
        self._render_constructs()
        self._render_particles()
        self._render_tiles()
        self._render_cursor()

    def _render_grid_bg(self):
        s = pygame.Surface((self.GRID_WIDTH, self.GRID_HEIGHT), pygame.SRCALPHA)
        s.fill(self.COLOR_GRID_BG)
        self.screen.blit(s, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET))

    def _render_tiles(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile_type = self.grid[r, c]
                if tile_type == -1: continue
                
                color = self.TILE_COLORS[tile_type]
                rect = pygame.Rect(
                    self.GRID_X_OFFSET + c * self.TILE_SIZE,
                    self.GRID_Y_OFFSET + r * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                pygame.draw.rect(self.screen, color, rect.inflate(-6, -6), border_radius=4)
                pygame.draw.rect(self.screen, tuple(min(255, x+50) for x in color), rect.inflate(-6, -6), 2, border_radius=4)

    def _render_cursor(self):
        r, c = self.cursor_pos
        rect = pygame.Rect(
            self.GRID_X_OFFSET + c * self.TILE_SIZE,
            self.GRID_Y_OFFSET + r * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        glow_color = (255, 255, 0)
        
        glow_size = int(10 + 4 * math.sin(self.steps * 0.2))
        for i in range(glow_size, 0, -2):
            alpha = 150 * (1 - i / glow_size)
            pygame.gfxdraw.rectangle(self.screen, rect.inflate(i, i), (*glow_color, alpha))
        
        pygame.draw.rect(self.screen, glow_color, rect, 2, border_radius=2)
        
        if self.selected_tile:
            sr, sc = self.selected_tile
            s_rect = pygame.Rect(
                self.GRID_X_OFFSET + sc * self.TILE_SIZE,
                self.GRID_Y_OFFSET + sr * self.TILE_SIZE,
                self.TILE_SIZE, self.TILE_SIZE
            )
            pygame.draw.rect(self.screen, (255, 255, 255), s_rect, 2, border_radius=2)

    def _render_portals(self):
        for portal in self.portals:
            x, y = int(portal['pos'][0]), int(portal['pos'][1])
            color = (100, 100, 200)
            pygame.gfxdraw.aacircle(self.screen, x, y, 25, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, 26, color)
            
            pulse_size = int(25 + 3 * math.sin(self.steps * 0.1))
            pygame.gfxdraw.aacircle(self.screen, x, y, pulse_size, (*color, 50))

    def _render_constructs(self):
        for con in self.constructs:
            x, y = int(con['pos'][0]), int(con['pos'][1])
            color = con['color']
            
            pulse = 1 + 0.1 * math.sin(self.steps * 0.15)
            size = int(15 * pulse)
            if con['type'] == 0:
                points = [(x, y - size), (x - size, y + size//2), (x + size, y + size//2)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            elif con['type'] == 1:
                rect = pygame.Rect(x - size//2, y - size//2, size, size)
                pygame.draw.rect(self.screen, color, rect)
            else:
                pygame.gfxdraw.aacircle(self.screen, x, y, size//2, color)
                pygame.gfxdraw.filled_circle(self.screen, x, y, size//2, color)

            bar_w, bar_h = 40, 5
            hp_perc = max(0, con['hp'] / con['max_hp'])
            pygame.draw.rect(self.screen, (50, 50, 50), (x - bar_w//2, y - 30, bar_w, bar_h))
            pygame.draw.rect(self.screen, (0, 255, 0), (x - bar_w//2, y - 30, int(bar_w * hp_perc), bar_h))

    def _render_nightmares(self):
        for n in self.nightmares:
            x, y = int(n['pos'][0]), int(n['pos'][1])
            size = int(8 + 2 * math.sin(self.steps * 0.3 + x))
            pygame.gfxdraw.aacircle(self.screen, x, y, size, n['color'])
            pygame.gfxdraw.filled_circle(self.screen, x, y, size, n['color'])

    def _render_particles(self):
        for p in self.particles:
            x, y = int(p['pos'][0]), int(p['pos'][1])
            alpha = int(255 * (p['lifespan'] / 40))
            color = (*p['color'], alpha)
            size = int(p['lifespan'] / 10)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, x, y, size, color)

    def _render_ui(self):
        steps_text = self.font_small.render(f"TIME: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        score_text = self.font_small.render(f"SCORE: {self.score:.0f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (10, 10))
        self.screen.blit(score_text, (10, 30))

        panel_x = self.GRID_X_OFFSET + self.GRID_WIDTH + 10
        panel_y = self.GRID_Y_OFFSET
        
        energy_text = self.font_small.render(f"ENERGY: {self.energy:.0f}", True, self.COLOR_ENERGY)
        self.screen.blit(energy_text, (panel_x, panel_y))
        bar_w, bar_h = 150, 20
        energy_perc = self.energy / self.max_energy
        pygame.draw.rect(self.screen, (50, 50, 30), (panel_x, panel_y + 20, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_ENERGY, (panel_x, panel_y + 20, int(bar_w * energy_perc), bar_h))
        pygame.draw.rect(self.screen, (255,255,255), (panel_x, panel_y + 20, bar_w, bar_h), 1)

        summon_text = self.font_small.render("SUMMON:", True, self.COLOR_UI_TEXT)
        self.screen.blit(summon_text, (panel_x, panel_y + 60))
        
        c_type = self.unlocked_constructs[self.selected_construct_idx]
        con_info = self._create_construct((0,0), c_type)
        cost = 15 + c_type * 10
        
        type_text = self.font_small.render(f"Type {c_type+1} (Cost: {cost})", True, con_info['color'])
        self.screen.blit(type_text, (panel_x, panel_y + 80))
        
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "VICTORY" if self.steps >= self.MAX_STEPS and not self._check_termination() else "DEFEAT"
            end_text = self.font_large.render(msg, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()
        
# Example usage:
if __name__ == "__main__":
    # The original code had a custom render mode "human_like" which isn't standard.
    # This block shows how to use the "rgb_array" mode to display the game.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup for human play
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Dreamfall Defenders")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting

        clock.tick(30) # Run at 30 FPS

    env.close()