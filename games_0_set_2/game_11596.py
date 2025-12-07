import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:13:23.130761
# Source Brief: brief_01596.md
# Brief Index: 1596
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Match colorful runes to summon warriors and cast powerful spells. "
        "Defeat waves of enemies before they overwhelm you in this puzzle-combat hybrid."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move the cursor. "
        "Press space to select a rune, then move to an adjacent rune and press space again to swap."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 8, 8
    GRID_X, GRID_Y = 180, 40
    RUNE_SIZE = 40
    RUNE_GAP = 4

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_GRID_BG = (25, 20, 40)
    COLOR_UI_BG = (40, 30, 60)
    COLOR_UI_FG = (200, 180, 255)
    COLOR_TEXT = (220, 210, 255)
    COLOR_CURSOR = (255, 255, 100)
    COLOR_SELECTED_GLOW = (255, 255, 255, 90)

    RUNE_DEFS = {
        1: {'name': 'Fire', 'color': (255, 80, 50), 'symbol': 'triangle'},
        2: {'name': 'Water', 'color': (50, 150, 255), 'symbol': 'circle'},
        3: {'name': 'Earth', 'color': (80, 200, 80), 'symbol': 'square'},
        4: {'name': 'Light', 'color': (255, 255, 100), 'symbol': 'star'},
        5: {'name': 'Dark', 'color': (180, 80, 255), 'symbol': 'hexagon'},
    }
    
    MAX_STEPS = 5000
    
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
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = 0
        self.max_player_health = 100
        self.enemies = []
        self.base_enemy_health = 10
        self.rune_grid = []
        self.warriors = []
        self.particles = []
        self.floating_texts = []
        self.cursor_pos = [0, 0]
        self.selected_rune_pos = None
        self.swap_info = None
        self.fall_info = None
        self.prev_space_held = False
        self.enemy_attack_timer = 0
        
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.max_player_health
        self.base_enemy_health = 10
        
        self._initialize_grid()
        self._initialize_enemies()
        
        self.warriors = []
        self.particles = []
        self.floating_texts = []
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_rune_pos = None
        
        self.swap_info = None # {'rune1_pos', 'rune2_pos', 'progress', 'reverting'}
        self.fall_info = None # Tracks falling runes
        
        self.prev_space_held = False
        self.enemy_attack_timer = 120 # Ticks before first attack

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        # --- Handle Player Input ---
        # Only process input if no animations are running
        if not self.swap_info and not self.fall_info:
            self._handle_movement(movement)
            if space_pressed:
                self._handle_selection()

        # --- Update Game Logic ---
        self._update_animations()
        self._update_warriors()
        reward += self._update_enemies()
        self._update_particles()
        self._update_floating_texts()
        
        # If animations just finished, check for matches
        if not self.swap_info and not self.fall_info:
            match_reward = self._find_and_process_matches()
            reward += match_reward

        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.base_enemy_health += 1
        
        # --- Calculate Rewards ---
        # Continuous rewards are handled via events
        
        terminated = self._check_termination()
        if terminated:
            if self.player_health <= 0:
                reward -= 100
            elif not self.enemies:
                reward += 100
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Gymnasium standard is to set terminated=True when truncated
        
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Private Helper Methods: Game Logic ---

    def _initialize_grid(self):
        self.rune_grid = [[0] * self.GRID_COLS for _ in range(self.GRID_ROWS)]
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                possible_types = list(self.RUNE_DEFS.keys())
                # Avoid initial matches
                if c >= 2 and self.rune_grid[r][c-1]['type'] == self.rune_grid[r][c-2]['type']:
                    if self.rune_grid[r][c-1]['type'] in possible_types:
                        possible_types.remove(self.rune_grid[r][c-1]['type'])
                if r >= 2 and self.rune_grid[r-1][c]['type'] == self.rune_grid[r-2][c]['type']:
                    if self.rune_grid[r-1][c]['type'] in possible_types:
                        possible_types.remove(self.rune_grid[r-1][c]['type'])
                
                rune_type = self.np_random.choice(possible_types)
                x = self.GRID_X + c * (self.RUNE_SIZE + self.RUNE_GAP)
                y = self.GRID_Y + r * (self.RUNE_SIZE + self.RUNE_GAP)
                self.rune_grid[r][c] = {'type': rune_type, 'visual_pos': [x, y], 'id': self.np_random.integers(1, 1e9)}

    def _initialize_enemies(self):
        self.enemies = []
        num_enemies = 3
        for i in range(num_enemies):
            x = self.SCREEN_WIDTH - 100
            y = 100 + i * 100
            self.enemies.append({
                'pos': [x, y],
                'health': self.base_enemy_health,
                'max_health': self.base_enemy_health,
                'charge': self.np_random.integers(0, 60),
                'charge_rate': 1 + (self.steps / 2000), # Increases over time
                'hit_timer': 0
            })

    def _handle_movement(self, movement):
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1  # Up
        elif movement == 2 and self.cursor_pos[1] < self.GRID_ROWS - 1: self.cursor_pos[1] += 1  # Down
        elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1  # Left
        elif movement == 4 and self.cursor_pos[0] < self.GRID_COLS - 1: self.cursor_pos[0] += 1  # Right
    
    def _handle_selection(self):
        # sound: select_sfx
        c, r = self.cursor_pos
        if self.selected_rune_pos is None:
            self.selected_rune_pos = [c, r]
        else:
            sc, sr = self.selected_rune_pos
            dist = abs(c - sc) + abs(r - sr)
            if dist == 0: # Clicked same rune
                self.selected_rune_pos = None
            elif dist == 1: # Adjacent rune
                self._initiate_swap([sc, sr], [c, r])
            else: # Clicked non-adjacent rune
                self.selected_rune_pos = [c, r]

    def _initiate_swap(self, pos1, pos2, reverting=False):
        # sound: swap_sfx
        self.swap_info = {
            'pos1': pos1, 'pos2': pos2,
            'progress': 0, 'reverting': reverting
        }
        self.selected_rune_pos = None

    def _update_animations(self):
        # Swap animation
        if self.swap_info:
            self.swap_info['progress'] += 0.15
            if self.swap_info['progress'] >= 1.0:
                c1, r1 = self.swap_info['pos1']
                c2, r2 = self.swap_info['pos2']
                
                # Perform the actual swap in the grid
                self.rune_grid[r1][c1], self.rune_grid[r2][c2] = self.rune_grid[r2][c2], self.rune_grid[r1][c1]
                
                if not self.swap_info['reverting']:
                    matches = self._find_matches()
                    if not matches:
                        # No match, revert the swap
                        self._initiate_swap(self.swap_info['pos1'], self.swap_info['pos2'], reverting=True)
                    else:
                        self.swap_info = None # Swap successful, processing will happen next tick
                else:
                    self.swap_info = None # Revert finished
        
        # Fall animation
        if self.fall_info:
            all_settled = True
            for fall_item in self.fall_info:
                rune = fall_item['rune']
                target_y = fall_item['target_y']
                
                current_y = rune['visual_pos'][1]
                if abs(target_y - current_y) > 1:
                    rune['visual_pos'][1] += (target_y - current_y) * 0.2
                    all_settled = False
                else:
                    rune['visual_pos'][1] = target_y
            
            if all_settled:
                self.fall_info = None

    def _find_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if self.rune_grid[r][c]['type'] == self.rune_grid[r][c+1]['type'] == self.rune_grid[r][c+2]['type']:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        # Vertical
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if self.rune_grid[r][c]['type'] == self.rune_grid[r+1][c]['type'] == self.rune_grid[r+2][c]['type']:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _find_and_process_matches(self):
        matches = self._find_matches()
        if not matches:
            return 0
        
        # sound: match_success_sfx
        reward = 0
        
        # Group matches by type and identify unique runes
        match_counts = {}
        unique_runes = {}
        for r, c in matches:
            rune = self.rune_grid[r][c]
            rune_type = rune['type']
            match_counts[rune_type] = match_counts.get(rune_type, 0) + 1
            if rune['id'] not in unique_runes:
                unique_runes[rune['id']] = {'rune': rune, 'pos': (r, c)}
        
        # Spawn warriors and spells
        for rune_type, count in match_counts.items():
            reward += count * 0.1
            self.score += count * 10
            
            # Find a position to spawn warrior/spell effect
            avg_x, avg_y = 0, 0
            for r_idx, c_idx in [p for p in matches if self.rune_grid[p[0]][p[1]]['type'] == rune_type]:
                avg_x += self.GRID_X + c_idx * (self.RUNE_SIZE + self.RUNE_GAP)
                avg_y += self.GRID_Y + r_idx * (self.RUNE_SIZE + self.RUNE_GAP)
            avg_x /= count
            avg_y /= count

            self._create_floating_text(f"+{count * 10}", (avg_x, avg_y), (255, 255, 150))
            
            if count >= 5: # Spell
                # sound: spell_cast_sfx
                reward += 5
                self.score += 50
                self._create_spell_effect((avg_x, avg_y), rune_type)
            elif count >= 3: # Warrior
                # sound: warrior_spawn_sfx
                reward += 1
                self.score += 20
                self._spawn_warrior(rune_type, (avg_x, avg_y))

        # Remove matched runes from grid (set to 0)
        for r, c in matches:
            self.rune_grid[r][c] = 0

        # Make runes fall
        self.fall_info = []
        for c in range(self.GRID_COLS):
            empty_count = 0
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.rune_grid[r][c] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    rune = self.rune_grid[r][c]
                    self.rune_grid[r + empty_count][c] = rune
                    self.rune_grid[r][c] = 0
                    
                    target_y = self.GRID_Y + (r + empty_count) * (self.RUNE_SIZE + self.RUNE_GAP)
                    self.fall_info.append({'rune': rune, 'target_y': target_y})
        
        # Add new runes at the top
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS):
                if self.rune_grid[r][c] == 0:
                    rune_type = self.np_random.choice(list(self.RUNE_DEFS.keys()))
                    x = self.GRID_X + c * (self.RUNE_SIZE + self.RUNE_GAP)
                    y = self.GRID_Y + (r - self.GRID_ROWS) * (self.RUNE_SIZE + self.RUNE_GAP) # Start off-screen
                    target_y = self.GRID_Y + r * (self.RUNE_SIZE + self.RUNE_GAP)
                    
                    new_rune = {'type': rune_type, 'visual_pos': [x, y], 'id': self.np_random.integers(1, 1e9)}
                    self.rune_grid[r][c] = new_rune
                    
                    if not self.fall_info: self.fall_info = []
                    self.fall_info.append({'rune': new_rune, 'target_y': target_y})
        
        return reward
        
    def _spawn_warrior(self, rune_type, pos):
        if not self.enemies: return
        target_enemy = self.np_random.choice(self.enemies)
        self.warriors.append({
            'pos': list(pos),
            'type': rune_type,
            'target': target_enemy,
            'speed': self.np_random.uniform(2.5, 3.5),
            'trail': []
        })

    def _create_spell_effect(self, pos, rune_type):
        num_particles = 50
        color = self.RUNE_DEFS[rune_type]['color']
        # Damage all enemies
        for enemy in self.enemies:
            damage = 5
            enemy['health'] -= damage
            enemy['hit_timer'] = 10
        
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'size': self.np_random.uniform(8, 15),
                'lifespan': self.np_random.integers(20, 40), 'color': color, 'max_lifespan': 40
            })
        self._create_floating_text("SPELL!", pos, (255, 255, 255), size='large')
        
    def _update_warriors(self):
        surviving_warriors = []
        for w in self.warriors:
            if w['target'] not in self.enemies: # Target already destroyed
                if self.enemies:
                    w['target'] = self.np_random.choice(self.enemies)
                else:
                    continue # No more targets
            
            target_pos = w['target']['pos']
            direction = [target_pos[0] - w['pos'][0], target_pos[1] - w['pos'][1]]
            dist = math.hypot(*direction)
            
            if dist < 20: # Hit target
                # sound: warrior_hit_sfx
                damage = 1
                w['target']['health'] -= damage
                w['target']['hit_timer'] = 10
                # Create impact particles
                for _ in range(10):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 3)
                    vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                    self.particles.append({
                        'pos': list(w['pos']), 'vel': vel, 'size': self.np_random.uniform(2, 5),
                        'lifespan': self.np_random.integers(10, 20), 'color': self.RUNE_DEFS[w['type']]['color'], 'max_lifespan': 20
                    })
            else:
                # Move towards target
                w['trail'].append(list(w['pos']))
                if len(w['trail']) > 5: w['trail'].pop(0)
                
                norm_dir = [d / dist for d in direction]
                w['pos'][0] += norm_dir[0] * w['speed']
                w['pos'][1] += norm_dir[1] * w['speed']
                surviving_warriors.append(w)
        self.warriors = surviving_warriors

    def _update_enemies(self):
        reward = 0
        surviving_enemies = []
        for e in self.enemies:
            if e['health'] <= 0:
                # sound: enemy_destroyed_sfx
                reward += 10
                self.score += 100
                self._create_floating_text("+100", e['pos'], (255, 200, 50))
                # Death explosion
                for _ in range(40):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 4)
                    vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                    self.particles.append({
                        'pos': list(e['pos']), 'vel': vel, 'size': self.np_random.uniform(3, 8),
                        'lifespan': self.np_random.integers(20, 40), 'color': (200, 200, 200), 'max_lifespan': 40
                    })
                continue
            
            e['charge'] += e['charge_rate']
            e['charge_rate'] = 1 + (self.steps / 2000)
            if e['charge'] >= 180: # Attack
                # sound: enemy_attack_sfx
                e['charge'] = 0
                self.player_health -= 10
                reward -= 10.0 # penalty for taking damage
                self._create_floating_text("-10 HP", (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT - 30), (255, 50, 50))
            
            if e['hit_timer'] > 0: e['hit_timer'] -= 1
            surviving_enemies.append(e)
        self.enemies = surviving_enemies
        return reward

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['size'] *= 0.95
            if p['lifespan'] > 0 and p['size'] > 0.5:
                active_particles.append(p)
        self.particles = active_particles

    def _create_floating_text(self, text, pos, color, lifespan=45, size='small'):
        self.floating_texts.append({
            'text': text, 'pos': list(pos), 'color': color,
            'lifespan': lifespan, 'max_lifespan': lifespan, 'size': size
        })

    def _update_floating_texts(self):
        active_texts = []
        for ft in self.floating_texts:
            ft['pos'][1] -= 0.7
            ft['lifespan'] -= 1
            if ft['lifespan'] > 0:
                active_texts.append(ft)
        self.floating_texts = active_texts

    def _check_termination(self):
        return self.player_health <= 0 or not self.enemies

    # --- Private Helper Methods: Rendering ---

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_bg()
        self._render_runes()
        self._render_cursor()
        self._render_warriors()
        self._render_enemies()
        self._render_particles()
        self._render_ui()
        self._render_floating_texts()

    def _render_grid_bg(self):
        grid_width = self.GRID_COLS * (self.RUNE_SIZE + self.RUNE_GAP) - self.RUNE_GAP
        grid_height = self.GRID_ROWS * (self.RUNE_SIZE + self.RUNE_GAP) - self.RUNE_GAP
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, 
                         (self.GRID_X, self.GRID_Y, grid_width, grid_height), border_radius=5)

    def _render_runes(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rune = self.rune_grid[r][c]
                if not rune: continue
                
                x, y = rune['visual_pos']
                
                # Handle swap animation
                if self.swap_info:
                    c1, r1 = self.swap_info['pos1']
                    c2, r2 = self.swap_info['pos2']
                    p = self.swap_info['progress']
                    if r == r1 and c == c1:
                        x2 = self.GRID_X + c2 * (self.RUNE_SIZE + self.RUNE_GAP)
                        y2 = self.GRID_Y + r2 * (self.RUNE_SIZE + self.RUNE_GAP)
                        x = x + (x2 - x) * p
                        y = y + (y2 - y) * p
                    elif r == r2 and c == c2:
                        x1 = self.GRID_X + c1 * (self.RUNE_SIZE + self.RUNE_GAP)
                        y1 = self.GRID_Y + r1 * (self.RUNE_SIZE + self.RUNE_GAP)
                        x = x + (x1 - x) * p
                        y = y + (y1 - y) * p

                self._draw_rune(rune['type'], (x, y))

    def _draw_rune(self, rune_type, pos):
        x, y = int(pos[0]), int(pos[1])
        size = self.RUNE_SIZE
        half = size // 2
        
        defn = self.RUNE_DEFS[rune_type]
        color = defn['color']
        shadow_color = [max(0, c-50) for c in color]

        pygame.draw.rect(self.screen, shadow_color, (x, y + 2, size, size), border_radius=8)
        pygame.draw.rect(self.screen, color, (x, y, size, size), border_radius=8)
        
        symbol_color = (255, 255, 255)
        center_x, center_y = x + half, y + half
        
        if defn['symbol'] == 'triangle':
            points = [(center_x, y + 8), (x + 8, y + size - 8), (x + size - 8, y + size - 8)]
            pygame.gfxdraw.aapolygon(self.screen, points, symbol_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, symbol_color)
        elif defn['symbol'] == 'circle':
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, half - 8, symbol_color)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, half - 8, symbol_color)
        elif defn['symbol'] == 'square':
            pygame.draw.rect(self.screen, symbol_color, (x + 8, y + 8, size - 16, size - 16))
        elif defn['symbol'] == 'star':
            points = []
            for i in range(10):
                radius = half - 8 if i % 2 == 0 else half - 16
                angle = i * math.pi / 5 + math.pi / 2
                points.append((center_x + radius * math.cos(angle), center_y - radius * math.sin(angle)))
            pygame.gfxdraw.aapolygon(self.screen, points, symbol_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, symbol_color)
        elif defn['symbol'] == 'hexagon':
            points = []
            for i in range(6):
                angle = i * math.pi / 3
                points.append((center_x + (half - 8) * math.cos(angle), center_y + (half - 8) * math.sin(angle)))
            pygame.gfxdraw.aapolygon(self.screen, points, symbol_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, symbol_color)

    def _render_cursor(self):
        c, r = self.cursor_pos
        x = self.GRID_X + c * (self.RUNE_SIZE + self.RUNE_GAP)
        y = self.GRID_Y + r * (self.RUNE_SIZE + self.RUNE_GAP)
        
        # Cursor
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (x-3, y-3, self.RUNE_SIZE+6, self.RUNE_SIZE+6), 3, border_radius=10)
        
        # Selected rune glow
        if self.selected_rune_pos:
            sc, sr = self.selected_rune_pos
            sx = self.GRID_X + sc * (self.RUNE_SIZE + self.RUNE_GAP)
            sy = self.GRID_Y + sr * (self.RUNE_SIZE + self.RUNE_GAP)
            
            glow_surf = pygame.Surface((self.RUNE_SIZE + 10, self.RUNE_SIZE + 10), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, self.COLOR_SELECTED_GLOW, glow_surf.get_rect(), border_radius=12)
            self.screen.blit(glow_surf, (sx - 5, sy - 5))

    def _render_warriors(self):
        for w in self.warriors:
            color = self.RUNE_DEFS[w['type']]['color']
            x, y = int(w['pos'][0]), int(w['pos'][1])
            
            # Trail
            if len(w['trail']) > 1:
                for i in range(len(w['trail']) - 1):
                    alpha = int(200 * (i / len(w['trail'])))
                    trail_color = (*color, alpha)
                    start_pos = [int(p) for p in w['trail'][i]]
                    end_pos = [int(p) for p in w['trail'][i+1]]
                    
                    temp_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                    pygame.draw.line(temp_surf, trail_color, start_pos, end_pos, width=max(1, int(8 * (i/len(w['trail'])))))
                    self.screen.blit(temp_surf, (0,0))
            
            # Body
            pygame.gfxdraw.filled_circle(self.screen, x, y, 6, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, 6, (255,255,255))
    
    def _render_enemies(self):
        for e in self.enemies:
            x, y = int(e['pos'][0]), int(e['pos'][1])
            color = (180, 40, 40) if e['hit_timer'] == 0 else (255, 150, 150)
            
            # Body
            pygame.draw.rect(self.screen, color, (x-20, y-20, 40, 40))
            pygame.draw.rect(self.screen, (255,255,255), (x-20, y-20, 40, 40), 2)
            
            # Health bar
            health_pct = max(0, e['health'] / e['max_health'])
            pygame.draw.rect(self.screen, (50,0,0), (x-20, y-30, 40, 5))
            pygame.draw.rect(self.screen, (255,0,0), (x-20, y-30, 40 * health_pct, 5))
            
            # Charge bar
            charge_pct = min(1, e['charge'] / 180)
            pygame.draw.rect(self.screen, (0,50,50), (x-20, y+25, 40, 5))
            pygame.draw.rect(self.screen, (0,255,255), (x-20, y+25, 40 * charge_pct, 5))
    
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan'])) if 'max_lifespan' in p else 255
            color = (*p['color'], alpha)
            pos = [int(c) for c in p['pos']]
            size = int(p['size'])
            
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (pos[0]-size, pos[1]-size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Player Health Bar
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, self.SCREEN_HEIGHT - 30, 150, 20))
        health_pct = max(0, self.player_health / self.max_player_health)
        pygame.draw.rect(self.screen, (50, 200, 50), (10, self.SCREEN_HEIGHT - 30, 150 * health_pct, 20))
        hp_text = self.font_small.render(f"HP: {self.player_health}/{self.max_player_health}", True, self.COLOR_TEXT)
        self.screen.blit(hp_text, (15, self.SCREEN_HEIGHT - 28))

        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 40))

    def _render_floating_texts(self):
        for ft in self.floating_texts:
            font = self.font_large if ft['size'] == 'large' else self.font_small
            alpha = int(255 * (ft['lifespan'] / ft['max_lifespan']))
            color = (*ft['color'], alpha)
            
            text_surf = font.render(ft['text'], True, color)
            text_surf.set_alpha(alpha)
            pos = (ft['pos'][0] - text_surf.get_width() / 2, ft['pos'][1] - text_surf.get_height() / 2)
            self.screen.blit(text_surf, pos)

    # --- Gymnasium Interface Compliance ---

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Code ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    
    screen_width, screen_height = GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Rune Conqueror")
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0

    clock = pygame.time.Clock()
    
    while not terminated and not truncated:
        # Event handling for manual control
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
            # Key presses for manual control
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    if movement in [1,2,3,4]: movement = 0
                if event.key == pygame.K_SPACE: space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 0

        # For this turn-based input style, we only want to register one move at a time
        # The auto_advance=True handles the continuous updates
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Reset discrete movement action after it's been processed
        movement = 0

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for smooth play

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()