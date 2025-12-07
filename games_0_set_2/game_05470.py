
# Generated: 2025-08-28T05:09:02.709302
# Source Brief: brief_05470.md
# Brief Index: 5470

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import heapq
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor. Space to place a tower. Shift to cycle tower types."
    )

    game_description = (
        "Defend your base from waves of zombies in this grid-based tower defense game. "
        "Place towers strategically to survive 10 waves."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.CELL_SIZE = 40
        self.MAX_STEPS = 1500 # Increased from brief's 1000 to allow for a full 10-wave game
        self.WIN_WAVE = 10

        # --- Colors ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_PATH = (50, 55, 60)
        self.COLOR_BASE = (0, 255, 100)
        self.COLOR_SPAWN = (100, 0, 0)
        self.COLOR_OBSTACLE = (80, 85, 90)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_CURSOR_VALID = (0, 255, 0, 100)
        self.COLOR_CURSOR_INVALID = (255, 0, 0, 100)

        # --- Tower Specifications ---
        self.TOWER_SPECS = {
            0: {"name": "Cannon", "cost": 50, "range": 2.5, "damage": 10, "fire_rate": 30, "color": (0, 150, 255), "proj_speed": 8},
            1: {"name": "Missile", "cost": 120, "range": 4.0, "damage": 35, "fire_rate": 75, "color": (200, 0, 255), "proj_speed": 6}
        }
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 72)
        
        # --- State Variables ---
        self.grid = None
        self.path = None
        self.base_pos = None
        self.spawn_pos = None
        self.towers = []
        self.zombies = []
        self.projectiles = []
        self.particles = []
        self.wave_number = 0
        self.wave_cooldown = 0
        self.zombies_to_spawn_count = 0
        self.zombie_spawn_timer = 0
        self.base_health = 0
        self.resources = 0
        self.score = 0
        self.steps = 0
        self.cursor_pos = [0, 0]
        self.selected_tower_type_idx = 0
        self.available_tower_types = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.game_over = False
        self.game_won = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = 100
        self.resources = 150
        self.wave_number = 0
        self.wave_cooldown = 150  # 5s at 30fps
        
        self._generate_level()

        self.towers = []
        self.zombies = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_tower_type_idx = 0
        self.available_tower_types = [0]
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        while True:
            self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
            self.spawn_pos = (0, self.np_random.integers(1, self.GRID_ROWS - 1))
            self.base_pos = (self.GRID_COLS - 1, self.np_random.integers(1, self.GRID_ROWS - 1))

            num_obstacles = self.np_random.integers(self.GRID_COLS, self.GRID_COLS * 2)
            for _ in range(num_obstacles):
                x, y = self.np_random.integers(1, self.GRID_COLS - 1), self.np_random.integers(0, self.GRID_ROWS)
                if (x, y) != self.spawn_pos and (x, y) != self.base_pos:
                    self.grid[y, x] = 2  # Obstacle

            self.path = self._find_path(self.spawn_pos, self.base_pos)
            if self.path:
                break
        
        for x, y in self.path:
            self.grid[y, x] = 1
        self.grid[self.spawn_pos[1], self.spawn_pos[0]] = 4
        self.grid[self.base_pos[1], self.base_pos[0]] = 3

    def _find_path(self, start, end):
        open_set = [(0, start)]
        heapq.heapify(open_set)
        came_from = {}
        g_score = { (c, r): float('inf') for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS) }
        g_score[start] = 0
        f_score = { (c, r): float('inf') for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS) }
        f_score[start] = abs(start[0] - end[0]) + abs(start[1] - end[1])

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                total_path = []
                while current in came_from:
                    total_path.append(current)
                    current = came_from[current]
                total_path.append(start)
                return total_path[::-1]

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.GRID_COLS and 0 <= neighbor[1] < self.GRID_ROWS):
                    continue
                if self.grid[neighbor[1], neighbor[0]] == 2: # Obstacle
                    continue

                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def step(self, action):
        if self.game_over or self.game_won:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        reward = 0
        self.steps += 1
        
        # --- Action Phase ---
        if self.wave_cooldown > 0:
            self._handle_cursor_movement(movement)
            if shift_pressed: self._cycle_tower_type()
            if space_pressed: self._place_tower()

        # --- Update Phase ---
        if self.wave_cooldown > 0:
            self.wave_cooldown -= 1
            if self.wave_cooldown == 0:
                self._start_next_wave()
        else: # Wave is active
            kill_reward = self._update_towers()
            reward += kill_reward
            self._update_projectiles()
            base_damage = self._update_zombies()
            if base_damage > 0:
                self.base_health = max(0, self.base_health - base_damage)
                # Sound: base_damage.wav
            self._update_particles()
            
            if self.wave_number > 0 and not self.zombies and self.zombies_to_spawn_count == 0:
                reward += 1.0 # Wave survived reward
                if self.wave_number >= self.WIN_WAVE:
                    self.game_won = True
                    reward += 50.0
                else:
                    self.wave_cooldown = 240 # 8s cooldown
                    self.resources += 75 + self.wave_number * 5
                    if self.wave_number == 3 and 1 not in self.available_tower_types:
                        self.available_tower_types.append(1)

        # --- Termination Check ---
        terminated = self.base_health <= 0 or self.game_won or self.steps >= self.MAX_STEPS
        if self.base_health <= 0 and not self.game_over:
            reward -= 50.0
            self.game_over = True
            # Sound: game_over.wav
        
        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_cursor_movement(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

    def _cycle_tower_type(self):
        self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.available_tower_types)
        # Sound: ui_cycle.wav

    def _place_tower(self):
        x, y = self.cursor_pos
        spec_idx = self.available_tower_types[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[spec_idx]
        
        if self.grid[y, x] == 0 and self.resources >= spec['cost']:
            self.resources -= spec['cost']
            self.grid[y, x] = 5 # Mark as tower
            self.towers.append({
                "pos": (x, y), "spec_idx": spec_idx, "cooldown": 0, "target": None
            })
            # Sound: place_tower.wav

    def _start_next_wave(self):
        self.wave_number += 1
        self.zombies_to_spawn_count = 4 + self.wave_number
        self.zombie_spawn_timer = 0
        self.zombie_spawn_interval = 30 # 1 zombie per second
        # Sound: wave_start.wav

    def _update_zombies(self):
        # Spawn new zombies
        if self.zombies_to_spawn_count > 0:
            self.zombie_spawn_timer -= 1
            if self.zombie_spawn_timer <= 0:
                self.zombie_spawn_timer = self.zombie_spawn_interval
                self.zombies_to_spawn_count -= 1
                
                speed = 0.5 + 0.05 * math.floor((self.wave_number - 1) / 2)
                health = 20 + 10 * (self.wave_number - 1)
                
                self.zombies.append({
                    "pos": [self.spawn_pos[0] * self.CELL_SIZE + self.CELL_SIZE/2, self.spawn_pos[1] * self.CELL_SIZE + self.CELL_SIZE/2],
                    "path_idx": 0,
                    "health": health,
                    "max_health": health,
                    "speed": speed,
                    "id": self.np_random.random()
                })
        
        base_damage = 0
        for z in reversed(self.zombies):
            if z['path_idx'] < len(self.path) - 1:
                target_node = self.path[z['path_idx'] + 1]
                target_pos = [target_node[0] * self.CELL_SIZE + self.CELL_SIZE/2, target_node[1] * self.CELL_SIZE + self.CELL_SIZE/2]
                
                direction = [target_pos[0] - z['pos'][0], target_pos[1] - z['pos'][1]]
                dist = math.hypot(*direction)
                if dist < z['speed']:
                    z['pos'] = target_pos
                    z['path_idx'] += 1
                else:
                    norm_dir = [d / dist for d in direction]
                    z['pos'][0] += norm_dir[0] * z['speed']
                    z['pos'][1] += norm_dir[1] * z['speed']
            else: # Reached base
                base_damage += 10 # Each zombie deals 10 damage
                self.zombies.remove(z)
                self._create_particles(z['pos'], self.COLOR_BASE, 15)

        return base_damage

    def _update_towers(self):
        kill_reward = 0
        for t in self.towers:
            spec = self.TOWER_SPECS[t['spec_idx']]
            t['cooldown'] = max(0, t['cooldown'] - 1)
            
            # Retarget if needed
            if t['target']:
                if t['target'] not in self.zombies:
                    t['target'] = None
                else:
                    dist = math.hypot(t['pos'][0] - t['target']['pos'][0]/self.CELL_SIZE, t['pos'][1] - t['target']['pos'][1]/self.CELL_SIZE)
                    if dist > spec['range']:
                        t['target'] = None

            # Find new target if no valid target
            if not t['target']:
                in_range_zombies = []
                for z in self.zombies:
                    dist = math.hypot(t['pos'][0] - z['pos'][0]/self.CELL_SIZE, t['pos'][1] - z['pos'][1]/self.CELL_SIZE)
                    if dist <= spec['range']:
                        in_range_zombies.append((dist, z))
                if in_range_zombies:
                    t['target'] = min(in_range_zombies, key=lambda i: i[0])[1]

            # Fire if ready and has target
            if t['cooldown'] == 0 and t['target']:
                t['cooldown'] = spec['fire_rate']
                tower_center = ((t['pos'][0] + 0.5) * self.CELL_SIZE, (t['pos'][1] + 0.5) * self.CELL_SIZE)
                self.projectiles.append({
                    "pos": list(tower_center), "target": t['target'], "spec_idx": t['spec_idx']
                })
                self._create_particles(tower_center, spec['color'], 3, 2, 1) # Muzzle flash
                # Sound: tower_fire.wav
        
        # Process projectile hits
        for p in reversed(self.projectiles):
            spec = self.TOWER_SPECS[p['spec_idx']]
            if p['target'] in self.zombies:
                target_pos = p['target']['pos']
                direction = [target_pos[0] - p['pos'][0], target_pos[1] - p['pos'][1]]
                dist = math.hypot(*direction)
                if dist < spec['proj_speed']:
                    p['target']['health'] -= spec['damage']
                    self._create_particles(p['pos'], self.COLOR_WHITE, 10)
                    if p['target']['health'] <= 0:
                        kill_reward += 0.1
                        self.resources += 2 # Small resource gain per kill
                        self._create_particles(p['target']['pos'], self.COLOR_ZOMBIE, 20)
                        self.zombies.remove(p['target'])
                        # Sound: zombie_die.wav
                    self.projectiles.remove(p)
                    # Sound: projectile_hit.wav
                else:
                    norm_dir = [d / dist for d in direction]
                    p['pos'][0] += norm_dir[0] * spec['proj_speed']
                    p['pos'][1] += norm_dir[1] * spec['proj_speed']
            else: # Target is gone
                self.projectiles.remove(p)
        return kill_reward

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count, speed_scale=3, life_scale=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * speed_scale
            life = self.np_random.uniform(10, 20) * life_scale / 20
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": life,
                "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_path()
        self._render_towers()
        self._render_zombies()
        self._render_projectiles()
        self._render_particles()
        if self.wave_cooldown > 0 and not self.game_over and not self.game_won:
            self._render_cursor()
        self._render_ui()
        if self.game_over or self.game_won:
            self._render_end_screen()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_and_path(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = (c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                cell_type = self.grid[r, c]
                color = self.COLOR_GRID
                if cell_type == 1: color = self.COLOR_PATH
                elif cell_type == 2: color = self.COLOR_OBSTACLE
                elif cell_type == 3: color = self.COLOR_BASE
                elif cell_type == 4: color = self.COLOR_SPAWN
                pygame.draw.rect(self.screen, color, rect)
        for r in range(self.GRID_ROWS + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, r * self.CELL_SIZE), (self.WIDTH, r * self.CELL_SIZE))
        for c in range(self.GRID_COLS + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (c * self.CELL_SIZE, 0), (c * self.CELL_SIZE, self.HEIGHT))

    def _render_towers(self):
        for t in self.towers:
            spec = self.TOWER_SPECS[t['spec_idx']]
            center_x = int((t['pos'][0] + 0.5) * self.CELL_SIZE)
            center_y = int((t['pos'][1] + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.4)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, spec['color'])
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_BG)

    def _render_zombies(self):
        for z in self.zombies:
            size = int(self.CELL_SIZE * 0.5)
            rect = pygame.Rect(z['pos'][0] - size/2, z['pos'][1] - size/2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, rect, border_radius=3)
            # Health bar
            hb_width = size
            hb_height = 4
            hb_y = rect.top - hb_height - 2
            health_pct = z['health'] / z['max_health']
            pygame.draw.rect(self.screen, (80,0,0), (rect.left, hb_y, hb_width, hb_height))
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, (rect.left, hb_y, int(hb_width * health_pct), hb_height))

    def _render_projectiles(self):
        for p in self.projectiles:
            spec = self.TOWER_SPECS[p['spec_idx']]
            if spec['name'] == "Cannon":
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 4, self.COLOR_WHITE)
            else: # Missile
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 5, spec['color'])
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 5, self.COLOR_WHITE)

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(p['life'] * 0.2))
            pygame.draw.rect(self.screen, p['color'], (p['pos'][0]-size/2, p['pos'][1]-size/2, size, size))

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        spec_idx = self.available_tower_types[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[spec_idx]
        
        # Cursor Box
        is_valid = self.grid[cy, cx] == 0 and self.resources >= spec['cost']
        cursor_color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill(cursor_color)
        self.screen.blit(s, (cx * self.CELL_SIZE, cy * self.CELL_SIZE))
        pygame.draw.rect(self.screen, self.COLOR_WHITE, (cx * self.CELL_SIZE, cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE), 1)

        # Range indicator
        range_px = spec['range'] * self.CELL_SIZE
        center_px = int((cx + 0.5) * self.CELL_SIZE), int((cy + 0.5) * self.CELL_SIZE)
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, center_px[0], center_px[1], int(range_px), (255, 255, 255, 30))
        pygame.gfxdraw.aacircle(s, center_px[0], center_px[1], int(range_px), (255, 255, 255, 80))
        self.screen.blit(s, (0,0))
    
    def _render_ui(self):
        # Top Bar
        bar_surf = pygame.Surface((self.WIDTH, 30), pygame.SRCALPHA)
        bar_surf.fill((0,0,0,128))
        self.screen.blit(bar_surf, (0,0))

        # Wave
        wave_text = f"Wave: {self.wave_number}/{self.WIN_WAVE}"
        if self.wave_cooldown > 0 and self.wave_number < self.WIN_WAVE:
             wave_text += f" (Next in {math.ceil(self.wave_cooldown/30)}s)"
        text_surf = self.font_small.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 5))

        # Base Health
        health_text = f"Base: {self.base_health}%"
        text_surf = self.font_small.render(health_text, True, self.COLOR_BASE)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 5))

        # Bottom Bar
        bar_surf = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        bar_surf.fill((0,0,0,128))
        self.screen.blit(bar_surf, (0, self.HEIGHT - 40))

        # Resources
        res_text = f"Resources: ${self.resources}"
        text_surf = self.font_medium.render(res_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, self.HEIGHT - 32))

        # Selected Tower
        if self.wave_cooldown > 0:
            spec_idx = self.available_tower_types[self.selected_tower_type_idx]
            spec = self.TOWER_SPECS[spec_idx]
            tower_text = f"Selected: {spec['name']} (Cost: ${spec['cost']})"
            cost_color = self.COLOR_TEXT if self.resources >= spec['cost'] else self.COLOR_ZOMBIE
            text_surf = self.font_medium.render(tower_text, True, cost_color)
            self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, self.HEIGHT - 32))

    def _render_end_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        msg = "YOU WIN!" if self.game_won else "GAME OVER"
        color = self.COLOR_BASE if self.game_won else self.COLOR_ZOMBIE
        text_surf = self.font_large.render(msg, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
        
        score_surf = self.font_medium.render(f"Final Score: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 30))

        self.screen.blit(overlay, (0,0))
        self.screen.blit(text_surf, text_rect)
        self.screen.blit(score_surf, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
            "zombies": len(self.zombies),
            "towers": len(self.towers)
        }
    
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op
    
    print(env.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Keyboard to Action Mapping ---
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Space
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()