import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:38:03.143229
# Source Brief: brief_01118.md
# Brief Index: 1118
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player builds a fractal fortress to defend a core
    against waves of enemies. The game features a build phase and a defense phase,
    with roguelike elements of unlocking new fractal patterns.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Build a fractal fortress to defend your core. Place turrets during the build phase "
        "and survive waves of incoming enemies in this tower defense game."
    )
    user_guide = (
        "Use arrow keys to move the build cursor. Press space to place a turret. "
        "Press shift to cycle between available turret types."
    )
    auto_advance = True

    # --- Constants ---
    # Colors (bright, high-contrast for clarity)
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_CORE = (0, 200, 255)
    COLOR_CORE_DMG = (255, 100, 100)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TOWER_CANNON = (0, 255, 150)
    COLOR_TOWER_LASER = (255, 0, 255)
    COLOR_PROJECTILE_CANNON = (200, 255, 220)
    COLOR_PROJECTILE_LASER = (255, 150, 255)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_VALUE = (255, 255, 255)
    COLOR_UI_RESOURCES = (50, 200, 50)

    # Game Parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 3000
    CORE_POS = np.array([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2])
    CORE_RADIUS = 15
    MAX_CORE_HEALTH = 100
    MAX_WAVES = 10

    # Turret Blueprints
    TURRET_STATS = [
        {
            "name": "Cannon", "cost": 100, "range": 80, "cooldown": 45,
            "color": COLOR_TOWER_CANNON, "proj_color": COLOR_PROJECTILE_CANNON, "proj_speed": 4, "proj_damage": 10
        },
        {
            "name": "Laser", "cost": 150, "range": 120, "cooldown": 90,
            "color": COLOR_TOWER_LASER, "proj_color": COLOR_PROJECTILE_LASER, "proj_speed": 8, "proj_damage": 25
        }
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_m = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- State variables initialized in reset() ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = 'BUILD' # 'BUILD' or 'DEFEND'
        self.core_health = 0
        self.resources = 0
        self.wave_number = 0
        self.build_phase_timer = 0
        
        self.grid_points = []
        self.grid_lines = []
        self.enemy_paths = {}
        self.spawn_points = []

        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.cursor_idx = 0
        self.selected_turret_type_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.cumulative_reward = 0.0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.cumulative_reward = 0.0
        self.game_over = False
        
        self.core_health = self.MAX_CORE_HEALTH
        self.resources = 250
        self.wave_number = 0
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.selected_turret_type_idx = 0
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True

        self._generate_fractal_grid(level=1)
        self.cursor_idx = 0

        self._start_build_phase()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        self.steps += 1

        # --- Action Processing ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # --- Game Logic Update ---
        if self.game_phase == 'BUILD':
            reward += self._update_build_phase(movement, space_pressed, shift_pressed)
        elif self.game_phase == 'DEFEND':
            reward += self._update_defense_phase()

        self._update_particles()
        
        # --- Termination and Phase Change ---
        terminated = self.core_health <= 0 or self.steps >= self.MAX_STEPS
        if self.core_health <= 0 and not self.game_over:
            reward -= 100.0  # Big penalty for losing
            self._create_explosion(self.CORE_POS, self.COLOR_CORE, 100)
            self.game_over = True
        
        if self.wave_number > self.MAX_WAVES and not self.game_over:
            reward += 10.0 # Big reward for winning level
            self.game_over = True
            terminated = True
            
        self.cumulative_reward += reward
        truncated = self.steps >= self.MAX_STEPS
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    # Game Logic Sub-routines
    #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#

    def _start_build_phase(self):
        self.game_phase = 'BUILD'
        self.wave_number += 1
        if self.wave_number > 1:
            self.resources += 100 + (self.wave_number - 1) * 20 # sfx: resource_gain
            self.score += 100 # Score for wave clear
        self.build_phase_timer = 450 # 15 seconds at 30fps

    def _start_defense_phase(self):
        self.game_phase = 'DEFEND'
        self._spawn_wave() # sfx: wave_start

    def _update_build_phase(self, movement, space_pressed, shift_pressed):
        if shift_pressed:
            self.selected_turret_type_idx = (self.selected_turret_type_idx + 1) % len(self.TURRET_STATS)
            # sfx: ui_cycle
        
        if movement != 0:
            self._move_cursor(movement)

        if space_pressed:
            self._place_turret()

        self.build_phase_timer -= 1
        if self.build_phase_timer <= 0:
            self._start_defense_phase()
        return 0.0

    def _update_defense_phase(self):
        reward = 0.0
        reward += self._update_towers()
        self._update_projectiles()
        reward += self._update_enemies()

        if not self.enemies and self.game_phase == 'DEFEND':
            reward += 1.0 # Wave complete reward
            self._start_build_phase()
        
        return reward

    def _move_cursor(self, direction):
        if not self.grid_points: return
        
        current_pos = self.grid_points[self.cursor_idx]
        best_target_idx = self.cursor_idx
        min_dist_sq = float('inf')

        # 1=up, 2=down, 3=left, 4=right
        for i, point in enumerate(self.grid_points):
            if i == self.cursor_idx: continue
            
            dx, dy = point[0] - current_pos[0], point[1] - current_pos[1]
            
            is_candidate = False
            if direction == 1 and dy < -1: is_candidate = True # Up
            elif direction == 2 and dy > 1: is_candidate = True # Down
            elif direction == 3 and dx < -1: is_candidate = True # Left
            elif direction == 4 and dx > 1: is_candidate = True # Right

            if is_candidate:
                dist_sq = dx*dx + dy*dy
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_target_idx = i
        
        self.cursor_idx = best_target_idx

    def _place_turret(self):
        turret_type = self.TURRET_STATS[self.selected_turret_type_idx]
        pos = self.grid_points[self.cursor_idx]

        # Check cost
        if self.resources < turret_type["cost"]:
            # sfx: error
            return

        # Check if location is occupied
        is_occupied = any(np.array_equal(t['pos'], pos) for t in self.towers)
        dist_to_core = np.linalg.norm(pos - self.CORE_POS)
        if is_occupied or dist_to_core < self.CORE_RADIUS + 10:
            # sfx: error
            return
        
        self.resources -= turret_type["cost"]
        self.towers.append({
            "pos": pos,
            "type_idx": self.selected_turret_type_idx,
            "cooldown": 0,
            "target": None
        })
        # sfx: place_turret

    def _spawn_wave(self):
        num_enemies = 3 + self.wave_number
        base_health = 20 + self.wave_number * 5
        base_speed = 0.5 + self.wave_number * 0.05
        
        for i in range(num_enemies):
            spawn_node = self.np_random.choice(self.spawn_points)
            path = self.enemy_paths[spawn_node]
            self.enemies.append({
                "pos": np.array(self.grid_points[spawn_node], dtype=float),
                "path": path,
                "path_idx": 0,
                "health": base_health + self.np_random.uniform(-5, 5),
                "max_health": base_health,
                "speed": base_speed + self.np_random.uniform(-0.1, 0.1),
                "id": self.steps + i
            })

    def _update_towers(self):
        reward = 0.0
        for tower in self.towers:
            stats = self.TURRET_STATS[tower['type_idx']]
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            # Find new target if needed
            if tower.get('target') is None or not any(e['id'] == tower['target']['id'] for e in self.enemies):
                tower['target'] = None
                closest_enemy = None
                min_dist = stats['range']
                for enemy in self.enemies:
                    dist = np.linalg.norm(enemy['pos'] - tower['pos'])
                    if dist < min_dist:
                        min_dist = dist
                        closest_enemy = enemy
                tower['target'] = closest_enemy

            # Fire if target in range
            if tower['target'] is not None and tower['cooldown'] <= 0:
                tower['cooldown'] = stats['cooldown']
                self.projectiles.append({
                    "pos": tower['pos'].copy(),
                    "target_pos": tower['target']['pos'].copy(),
                    "speed": stats['proj_speed'],
                    "damage": stats['proj_damage'],
                    "color": stats['proj_color']
                })
                # sfx: shoot_cannon or shoot_laser
        return reward

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            direction = proj['target_pos'] - proj['pos']
            dist = np.linalg.norm(direction)
            if dist < proj['speed']:
                proj['pos'] = proj['target_pos']
            else:
                proj['pos'] += (direction / dist) * proj['speed']
            
            # Check for collision
            hit_enemy = None
            for enemy in self.enemies:
                if np.linalg.norm(proj['pos'] - enemy['pos']) < 8:
                    hit_enemy = enemy
                    break
            
            if hit_enemy:
                hit_enemy['health'] -= proj['damage']
                self._create_explosion(proj['pos'], proj['color'], 5)
                self.projectiles.remove(proj)
                # sfx: hit_enemy

    def _update_enemies(self):
        reward = 0.0
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                self.score += 10
                reward += 0.1
                self._create_explosion(enemy['pos'], self.COLOR_ENEMY, 15)
                self.enemies.remove(enemy)
                # sfx: enemy_destroyed
                continue

            # Movement
            if enemy['path_idx'] < len(enemy['path']):
                target_node_idx = enemy['path'][enemy['path_idx']]
                target_pos = self.grid_points[target_node_idx]
                direction = target_pos - enemy['pos']
                dist = np.linalg.norm(direction)
                
                if dist < enemy['speed']:
                    enemy['pos'] = np.array(target_pos, dtype=float)
                    enemy['path_idx'] += 1
                else:
                    enemy['pos'] += (direction / dist) * enemy['speed']
            else: # Reached the core
                self.core_health -= 10
                reward -= 0.1 * 10 # -1 for reaching core
                self._create_explosion(enemy['pos'], self.COLOR_CORE_DMG, 20)
                self.enemies.remove(enemy)
                # sfx: core_damage
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 31),
                "color": color,
                "size": self.np_random.uniform(1, 3)
            })

    #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    # Fractal Grid Generation
    #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#

    def _generate_fractal_grid(self, level=1):
        # Using a simple L-system for fractal generation
        axiom = "F"
        rules = {"F": "F+F-F"}
        iterations = 3
        angle_deg = 60
        length = 40
        
        # Build the L-system string
        current_string = axiom
        for _ in range(iterations):
            next_string = ""
            for char in current_string:
                next_string += rules.get(char, char)
            current_string = next_string
        
        # Interpret the string to generate lines
        pos = self.CORE_POS.copy()
        angle = -90
        saved_states = []
        points = {tuple(pos.astype(int)): 0}
        lines = set()
        point_counter = 1

        # Center the fractal
        # This is a simplified pre-calculation of bounds
        min_x, max_x = self.CORE_POS[0], self.CORE_POS[0]
        min_y, max_y = self.CORE_POS[1], self.CORE_POS[1]
        temp_pos = np.array([0.0, 0.0])
        temp_angle = -90
        for char in current_string:
            if char == 'F':
                rad = math.radians(temp_angle)
                temp_pos += np.array([math.cos(rad), math.sin(rad)]) * length
                min_x, max_x = min(min_x, temp_pos[0]), max(max_x, temp_pos[0])
                min_y, max_y = min(min_y, temp_pos[1]), max(max_y, temp_pos[1])
            elif char == '+': temp_angle += angle_deg
            elif char == '-': temp_angle -= angle_deg
        
        offset = self.CORE_POS - np.array([(min_x+max_x)/2, (min_y+max_y)/2])
        pos = self.CORE_POS.copy() # Start drawing from core

        # Actual drawing pass
        for char in current_string:
            start_pos_tuple = tuple(pos.astype(int))
            if start_pos_tuple not in points:
                points[start_pos_tuple] = point_counter
                point_counter += 1

            if char == 'F':
                rad = math.radians(angle)
                end_pos = pos + np.array([math.cos(rad), math.sin(rad)]) * length
                end_pos_tuple = tuple(end_pos.astype(int))
                
                if end_pos_tuple not in points:
                    points[end_pos_tuple] = point_counter
                    point_counter += 1
                
                p1_idx, p2_idx = points[start_pos_tuple], points[end_pos_tuple]
                lines.add(tuple(sorted((p1_idx, p2_idx))))
                pos = end_pos
            elif char == '+': angle += angle_deg
            elif char == '-': angle -= angle_deg
            elif char == '[': saved_states.append((pos.copy(), angle))
            elif char == ']': pos, angle = saved_states.pop()

        # Finalize data structures
        idx_to_point = {v: k for k, v in points.items()}
        self.grid_points = [np.array(idx_to_point[i]) for i in range(len(idx_to_point))]
        self.grid_lines = [(self.grid_points[p1], self.grid_points[p2]) for p1, p2 in lines]
        
        # Pathfinding using BFS from the core (node 0)
        adj = {i: [] for i in range(len(self.grid_points))}
        for p1, p2 in lines:
            adj[p1].append(p2)
            adj[p2].append(p1)

        q = deque([0])
        parent = {0: None}
        visited = {0}
        max_dist, farthest_nodes = 0, []

        while q:
            u = q.popleft()
            dist_u = 0
            curr = u
            while parent[curr] is not None:
                dist_u += 1
                curr = parent[curr]
            
            if dist_u > max_dist:
                max_dist = dist_u
                farthest_nodes = [u]
            elif dist_u == max_dist:
                farthest_nodes.append(u)

            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    q.append(v)
        
        self.spawn_points = farthest_nodes
        self.enemy_paths = {}
        for i in range(len(self.grid_points)):
            if i in parent:
                path = []
                curr = i
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                self.enemy_paths[i] = path[::-1]


    #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    # Rendering
    #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Grid
        for p1, p2 in self.grid_lines:
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        
        # Core
        core_health_ratio = self.core_health / self.MAX_CORE_HEALTH
        core_color = self.COLOR_CORE
        if self.core_health < self.MAX_CORE_HEALTH:
            core_color = tuple(np.clip(
                np.array(self.COLOR_CORE) * core_health_ratio + np.array(self.COLOR_CORE_DMG) * (1-core_health_ratio), 0, 255
            ))
        pulse = 1 + math.sin(self.steps * 0.1) * 0.1
        self._draw_glowing_circle(self.CORE_POS, self.CORE_RADIUS * pulse, core_color)
        
        # Towers
        for tower in self.towers:
            stats = self.TURRET_STATS[tower['type_idx']]
            self._draw_glowing_circle(tower['pos'], 6, stats['color'])
            if tower['cooldown'] > 0:
                cooldown_ratio = tower['cooldown'] / stats['cooldown']
                pygame.draw.circle(self.screen, (0,0,0,150), tower['pos'], 4)
                pygame.draw.arc(self.screen, stats['color'], (*(tower['pos']-4), 8, 8), -math.pi/2, -math.pi/2 + 2*math.pi*cooldown_ratio, 2)
        
        # Enemies
        for enemy in self.enemies:
            self._draw_glowing_rect(enemy['pos'], 8, self.COLOR_ENEMY)
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_len = 12
            pygame.draw.rect(self.screen, (50,0,0), (enemy['pos'][0]-bar_len/2, enemy['pos'][1]-10, bar_len, 3))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (enemy['pos'][0]-bar_len/2, enemy['pos'][1]-10, bar_len*health_ratio, 3))

        # Projectiles
        for proj in self.projectiles:
            self._draw_glowing_circle(proj['pos'], 3, proj['color'])
        
        # Particles
        for p in self.particles:
            alpha = p['lifespan'] / 30.0
            color = (*p['color'], int(alpha * 255))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color)

        # Cursor in build phase
        if self.game_phase == 'BUILD' and self.grid_points:
            pos = self.grid_points[self.cursor_idx]
            self._draw_glowing_circle(pos, 8, self.COLOR_CURSOR, hollow=True)
            turret_type = self.TURRET_STATS[self.selected_turret_type_idx]
            if self.resources >= turret_type['cost']:
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 6, (*turret_type['color'], 100))
                self._draw_range_indicator(pos, turret_type['range'], (255,255,255,50))
            else:
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 6, (255,0,0,100))

    def _render_ui(self):
        # Top Bar: Wave and Build Timer
        wave_text = f"WAVE {self.wave_number}/{self.MAX_WAVES}"
        self._draw_text(wave_text, (self.SCREEN_WIDTH/2, 20), self.font_l, self.COLOR_UI_VALUE, center=True)

        if self.game_phase == 'BUILD':
            timer_sec = self.build_phase_timer / 30.0
            timer_text = f"BUILD PHASE: {timer_sec:.1f}s"
            self._draw_text(timer_text, (self.SCREEN_WIDTH/2, 45), self.font_m, self.COLOR_UI_TEXT, center=True)
        
        # Top-Left: Resources
        self._draw_text("RESOURCES", (10, 15), self.font_s, self.COLOR_UI_TEXT)
        self._draw_text(f"$ {self.resources}", (15, 30), self.font_m, self.COLOR_UI_RESOURCES)
        
        # Top-Right: Score
        self._draw_text("SCORE", (self.SCREEN_WIDTH - 10, 15), self.font_s, self.COLOR_UI_TEXT, align="right")
        self._draw_text(f"{self.score}", (self.SCREEN_WIDTH - 15, 30), self.font_m, self.COLOR_UI_VALUE, align="right")

        # Bottom-Left: Selected Turret Info
        if self.game_phase == 'BUILD':
            turret = self.TURRET_STATS[self.selected_turret_type_idx]
            self._draw_text(f"Selected: {turret['name']}", (10, self.SCREEN_HEIGHT - 60), self.font_s, self.COLOR_UI_TEXT)
            self._draw_text(f"Cost: ${turret['cost']}", (15, self.SCREEN_HEIGHT - 45), self.font_s, self.COLOR_UI_RESOURCES)
            self._draw_text(f"Range: {turret['range']}", (15, self.SCREEN_HEIGHT - 30), self.font_s, self.COLOR_UI_TEXT)
            self._draw_text(f"Damage: {turret['proj_damage']}", (15, self.SCREEN_HEIGHT - 15), self.font_s, self.COLOR_UI_TEXT)

        # Core Health Bar
        health_angle = (self.core_health / self.MAX_CORE_HEALTH) * 2 * math.pi
        if health_angle > 0:
            pygame.draw.arc(self.screen, self.COLOR_CORE, (*(self.CORE_POS - self.CORE_RADIUS - 5), (self.CORE_RADIUS+5)*2, (self.CORE_RADIUS+5)*2), -math.pi/2, -math.pi/2 + health_angle, 3)

        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "LEVEL COMPLETE" if self.core_health > 0 else "CORE DESTROYED"
            self._draw_text(msg, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20), self.font_l, self.COLOR_UI_VALUE, center=True)
            self._draw_text(f"Final Score: {self.score}", (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20), self.font_m, self.COLOR_UI_TEXT, center=True)


    def _draw_glowing_circle(self, pos, radius, color, hollow=False):
        pos_i = pos.astype(int)
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], int(radius * 1.5), (*color, 30))
        pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], int(radius * 1.2), (*color, 60))
        # Main circle
        if hollow:
            pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], int(radius), color)
            pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], int(radius-1), color)
        else:
            pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], int(radius), color)
            pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], int(radius), color)
    
    def _draw_glowing_rect(self, center_pos, size, color):
        pos_i = (center_pos - size/2).astype(int)
        rect = pygame.Rect(*pos_i, size, size)
        glow_rect = rect.inflate(size*1.0, size*1.0)
        # Pygame doesn't have a good glow for rects, so we fake it with a surface
        shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, (*color, 80), shape_surf.get_rect(), border_radius=3)
        self.screen.blit(shape_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, color, rect, border_radius=2)

    def _draw_range_indicator(self, pos, radius, color):
        pos_i = pos.astype(int)
        pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], int(radius), color)

    def _draw_text(self, text, pos, font, color, align="left", center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        elif align == "left":
            text_rect.topleft = pos
        elif align == "right":
            text_rect.topright = pos
        self.screen.blit(text_surface, text_rect)

    #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
    # Gymnasium Interface Compliance
    #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "core_health": self.core_health,
            "resources": self.resources,
            "wave": self.wave_number,
            "phase": self.game_phase,
            "towers": len(self.towers),
            "enemies": len(self.enemies)
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # Example usage:
    # This allows the game to be played by a human for testing and demonstration.
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Fractal Fortress Defender")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement, space_held, shift_held = 0, 0, 0
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Termination ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("-" * 20)
                print(f"Resetting game. Final score: {info['score']}, Total reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print("-" * 20)
            print(f"Episode finished. Final score: {info['score']}, Total reward: {total_reward:.2f}")
            print("Press 'R' to restart.")
            # Wait for reset
            game_ended = True
            while game_ended:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        game_ended = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        game_ended = False
        
        env.clock.tick(30) # Lock to 30 FPS

    env.close()