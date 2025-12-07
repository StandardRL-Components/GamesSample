import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:47:23.793201
# Source Brief: brief_00661.md
# Brief Index: 661
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Prism Guardian: A tower defense game where you clone prism guardians to defend a crystalline network.
    The goal is to survive 20 waves of geometric invaders aiming to destroy the network core.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = "Defend a crystalline network core from waves of geometric invaders by placing prism guardians on a grid."
    user_guide = "Use arrow keys to move the selector, space to place a selected prism, and shift to cycle through available prism types."
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.render_mode = render_mode
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 64)

        # --- Game Constants ---
        self.MAX_STEPS = 3000
        self.MAX_WAVES = 20
        self.GRID_COLS, self.GRID_ROWS = 7, 5
        self.NODE_RADIUS = 12
        self.CORE_RADIUS = 20
        self.INITIAL_RESOURCES = 150
        self.CORE_NODE_IDX = -1 # Set in _setup_network

        # --- Color Palette ---
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_GRID = (50, 70, 120)
        self.COLOR_NODE = (100, 150, 255)
        self.COLOR_NODE_GLOW = (150, 200, 255, 50)
        self.COLOR_CORE = (0, 255, 255)
        self.COLOR_CORE_GLOW = (100, 255, 255, 80)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_GREEN = (0, 255, 100)
        self.COLOR_RED = (255, 50, 50)
        self.COLOR_YELLOW = (255, 255, 0)

        # --- Entity Configurations ---
        self.PRISM_TYPES = [
            {'name': 'Pulse Cannon', 'cost': 50, 'range': 100, 'damage': 10, 'fire_rate': 45, 'proj_speed': 5, 'color': (0, 150, 255), 'unlock_wave': 1, 'shape': 'triangle'},
            {'name': 'Beam Laser', 'cost': 80, 'range': 150, 'damage': 5, 'fire_rate': 15, 'proj_speed': 20, 'color': (255, 0, 150), 'unlock_wave': 5, 'shape': 'diamond'},
            {'name': 'Shockwave', 'cost': 120, 'range': 80, 'damage': 25, 'fire_rate': 90, 'proj_speed': 4, 'color': (255, 150, 0), 'unlock_wave': 10, 'aoe_radius': 40, 'shape': 'hexagon'},
        ]
        self.ENEMY_TYPES = [
            {'name': 'Scout', 'health': 50, 'speed': 1.0, 'damage': 5, 'color': (255, 50, 50), 'value': 10, 'shape': 'square'},
            {'name': 'Tank', 'health': 150, 'speed': 0.6, 'damage': 10, 'color': (200, 50, 100), 'value': 20, 'shape': 'pentagon'},
            {'name': 'Swarm', 'health': 30, 'speed': 1.5, 'damage': 3, 'color': (255, 100, 100), 'value': 5, 'shape': 'small_square'},
        ]

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.resources = 0
        self.core_health = 100.0
        self.network_nodes = []
        self.prisms = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.selector_pos = [0, 0]
        self.selected_prism_type_idx = 0
        self.unlocked_prism_indices = []
        self.wave_in_progress = False
        self.wave_complete_timer = 0
        self.next_enemy_spawn_timer = 0
        self.wave_spawn_list = []
        self.last_space_held = False
        self.last_shift_held = False
        self.next_enemy_id = 0
        self.reward_this_step = 0
        self.wave_transition_text = None
        self.wave_transition_alpha = 0
        
        # self.reset() # This is called by the harness
        # self.validate_implementation() # This is a developer tool, not for production
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.resources = self.INITIAL_RESOURCES
        self.core_health = 100.0
        
        self._setup_network()
        
        self.prisms = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.selector_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_prism_type_idx = 0
        self.unlocked_prism_indices = [i for i, p in enumerate(self.PRISM_TYPES) if p['unlock_wave'] <= 1]
        
        self.wave_in_progress = False
        self.wave_complete_timer = 120 # Initial delay before first wave
        self.next_enemy_spawn_timer = 0
        self.wave_spawn_list = []
        self.next_enemy_id = 0
        self.reward_this_step = 0

        self.last_space_held = False
        self.last_shift_held = False
        
        self._start_next_wave()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        
        self._handle_input(movement, space_pressed, shift_pressed)
        self._update_game_logic()

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        self.steps += 1
        
        terminated = self._check_termination()
        reward = self.reward_this_step
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _setup_network(self):
        self.network_nodes = []
        w_pad = self.width * 0.15
        h_pad = self.height * 0.15
        x_spacing = (self.width - 2 * w_pad) / (self.GRID_COLS - 1)
        y_spacing = (self.height - 2 * h_pad) / (self.GRID_ROWS - 1)
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                is_core = (c == self.GRID_COLS // 2 and r == self.GRID_ROWS // 2)
                node = {
                    'pos': (w_pad + c * x_spacing, h_pad + r * y_spacing),
                    'coords': (c, r),
                    'health': 100,
                    'is_core': is_core,
                    'prism': None,
                    'damage_flash': 0
                }
                self.network_nodes.append(node)
                if is_core:
                    self.CORE_NODE_IDX = len(self.network_nodes) - 1

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Selector movement
        if movement != 0:
            c, r = self.selector_pos
            if movement == 1: r = max(0, r - 1)
            elif movement == 2: r = min(self.GRID_ROWS - 1, r + 1)
            elif movement == 3: c = max(0, c - 1)
            elif movement == 4: c = min(self.GRID_COLS - 1, c + 1)
            self.selector_pos = [c, r]
            
        # Place prism
        if space_pressed:
            self._place_prism()
            
        # Cycle prism type
        if shift_pressed and self.unlocked_prism_indices:
            self.selected_prism_type_idx = (self.selected_prism_type_idx + 1) % len(self.unlocked_prism_indices)

    def _place_prism(self):
        node_idx = self.selector_pos[1] * self.GRID_COLS + self.selector_pos[0]
        node = self.network_nodes[node_idx]
        
        if node['prism'] is None and not node['is_core']:
            prism_config = self.PRISM_TYPES[self.unlocked_prism_indices[self.selected_prism_type_idx]]
            if self.resources >= prism_config['cost']:
                self.resources -= prism_config['cost']
                prism = {
                    'type_idx': self.unlocked_prism_indices[self.selected_prism_type_idx],
                    'node_idx': node_idx,
                    'pos': node['pos'],
                    'rotation': random.uniform(0, 360),
                    'cooldown': 0,
                }
                self.prisms.append(prism)
                node['prism'] = prism
                # sfx: prism placed

    def _update_game_logic(self):
        self._update_wave_manager()
        self._update_prisms()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        
        for node in self.network_nodes:
            if node['damage_flash'] > 0:
                node['damage_flash'] -= 1

        if self.wave_transition_alpha > 0:
            self.wave_transition_alpha -= 3

    def _update_wave_manager(self):
        if not self.wave_in_progress:
            if not self.enemies:
                self.wave_complete_timer -= 1
                if self.wave_complete_timer <= 0:
                    self._start_next_wave()
        else: # Wave is in progress
            if not self.enemies and not self.wave_spawn_list:
                self.wave_in_progress = False
                self.wave_complete_timer = 180 # Time until next wave
                self.reward_this_step += 1.0 # Wave complete bonus
                # sfx: wave complete
    
    def _start_next_wave(self):
        if self.wave_number >= self.MAX_WAVES:
            return
            
        self.wave_number += 1
        self.wave_in_progress = True
        
        # Unlock new prisms
        self.unlocked_prism_indices = [i for i, p in enumerate(self.PRISM_TYPES) if p['unlock_wave'] <= self.wave_number]

        # Generate enemies for the wave
        num_enemies = 3 + self.wave_number * 2
        
        available_enemy_types = []
        if self.wave_number < 5: available_enemy_types = [0]
        elif self.wave_number < 10: available_enemy_types = [0, 1]
        else: available_enemy_types = [0, 1, 2]
        
        self.wave_spawn_list = [random.choice(available_enemy_types) for _ in range(num_enemies)]
        self.next_enemy_spawn_timer = 0

        # Wave transition text
        self.wave_transition_text = f"WAVE {self.wave_number}"
        self.wave_transition_alpha = 255

    def _spawn_enemy(self, type_idx):
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top': pos = (random.uniform(0, self.width), -20)
        elif edge == 'bottom': pos = (random.uniform(0, self.width), self.height + 20)
        elif edge == 'left': pos = (-20, random.uniform(0, self.height))
        elif edge == 'right': pos = (self.width + 20, random.uniform(0, self.height))

        config = self.ENEMY_TYPES[type_idx]
        health_multiplier = 1 + (self.wave_number - 1) * 0.05
        speed_multiplier = 1 + (self.wave_number - 1) * 0.05

        enemy = {
            'id': self.next_enemy_id,
            'type_idx': type_idx,
            'pos': np.array(pos, dtype=float),
            'max_health': config['health'] * health_multiplier,
            'health': config['health'] * health_multiplier,
            'speed': config['speed'] * speed_multiplier,
            'damage': config['damage'],
            'value': config['value'],
            'rotation': 0
        }
        self.enemies.append(enemy)
        self.next_enemy_id += 1

    def _update_prisms(self):
        for prism in self.prisms:
            prism['rotation'] = (prism['rotation'] + 0.5) % 360
            if prism['cooldown'] > 0:
                prism['cooldown'] -= 1
                continue

            config = self.PRISM_TYPES[prism['type_idx']]
            target = None
            min_dist = config['range'] ** 2

            for enemy in self.enemies:
                dist_sq = (enemy['pos'][0] - prism['pos'][0])**2 + (enemy['pos'][1] - prism['pos'][1])**2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = enemy
            
            if target:
                prism['cooldown'] = config['fire_rate']
                proj = {
                    'pos': np.array(prism['pos'], dtype=float),
                    'target_id': target['id'],
                    'speed': config['proj_speed'],
                    'damage': config['damage'],
                    'color': config['color'],
                    'is_aoe': 'aoe_radius' in config,
                    'aoe_radius': config.get('aoe_radius', 0)
                }
                self.projectiles.append(proj)
                # sfx: prism fire

    def _update_enemies(self):
        if self.wave_in_progress and self.wave_spawn_list:
            self.next_enemy_spawn_timer -= 1
            if self.next_enemy_spawn_timer <= 0:
                self._spawn_enemy(self.wave_spawn_list.pop(0))
                self.next_enemy_spawn_timer = max(15, 60 - self.wave_number * 2)

        core_pos = self.network_nodes[self.CORE_NODE_IDX]['pos']
        for enemy in self.enemies[:]:
            direction = np.array(core_pos) - enemy['pos']
            dist = np.linalg.norm(direction)
            
            if dist < 10: # Reached core
                self.core_health -= enemy['damage']
                self.network_nodes[self.CORE_NODE_IDX]['damage_flash'] = 15
                self.reward_this_step -= 0.5 # Heavier penalty for core hit
                self.enemies.remove(enemy)
                self._create_particles(enemy['pos'], enemy['damage'], self.COLOR_CORE)
                # sfx: core damage
                continue

            direction /= dist
            enemy['pos'] += direction * enemy['speed']
            enemy['rotation'] = (enemy['rotation'] + enemy['speed']) % 360

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            target_enemy = next((e for e in self.enemies if e['id'] == proj['target_id']), None)
            
            if not target_enemy:
                self.projectiles.remove(proj)
                continue

            direction = target_enemy['pos'] - proj['pos']
            dist = np.linalg.norm(direction)

            if dist < 10: # Hit target
                if proj['is_aoe']:
                    self._create_particles(proj['pos'], 20, proj['color'], radius=proj['aoe_radius'])
                    for enemy in self.enemies[:]:
                        if np.linalg.norm(enemy['pos'] - proj['pos']) < proj['aoe_radius']:
                            enemy['health'] -= proj['damage']
                else:
                    target_enemy['health'] -= proj['damage']
                    self._create_particles(target_enemy['pos'], 10, proj['color'])
                
                # sfx: enemy hit
                self.projectiles.remove(proj)
            else:
                direction /= dist
                proj['pos'] += direction * proj['speed']
        
        # Check for dead enemies after all projectiles have resolved
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                self.resources += enemy['value']
                self.reward_this_step += 0.1
                self._create_particles(enemy['pos'], 30, enemy['color'])
                self.enemies.remove(enemy)
                # sfx: enemy destroyed

    def _create_particles(self, pos, count, color, radius=10):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': np.array(pos, dtype=float) + np.random.randn(2) * radius * 0.1,
                'vel': np.array(vel, dtype=float),
                'lifetime': random.randint(15, 30),
                'color': color,
                'size': random.uniform(1, 3)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Drag
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.core_health <= 0:
            self.game_over = True
            self.reward_this_step -= 100
            # sfx: game over
            return True
        if self.wave_number > self.MAX_WAVES and not self.enemies:
            self.game_over = True
            self.reward_this_step += 100
            # sfx: victory
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render network connections
        for i, n1 in enumerate(self.network_nodes):
            for j, n2 in enumerate(self.network_nodes):
                if i >= j: continue
                dist_sq = (n1['pos'][0] - n2['pos'][0])**2 + (n1['pos'][1] - n2['pos'][1])**2
                if dist_sq < 150**2: # Arbitrary connection distance
                    pygame.draw.aaline(self.screen, self.COLOR_GRID, n1['pos'], n2['pos'])
        
        # Render nodes
        for node in self.network_nodes:
            radius = self.CORE_RADIUS if node['is_core'] else self.NODE_RADIUS
            color = self.COLOR_CORE if node['is_core'] else self.COLOR_NODE
            glow_color = self.COLOR_CORE_GLOW if node['is_core'] else self.COLOR_NODE_GLOW
            
            if node['damage_flash'] > 0:
                flash_color = self.COLOR_RED
                pygame.gfxdraw.filled_circle(self.screen, int(node['pos'][0]), int(node['pos'][1]), radius, flash_color)
            else:
                self._draw_glowing_circle(int(node['pos'][0]), int(node['pos'][1]), radius, color, glow_color)

        # Render selector
        sel_node_pos = self.network_nodes[self.selector_pos[1] * self.GRID_COLS + self.selector_pos[0]]['pos']
        self._draw_selector(sel_node_pos)
        
        # Render projectiles
        for proj in self.projectiles:
            target_enemy = next((e for e in self.enemies if e['id'] == proj['target_id']), None)
            direction = np.array([1.0, 0.0])  # Default direction
            if target_enemy:
                dir_vector = target_enemy['pos'] - proj['pos']
                norm = np.linalg.norm(dir_vector)
                if norm > 1e-6:
                    direction = dir_vector / norm
            
            # Draw a "leading line" to show velocity
            velocity_vector = direction * proj['speed']
            end_point = proj['pos'] + velocity_vector * 2.0
            
            pygame.draw.aaline(self.screen, proj['color'], proj['pos'], end_point, True)

        # Render prisms
        for prism in self.prisms:
            config = self.PRISM_TYPES[prism['type_idx']]
            self._draw_geometric_shape(prism['pos'], 12, config['shape'], prism['rotation'], config['color'])
        
        # Render enemies
        for enemy in self.enemies:
            config = self.ENEMY_TYPES[enemy['type_idx']]
            self._draw_geometric_shape(enemy['pos'], 10, config['shape'], enemy['rotation'], config['color'])
            # Health bar
            if enemy['health'] < enemy['max_health']:
                bar_w = 20
                bar_h = 3
                fill_w = int((enemy['health'] / enemy['max_health']) * bar_w)
                pygame.draw.rect(self.screen, self.COLOR_RED, (enemy['pos'][0] - bar_w/2, enemy['pos'][1] - 20, bar_w, bar_h))
                pygame.draw.rect(self.screen, self.COLOR_GREEN, (enemy['pos'][0] - bar_w/2, enemy['pos'][1] - 20, fill_w, bar_h))
        
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 30))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

    def _render_ui(self):
        # Top-left: Wave
        wave_text = self.font_m.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        # Top-right: Core Health
        health_str = f"CORE HEALTH: {max(0, int(self.core_health))}%"
        health_color = self.COLOR_GREEN if self.core_health > 50 else (self.COLOR_YELLOW if self.core_health > 20 else self.COLOR_RED)
        health_text = self.font_m.render(health_str, True, health_color)
        self.screen.blit(health_text, (self.width - health_text.get_width() - 10, 10))
        
        # Bottom-center: Resources
        res_text = self.font_m.render(f"RESOURCES: {self.resources}", True, self.COLOR_YELLOW)
        self.screen.blit(res_text, (self.width/2 - res_text.get_width()/2, self.height - 40))
        
        # Bottom-right: Selected Prism
        if self.unlocked_prism_indices:
            prism_config = self.PRISM_TYPES[self.unlocked_prism_indices[self.selected_prism_type_idx]]
            sel_text = self.font_s.render(f"Selected: {prism_config['name']}", True, self.COLOR_TEXT)
            cost_text = self.font_s.render(f"Cost: {prism_config['cost']}", True, self.COLOR_YELLOW)
            self.screen.blit(sel_text, (self.width - sel_text.get_width() - 10, self.height - 50))
            self.screen.blit(cost_text, (self.width - cost_text.get_width() - 10, self.height - 30))
        
        # Wave transition text
        if self.wave_transition_alpha > 0 and self.wave_transition_text:
            wave_surf = self.font_l.render(self.wave_transition_text, True, self.COLOR_TEXT)
            wave_surf.set_alpha(self.wave_transition_alpha)
            self.screen.blit(wave_surf, (self.width/2 - wave_surf.get_width()/2, self.height/2 - wave_surf.get_height()/2))

    def _draw_glowing_circle(self, x, y, r, color, glow_color):
        pygame.gfxdraw.filled_circle(self.screen, x, y, r + 4, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, r, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, r, color)

    def _draw_selector(self, pos):
        x, y = int(pos[0]), int(pos[1])
        r = self.NODE_RADIUS + 6
        points = []
        for i in range(8):
            angle = (i / 8) * 2 * math.pi + (self.steps * 0.05)
            points.append((x + r * math.cos(angle), y + r * math.sin(angle)))
        pygame.draw.aalines(self.screen, self.COLOR_SELECTOR, True, points)

    def _draw_geometric_shape(self, pos, size, shape_type, angle_deg, color):
        angle_rad = math.radians(angle_deg)
        if shape_type == 'triangle': num_sides = 3
        elif shape_type == 'square': num_sides = 4
        elif shape_type == 'pentagon': num_sides = 5
        elif shape_type == 'hexagon': num_sides = 6
        elif shape_type == 'diamond': num_sides = 4
        elif shape_type == 'small_square': num_sides, size = 4, 6
        else: num_sides = 3
        
        points = []
        for i in range(num_sides):
            a = angle_rad + (i / num_sides) * 2 * math.pi
            px = pos[0] + size * math.cos(a)
            py = pos[1] + size * math.sin(a)
            if shape_type == 'diamond':
                px = pos[0] + size * (1.5 if i % 2 == 0 else 0.7) * math.cos(a)
                py = pos[1] + size * (0.7 if i % 2 == 0 else 1.5) * math.sin(a)
            points.append((px, py))
        
        glow_color = (*color, 60)
        # Glow effect
        for i in range(3, 0, -1):
            glow_points = [(p[0] + (p[0] - pos[0]) * i * 0.1, p[1] + (p[1] - pos[1]) * i * 0.1) for p in points]
            pygame.gfxdraw.aapolygon(self.screen, glow_points, glow_color)
        
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "core_health": self.core_health, "resources": self.resources}

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Prism Guardian")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0

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
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}, Steps: {info['steps']}")
            running = False
            pygame.time.wait(2000)

        clock.tick(env.metadata['render_fps'])
        
    env.close()