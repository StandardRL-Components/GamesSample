
# Generated: 2025-08-28T05:08:34.357519
# Source Brief: brief_02529.md
# Brief Index: 2529

        
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

    user_guide = (
        "Controls: ↑↓←→ to move the cursor. Press shift to cycle tower types. Press space to build a tower."
    )

    game_description = (
        "Defend your base from waves of invading enemies by strategically placing towers in an isometric 2D world."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.width, self.height = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Colors ---
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_PATH = (40, 45, 60)
        self.COLOR_GRID = (25, 30, 45)
        self.COLOR_BASE = (50, 150, 80)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_PROJECTILE = (100, 200, 255)
        self.COLOR_TOWER_1 = (230, 180, 50)
        self.COLOR_TOWER_2 = (180, 80, 230)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_GREEN = (0, 200, 0)
        self.COLOR_HEALTH_RED = (200, 0, 0)
        self.COLOR_CURSOR_VALID = (0, 255, 0, 100)
        self.COLOR_CURSOR_INVALID = (255, 0, 0, 100)

        # --- Game Constants ---
        self.MAX_STEPS = 30 * 60 # 60 seconds at 30fps
        self.TOTAL_WAVES = 10
        self.STARTING_BASE_HEALTH = 100
        self.STARTING_RESOURCES = 150
        self.INTER_WAVE_DELAY = 150 # 5 seconds
        self.ENEMY_SPAWN_DELAY = 15 # 0.5 seconds

        # --- Isometric Projection ---
        self.tile_width = 48
        self.tile_height = 24
        self.iso_offset_x = self.width // 2
        self.iso_offset_y = 80
        self.grid_size = (12, 12)

        # --- Game Map ---
        self.path_coords = [
            (0, 5), (1, 5), (2, 5), (3, 5), (3, 4), (3, 3), (3, 2),
            (4, 2), (5, 2), (6, 2), (7, 2), (7, 3), (7, 4), (7, 5),
            (7, 6), (7, 7), (8, 7), (9, 7), (10, 7)
        ]
        self.base_pos = (11, 7)
        self.placement_spots = self._generate_placement_spots()
        self.tower_definitions = [
            {'name': 'Gatling', 'cost': 50, 'range': 80, 'damage': 5, 'fire_rate': 5, 'color': self.COLOR_TOWER_1},
            {'name': 'Cannon', 'cost': 120, 'range': 120, 'damage': 35, 'fire_rate': 1, 'color': self.COLOR_TOWER_2},
        ]

        self.reset()
        self.validate_implementation()

    def _grid_to_iso(self, x, y):
        iso_x = (x - y) * (self.tile_width / 2) + self.iso_offset_x
        iso_y = (x + y) * (self.tile_height / 2) + self.iso_offset_y
        return int(iso_x), int(iso_y)
    
    def _generate_placement_spots(self):
        spots = set()
        path_set = set(self.path_coords)
        path_set.add(self.base_pos)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if (x, y) not in path_set:
                    # check if adjacent to path
                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                        if (x+dx, y+dy) in path_set:
                            spots.add((x, y))
                            break
        return list(spots)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = self.STARTING_BASE_HEALTH
        self.resources = self.STARTING_RESOURCES
        self.wave_number = 0
        self.wave_in_progress = False
        self.inter_wave_timer = self.INTER_WAVE_DELAY // 3 # Start first wave faster

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.grid_size[0] // 2, self.grid_size[1] // 2]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        self.reward_this_step = 0

        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.TOTAL_WAVES:
            self.game_over = True # VICTORY
            self.reward_this_step += 50
            return

        self.wave_in_progress = True
        self.inter_wave_timer = 0
        
        difficulty_mod = 1 + (self.wave_number - 1) * 0.05
        enemy_count = 10 + (self.wave_number - 1) * 2
        
        self.enemies_to_spawn = []
        for i in range(enemy_count):
            start_pos = self._grid_to_iso(self.path_coords[0][0], self.path_coords[0][1])
            enemy = {
                'path_index': 0,
                'distance_on_segment': -i * 20, # Stagger spawn
                'pos': list(start_pos),
                'health': 100 * difficulty_mod,
                'max_health': 100 * difficulty_mod,
                'speed': 1.0 * difficulty_mod,
                'value': 10,
                'id': self.np_random.random()
            }
            self.enemies_to_spawn.append(enemy)
        
        self.spawn_timer = 0
        self.reward_this_step += 1 # Wave start bonus

    def step(self, action):
        self.reward_this_step = 0
        self.game_over = self._check_termination()
        
        if not self.game_over:
            self._handle_input(action)
            self._update_wave_manager()
            self._update_towers()
            self._update_projectiles()
            self._update_enemies()
            self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        reward = self.reward_this_step
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.grid_size[0] - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.grid_size[1] - 1)

        # Cycle tower (on press)
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_definitions)
        
        # Place tower (on press)
        if space_held and not self.last_space_held:
            self._place_tower()

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
    def _place_tower(self):
        pos_tuple = tuple(self.cursor_pos)
        tower_def = self.tower_definitions[self.selected_tower_type]
        
        is_valid_spot = pos_tuple in self.placement_spots
        is_occupied = any(t['grid_pos'] == pos_tuple for t in self.towers)
        can_afford = self.resources >= tower_def['cost']

        if is_valid_spot and not is_occupied and can_afford:
            self.resources -= tower_def['cost']
            iso_pos = self._grid_to_iso(pos_tuple[0], pos_tuple[1])
            new_tower = {
                'grid_pos': pos_tuple,
                'screen_pos': iso_pos,
                'type_index': self.selected_tower_type,
                'cooldown': 0,
            }
            self.towers.append(new_tower)
            # sfx: build_tower.wav
            
    def _update_wave_manager(self):
        if not self.wave_in_progress:
            self.inter_wave_timer += 1
            if self.inter_wave_timer >= self.INTER_WAVE_DELAY:
                self._start_next_wave()
        else:
            if not self.enemies and not self.enemies_to_spawn:
                self.wave_in_progress = False
                self.inter_wave_timer = 0
            
            self.spawn_timer += 1
            if self.enemies_to_spawn and self.spawn_timer >= self.ENEMY_SPAWN_DELAY:
                self.enemies.append(self.enemies_to_spawn.pop(0))
                self.spawn_timer = 0

    def _update_enemies(self):
        for enemy in reversed(self.enemies):
            enemy['distance_on_segment'] += enemy['speed']
            
            p1_idx = enemy['path_index']
            p2_idx = p1_idx + 1
            
            if p2_idx >= len(self.path_coords):
                self.enemies.remove(enemy)
                self.base_health -= 10
                self._create_explosion(enemy['pos'], self.COLOR_BASE, 15)
                # sfx: base_damage.wav
                continue
                
            start_node = self._grid_to_iso(*self.path_coords[p1_idx])
            end_node = self._grid_to_iso(*self.path_coords[p2_idx])
            
            segment_vec = (end_node[0] - start_node[0], end_node[1] - start_node[1])
            segment_len = math.hypot(*segment_vec)
            
            if segment_len > 0 and enemy['distance_on_segment'] >= segment_len:
                enemy['path_index'] += 1
                enemy['distance_on_segment'] -= segment_len
            
            if segment_len > 0:
                progress = enemy['distance_on_segment'] / segment_len
                enemy['pos'][0] = start_node[0] + segment_vec[0] * progress
                enemy['pos'][1] = start_node[1] + segment_vec[1] * progress

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue
            
            tower_def = self.tower_definitions[tower['type_index']]
            
            target = None
            max_path_dist = -1

            for enemy in self.enemies:
                dist = math.hypot(enemy['pos'][0] - tower['screen_pos'][0], enemy['pos'][1] - tower['screen_pos'][1])
                if dist <= tower_def['range']:
                    path_dist = enemy['path_index'] + (enemy['distance_on_segment'] / 1000) # Prioritize enemies further along
                    if path_dist > max_path_dist:
                        max_path_dist = path_dist
                        target = enemy
            
            if target:
                tower['cooldown'] = 60 / tower_def['fire_rate'] # 60fps assumed for rate
                proj_start_pos = (tower['screen_pos'][0], tower['screen_pos'][1] - 15)
                new_proj = {
                    'pos': list(proj_start_pos),
                    'target_id': target['id'],
                    'speed': 8,
                    'damage': tower_def['damage'],
                }
                self.projectiles.append(new_proj)
                # sfx: tower_shoot.wav

    def _update_projectiles(self):
        for proj in reversed(self.projectiles):
            target_enemy = next((e for e in self.enemies if e['id'] == proj['target_id']), None)

            if not target_enemy:
                self.projectiles.remove(proj)
                continue

            target_pos = target_enemy['pos']
            dx = target_pos[0] - proj['pos'][0]
            dy = target_pos[1] - proj['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < 5: # Hit
                target_enemy['health'] -= proj['damage']
                self.projectiles.remove(proj)
                self._create_explosion(proj['pos'], self.COLOR_PROJECTILE, 5)
                if target_enemy['health'] <= 0:
                    self.resources += target_enemy['value']
                    self.reward_this_step += 0.1
                    self._create_explosion(target_enemy['pos'], self.COLOR_ENEMY, 20)
                    # sfx: enemy_die.wav
                    self.enemies.remove(target_enemy)
                continue

            proj['pos'][0] += (dx / dist) * proj['speed']
            proj['pos'][1] += (dy / dist) * proj['speed']

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 2
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                'pos': list(pos),
                'vel': vel,
                'lifespan': 20 + self.np_random.integers(0, 10),
                'max_lifespan': 30,
                'color': color,
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.game_over: # Already set by win condition
            return True
        if self.base_health <= 0:
            if not self.game_over: # First frame of loss
                self.reward_this_step -= 50
                self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_map()
        
        # Sort all dynamic entities by Y-pos for correct isometric drawing
        render_queue = []
        for t in self.towers:
            render_queue.append(('tower', t))
        for e in self.enemies:
            render_queue.append(('enemy', e))
        
        render_queue.sort(key=lambda item: item[1]['pos'][1] if 'pos' in item[1] else item[1]['screen_pos'][1])

        for item_type, item_data in render_queue:
            if item_type == 'tower':
                self._render_tower(item_data)
            elif item_type == 'enemy':
                self._render_enemy(item_data)

        for proj in self.projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (int(proj['pos'][0]), int(proj['pos'][1])), 3)
        
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

    def _render_map(self):
        # Draw grid
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                iso_pos = self._grid_to_iso(x, y)
                points = [
                    self._grid_to_iso(x, y),
                    self._grid_to_iso(x + 1, y),
                    self._grid_to_iso(x + 1, y + 1),
                    self._grid_to_iso(x, y + 1)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Draw path
        for x, y in self.path_coords:
            points = [
                self._grid_to_iso(x, y), self._grid_to_iso(x + 1, y),
                self._grid_to_iso(x + 1, y + 1), self._grid_to_iso(x, y + 1)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PATH)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Draw base
        bx, by = self.base_pos
        points = [
            self._grid_to_iso(bx, by), self._grid_to_iso(bx + 1, by),
            self._grid_to_iso(bx + 1, by + 1), self._grid_to_iso(bx, by + 1)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_BASE)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)
        
        # Draw placement spots and cursor
        cursor_tuple = tuple(self.cursor_pos)
        tower_def = self.tower_definitions[self.selected_tower_type]
        is_valid_spot = cursor_tuple in self.placement_spots
        is_occupied = any(t['grid_pos'] == cursor_tuple for t in self.towers)
        can_afford = self.resources >= tower_def['cost']
        cursor_color = self.COLOR_CURSOR_VALID if is_valid_spot and not is_occupied and can_afford else self.COLOR_CURSOR_INVALID

        for x, y in self.placement_spots:
            color = (80, 80, 80, 20)
            points = [self._grid_to_iso(x,y), self._grid_to_iso(x+1,y), self._grid_to_iso(x+1,y+1), self._grid_to_iso(x,y+1)]
            temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.gfxdraw.filled_polygon(temp_surf, points, color)
            self.screen.blit(temp_surf, (0,0))
            
        cursor_points = [self._grid_to_iso(*cursor_tuple), self._grid_to_iso(cursor_tuple[0]+1,cursor_tuple[1]), self._grid_to_iso(cursor_tuple[0]+1,cursor_tuple[1]+1), self._grid_to_iso(cursor_tuple[0],cursor_tuple[1]+1)]
        temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        pygame.gfxdraw.filled_polygon(temp_surf, cursor_points, cursor_color)
        pygame.gfxdraw.aapolygon(temp_surf, cursor_points, (255,255,255,150))
        self.screen.blit(temp_surf, (0,0))


    def _render_tower(self, tower):
        tower_def = self.tower_definitions[tower['type_index']]
        x, y = tower['screen_pos']
        color = tower_def['color']
        
        base_rect = pygame.Rect(x - 8, y - 4, 16, 8)
        pygame.draw.rect(self.screen, color, base_rect, border_radius=2)
        
        turret_rect = pygame.Rect(x - 4, y - 16, 8, 12)
        pygame.draw.rect(self.screen, color, turret_rect, border_radius=2)
        
        # Cooldown indicator
        if tower['cooldown'] > 0:
            cooldown_ratio = tower['cooldown'] / (60 / tower_def['fire_rate'])
            cooldown_color = (255, 255, 255, 100)
            temp_surf = pygame.Surface((16, 16), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, cooldown_color, (8,8), 8)
            pygame.draw.arc(temp_surf, self.COLOR_BG, (0,0,16,16), 0, cooldown_ratio * 2 * math.pi, 8)
            self.screen.blit(temp_surf, (x - 8, y - 8))


    def _render_enemy(self, enemy):
        x, y = int(enemy['pos'][0]), int(enemy['pos'][1])
        size = 8
        
        # Body
        body_rect = pygame.Rect(x - size//2, y - size//2 - 2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_ENEMY, body_rect, border_radius=1)
        
        # Health bar
        health_ratio = enemy['health'] / enemy['max_health']
        bar_width = 12
        bar_height = 3
        bar_x = x - bar_width // 2
        bar_y = y - size//2 - 8
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (bar_x, bar_y, int(bar_width * health_ratio), bar_height))


    def _render_ui(self):
        # Top-left UI: Resources, Base Health, Wave
        resource_text = self.font_small.render(f"$: {int(self.resources)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(resource_text, (10, 10))

        health_text = self.font_small.render(f"Base: {int(self.base_health)}/{self.STARTING_BASE_HEALTH}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 30))

        wave_text = self.font_small.render(f"Wave: {self.wave_number}/{self.TOTAL_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 50))
        
        # Bottom-center UI: Selected Tower
        tower_def = self.tower_definitions[self.selected_tower_type]
        tower_info_text = self.font_small.render(
            f"Build: {tower_def['name']} (Cost: {tower_def['cost']}) - [SHIFT] to cycle", True, self.COLOR_UI_TEXT
        )
        text_rect = tower_info_text.get_rect(centerx=self.width // 2, bottom=self.height - 10)
        self.screen.blit(tower_info_text, text_rect)

        # Game Over / Victory Text
        if self.game_over:
            if self.base_health <= 0:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY)
            else:
                end_text = self.font_large.render("VICTORY!", True, self.COLOR_BASE)
            text_rect = end_text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(end_text, text_rect)
            
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
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
        
        # Transpose obs for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000)
            obs, info = env.reset()

        clock.tick(30) # Match auto_advance rate

    pygame.quit()