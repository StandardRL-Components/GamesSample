
# Generated: 2025-08-28T02:46:43.461853
# Source Brief: brief_01809.md
# Brief Index: 1809

        
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
        "Controls: Arrows to move cursor, Shift to cycle tower type, Space to place tower or start wave."
    )

    game_description = (
        "A top-down tower defense game. Place towers to defend your base against waves of enemies."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.TILE_SIZE = self.WIDTH // self.GRID_COLS
        
        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_PATH = (40, 40, 60)
        self.COLOR_GRID = (50, 50, 70)
        self.COLOR_GRID_CURSOR = (255, 255, 0)
        self.COLOR_START_BTN = (0, 150, 200)
        self.COLOR_BASE = (0, 200, 100)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_ENEMY_HEALTH_BG = (80, 20, 20)
        self.COLOR_ENEMY_HEALTH = (255, 80, 80)
        self.TOWER_COLORS = [(80, 120, 255), (255, 200, 50), (200, 80, 255)]
        self.PROJECTILE_COLORS = [(120, 180, 255), (255, 220, 100), (220, 120, 255)]
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR_BG = (80, 0, 0)
        self.COLOR_HEALTH_BAR = (0, 200, 100)

        # Game parameters
        self.MAX_WAVES = 10
        self.INITIAL_BASE_HEALTH = 100
        self.INITIAL_RESOURCES = 150
        self.WAVE_COMPLETION_RESOURCES = 100
        self.MAX_STEPS = 5000

        self.TOWER_SPECS = {
            0: {'name': 'Gatling', 'cost': 50, 'range': 80, 'cooldown': 10, 'damage': 10, 'shape': 'triangle'},
            1: {'name': 'Cannon', 'cost': 75, 'range': 120, 'cooldown': 25, 'damage': 30, 'shape': 'square'},
            2: {'name': 'Artillery', 'cost': 100, 'range': 200, 'cooldown': 50, 'damage': 80, 'shape': 'pentagon'}
        }
        self.START_WAVE_POS = (0, self.GRID_ROWS - 1)

        # --- Gymnasium Interface ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # --- Game State Initialization ---
        self._setup_layout()
        self.reset()

        self.validate_implementation()

    def _setup_layout(self):
        y_center = self.HEIGHT // 2
        self.path_waypoints = [
            pygame.Vector2(-20, y_center),
            pygame.Vector2(160, y_center),
            pygame.Vector2(160, 300),
            pygame.Vector2(480, 300),
            pygame.Vector2(480, y_center),
            pygame.Vector2(self.WIDTH + 20, y_center)
        ]
        self.base_pos = pygame.Vector2(self.WIDTH - 20, y_center)

        self.placement_zones = {}
        path_rects = []
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            path_rects.append(pygame.Rect(min(p1.x, p2.x) - self.TILE_SIZE//2, min(p1.y, p2.y) - self.TILE_SIZE//2, 
                                          abs(p1.x - p2.x) + self.TILE_SIZE, abs(p1.y - p2.y) + self.TILE_SIZE))

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                zone_rect = pygame.Rect(c * self.TILE_SIZE, r * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                on_path = any(zone_rect.colliderect(pr) for pr in path_rects)
                is_start_button = (c, r) == self.START_WAVE_POS
                if not on_path and not is_start_button:
                    self.placement_zones[(c, r)] = {
                        "center": (c * self.TILE_SIZE + self.TILE_SIZE // 2, r * self.TILE_SIZE + self.TILE_SIZE // 2),
                        "occupied": False
                    }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = "PLACEMENT"
        self.wave_number = 1
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = (self.GRID_COLS // 2, self.GRID_ROWS // 2)
        self.selected_tower_type = 0
        self.last_shift_held = False
        self.last_space_held = False
        self.screen_shake = 0

        for zone in self.placement_zones.values():
            zone["occupied"] = False

        self._prepare_next_wave()
        
        return self._get_observation(), self._get_info()

    def _prepare_next_wave(self):
        self.enemies_spawned_this_wave = 0
        base_enemy_count = 5
        self.enemies_to_spawn_total = base_enemy_count + (self.wave_number - 1) * 2
        self.spawn_timer = 0
        self.spawn_interval = max(10, 30 - self.wave_number)
        
        base_health = 10
        health_increase_factor = 1.15
        self.enemy_health = int(base_health * (health_increase_factor ** (self.wave_number - 1)))
        self.enemy_speed = 1.0 + self.wave_number * 0.1

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.game_over = False

        if self.game_phase == "PLACEMENT":
            self._handle_placement_phase(movement, space_pressed, shift_pressed)
        
        if self.game_phase == "WAVE":
            reward += self._update_wave_phase()
            if self.enemies_spawned_this_wave >= self.enemies_to_spawn_total and not self.enemies:
                reward += self._end_wave()

        # Update persistent state
        if self.screen_shake > 0: self.screen_shake -= 1
        self.steps += 1
        
        # Check termination conditions
        terminated = self.base_health <= 0 or (self.wave_number > self.MAX_WAVES and self.game_phase == "PLACEMENT")
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.base_health > 0 and self.wave_number > self.MAX_WAVES:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty

        self.score += reward

        self.last_shift_held = shift_pressed
        self.last_space_held = space_pressed
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_placement_phase(self, movement, space_pressed, shift_pressed):
        # Cycle tower type
        if shift_pressed and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: UI_cycle.wav

        # Move cursor
        if movement != 0:
            c, r = self.cursor_pos
            if movement == 1: r -= 1 # Up
            elif movement == 2: r += 1 # Down
            elif movement == 3: c -= 1 # Left
            elif movement == 4: c += 1 # Right
            self.cursor_pos = (max(0, min(self.GRID_COLS - 1, c)), max(0, min(self.GRID_ROWS - 1, r)))

        # Handle space press (place tower or start wave)
        if space_pressed and not self.last_space_held:
            if self.cursor_pos == self.START_WAVE_POS:
                self.game_phase = "WAVE"
                # sfx: wave_start.wav
            elif self.cursor_pos in self.placement_zones and not self.placement_zones[self.cursor_pos]["occupied"]:
                spec = self.TOWER_SPECS[self.selected_tower_type]
                if self.resources >= spec['cost']:
                    self.resources -= spec['cost']
                    center_pos = self.placement_zones[self.cursor_pos]["center"]
                    self.towers.append({
                        'pos': pygame.Vector2(center_pos), 'type': self.selected_tower_type,
                        'cooldown': 0, 'target': None
                    })
                    self.placement_zones[self.cursor_pos]["occupied"] = True
                    # sfx: tower_place.wav
                else:
                    # sfx: error.wav
                    pass

    def _update_wave_phase(self):
        reward = 0
        
        # 1. Spawn enemies
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval and self.enemies_spawned_this_wave < self.enemies_to_spawn_total:
            self.spawn_timer = 0
            self.enemies_spawned_this_wave += 1
            self.enemies.append({
                'pos': self.path_waypoints[0].copy(), 'health': self.enemy_health, 'max_health': self.enemy_health,
                'speed': self.enemy_speed, 'waypoint_idx': 1, 'value': 0.1
            })

        # 2. Update towers
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
            
            spec = self.TOWER_SPECS[tower['type']]
            # Invalidate target if dead or out of range
            if tower.get('target') and (tower['target'] not in self.enemies or tower['pos'].distance_to(tower['target']['pos']) > spec['range']):
                tower['target'] = None

            # Find new target if needed
            if not tower.get('target'):
                possible_targets = [e for e in self.enemies if tower['pos'].distance_to(e['pos']) <= spec['range']]
                if possible_targets:
                    # Target enemy furthest along the path
                    tower['target'] = max(possible_targets, key=lambda e: e['waypoint_idx'] + e['pos'].distance_to(self.path_waypoints[e['waypoint_idx']-1]) / e['pos'].distance_to(self.path_waypoints[e['waypoint_idx']]))

            # Fire if ready and has target
            if tower.get('target') and tower['cooldown'] <= 0:
                tower['cooldown'] = spec['cooldown']
                target_pos = tower['target']['pos']
                direction = (target_pos - tower['pos']).normalize()
                self.projectiles.append({
                    'pos': tower['pos'].copy(), 'vel': direction * 8, 'damage': spec['damage'],
                    'color': self.PROJECTILE_COLORS[tower['type']], 'lifespan': spec['range'] // 6
                })
                # sfx: shoot_X.wav

        # 3. Update projectiles and check collisions
        for proj in self.projectiles[:]:
            proj['pos'] += proj['vel']
            proj['lifespan'] -= 1
            if proj['lifespan'] <= 0:
                self.projectiles.remove(proj)
                continue
            
            for enemy in self.enemies[:]:
                if proj['pos'].distance_to(enemy['pos']) < self.TILE_SIZE / 3:
                    enemy['health'] -= proj['damage']
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    
                    if enemy['health'] <= 0:
                        reward += enemy['value']
                        self._create_explosion(enemy['pos'], self.COLOR_ENEMY)
                        self.enemies.remove(enemy)
                        # sfx: enemy_die.wav
                    break
        
        # 4. Update enemies
        for enemy in self.enemies[:]:
            if enemy['waypoint_idx'] >= len(self.path_waypoints):
                self.base_health = max(0, self.base_health - 10)
                reward -= 1.0 # -0.1 per health point * 10
                self.enemies.remove(enemy)
                self.screen_shake = 10
                # sfx: base_damage.wav
                continue

            target_waypoint = self.path_waypoints[enemy['waypoint_idx']]
            direction = (target_waypoint - enemy['pos'])
            if direction.length() < enemy['speed']:
                enemy['waypoint_idx'] += 1
            else:
                enemy['pos'] += direction.normalize() * enemy['speed']
        
        # 5. Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        return reward

    def _end_wave(self):
        self.game_phase = "PLACEMENT"
        self.wave_number += 1
        if self.wave_number <= self.MAX_WAVES:
            self.resources += self.WAVE_COMPLETION_RESOURCES
            self._prepare_next_wave()
            # sfx: wave_complete.wav
            return 1.0 # Wave completion reward
        return 0

    def _create_explosion(self, pos, color):
        for _ in range(20):
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(random.uniform(-3, 3), random.uniform(-3, 3)),
                'life': random.randint(10, 20),
                'color': color
            })

    def _get_observation(self):
        # Apply screen shake
        shake_offset = (0, 0)
        if self.screen_shake > 0:
            shake_offset = (random.randint(-4, 4), random.randint(-4, 4))
        
        # Render all game elements
        self._render_game(shake_offset)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, offset):
        self.screen.fill(self.COLOR_BG)
        ox, oy = offset

        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, [(p.x + ox, p.y + oy) for p in self.path_waypoints], self.TILE_SIZE)
        
        # Base
        pygame.draw.circle(self.screen, self.COLOR_BASE, (self.base_pos.x + ox, self.base_pos.y + oy), 15)

        # Placement Zones & Towers
        if self.game_phase == "PLACEMENT":
            for pos, zone in self.placement_zones.items():
                rect = pygame.Rect(pos[0] * self.TILE_SIZE + ox, pos[1] * self.TILE_SIZE + oy, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        for tower in self.towers:
            self._draw_tower(tower, offset)

        # Projectiles
        for proj in self.projectiles:
            pygame.draw.circle(self.screen, proj['color'], (int(proj['pos'].x + ox), int(proj['pos'].y + oy)), 3)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x + ox), int(enemy['pos'].y + oy))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            bar_w = 16
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_HEALTH_BG, (pos[0] - bar_w/2, pos[1] - 16, bar_w, 4))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_HEALTH, (pos[0] - bar_w/2, pos[1] - 16, bar_w * health_pct, 4))

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x + ox), int(p['pos'].y + oy)), max(0, int(p['life'] / 5)))

        # Cursor and UI for placement phase
        if self.game_phase == "PLACEMENT":
            self._render_cursor(offset)

    def _draw_tower(self, tower, offset):
        spec = self.TOWER_SPECS[tower['type']]
        color = self.TOWER_COLORS[tower['type']]
        pos = (tower['pos'].x + offset[0], tower['pos'].y + offset[1])
        
        # Draw range circle if targeted or in placement phase
        if tower.get('target') or self.game_phase == 'PLACEMENT':
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), spec['range'], (*color, 60))

        if spec['shape'] == 'triangle':
            points = [ (pos[0], pos[1] - 10), (pos[0] - 10, pos[1] + 8), (pos[0] + 10, pos[1] + 8) ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif spec['shape'] == 'square':
            rect = pygame.Rect(pos[0] - 8, pos[1] - 8, 16, 16)
            pygame.draw.rect(self.screen, color, rect)
        elif spec['shape'] == 'pentagon':
            points = []
            for i in range(5):
                angle = math.radians(90 + i * 72)
                points.append((pos[0] + 12 * math.cos(angle), pos[1] - 12 * math.sin(angle)))
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_cursor(self, offset):
        c, r = self.cursor_pos
        ox, oy = offset
        cursor_rect = pygame.Rect(c * self.TILE_SIZE + ox, r * self.TILE_SIZE + oy, self.TILE_SIZE, self.TILE_SIZE)
        
        if self.cursor_pos == self.START_WAVE_POS:
            pygame.draw.rect(self.screen, self.COLOR_START_BTN, cursor_rect, 0)
            pygame.draw.rect(self.screen, self.COLOR_GRID_CURSOR, cursor_rect, 2)
            # Draw play icon
            points = [(cursor_rect.centerx - 5, cursor_rect.centery - 8),
                      (cursor_rect.centerx - 5, cursor_rect.centery + 8),
                      (cursor_rect.centerx + 8, cursor_rect.centery)]
            pygame.draw.polygon(self.screen, (255,255,255), points)
        else:
            pygame.draw.rect(self.screen, self.COLOR_GRID_CURSOR, cursor_rect, 2)
            if self.cursor_pos in self.placement_zones and not self.placement_zones[self.cursor_pos]["occupied"]:
                # Preview tower
                spec = self.TOWER_SPECS[self.selected_tower_type]
                color = self.TOWER_COLORS[self.selected_tower_type]
                pygame.gfxdraw.aacircle(self.screen, cursor_rect.centerx, cursor_rect.centery, spec['range'], (*color, 40))
                # Add a ghost tower drawing here if desired

    def _render_ui(self):
        # Top bar background
        pygame.draw.rect(self.screen, self.COLOR_BG, (0, 0, self.WIDTH, 30))
        
        # Base Health
        health_text = self.font_small.render(f"Base HP: {self.base_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 7))
        health_pct = self.base_health / self.INITIAL_BASE_HEALTH
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (100, 8, 100, 14))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (100, 8, 100 * health_pct, 14))

        # Wave
        wave_text = self.font_large.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH // 2 - wave_text.get_width() // 2, 3))

        # Resources
        res_text = self.font_large.render(f"${self.resources}", True, (255, 215, 0))
        self.screen.blit(res_text, (self.WIDTH - res_text.get_width() - 10, 3))

        # Selected Tower Info (in placement phase)
        if self.game_phase == "PLACEMENT":
            spec = self.TOWER_SPECS[self.selected_tower_type]
            color = self.TOWER_COLORS[self.selected_tower_type]
            
            info_text = f"Build: {spec['name']} (${spec['cost']})"
            info_render = self.font_small.render(info_text, True, color)
            self.screen.blit(info_render, (10, self.HEIGHT - 25))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.base_health > 0 else "DEFEAT"
            color = self.COLOR_BASE if self.base_health > 0 else self.COLOR_ENEMY
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
            "resources": self.resources,
            "game_phase": self.game_phase,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a display for human playing
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = np.array([movement, space_held, shift_held])
        # --- End Human Controls ---

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if info['game_phase'] == 'WAVE':
            clock.tick(30) # Run at 30 FPS during waves
        else:
            clock.tick(15) # Slower update during placement

    print(f"Game Over. Final Score: {info['score']}, Wave: {info['wave']}")
    env.close()