
# Generated: 2025-08-27T22:25:41.870074
# Source Brief: brief_03120.md
# Brief Index: 3120

        
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
        "Controls: Arrow keys to move cursor. Space to place a tower. Shift to cycle tower type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in this isometric tower defense game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 16)
        self.font_l = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Game Constants ---
        self.max_steps = 1000
        self.initial_base_health = 20
        self.initial_resources = 100
        self.grid_w, self.grid_h = 20, 12
        self.tile_w, self.tile_h = 40, 20
        self.origin_x = self.screen_width / 2
        self.origin_y = 60

        # --- Colors ---
        self.COLOR_BG = (25, 35, 50)
        self.COLOR_PATH = (60, 70, 90)
        self.COLOR_GRASS = (40, 50, 70)
        self.COLOR_PLACEABLE = (50, 65, 85)
        self.COLOR_BASE = (0, 150, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_CURSOR_VALID = (255, 255, 255, 100)
        self.COLOR_CURSOR_INVALID = (255, 0, 0, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_HEALTH_GREEN = (0, 200, 0)
        self.COLOR_HEALTH_RED = (200, 0, 0)
        self.TOWER_COLORS = [(100, 255, 100), (255, 150, 50)]

        # --- Game World Definition ---
        self._define_world()
        self._define_towers()
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.resources = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.enemy_spawn_timer = 0.0
        self.enemy_spawn_rate = 60.0 # Ticks per spawn
        self.enemy_health_multiplier = 1.0

        self.reset()
        self.validate_implementation()

    def _define_world(self):
        self.path_coords = [
            (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (5, 2),
            (6, 2), (7, 2), (8, 2), (9, 2), (10, 2), (11, 2), (11, 3), (11, 4),
            (11, 5), (11, 6), (11, 7), (12, 7), (13, 7), (14, 7), (15, 7), (16, 7),
            (17, 7), (18, 7), (19, 7)
        ]
        self.base_pos = (19, 7)
        self.tower_spots = [
            (2, 3), (2, 7), (4, 1), (4, 7), (7, 4), (9, 4), (9, 0),
            (13, 5), (13, 9), (15, 5), (15, 9), (17, 5), (17, 9)
        ]
        
        # Pre-calculate tile polygons for rendering
        self.tile_polygons = {}
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                iso_x, iso_y = self._grid_to_iso(x, y)
                poly = [
                    (iso_x, iso_y),
                    (iso_x + self.tile_w / 2, iso_y + self.tile_h / 2),
                    (iso_x, iso_y + self.tile_h),
                    (iso_x - self.tile_w / 2, iso_y + self.tile_h / 2)
                ]
                
                color = self.COLOR_GRASS
                if (x, y) in self.path_coords:
                    color = self.COLOR_PATH
                elif (x, y) in self.tower_spots:
                    color = self.COLOR_PLACEABLE
                self.tile_polygons[(x, y)] = (poly, color)

    def _define_towers(self):
        self.tower_types = [
            {
                "name": "Gun Turret", "cost": 25, "damage": 5, "range": 100, 
                "fire_rate": 20, "proj_speed": 8, "color": self.TOWER_COLORS[0]
            },
            {
                "name": "Cannon", "cost": 60, "damage": 15, "range": 120, 
                "fire_rate": 60, "proj_speed": 6, "color": self.TOWER_COLORS[1]
            }
        ]

    def _grid_to_iso(self, grid_x, grid_y):
        iso_x = self.origin_x + (grid_x - grid_y) * (self.tile_w / 2)
        iso_y = self.origin_y + (grid_x + grid_y) * (self.tile_h / 2)
        return iso_x, iso_y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = self.initial_base_health
        self.resources = self.initial_resources
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = self.tower_spots[0]
        self.selected_tower_type = 0
        
        self.last_space_held = False
        self.last_shift_held = False

        self.enemy_spawn_timer = 0
        self.enemy_spawn_rate = 60.0
        self.enemy_health_multiplier = 1.0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01  # Small penalty for time passing
        
        # --- 1. Handle Player Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Cycle tower type on SHIFT press
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_types)
        
        # Move cursor
        if movement != 0:
            current_index = self.tower_spots.index(self.cursor_pos)
            if movement == 1: # Up
                candidates = [i for i, spot in enumerate(self.tower_spots) if spot[1] < self.cursor_pos[1]]
                if candidates:
                    next_spot_idx = min(candidates, key=lambda i: (self.cursor_pos[1]-self.tower_spots[i][1])**2 + (self.cursor_pos[0]-self.tower_spots[i][0])**2)
                    self.cursor_pos = self.tower_spots[next_spot_idx]
            elif movement == 2: # Down
                candidates = [i for i, spot in enumerate(self.tower_spots) if spot[1] > self.cursor_pos[1]]
                if candidates:
                    next_spot_idx = min(candidates, key=lambda i: (self.tower_spots[i][1]-self.cursor_pos[1])**2 + (self.tower_spots[i][0]-self.cursor_pos[0])**2)
                    self.cursor_pos = self.tower_spots[next_spot_idx]
            elif movement == 3: # Left
                candidates = [i for i, spot in enumerate(self.tower_spots) if spot[0] < self.cursor_pos[0]]
                if candidates:
                    next_spot_idx = min(candidates, key=lambda i: (self.cursor_pos[0]-self.tower_spots[i][0])**2 + (self.cursor_pos[1]-self.tower_spots[i][1])**2)
                    self.cursor_pos = self.tower_spots[next_spot_idx]
            elif movement == 4: # Right
                candidates = [i for i, spot in enumerate(self.tower_spots) if spot[0] > self.cursor_pos[0]]
                if candidates:
                    next_spot_idx = min(candidates, key=lambda i: (self.tower_spots[i][0]-self.cursor_pos[0])**2 + (self.cursor_pos[1]-self.tower_spots[i][1])**2)
                    self.cursor_pos = self.tower_spots[next_spot_idx]

        # Place tower on SPACE press
        tower_spec = self.tower_types[self.selected_tower_type]
        is_spot_empty = not any(t['pos'] == self.cursor_pos for t in self.towers)
        if space_held and not self.last_space_held and is_spot_empty and self.resources >= tower_spec['cost']:
            self.resources -= tower_spec['cost']
            iso_x, iso_y = self._grid_to_iso(*self.cursor_pos)
            self.towers.append({
                "pos": self.cursor_pos, "iso_pos": (iso_x, iso_y),
                "spec": tower_spec, "cooldown": 0, "target": None
            })
            # Sfx: place_tower.wav
            self._create_particles(iso_x, iso_y, tower_spec['color'], 20)

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- 2. Update Game Logic ---
        # Update difficulty scaling
        self.enemy_spawn_rate = max(20.0, 60.0 * (1 - (self.steps // 100) * 0.01))
        self.enemy_health_multiplier = 1.0 + (self.steps // 200) * 0.02

        # Spawn enemies
        self.enemy_spawn_timer += 1
        if self.enemy_spawn_timer >= self.enemy_spawn_rate:
            self.enemy_spawn_timer = 0
            start_pos = self.path_coords[0]
            iso_x, iso_y = self._grid_to_iso(*start_pos)
            self.enemies.append({
                "pos": [iso_x, iso_y], "path_idx": 0, "speed": self.np_random.uniform(0.7, 1.0),
                "health": 20 * self.enemy_health_multiplier, "max_health": 20 * self.enemy_health_multiplier,
                "value": 5
            })

        # Update towers
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                # Find target
                target = None
                min_dist = tower['spec']['range'] ** 2
                for enemy in self.enemies:
                    dist_sq = (tower['iso_pos'][0] - enemy['pos'][0])**2 + (tower['iso_pos'][1] - enemy['pos'][1])**2
                    if dist_sq < min_dist:
                        min_dist = dist_sq
                        target = enemy
                
                if target:
                    tower['cooldown'] = tower['spec']['fire_rate']
                    self.projectiles.append({
                        "pos": list(tower['iso_pos']), "target": target, "spec": tower['spec']
                    })
                    # Sfx: tower_fire.wav

        # Update projectiles
        for proj in self.projectiles[:]:
            target_pos = proj['target']['pos']
            dx = target_pos[0] - proj['pos'][0]
            dy = target_pos[1] - proj['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < proj['spec']['proj_speed']:
                # Hit target
                proj['target']['health'] -= proj['spec']['damage']
                reward += 0.1
                self.score += 1
                self._create_particles(proj['pos'][0], proj['pos'][1], (255, 255, 0), 5)
                self.projectiles.remove(proj)
                # Sfx: enemy_hit.wav
            else:
                proj['pos'][0] += (dx / dist) * proj['spec']['proj_speed']
                proj['pos'][1] += (dy / dist) * proj['spec']['proj_speed']

        # Update enemies
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                # Enemy defeated
                reward += 1
                self.score += 10
                self.resources += enemy['value']
                self._create_particles(enemy['pos'][0], enemy['pos'][1], self.COLOR_ENEMY, 30, 5)
                self.enemies.remove(enemy)
                # Sfx: enemy_die.wav
                continue

            # Move enemy
            if enemy['path_idx'] < len(self.path_coords) - 1:
                target_grid_pos = self.path_coords[enemy['path_idx'] + 1]
                target_iso_pos = self._grid_to_iso(*target_grid_pos)
                
                dx = target_iso_pos[0] - enemy['pos'][0]
                dy = target_iso_pos[1] - enemy['pos'][1]
                dist = math.hypot(dx, dy)
                
                if dist < enemy['speed']:
                    enemy['path_idx'] += 1
                    enemy['pos'] = list(target_iso_pos)
                else:
                    enemy['pos'][0] += (dx / dist) * enemy['speed']
                    enemy['pos'][1] += (dy / dist) * enemy['speed']
            else:
                # Reached base
                self.base_health -= 1
                self._create_particles(enemy['pos'][0], enemy['pos'][1], self.COLOR_BASE, 20, 5)
                self.enemies.remove(enemy)
                # Sfx: base_damage.wav
        
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # --- 3. Check Termination ---
        self.steps += 1
        terminated = self.base_health <= 0 or self.steps >= self.max_steps
        if terminated:
            self.game_over = True
            if self.base_health > 0: # Won
                reward += 100
                self.score += 1000
            else: # Lost
                reward -= 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, x, y, color, count, max_life=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(max_life // 2, max_life),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _render_game(self):
        # Draw base tiles
        for poly, color in self.tile_polygons.values():
            pygame.gfxdraw.filled_polygon(self.screen, poly, color)
            pygame.gfxdraw.aapolygon(self.screen, poly, (0,0,0,50))

        # Draw base
        base_iso_x, base_iso_y = self._grid_to_iso(*self.base_pos)
        base_rect = pygame.Rect(0, 0, 30, 30)
        base_rect.center = (base_iso_x, base_iso_y + self.tile_h / 2)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=4)
        
        # Combine and sort drawable entities (towers and enemies) by Y-coordinate for correct occlusion
        draw_list = []
        for t in self.towers:
            draw_list.append({'type': 'tower', 'y': t['iso_pos'][1], 'data': t})
        for e in self.enemies:
            draw_list.append({'type': 'enemy', 'y': e['pos'][1], 'data': e})
        
        draw_list.sort(key=lambda item: item['y'])

        for item in draw_list:
            if item['type'] == 'tower':
                self._draw_tower(item['data'])
            elif item['type'] == 'enemy':
                self._draw_enemy(item['data'])
        
        # Draw projectiles
        for proj in self.projectiles:
            pygame.draw.circle(self.screen, proj['spec']['color'], (int(proj['pos'][0]), int(proj['pos'][1])), 3)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*p['color'], alpha)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)
        
        # Draw cursor
        self._draw_cursor()

    def _draw_tower(self, tower):
        iso_x, iso_y = tower['iso_pos']
        spec = tower['spec']
        center_y = iso_y + self.tile_h / 2
        
        # Base
        pygame.draw.rect(self.screen, (30,30,30), (iso_x - 12, center_y - 6, 24, 12), border_radius=3)
        # Turret
        pygame.draw.rect(self.screen, spec['color'], (iso_x - 8, center_y - 10, 16, 10), border_radius=3)
        
    def _draw_enemy(self, enemy):
        x, y = int(enemy['pos'][0]), int(enemy['pos'][1])
        pygame.draw.circle(self.screen, self.COLOR_ENEMY, (x, y), 6)
        pygame.draw.circle(self.screen, (0,0,0), (x, y), 6, 1)

        # Health bar
        health_pct = enemy['health'] / enemy['max_health']
        bar_w = 16
        bar_h = 4
        bar_x = x - bar_w / 2
        bar_y = y - 15
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (bar_x, bar_y, bar_w * health_pct, bar_h))

    def _draw_cursor(self):
        iso_x, iso_y = self._grid_to_iso(*self.cursor_pos)
        tower_spec = self.tower_types[self.selected_tower_type]
        is_spot_empty = not any(t['pos'] == self.cursor_pos for t in self.towers)
        can_afford = self.resources >= tower_spec['cost']

        # Draw range indicator
        if is_spot_empty:
            color = (255, 255, 255, 20) if can_afford else (255, 0, 0, 20)
            pygame.gfxdraw.filled_circle(self.screen, int(iso_x), int(iso_y + self.tile_h / 2), tower_spec['range'], color)
        
        # Draw placement tile highlight
        poly, _ = self.tile_polygons[self.cursor_pos]
        color = self.COLOR_CURSOR_VALID if is_spot_empty and can_afford else self.COLOR_CURSOR_INVALID
        pygame.gfxdraw.filled_polygon(self.screen, poly, color)
        pygame.gfxdraw.aapolygon(self.screen, poly, (255, 255, 255))

    def _render_ui(self):
        # --- Top Bar ---
        top_bar_rect = pygame.Rect(0, 0, self.screen_width, 40)
        pygame.draw.rect(self.screen, (15, 20, 30), top_bar_rect)
        
        # Resources
        res_text = self.font_l.render(f"{self.resources}", True, self.COLOR_GOLD)
        self.screen.blit(res_text, (20, 8))
        
        # Base Health
        health_text = self.font_l.render(f"Base: {self.base_health}/{self.initial_base_health}", True, self.COLOR_BASE)
        self.screen.blit(health_text, (120, 8))

        # Time/Steps
        time_text = self.font_l.render(f"Time: {self.steps}/{self.max_steps}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(right=self.screen_width - 20, centery=20)
        self.screen.blit(time_text, time_rect)

        # --- Bottom Bar (Tower Info) ---
        bottom_bar_rect = pygame.Rect(0, self.screen_height - 40, self.screen_width, 40)
        pygame.draw.rect(self.screen, (15, 20, 30), bottom_bar_rect)
        
        spec = self.tower_types[self.selected_tower_type]
        tower_info_text = self.font_s.render(
            f"Selected: {spec['name']} | Cost: {spec['cost']} | Dmg: {spec['damage']} | Rng: {spec['range']} | Rate: {spec['fire_rate']}",
            True, self.COLOR_TEXT
        )
        info_rect = tower_info_text.get_rect(centerx=self.screen_width/2, centery=self.screen_height - 20)
        self.screen.blit(tower_info_text, info_rect)
        
        # Game Over Text
        if self.game_over:
            s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            
            msg = "VICTORY!" if self.base_health > 0 else "DEFEAT"
            color = self.COLOR_HEALTH_GREEN if self.base_health > 0 else self.COLOR_ENEMY
            
            end_text = self.font_l.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(end_text, end_rect)


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
            "resources": self.resources,
            "base_health": self.base_health,
            "enemies": len(self.enemies),
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

# Example usage to run and display the game
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need a Pygame screen
    pygame.display.set_caption("Tower Defense")
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    obs, info = env.reset()
    terminated = False
    
    # Store key presses
    key_map = {
        pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4,
    }
    
    running = True
    while running:
        # --- Human Input ---
        movement = 0
        space_pressed = 0
        shift_pressed = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement = move_val
                break # Only one movement key at a time
        
        if keys[pygame.K_SPACE]:
            space_pressed = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_pressed = 1
        
        action = [movement, space_pressed, shift_pressed]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Score: {info['score']}, Steps: {info['steps']}")
            # Simple delay and reset on game over
            pygame.time.delay(3000)
            obs, info = env.reset()

        # --- Rendering ---
        # The observation is already a rendered frame. We just need to display it.
        # Need to transpose it back for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we need to control the frame rate of the human player loop
        env.clock.tick(30) # Limit human play to 30 FPS

    env.close()