
# Generated: 2025-08-27T23:33:11.191710
# Source Brief: brief_03502.md
# Brief Index: 3502

        
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
        "Controls: Use arrow keys to select a tower slot. Press Shift to cycle tower types. Press Space to build or upgrade a tower."
    )

    game_description = (
        "Defend your base from waves of enemies by placing and upgrading towers in this isometric tower defense game."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.MAX_WAVES = 10

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Colors ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_PATH = (50, 60, 70)
        self.COLOR_PATH_BORDER = (40, 50, 60)
        self.COLOR_SLOT = (70, 80, 90)
        self.COLOR_BASE = (0, 150, 200)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_RESOURCE = (255, 200, 0)
        self.COLOR_HEALTH_BG = (100, 0, 0)
        self.COLOR_HEALTH_FG = (0, 200, 0)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_INVALID = (255, 50, 50)
        
        # --- Isometric Grid ---
        self.TILE_W, self.TILE_H = 48, 24
        self.ORIGIN_X, self.ORIGIN_Y = self.WIDTH // 2, 80

        # --- Game Design ---
        self.ENEMY_PATH = [(i, 9) for i in range(12)] + [(11, i) for i in range(8, -1, -1)]
        self.TOWER_SLOTS = [(2, 7), (5, 7), (8, 7), (3, 4), (6, 4), (9, 4)]
        self.BASE_POS = (11, 0)
        self.INITIAL_RESOURCES = 100
        self.INITIAL_BASE_HEALTH = 50
        self.INTERMISSION_DURATION = 150 # 5 seconds at 30fps

        self.TOWER_SPECS = {
            "arrow": {
                "name": "Arrow Tower",
                "cost": [50, 75],
                "damage": [10, 15],
                "range": [2.5, 3],
                "cooldown": [30, 25],
                "proj_speed": 8,
                "color": [(150, 150, 150), (200, 200, 200)],
            },
            "cannon": {
                "name": "Cannon",
                "cost": [100, 150],
                "damage": [25, 40],
                "range": [2.0, 2.5],
                "cooldown": [60, 50],
                "proj_speed": 6,
                "color": [(200, 100, 0), (255, 150, 50)],
            }
        }
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = 0
        self.resources = 0
        self.wave_number = 0
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.cursor_index = 0
        self.selected_tower_type_idx = 0
        self.available_tower_types = []
        self.wave_state = "intermission" # or "active"
        self.intermission_timer = 0
        self.enemies_to_spawn_this_wave = []
        self.spawn_timer = 0
        self.last_space_press = False
        self.last_shift_press = False
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.wave_number = 0
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.cursor_index = 0
        self.selected_tower_type_idx = 0
        self.available_tower_types = ["arrow"]
        self.wave_state = "intermission"
        self.intermission_timer = self.INTERMISSION_DURATION
        self.enemies_to_spawn_this_wave = []
        self.spawn_timer = 0
        self.last_space_press = False
        self.last_shift_press = False
        
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        self.steps += 1

        # --- 1. Handle Player Input ---
        if movement in [1, 4]: # next slot (right/up)
            self.cursor_index = (self.cursor_index + 1) % len(self.TOWER_SLOTS)
        elif movement in [2, 3]: # prev slot (down/left)
            self.cursor_index = (self.cursor_index - 1 + len(self.TOWER_SLOTS)) % len(self.TOWER_SLOTS)

        if shift_held and not self.last_shift_press:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.available_tower_types)
        
        if space_held and not self.last_space_press:
            reward += self._handle_build_or_upgrade()

        self.last_space_press = space_held
        self.last_shift_press = shift_held

        # --- 2. Update Game State ---
        if self.wave_state == "intermission":
            self.intermission_timer -= 1
            if self.intermission_timer <= 0:
                self._start_next_wave()
        
        if self.wave_state == "active":
            # Spawn enemies
            self.spawn_timer -= 1
            if self.spawn_timer <= 0 and self.enemies_to_spawn_this_wave:
                self.enemies.append(self.enemies_to_spawn_this_wave.pop(0))
                self.spawn_timer = 15 # spawn interval

            # Update Towers
            for tower in self.towers:
                tower["cooldown_timer"] = max(0, tower["cooldown_timer"] - 1)
                if tower["cooldown_timer"] == 0:
                    target = self._find_target(tower)
                    if target:
                        self._fire_projectile(tower, target)
                        # sfx: tower_fire.wav
                        self._create_particles(self._grid_to_screen(tower['pos']), 5, (255, 255, 200), 1, 3)

            # Update Projectiles
            for proj in self.projectiles[:]:
                proj['pos'] = (proj['pos'][0] + proj['vel'][0], proj['pos'][1] + proj['vel'][1])
                proj['lifespan'] -= 1
                if proj['lifespan'] <= 0:
                    self.projectiles.remove(proj)
                    continue
                
                # Simple point-rect collision for isometric view
                enemy_screen_pos = self._grid_to_screen(proj['target']['pos'])
                if pygame.Rect(enemy_screen_pos[0]-8, enemy_screen_pos[1]-16, 16, 24).collidepoint(proj['pos']):
                    proj['target']['health'] -= proj['damage']
                    reward += 0.1
                    self.score += 1
                    self._create_particles(proj['pos'], 10, (255, 150, 0), 2, 4)
                    self.projectiles.remove(proj)
                    # sfx: hit.wav

            # Update Enemies
            for enemy in self.enemies[:]:
                if enemy['health'] <= 0:
                    reward += 1.0
                    self.score += 10
                    self.resources += enemy['bounty']
                    self._create_particles(self._grid_to_screen(enemy['pos']), 20, (200, 50, 50), 3, 6)
                    self.enemies.remove(enemy)
                    # sfx: enemy_die.wav
                    continue

                path_idx = enemy['path_index']
                if path_idx < len(self.ENEMY_PATH) - 1:
                    start_node = self.ENEMY_PATH[path_idx]
                    end_node = self.ENEMY_PATH[path_idx + 1]
                    enemy['distance_on_segment'] += enemy['speed']
                    
                    if enemy['distance_on_segment'] >= 1.0:
                        enemy['distance_on_segment'] = 0.0
                        enemy['path_index'] += 1
                        path_idx += 1
                        if path_idx >= len(self.ENEMY_PATH) - 1: # Reached end
                            self.base_health = max(0, self.base_health - enemy['damage'])
                            self.enemies.remove(enemy)
                            # sfx: base_damage.wav
                            self._create_particles(self._grid_to_screen(self.BASE_POS), 30, (255, 0, 0), 5, 8, -math.pi/2, math.pi)
                            continue
                    
                    start_node = self.ENEMY_PATH[path_idx]
                    end_node = self.ENEMY_PATH[path_idx + 1]
                    enemy['pos'] = (
                        start_node[0] + (end_node[0] - start_node[0]) * enemy['distance_on_segment'],
                        start_node[1] + (end_node[1] - start_node[1]) * enemy['distance_on_segment']
                    )

            # Check for wave end
            if not self.enemies and not self.enemies_to_spawn_this_wave:
                reward += 5.0
                self.score += 50
                self.wave_state = "intermission"
                self.intermission_timer = self.INTERMISSION_DURATION
                if self.wave_number >= self.MAX_WAVES:
                    self.game_won = True
                    self.game_over = True
                else: # Unlock new tower
                    if self.wave_number == 3 and "cannon" not in self.available_tower_types:
                        self.available_tower_types.append("cannon")

        # Update Particles
        for p in self.particles[:]:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['vel'] = (p['vel'][0] * 0.95, p['vel'][1] * 0.95 + 0.1) # gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

        # --- 3. Check Termination ---
        terminated = False
        if self.base_health <= 0:
            self.game_over = True
            reward = -100.0
        if self.game_won:
            reward = 100.0
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        if self.game_over:
            terminated = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES: return

        self.wave_state = "active"
        num_enemies = 2 + self.wave_number
        health_multiplier = 1 + (self.wave_number - 1) * 0.05
        speed_multiplier = 1 + (self.wave_number - 1) * 0.05
        
        self.enemies_to_spawn_this_wave = []
        for _ in range(num_enemies):
            start_pos = self.ENEMY_PATH[0]
            enemy = {
                "health": 100 * health_multiplier,
                "max_health": 100 * health_multiplier,
                "speed": 0.02 * speed_multiplier,
                "pos": (float(start_pos[0]), float(start_pos[1])),
                "path_index": 0,
                "distance_on_segment": 0.0,
                "damage": 1,
                "bounty": 10 + self.wave_number,
            }
            self.enemies_to_spawn_this_wave.append(enemy)
        self.spawn_timer = 0

    def _handle_build_or_upgrade(self):
        slot_pos = self.TOWER_SLOTS[self.cursor_index]
        tower_at_slot = next((t for t in self.towers if t['pos'] == slot_pos), None)
        tower_type_key = self.available_tower_types[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[tower_type_key]

        if tower_at_slot: # Upgrade
            if tower_at_slot['type'] == tower_type_key and tower_at_slot['level'] < len(spec['cost']):
                cost = spec['cost'][tower_at_slot['level']]
                if self.resources >= cost:
                    self.resources -= cost
                    tower_at_slot['level'] += 1
                    # sfx: upgrade.wav
                    self._create_particles(self._grid_to_screen(slot_pos), 20, self.COLOR_RESOURCE, 2, 4)
                    return 0.5 # Small reward for upgrading
        else: # Build
            cost = spec['cost'][0]
            if self.resources >= cost:
                self.resources -= cost
                new_tower = {
                    "pos": slot_pos,
                    "type": tower_type_key,
                    "level": 1,
                    "cooldown_timer": 0,
                }
                self.towers.append(new_tower)
                # sfx: build.wav
                self._create_particles(self._grid_to_screen(slot_pos), 20, (200, 200, 255), 2, 4)
                return 0.5 # Small reward for building
        return 0.0

    def _find_target(self, tower):
        spec = self.TOWER_SPECS[tower['type']]
        tower_range = spec['range'][tower['level'] - 1]
        
        potential_targets = []
        for enemy in self.enemies:
            dist = math.hypot(tower['pos'][0] - enemy['pos'][0], tower['pos'][1] - enemy['pos'][1])
            if dist <= tower_range:
                enemy_path_dist = enemy['path_index'] + enemy['distance_on_segment']
                potential_targets.append((enemy_path_dist, enemy))
        
        if not potential_targets:
            return None
        
        # Target enemy furthest along the path
        potential_targets.sort(key=lambda x: x[0], reverse=True)
        return potential_targets[0][1]

    def _fire_projectile(self, tower, target):
        spec = self.TOWER_SPECS[tower['type']]
        level_idx = tower['level'] - 1
        tower['cooldown_timer'] = spec['cooldown'][level_idx]
        
        start_pos = self._grid_to_screen(tower['pos'])
        target_pos = self._grid_to_screen(target['pos'])
        
        angle = math.atan2(target_pos[1] - start_pos[1], target_pos[0] - start_pos[0])
        speed = spec['proj_speed']
        
        new_proj = {
            "pos": start_pos,
            "vel": (math.cos(angle) * speed, math.sin(angle) * speed),
            "damage": spec['damage'][level_idx],
            "lifespan": 60,
            "target": target,
            "type": tower['type'],
        }
        self.projectiles.append(new_proj)

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
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
        }

    def _grid_to_screen(self, pos):
        x, y = pos
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_W / 2
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_H / 2
        return int(screen_x), int(screen_y)

    def _draw_iso_rect(self, pos, color, size_w=1, size_h=1, border_color=None, border_width=2):
        x, y = self._grid_to_screen(pos)
        points = [
            (x, y - (size_h - 1) * self.TILE_H),
            (x + size_w * self.TILE_W / 2, y + (size_w / 2 - (size_h - 1)) * self.TILE_H),
            (x, y + (size_w + size_h - 1) * self.TILE_H / 2),
            (x - size_w * self.TILE_W / 2, y + (size_w / 2 - (size_h - 1)) * self.TILE_H)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        if border_color:
            pygame.draw.aalines(self.screen, border_color, True, points, border_width)

    def _create_particles(self, pos, count, color, min_speed, max_speed, angle_min=0, angle_max=2*math.pi):
        for _ in range(count):
            angle = self.rng.uniform(angle_min, angle_max)
            speed = self.rng.uniform(min_speed, max_speed)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": self.rng.integers(15, 30),
                "color": color,
            })

    def _render_game(self):
        # --- Draw Path and Slots ---
        for i in range(len(self.ENEMY_PATH) - 1):
            p1 = self._grid_to_screen(self.ENEMY_PATH[i])
            p2 = self._grid_to_screen(self.ENEMY_PATH[i+1])
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, self.TILE_H * 2)
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, p1, p2, self.TILE_H * 2 + 4)
        for p in self.ENEMY_PATH:
            self._draw_iso_rect(p, self.COLOR_PATH, border_color=self.COLOR_PATH_BORDER)

        for slot_pos in self.TOWER_SLOTS:
            self._draw_iso_rect(slot_pos, self.COLOR_SLOT)

        # --- Sort and Draw Dynamic Objects ---
        drawable_objects = []
        for enemy in self.enemies:
            drawable_objects.append(("enemy", enemy))
        for tower in self.towers:
            drawable_objects.append(("tower", tower))
        
        drawable_objects.sort(key=lambda item: self._grid_to_screen(item[1]['pos'])[1])

        for obj_type, obj in drawable_objects:
            if obj_type == "enemy":
                self._render_enemy(obj)
            elif obj_type == "tower":
                self._render_tower(obj)

        # --- Draw Base ---
        self._draw_iso_rect(self.BASE_POS, self.COLOR_BASE, size_w=1.2, size_h=1.2)
        
        # --- Draw Cursor ---
        cursor_pos = self.TOWER_SLOTS[self.cursor_index]
        screen_pos = self._grid_to_screen(cursor_pos)
        tower_at_slot = next((t for t in self.towers if t['pos'] == cursor_pos), None)
        tower_type_key = self.available_tower_types[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[tower_type_key]
        can_afford = False
        if tower_at_slot:
            if tower_at_slot['type'] == tower_type_key and tower_at_slot['level'] < len(spec['cost']):
                can_afford = self.resources >= spec['cost'][tower_at_slot['level']]
        else:
            can_afford = self.resources >= spec['cost'][0]
        
        cursor_color = self.COLOR_CURSOR if can_afford else self.COLOR_CURSOR_INVALID
        pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], self.TILE_W // 2, cursor_color)

        # --- Draw Projectiles ---
        for proj in self.projectiles:
            color = self.TOWER_SPECS[proj['type']]['color'][0]
            pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), 4, color)
            pygame.gfxdraw.aacircle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), 4, (255,255,255))

        # --- Draw Particles ---
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 20))))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 4, 4))
            self.screen.blit(temp_surf, (int(p['pos'][0] - 2), int(p['pos'][1] - 2)))

    def _render_enemy(self, enemy):
        screen_pos = self._grid_to_screen(enemy['pos'])
        pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1] - 8, 8, (200, 50, 50))
        pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1] - 8, 8, (255, 100, 100))
        
        # Health bar
        health_pct = enemy['health'] / enemy['max_health']
        bar_w = 20
        bar_h = 4
        bar_x = screen_pos[0] - bar_w // 2
        bar_y = screen_pos[1] - 25
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (bar_x, bar_y, int(bar_w * health_pct), bar_h))

    def _render_tower(self, tower):
        spec = self.TOWER_SPECS[tower['type']]
        level_idx = tower['level'] - 1
        color = spec['color'][level_idx]
        self._draw_iso_rect(tower['pos'], color, size_w=0.8, size_h=0.8)
        if tower['level'] > 1:
            pos_x, pos_y = self._grid_to_screen(tower['pos'])
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y-8, 4, (255, 255, 0))

    def _render_ui(self):
        # --- Top Bar ---
        pygame.draw.rect(self.screen, (15, 20, 25), (0, 0, self.WIDTH, 30))
        
        # Resources
        res_text = self.font_medium.render(f"${self.resources}", True, self.COLOR_RESOURCE)
        self.screen.blit(res_text, (10, 2))
        
        # Base Health
        health_text = self.font_medium.render(f"Base: {self.base_health}/{self.INITIAL_BASE_HEALTH}", True, self.COLOR_BASE)
        self.screen.blit(health_text, (self.WIDTH - health_text.get_width() - 10, 2))
        
        # Wave Info
        if self.wave_state == "intermission":
            wave_text_str = f"Wave {self.wave_number+1} starts in {math.ceil(self.intermission_timer/self.FPS)}s"
        else:
            wave_text_str = f"Wave {self.wave_number}/{self.MAX_WAVES}"
        wave_text = self.font_medium.render(wave_text_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH//2 - wave_text.get_width()//2, 2))

        # --- Bottom Bar (Tower Info) ---
        pygame.draw.rect(self.screen, (15, 20, 25), (0, self.HEIGHT - 40, self.WIDTH, 40))
        
        tower_type_key = self.available_tower_types[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[tower_type_key]
        
        info_text_str = f"Selected: {spec['name']} | Cost: ${spec['cost'][0]}"
        if len(spec['cost']) > 1:
            info_text_str += f" | Upgrade: ${spec['cost'][1]}"
        
        info_text = self.font_medium.render(info_text_str, True, self.COLOR_TEXT)
        self.screen.blit(info_text, (self.WIDTH//2 - info_text.get_width()//2, self.HEIGHT - 35))

        # --- Game Over/Win Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 50, 50)
            
            text = self.font_large.render(message, True, color)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(text, text_rect)
            
            score_text = self.font_medium.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 30))
            self.screen.blit(score_text, score_rect)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        # --- Human Input ---
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
        
        # --- Environment Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Display ---
        # The environment's observation is already a rendered frame.
        # We just need to display it.
        display_surface = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(display_surface, frame)
        pygame.display.flip()
        
        # --- Control Framerate ---
        env.clock.tick(env.FPS)
        
        if terminated:
            # Wait a bit on the game over screen before closing
            pygame.time.wait(3000)
            running = False

    env.close()