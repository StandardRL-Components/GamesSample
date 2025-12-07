
# Generated: 2025-08-28T07:03:29.152450
# Source Brief: brief_03111.md
# Brief Index: 3111

        
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
        "Controls: Arrow keys to select a build location. "
        "Shift to cycle tower type. Space to build."
    )

    game_description = (
        "A minimalist tower defense game. Defend your base from waves of enemies by "
        "strategically placing towers on the grid. Survive all 10 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 18000  # 10 minutes at 30fps
    MAX_WAVES = 10

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_PATH = (40, 40, 60)
    COLOR_PATH_OUTLINE = (60, 60, 80)
    COLOR_BASE = (0, 200, 100)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 30, 45)
    COLOR_HEALTH_BAR = (40, 220, 110)
    COLOR_HEALTH_BAR_BG = (80, 40, 40)
    COLOR_ZONE_VALID = (255, 255, 255, 40)
    COLOR_ZONE_INVALID = (255, 0, 0, 40)
    COLOR_ZONE_SELECTED = (255, 255, 0, 120)

    # Tower Stats
    TOWER_STATS = [
        {
            "name": "Cannon",
            "cost": 25, "range": 80, "damage": 2, "fire_rate": 30, # 1 shot/sec
            "color": (50, 150, 255), "proj_color": (150, 200, 255)
        },
        {
            "name": "Missile",
            "cost": 60, "range": 120, "damage": 8, "fire_rate": 90, # 1 shot/3 sec
            "color": (255, 150, 50), "proj_color": (255, 200, 150)
        },
        {
            "name": "Laser",
            "cost": 40, "range": 100, "damage": 0.2, "fire_rate": 3, # 10 shots/sec
            "color": (200, 50, 255), "proj_color": (255, 150, 255)
        }
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 20, bold=True)

        self._define_game_geometry()
        
        # These will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.resources = 0
        self.current_wave = 0
        self.wave_state = "INTERMISSION" # or "ACTIVE"
        self.wave_timer = 0
        self.spawn_timer = 0
        self.enemies_to_spawn = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.selected_zone_idx = (0, 0)
        self.selected_tower_type = 0
        self.last_movement_action = 0
        self.last_shift_press = False
        self.last_space_press = False
        
        self.reset()
        self.validate_implementation()

    def _define_game_geometry(self):
        self.path_points = [
            pygame.Vector2(100, -20),
            pygame.Vector2(100, 150),
            pygame.Vector2(540, 150),
            pygame.Vector2(540, 300),
            pygame.Vector2(320, 300)
        ]
        self.path_segments = []
        self.path_total_length = 0
        for i in range(len(self.path_points) - 1):
            start, end = self.path_points[i], self.path_points[i+1]
            length = start.distance_to(end)
            direction = (end - start).normalize() if length > 0 else pygame.Vector2(0)
            self.path_segments.append({"start": start, "end": end, "length": length, "dir": direction})
            self.path_total_length += length
        
        self.base_pos = self.path_points[-1]
        self.base_rect = pygame.Rect(self.base_pos.x - 15, self.base_pos.y - 15, 30, 30)
        
        self.placement_zones = []
        grid_size = 4
        x_offset, y_offset = 180, 40
        x_spacing, y_spacing = 100, 80
        for r in range(grid_size):
            row = []
            for c in range(grid_size):
                pos = pygame.Vector2(x_offset + c * x_spacing, y_offset + r * y_spacing)
                is_on_path = False
                for seg in self.path_segments:
                    p_rect = pygame.Rect(pos.x - 20, pos.y - 20, 40, 40)
                    if p_rect.clipline(seg["start"], seg["end"]):
                        is_on_path = True
                        break
                row.append({"pos": pos, "occupied": False, "valid": not is_on_path})
            self.placement_zones.append(row)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 100
        self.resources = 80
        self.current_wave = 0

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        for r in self.placement_zones:
            for z in r:
                z["occupied"] = False

        self.selected_zone_idx = (0, 0)
        self.selected_tower_type = 0
        self.last_movement_action = 0
        self.last_shift_press = False
        self.last_space_press = False
        
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            return
        self.wave_state = "ACTIVE"
        self.enemies_to_spawn = 3 + (self.current_wave - 1) // 2
        self.spawn_timer = 90 # Time until first spawn
        self.wave_timer = 60 # Time between spawns

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.001 # Small time penalty
        self.steps += 1
        
        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game State ---
        self._update_wave_logic()
        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        if self.base_health <= 0:
            reward = -100.0
            terminated = True
            self.game_over = True
        elif self.current_wave > self.MAX_WAVES and not self.enemies:
            reward = 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Cycle tower type (on key press)
        if shift_held and not self.last_shift_press:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_STATS)
        self.last_shift_press = shift_held
        
        # Select placement zone (on key press)
        if movement != 0 and movement != self.last_movement_action:
            r, c = self.selected_zone_idx
            if movement == 1: r = (r - 1 + 4) % 4 # Up
            elif movement == 2: r = (r + 1) % 4 # Down
            elif movement == 3: c = (c - 1 + 4) % 4 # Left
            elif movement == 4: c = (c + 1) % 4 # Right
            self.selected_zone_idx = (r, c)
        self.last_movement_action = movement

        # Place tower (on key press)
        if space_held and not self.last_space_press:
            r, c = self.selected_zone_idx
            zone = self.placement_zones[r][c]
            tower_spec = self.TOWER_STATS[self.selected_tower_type]
            if zone["valid"] and not zone["occupied"] and self.resources >= tower_spec["cost"]:
                self.resources -= tower_spec["cost"]
                zone["occupied"] = True
                self.towers.append({
                    "pos": zone["pos"], "type_idx": self.selected_tower_type,
                    "cooldown": 0, "id": self.np_random.integers(1, 1_000_000)
                })
                # SFX: place_tower
                self._create_particles(zone["pos"], 10, tower_spec["color"], 1, 5, 15)

        self.last_space_press = space_held

    def _update_wave_logic(self):
        if self.wave_state == "ACTIVE":
            if self.enemies_to_spawn > 0:
                self.spawn_timer -= 1
                if self.spawn_timer <= 0:
                    self.spawn_timer = self.wave_timer
                    self.enemies_to_spawn -= 1
                    self.enemies.append({
                        "pos": pygame.Vector2(self.path_points[0]),
                        "dist": 0,
                        "health": 1 + self.current_wave,
                        "max_health": 1 + self.current_wave,
                        "speed": 1.0 + self.current_wave * 0.05,
                        "id": self.np_random.integers(1, 1_000_000)
                    })
            elif not self.enemies: # Wave cleared
                self.wave_state = "INTERMISSION"
                self.wave_timer = 150 # 5 seconds
                if self.current_wave < self.MAX_WAVES:
                    self.resources += 20 + self.current_wave * 5 # Wave clear bonus
        
        elif self.wave_state == "INTERMISSION":
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._start_next_wave()

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] > 0:
                continue

            spec = self.TOWER_STATS[tower['type_idx']]
            target = None
            min_dist = spec['range'] ** 2

            for enemy in self.enemies:
                dist_sq = tower['pos'].distance_squared_to(enemy['pos'])
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = enemy
            
            if target:
                # SFX: tower_shoot
                tower['cooldown'] = spec['fire_rate']
                self.projectiles.append({
                    "pos": pygame.Vector2(tower['pos']),
                    "target_id": target['id'],
                    "type_idx": tower['type_idx'],
                    "speed": 8 if spec['name'] != 'Laser' else 20,
                })
        return reward

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        for proj in self.projectiles:
            target = next((e for e in self.enemies if e['id'] == proj['target_id']), None)
            if not target:
                projectiles_to_remove.append(proj)
                continue
            
            direction = (target['pos'] - proj['pos']).normalize() if target['pos'] != proj['pos'] else pygame.Vector2(0)
            proj['pos'] += direction * proj['speed']
            
            if proj['pos'].distance_squared_to(target['pos']) < 100: # Hit
                # SFX: enemy_hit
                spec = self.TOWER_STATS[proj['type_idx']]
                target['health'] -= spec['damage']
                reward += 0.01 # Small reward for hit
                projectiles_to_remove.append(proj)
                self._create_particles(proj['pos'], 5, spec['proj_color'], 1, 3, 10)
                if target['health'] <= 0:
                    # SFX: enemy_destroy
                    reward += 1.0 # Kill reward
                    self.score += self.current_wave
                    self.resources += 2 + self.current_wave // 2
                    target['health'] = 0 # Mark for removal
                    self._create_particles(target['pos'], 20, self.COLOR_ENEMY, 2, 8, 20)

        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        return reward

    def _update_enemies(self):
        reward = 0
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy['health'] <= 0:
                enemies_to_remove.append(enemy)
                continue
            
            enemy['dist'] += enemy['speed']
            if enemy['dist'] >= self.path_total_length:
                # SFX: base_damage
                self.base_health -= 10
                reward -= 10
                enemies_to_remove.append(enemy)
                self._create_particles(self.base_pos, 30, self.COLOR_BASE_DMG, 5, 10, 25)
                continue
            
            dist_acc = 0
            for seg in self.path_segments:
                if dist_acc + seg['length'] >= enemy['dist']:
                    dist_on_seg = enemy['dist'] - dist_acc
                    enemy['pos'] = seg['start'] + seg['dir'] * dist_on_seg
                    break
                dist_acc += seg['length']
        
        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        return reward

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p['lifetime'] -= 1
            p['radius'] += p['growth']
            p['pos'] += p['vel']
            if p['lifetime'] <= 0:
                particles_to_remove.append(p)
        self.particles = [p for p in self.particles if p not in particles_to_remove]

    def _create_particles(self, pos, count, color, min_speed, max_speed, lifetime):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed / self.FPS
            self.particles.append({
                "pos": pygame.Vector2(pos), "vel": vel, "radius": 1,
                "color": color, "lifetime": lifetime, "growth": 0.5
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Path
        for i in range(len(self.path_points) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH_OUTLINE, self.path_points[i], self.path_points[i+1], 26)
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path_points[i], self.path_points[i+1], 20)

        # Base
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.base_rect)
        pygame.gfxdraw.rectangle(self.screen, self.base_rect, (*self.COLOR_BASE, 150))

        # Placement Zones
        for r_idx, row in enumerate(self.placement_zones):
            for c_idx, zone in enumerate(row):
                color = self.COLOR_ZONE_VALID if zone["valid"] else self.COLOR_ZONE_INVALID
                if (r_idx, c_idx) == self.selected_zone_idx:
                    color = self.COLOR_ZONE_SELECTED
                pygame.gfxdraw.filled_circle(self.screen, int(zone["pos"].x), int(zone["pos"].y), 20, color)
                pygame.gfxdraw.aacircle(self.screen, int(zone["pos"].x), int(zone["pos"].y), 20, (*color[:3], 150))

        # Towers
        for tower in self.towers:
            spec = self.TOWER_STATS[tower['type_idx']]
            pygame.gfxdraw.filled_circle(self.screen, int(tower['pos'].x), int(tower['pos'].y), 12, spec['color'])
            pygame.gfxdraw.aacircle(self.screen, int(tower['pos'].x), int(tower['pos'].y), 12, (255,255,255))

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 7, (255,255,255))
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_w = 14
            bar_x = pos[0] - bar_w/2
            bar_y = pos[1] - 14
            pygame.draw.rect(self.screen, (80,0,0), (bar_x, bar_y, bar_w, 3))
            pygame.draw.rect(self.screen, (0,200,0), (bar_x, bar_y, bar_w * health_ratio, 3))

        # Projectiles
        for proj in self.projectiles:
            spec = self.TOWER_STATS[proj['type_idx']]
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, spec['proj_color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, (255,255,255, 200))

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifetime'] / 20))))
            color = (*p['color'], alpha)
            pos = (int(p['pos'].x), int(p['pos'].y))
            rad = int(p['radius'])
            if rad > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], rad, color)

    def _render_ui(self):
        # UI Background
        ui_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 40, self.SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, (80, 80, 100), (0, self.SCREEN_HEIGHT - 40), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT - 40), 1)

        # Base Health
        health_text = self.font_small.render("BASE HP", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, self.SCREEN_HEIGHT - 32))
        health_bar_rect = pygame.Rect(80, self.SCREEN_HEIGHT - 30, 100, 14)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, health_bar_rect)
        health_ratio = max(0, self.base_health / 100)
        current_health_rect = pygame.Rect(80, self.SCREEN_HEIGHT - 30, 100 * health_ratio, 14)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, current_health_rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, health_bar_rect, 1)

        # Resources
        res_text = self.font_small.render(f"RES: ${self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (200, self.SCREEN_HEIGHT - 32))

        # Wave Info
        wave_str = f"WAVE: {self.current_wave}/{self.MAX_WAVES}"
        if self.wave_state == "INTERMISSION" and self.current_wave < self.MAX_WAVES:
             wave_str += f" (Next in {self.wave_timer // self.FPS}s)"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (300, self.SCREEN_HEIGHT - 32))

        # Selected Tower
        spec = self.TOWER_STATS[self.selected_tower_type]
        tower_text = self.font_small.render(f"BUILD: {spec['name']} (${spec['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(tower_text, (480, self.SCREEN_HEIGHT - 32))
        pygame.gfxdraw.filled_circle(self.screen, 465, self.SCREEN_HEIGHT - 25, 8, spec['color'])
        pygame.gfxdraw.aacircle(self.screen, 465, self.SCREEN_HEIGHT - 25, 8, (255,255,255))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
            "resources": self.resources,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    # Game loop
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            running = False
            pygame.time.wait(2000)

        clock.tick(env.FPS)

    env.close()