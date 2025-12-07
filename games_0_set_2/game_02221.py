import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move placement cursor. "
        "Shift to cycle tower type. Space to build tower."
    )

    game_description = (
        "Defend your base from waves of invaders by strategically placing "
        "attack towers in this isometric tower defense game."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)

        # --- Colors ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_PATH = (50, 60, 70)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_BASE = (0, 100, 200)
        self.COLOR_BASE_DMG = (200, 50, 50)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_SHADOW = (10, 10, 15)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_WAVE = (150, 150, 180)
        self.COLOR_HEALTH_GREEN = (40, 200, 40)
        self.COLOR_HEALTH_YELLOW = (200, 200, 40)
        self.COLOR_HEALTH_RED = (200, 40, 40)
        self.CURSOR_VALID = (0, 255, 0, 100)
        self.CURSOR_INVALID = (255, 0, 0, 100)

        # --- Game World ---
        self.GRID_SIZE = (15, 11)
        self.TILE_WIDTH_HALF = 22
        self.TILE_HEIGHT_HALF = 11
        self.ORIGIN = (self.screen_width // 2, 80)

        self.path_coords = [
            (0, 5), (1, 5), (2, 5), (3, 5), (3, 4), (3, 3), (3, 2), (4, 2),
            (5, 2), (6, 2), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
            (8, 7), (9, 7), (10, 7), (11, 7), (11, 6), (11, 5), (11, 4),
            (12, 4), (13, 4), (14, 4)
        ]
        
        self.base_pos = (14, 3)
        self.tower_placement_coords = self._get_placement_coords()

        # --- Game Config ---
        self.MAX_STEPS = 5000
        self.MAX_WAVES = 10
        self.INITIAL_HEALTH = 100
        self.INITIAL_GOLD = 150
        self.WAVE_PREP_TIME = 150 # steps between waves

        # --- Tower Config ---
        self.TOWER_SPECS = {
            0: {"name": "Gatling", "cost": 50, "range": 100, "damage": 4, "fire_rate": 10, "color": (0, 150, 255), "proj_speed": 8},
            1: {"name": "Cannon", "cost": 120, "range": 140, "damage": 25, "fire_rate": 60, "color": (255, 100, 0), "proj_speed": 5},
        }

        # --- Enemy Config ---
        self.ENEMY_SPECS = {
            "grunt": {"health": 50, "speed": 0.8, "gold": 5, "size": 6, "color": (200, 60, 60)},
            "runner": {"health": 30, "speed": 1.5, "gold": 8, "size": 5, "color": (255, 120, 120)},
            "tank": {"health": 150, "speed": 0.5, "gold": 15, "size": 8, "color": (150, 40, 40)},
        }
        
        self.WAVE_COMPOSITION = self._generate_waves()

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = 0
        self.gold = 0
        self.current_wave = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = (0, 0)
        self.selected_tower_type = 0
        self.prev_action = [0, 0, 0]

        self.wave_spawning = False
        self.wave_spawn_timer = 0
        self.wave_spawn_idx = 0
        self.inter_wave_timer = 0
        self.reward_events = []
        
        # Initialize state
        self.reset()

    def _get_placement_coords(self):
        placement = set()
        path_set = set(self.path_coords)
        for r in range(self.GRID_SIZE[1]):
            for c in range(self.GRID_SIZE[0]):
                if (c, r) not in path_set and (c,r) != self.base_pos:
                    placement.add((c, r))
        return placement

    def _generate_waves(self):
        waves = []
        for i in range(self.MAX_WAVES):
            wave = []
            num_enemies = 3 + i * 2
            for _ in range(num_enemies):
                spawn_delay = random.randint(15, 45)
                enemy_type = "grunt"
                if i > 2 and random.random() < 0.3 + (i * 0.02):
                    enemy_type = "runner"
                if i > 4 and random.random() < 0.2 + (i * 0.03):
                    enemy_type = "tank"
                wave.append((enemy_type, spawn_delay))
            waves.append(wave)
        return waves
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = self.INITIAL_HEALTH
        self.gold = self.INITIAL_GOLD
        self.current_wave = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = (self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2)
        self.selected_tower_type = 0
        self.prev_action = [0, 0, 0]

        self.wave_spawning = False
        self.wave_spawn_timer = 0
        self.wave_spawn_idx = 0
        self.inter_wave_timer = self.WAVE_PREP_TIME // 2

        self.reward_events = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        self.reward_events = ["timestep"]
        
        is_noop = action[0] == 0 and action[1] == 0 and action[2] == 0
        
        self._handle_input(action)

        if not is_noop:
            self._update_game_state()
            
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()

        self.prev_action = action

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        cx, cy = self.cursor_pos
        if movement == 1: cy -= 1 # Up
        elif movement == 2: cy += 1 # Down
        elif movement == 3: cx -= 1 # Left
        elif movement == 4: cx += 1 # Right
        self.cursor_pos = (
            max(0, min(self.GRID_SIZE[0] - 1, cx)),
            max(0, min(self.GRID_SIZE[1] - 1, cy))
        )

        # --- Cycle Tower (on press) ---
        shift_pressed = shift_held and not self.prev_action[2] == 1
        if shift_pressed:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: UI_cycle

        # --- Place Tower (on press) ---
        space_pressed = space_held and not self.prev_action[1] == 1
        if space_pressed:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            is_valid_spot = self.cursor_pos in self.tower_placement_coords
            is_occupied = any(t['grid_pos'] == self.cursor_pos for t in self.towers)
            
            if is_valid_spot and not is_occupied and self.gold >= spec['cost']:
                self.gold -= spec['cost']
                self.towers.append({
                    "grid_pos": self.cursor_pos,
                    "type": self.selected_tower_type,
                    "cooldown": 0,
                    "target": None
                })
                self._create_particles(self._iso_to_screen(*self.cursor_pos), 20, spec['color'])
                # sfx: tower_place
            else:
                # sfx: UI_error
                pass

    def _update_game_state(self):
        self._update_waves()
        self._update_towers()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()

    def _update_waves(self):
        # If wave is over and not all waves are done
        if not self.wave_spawning and not self.enemies and self.current_wave < self.MAX_WAVES:
            self.inter_wave_timer -= 1
            if self.inter_wave_timer <= 0:
                self.current_wave += 1
                self.wave_spawning = True
                self.wave_spawn_idx = 0
                if self.current_wave > 1:
                    self.reward_events.append("wave_survived")
                    # sfx: wave_complete

        # If currently spawning a wave
        if self.wave_spawning:
            self.wave_spawn_timer -= 1
            if self.wave_spawn_timer <= 0:
                wave_data = self.WAVE_COMPOSITION[self.current_wave - 1]
                if self.wave_spawn_idx < len(wave_data):
                    enemy_type, spawn_delay = wave_data[self.wave_spawn_idx]
                    self._spawn_enemy(enemy_type)
                    self.wave_spawn_timer = spawn_delay
                    self.wave_spawn_idx += 1
                else:
                    self.wave_spawning = False

    def _spawn_enemy(self, enemy_type):
        spec = self.ENEMY_SPECS[enemy_type]
        scale = 1 + (self.current_wave - 1) * 0.05
        self.enemies.append({
            "type": enemy_type,
            "path_idx": 0,
            "sub_pos": 0.0,
            "health": spec["health"] * scale,
            "max_health": spec["health"] * scale,
            "speed": spec["speed"],
            "id": self.steps + random.random()
        })
    
    def _update_enemies(self):
        for enemy in self.enemies[:]:
            enemy["sub_pos"] += enemy["speed"]
            while enemy["sub_pos"] >= 1.0 and enemy["path_idx"] < len(self.path_coords) - 1:
                enemy["sub_pos"] -= 1.0
                enemy["path_idx"] += 1
            
            if enemy["path_idx"] >= len(self.path_coords) - 1:
                self.base_health -= 10
                self.reward_events.append("base_hit")
                self._create_particles(self._iso_to_screen(*self.base_pos), 30, self.COLOR_BASE_DMG)
                self.enemies.remove(enemy)
                # sfx: base_damage
                continue

            if enemy["health"] <= 0:
                spec = self.ENEMY_SPECS[enemy["type"]]
                self.gold += spec["gold"]
                self.reward_events.append("enemy_defeated")
                pos = self._get_enemy_screen_pos(enemy)
                self._create_particles(pos, 15, spec["color"])
                self.enemies.remove(enemy)
                # sfx: enemy_explode
                continue

    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue

            # Invalidate target if it's dead or out of range
            if tower.get("target"):
                target_enemy = next((e for e in self.enemies if e["id"] == tower["target"]), None)
                if not target_enemy:
                    tower["target"] = None
                else:
                    tower_pos = self._iso_to_screen(*tower["grid_pos"])
                    enemy_pos = self._get_enemy_screen_pos(target_enemy)
                    if math.hypot(tower_pos[0] - enemy_pos[0], tower_pos[1] - enemy_pos[1]) > spec["range"]:
                        tower["target"] = None

            # Find new target if needed
            if not tower.get("target"):
                tower_pos = self._iso_to_screen(*tower["grid_pos"])
                potential_targets = []
                for enemy in self.enemies:
                    enemy_pos = self._get_enemy_screen_pos(enemy)
                    dist = math.hypot(tower_pos[0] - enemy_pos[0], tower_pos[1] - enemy_pos[1])
                    if dist <= spec["range"]:
                        potential_targets.append((dist, enemy))
                
                if potential_targets:
                    # Target the enemy furthest along the path
                    best_target = max(potential_targets, key=lambda t: (t[1]['path_idx'], t[1]['sub_pos']))[1]
                    tower["target"] = best_target["id"]
            
            # Fire if target is valid
            if tower.get("target"):
                target_enemy = next((e for e in self.enemies if e["id"] == tower["target"]), None)
                if target_enemy:
                    tower["cooldown"] = spec["fire_rate"]
                    tower_pos = self._iso_to_screen(*tower["grid_pos"])
                    self.projectiles.append({
                        "start_pos": tower_pos,
                        "pos": list(tower_pos),
                        "target_id": target_enemy["id"],
                        "spec": spec
                    })
                    # sfx: tower_fire

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            target_enemy = next((e for e in self.enemies if e["id"] == proj["target_id"]), None)
            if not target_enemy:
                self.projectiles.remove(proj)
                continue

            target_pos = self._get_enemy_screen_pos(target_enemy)
            direction = [target_pos[0] - proj["pos"][0], target_pos[1] - proj["pos"][1]]
            dist = math.hypot(*direction)

            if dist < proj["spec"]["proj_speed"]:
                target_enemy["health"] -= proj["spec"]["damage"]
                self.reward_events.append("enemy_damaged")
                self._create_particles(target_pos, 5, proj["spec"]["color"])
                self.projectiles.remove(proj)
                # sfx: enemy_hit
                continue

            direction[0] /= dist
            direction[1] /= dist
            
            proj["pos"][0] += direction[0] * proj["spec"]["proj_speed"]
            proj["pos"][1] += direction[1] * proj["spec"]["proj_speed"]

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _calculate_reward(self):
        reward = 0
        for event in self.reward_events:
            if event == "timestep": reward -= 0.001
            elif event == "enemy_damaged": reward += 0.01
            elif event == "enemy_defeated": reward += 1.0
            elif event == "wave_survived": reward += 5.0
            elif event == "base_hit": reward -= 2.0
        
        if self.game_over:
            reward += -100.0
        if self.game_won:
            reward += 100.0
            
        return reward

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.current_wave == self.MAX_WAVES and not self.enemies and not self.wave_spawning:
            self.game_won = True
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

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "gold": self.gold,
            "wave": f"{self.current_wave}/{self.MAX_WAVES}",
            "enemies": len(self.enemies),
            "towers": len(self.towers)
        }

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.ORIGIN[0] + (grid_x - grid_y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN[1] + (grid_x + grid_y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _render_iso_tile(self, surface, color, grid_pos, height_offset=0):
        x, y = self._iso_to_screen(grid_pos[0], grid_pos[1])
        y -= height_offset
        points = [
            (x, y - self.TILE_HEIGHT_HALF),
            (x + self.TILE_WIDTH_HALF, y),
            (x, y + self.TILE_HEIGHT_HALF),
            (x - self.TILE_WIDTH_HALF, y)
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)
    
    def _render_iso_cube(self, surface, color, grid_pos, size):
        x, y = self._iso_to_screen(*grid_pos)
        
        top_points = [
            (x, y - size),
            (x + self.TILE_WIDTH_HALF * size/self.TILE_HEIGHT_HALF * 0.5, y - size/2),
            (x, y),
            (x - self.TILE_WIDTH_HALF * size/self.TILE_HEIGHT_HALF * 0.5, y - size/2)
        ]
        
        side_color_1 = tuple(max(0, c-30) for c in color)
        side_color_2 = tuple(max(0, c-60) for c in color)

        # Left side
        pygame.draw.polygon(surface, side_color_2, [top_points[3], top_points[2], (top_points[2][0], top_points[2][1]+size), (top_points[3][0], top_points[3][1]+size)])
        # Right side
        pygame.draw.polygon(surface, side_color_1, [top_points[2], top_points[1], (top_points[1][0], top_points[1][1]+size), (top_points[2][0], top_points[2][1]+size)])
        # Top
        pygame.draw.polygon(surface, color, top_points)

    def _render_text_with_shadow(self, text, font, color, pos):
        shadow_pos = (pos[0] + 2, pos[1] + 2)
        text_surf = font.render(text, True, self.COLOR_SHADOW)
        self.screen.blit(text_surf, shadow_pos)
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _get_enemy_screen_pos(self, enemy):
        p1 = self.path_coords[enemy["path_idx"]]
        p2 = self.path_coords[min(len(self.path_coords) - 1, enemy["path_idx"] + 1)]
        
        s_p1 = self._iso_to_screen(*p1)
        s_p2 = self._iso_to_screen(*p2)

        x = s_p1[0] + (s_p2[0] - s_p1[0]) * enemy["sub_pos"]
        y = s_p1[1] + (s_p2[1] - s_p1[1]) * enemy["sub_pos"]
        return int(x), int(y)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed - 2],
                "life": random.randint(15, 30),
                "color": color,
                "size": random.randint(2, 4)
            })

    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_SIZE[1]):
            for c in range(self.GRID_SIZE[0]):
                self._render_iso_tile(self.screen, self.COLOR_GRID, (c, r))
        
        # Draw path
        for pos in self.path_coords:
            self._render_iso_tile(self.screen, self.COLOR_PATH, pos)
        
        # Draw base
        base_color = self.COLOR_BASE if self.base_health > 30 else self.COLOR_BASE_DMG
        self._render_iso_cube(self.screen, base_color, self.base_pos, 12)

        # Draw towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            self._render_iso_cube(self.screen, spec['color'], tower['grid_pos'], 8)
            # Draw range indicator when placing or maybe always for selected tower
            if self.cursor_pos == tower['grid_pos']:
                pos = self._iso_to_screen(*tower['grid_pos'])
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], spec['range'], (*spec['color'], 80))


        # Draw cursor
        is_valid = self.cursor_pos in self.tower_placement_coords and not any(t['grid_pos'] == self.cursor_pos for t in self.towers)
        cursor_color = self.CURSOR_VALID if is_valid else self.CURSOR_INVALID
        self._render_iso_tile(self.screen, cursor_color, self.cursor_pos)
        tower_spec = self.TOWER_SPECS[self.selected_tower_type]
        pos = self._iso_to_screen(*self.cursor_pos)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], tower_spec['range'], cursor_color)

        # Draw enemies
        for enemy in sorted(self.enemies, key=lambda e: self._get_enemy_screen_pos(e)[1]):
            spec = self.ENEMY_SPECS[enemy["type"]]
            pos = self._get_enemy_screen_pos(enemy)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], spec["size"], spec["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], spec["size"], spec["color"])
            
            # Health bar
            health_pct = enemy["health"] / enemy["max_health"]
            bar_width = 20
            bar_height = 4
            bar_pos = (pos[0] - bar_width // 2, pos[1] - spec["size"] - 10)
            
            health_color = self.COLOR_HEALTH_GREEN
            if health_pct < 0.6: health_color = self.COLOR_HEALTH_YELLOW
            if health_pct < 0.3: health_color = self.COLOR_HEALTH_RED

            pygame.draw.rect(self.screen, (50, 50, 50), (*bar_pos, bar_width, bar_height))
            pygame.draw.rect(self.screen, health_color, (*bar_pos, int(bar_width * health_pct), bar_height))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            color = proj['spec']['color']
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, (*color, 50))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, (*color, 50))
            # Core
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            if len(color) == 4:
                # Create a temporary surface for transparency
                temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.rect(temp_surf, color, (0, 0, p['size'], p['size']))
                self.screen.blit(temp_surf, (pos[0] - p['size']//2, pos[1] - p['size']//2))

    def _render_ui(self):
        # Top Bar
        bar_surf = pygame.Surface((self.screen_width, 40), pygame.SRCALPHA)
        bar_surf.fill((10, 15, 20, 200))
        self.screen.blit(bar_surf, (0, 0))

        # Base Health
        self._render_text_with_shadow(f"Base: {max(0, int(self.base_health))}", self.font_medium, self.COLOR_TEXT, (10, 8))
        # Gold
        self._render_text_with_shadow(f"Gold: {self.gold}", self.font_medium, self.COLOR_GOLD, (180, 8))
        # Wave
        wave_text = f"Wave: {self.current_wave}/{self.MAX_WAVES}"
        if not self.enemies and not self.wave_spawning and self.current_wave < self.MAX_WAVES:
            wave_text = f"Next wave in {self.inter_wave_timer // 30 + 1}s"
        self._render_text_with_shadow(wave_text, self.font_medium, self.COLOR_WAVE, (320, 8))

        # Selected Tower UI
        spec = self.TOWER_SPECS[self.selected_tower_type]
        cost_color = self.COLOR_GOLD if self.gold >= spec['cost'] else self.COLOR_HEALTH_RED
        self._render_text_with_shadow(f"Build: {spec['name']}", self.font_medium, self.COLOR_TEXT, (480, 8))
        self._render_text_with_shadow(f"Cost: {spec['cost']}", self.font_small, cost_color, (482, 30))

        # Game Over/Win Message
        if self.game_over:
            self._render_text_with_shadow("GAME OVER", self.font_large, self.COLOR_HEALTH_RED, (190, 180))
        elif self.game_won:
            self._render_text_with_shadow("VICTORY!", self.font_large, self.COLOR_GOLD, (220, 180))

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3), f"Obs shape is {test_obs.shape}"
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    env.reset()
    
    # --- Manual Play ---
    # This allows a human to play the game for testing purposes
    # Requires pygame.display to be set up
    
    # Unset the dummy video driver to allow a window to be created
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    action = [0, 0, 0] # no-op, release all

    while running:
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        
        mov = 0 # none
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- ENV RESET ---")

        # --- Step Environment ---
        # For human play, we advance state every frame if an action is held
        # or if it's a no-op (to let the game run).
        # An RL agent would decide when to send no-ops.
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render to Screen ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode Finished. Total Reward: {total_reward}")
            print("Press 'R' to reset.")

        clock.tick(30) # Run at 30 FPS

    pygame.quit()