
# Generated: 2025-08-28T04:26:50.795873
# Source Brief: brief_02309.md
# Brief Index: 2309

        
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
        "Controls: Arrow keys to move selector. Space to place a tower. Shift to cycle tower type."
    )

    game_description = (
        "Defend your base from waves of zombies by strategically placing towers along their path."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    GRID_W, GRID_H = 32, 20
    CELL_W, CELL_H = WIDTH // GRID_W, HEIGHT // GRID_H

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_PATH = (40, 50, 60)
    COLOR_PATH_BORDER = (50, 60, 70)
    COLOR_BASE = (0, 150, 200)
    COLOR_BASE_DAMAGED = (200, 150, 0)
    
    COLOR_TOWER_SPOT = (0, 80, 40, 100)
    COLOR_TOWER_SPOT_HOVER = (0, 150, 80, 200)

    COLOR_ZOMBIE = (200, 50, 50)
    COLOR_ZOMBIE_FLASH = (255, 255, 255)
    
    COLOR_TEXT = (220, 220, 220)
    COLOR_RESOURCES = (255, 200, 0)
    COLOR_HEALTH_BAR = (0, 200, 0)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)

    TOWER_SPECS = [
        {"name": "Gatling", "cost": 25, "range": 80, "cooldown": 10, "damage": 1, "color": (80, 150, 255), "projectile_speed": 10, "type": "basic"},
        {"name": "Frost", "cost": 40, "range": 60, "cooldown": 30, "slow_factor": 0.5, "slow_duration": 60, "color": (255, 255, 100), "type": "slow"},
        {"name": "Cannon", "cost": 75, "range": 100, "cooldown": 60, "damage": 3, "radius": 30, "color": (200, 100, 255), "projectile_speed": 7, "type": "splash"},
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_m = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.path_waypoints = []
        self.tower_spots = []
        self.base_pos = (0, 0)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = 0
        self.resources = 0
        self.wave_number = 0
        self.zombies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.wave_active = False
        self.wave_zombies_to_spawn = 0
        self.wave_spawn_timer = 0
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.last_space_state = 0
        self.last_shift_state = 0
        self.last_move_time = 0

        self.reset()
        self.validate_implementation()

    def _generate_map(self):
        self.path_waypoints = []
        path_y = self.np_random.integers(self.GRID_H // 4, self.GRID_H * 3 // 4)
        path_x = 0
        self.path_waypoints.append((path_x * self.CELL_W, path_y * self.CELL_H))

        while path_x < self.GRID_W - 3:
            path_x += self.np_random.integers(4, 10)
            path_x = min(path_x, self.GRID_W - 3)
            self.path_waypoints.append((path_x * self.CELL_W, path_y * self.CELL_H))
            if path_x < self.GRID_W - 3:
                new_y = self.np_random.integers(self.GRID_H // 4, self.GRID_H * 3 // 4)
                self.path_waypoints.append((path_x * self.CELL_W, new_y * self.CELL_H))
                path_y = new_y
        
        self.base_pos = (self.WIDTH - self.CELL_W * 1.5, path_y * self.CELL_H)
        self.path_waypoints.append(self.base_pos)

        self.tower_spots = []
        for i in range(len(self.path_waypoints) - 1):
            p1 = pygame.Vector2(self.path_waypoints[i])
            p2 = pygame.Vector2(self.path_waypoints[i+1])
            if p1.x == p2.x: # Vertical segment
                side = 1 if p1.x < self.WIDTH / 2 else -1
                for d in [-2, 2]:
                    spot = (int(p1.x + d * self.CELL_W * 1.5), int((p1.y + p2.y) / 2))
                    if 0 < spot[0] < self.WIDTH and 0 < spot[1] < self.HEIGHT:
                        self.tower_spots.append(spot)
            else: # Horizontal segment
                side = 1 if p1.y < self.HEIGHT / 2 else -1
                for d in [-2, 2]:
                    spot = (int((p1.x + p2.x) / 2), int(p1.y + d * self.CELL_H * 1.5))
                    if 0 < spot[0] < self.WIDTH and 0 < spot[1] < self.HEIGHT:
                        self.tower_spots.append(spot)
        self.tower_spots = list(set(self.tower_spots)) # Remove duplicates

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = 100
        self.resources = 80
        self.wave_number = 0
        
        self.zombies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.wave_active = False
        self._generate_map()
        self._start_next_wave()
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_type = 0
        self.last_space_state = 0
        self.last_shift_state = 0
        self.last_move_time = 0
        
        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.wave_number += 1
        self.wave_active = True
        
        if self.wave_number > 10:
            self.game_won = True
            return

        self.wave_zombies_to_spawn = 8 + self.wave_number * 2
        self.zombie_speed = 0.5 + (self.wave_number // 2) * 0.1
        self.zombie_health = 1 + (self.wave_number // 2)
        self.wave_spawn_timer = 0

    def step(self, action):
        reward = 0
        self.game_over = self.base_health <= 0
        
        if not self.game_over and not self.game_won:
            self._handle_input(action)
            self._update_wave()
            self._update_towers()
            reward += self._update_projectiles()
            reward += self._update_zombies()
            self._update_particles()
        
        self.score += reward
        self.steps += 1
        
        terminated = self.base_health <= 0 or self.game_won or self.steps >= 20000

        if terminated:
            if self.game_won:
                reward += 100
            elif self.base_health <= 0:
                reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action
        
        # --- Cursor Movement ---
        now = pygame.time.get_ticks()
        if now - self.last_move_time > 100: # 100ms cooldown for cursor move
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_H - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_W - 1, self.cursor_pos[0] + 1)
            if movement != 0: self.last_move_time = now

        # --- Tower Selection (Shift) ---
        if shift_held and not self.last_shift_state:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: menu_click
        self.last_shift_state = shift_held

        # --- Tower Placement (Space) ---
        if space_held and not self.last_space_state:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.resources >= spec["cost"]:
                cursor_px = (self.cursor_pos[0] * self.CELL_W + self.CELL_W / 2, self.cursor_pos[1] * self.CELL_H + self.CELL_H / 2)
                for spot in self.tower_spots:
                    if math.hypot(cursor_px[0] - spot[0], cursor_px[1] - spot[1]) < self.CELL_W:
                        is_occupied = any(math.hypot(t['pos'][0] - spot[0], t['pos'][1] - spot[1]) < 1 for t in self.towers)
                        if not is_occupied:
                            self.resources -= spec["cost"]
                            self.towers.append({
                                "pos": spot, "spec": spec, "cooldown": 0, "target": None
                            })
                            # sfx: place_tower
                            break
        self.last_space_state = space_held
    
    def _update_wave(self):
        if self.wave_active:
            self.wave_spawn_timer -= 1
            if self.wave_spawn_timer <= 0 and self.wave_zombies_to_spawn > 0:
                spawn_pos = pygame.Vector2(self.path_waypoints[0])
                self.zombies.append({
                    "pos": spawn_pos, "health": self.zombie_health, "max_health": self.zombie_health,
                    "speed": self.zombie_speed, "waypoint_idx": 1, "flash_timer": 0, "slow_timer": 0
                })
                self.wave_zombies_to_spawn -= 1
                self.wave_spawn_timer = 45 # Spawn every 1.5 seconds
                # sfx: zombie_spawn
        elif not self.zombies and not self.game_won:
            self._start_next_wave()

    def _update_towers(self):
        for t in self.towers:
            t["cooldown"] = max(0, t["cooldown"] - 1)
            if t["cooldown"] > 0: continue

            # Find target
            in_range_zombies = []
            for z in self.zombies:
                dist = math.hypot(t["pos"][0] - z["pos"].x, t["pos"][1] - z["pos"].y)
                if dist <= t["spec"]["range"]:
                    in_range_zombies.append(z)
            
            if not in_range_zombies: continue
            target_zombie = min(in_range_zombies, key=lambda z: math.hypot(self.base_pos[0] - z["pos"].x, self.base_pos[1] - z["pos"].y))

            # Fire
            t["cooldown"] = t["spec"]["cooldown"]
            spec = t["spec"]
            # sfx: tower_shoot
            if spec["type"] == "basic":
                self.projectiles.append({"pos": pygame.Vector2(t["pos"]), "target": target_zombie, "spec": spec})
            elif spec["type"] == "slow":
                target_zombie["slow_timer"] = spec["slow_duration"]
                self._create_particles(target_zombie["pos"], spec["color"], 10, 1.5)
            elif spec["type"] == "splash":
                self.projectiles.append({"pos": pygame.Vector2(t["pos"]), "target": target_zombie, "spec": spec})

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            target_pos = p["target"]["pos"]
            direction = (target_pos - p["pos"]).normalize()
            p["pos"] += direction * p["spec"]["projectile_speed"]
            
            if math.hypot(p["pos"].x - target_pos.x, p["pos"].y - target_pos.y) < p["spec"]["projectile_speed"]:
                # Hit
                spec = p["spec"]
                if spec["type"] == "basic":
                    p["target"]["health"] -= spec["damage"]
                    p["target"]["flash_timer"] = 5
                    if p["target"]["health"] <= 0: reward += 0.1
                elif spec["type"] == "splash":
                    for z in self.zombies:
                        if math.hypot(z["pos"].x - target_pos.x, z["pos"].y - target_pos.y) < spec["radius"]:
                            z["health"] -= spec["damage"]
                            z["flash_timer"] = 5
                            if z["health"] <= 0 and p["target"] != z: reward += 0.1 # Avoid double counting
                    if p["target"]["health"] <= 0: reward += 0.1

                self._create_particles(p["pos"], spec["color"], 15, 3)
                self.projectiles.remove(p)
                # sfx: projectile_hit
        return reward

    def _update_zombies(self):
        reward = 0
        for z in self.zombies[:]:
            z["flash_timer"] = max(0, z["flash_timer"] - 1)
            z["slow_timer"] = max(0, z["slow_timer"] - 1)
            
            if z["health"] <= 0:
                self.resources += 5
                self.zombies.remove(z)
                self._create_particles(z["pos"], self.COLOR_ZOMBIE, 20, 2)
                # sfx: zombie_die
                continue

            target_waypoint = pygame.Vector2(self.path_waypoints[z["waypoint_idx"]])
            direction = (target_waypoint - z["pos"])
            
            if direction.length() < z["speed"]:
                z["pos"] = target_waypoint
                if z["waypoint_idx"] < len(self.path_waypoints) - 1:
                    z["waypoint_idx"] += 1
                else: # Reached base
                    self.base_health -= 10
                    self.zombies.remove(z)
                    self._create_particles(self.base_pos, self.COLOR_BASE_DAMAGED, 30, 4)
                    reward -= 1
                    # sfx: base_damage
            else:
                speed = z["speed"] * (z["spec"]["slow_factor"] if z["slow_timer"] > 0 else 1)
                z["pos"] += direction.normalize() * speed
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "life": self.np_random.integers(10, 20), "color": color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Path
        for i in range(len(self.path_waypoints) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, self.path_waypoints[i], self.path_waypoints[i+1], self.CELL_H + 4)
        for i in range(len(self.path_waypoints) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path_waypoints[i], self.path_waypoints[i+1], self.CELL_H)
        
        # Base
        base_rect = pygame.Rect(0, 0, self.CELL_W * 2, self.CELL_H * 2)
        base_rect.center = self.base_pos
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)
        
        # Tower Spots
        cursor_px = (self.cursor_pos[0] * self.CELL_W + self.CELL_W / 2, self.cursor_pos[1] * self.CELL_H + self.CELL_H / 2)
        for spot in self.tower_spots:
            is_occupied = any(math.hypot(t['pos'][0] - spot[0], t['pos'][1] - spot[1]) < 1 for t in self.towers)
            if not is_occupied:
                color = self.COLOR_TOWER_SPOT_HOVER if math.hypot(cursor_px[0] - spot[0], cursor_px[1] - spot[1]) < self.CELL_W else self.COLOR_TOWER_SPOT
                s = pygame.Surface((self.CELL_W, self.CELL_W), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (self.CELL_W // 2, self.CELL_W // 2), self.CELL_W // 2)
                self.screen.blit(s, (spot[0] - self.CELL_W // 2, spot[1] - self.CELL_W // 2))

        # Towers
        for t in self.towers:
            pos, spec = t["pos"], t["spec"]
            # Range indicator
            s = pygame.Surface((spec["range"] * 2, spec["range"] * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 255, 255, 20), (spec["range"], spec["range"]), spec["range"])
            self.screen.blit(s, (pos[0] - spec["range"], pos[1] - spec["range"]))
            # Tower body
            if spec["type"] == "basic":
                points = [ (pos[0], pos[1] - 8), (pos[0] - 7, pos[1] + 5), (pos[0] + 7, pos[1] + 5) ]
                pygame.gfxdraw.aapolygon(self.screen, points, spec["color"])
                pygame.gfxdraw.filled_polygon(self.screen, points, spec["color"])
            else:
                 pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 8, spec["color"])
                 pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 8, spec["color"])
        
        # Projectiles
        for p in self.projectiles:
            pygame.draw.line(self.screen, p["spec"]["color"], p["pos"], p["pos"] - pygame.Vector2(p["spec"]["projectile_speed"],0).rotate_rad( (p["target"]["pos"]-p["pos"]).angle_to(pygame.Vector2(1,0)) * math.pi/180 ), 3)
        
        # Zombies
        for z in self.zombies:
            color = self.COLOR_ZOMBIE_FLASH if z["flash_timer"] > 0 else self.COLOR_ZOMBIE
            slow_color = self.TOWER_SPECS[1]["color"]
            pos = (int(z["pos"].x), int(z["pos"].y))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 6, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, color)
            if z["slow_timer"] > 0:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, slow_color)
            # Health bar
            bar_w = 12
            health_pct = z["health"] / z["max_health"]
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (pos[0] - bar_w/2, pos[1] - 12, bar_w, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (pos[0] - bar_w/2, pos[1] - 12, bar_w * health_pct, 3))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20))
            color = p["color"] + (alpha,)
            s = pygame.Surface((4, 4), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p["pos"].x-2), int(p["pos"].y-2)))

    def _render_ui(self):
        # Cursor
        cursor_rect = (self.cursor_pos[0] * self.CELL_W, self.cursor_pos[1] * self.CELL_H, self.CELL_W, self.CELL_H)
        pygame.draw.rect(self.screen, (255, 255, 255), cursor_rect, 1)

        # UI Panels
        pygame.draw.rect(self.screen, (0, 0, 0, 150), (0, 0, self.WIDTH, 30))
        pygame.draw.rect(self.screen, (0, 0, 0, 150), (0, self.HEIGHT - 40, self.WIDTH, 40))

        # Top UI
        wave_text = self.font_s.render(f"Wave: {self.wave_number}/10", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 7))

        health_text = self.font_s.render(f"Base Health: {max(0, self.base_health)}%", True, self.COLOR_HEALTH_BAR)
        self.screen.blit(health_text, (self.WIDTH - health_text.get_width() - 10, 7))
        
        # Bottom UI
        res_text = self.font_m.render(f"${self.resources}", True, self.COLOR_RESOURCES)
        self.screen.blit(res_text, (10, self.HEIGHT - 32))

        selected_spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_text = self.font_s.render(f"Selected: {selected_spec['name']} (${selected_spec['cost']})", True, selected_spec['color'])
        self.screen.blit(tower_text, (self.WIDTH - tower_text.get_width() - 10, self.HEIGHT - 28))

        # Game Over/Win Message
        if self.game_over:
            text = self.font_l.render("GAME OVER", True, self.COLOR_ZOMBIE)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)
        elif self.game_won:
            text = self.font_l.render("VICTORY!", True, self.COLOR_BASE)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
            "resources": self.resources,
            "zombies": len(self.zombies),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")