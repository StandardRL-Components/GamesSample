
# Generated: 2025-08-28T05:16:47.821198
# Source Brief: brief_05520.md
# Brief Index: 5520

        
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
        "Controls: Use arrow keys to move the placement cursor. Press Shift to cycle through tower types. "
        "Press Space to build the selected tower on the highlighted tile."
    )

    game_description = (
        "An isometric tower defense game. Strategically place towers to defend your base against waves of incoming enemies. "
        "Survive all 10 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 30000  # Approx 16 minutes at 30fps
    FPS = 30
    
    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 50, 60)
    COLOR_PATH = (60, 70, 80)
    COLOR_BASE = (0, 150, 200)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_PROJECTILE = (0, 200, 255)
    COLOR_GOLD = (255, 200, 0)
    COLOR_TEXT = (230, 230, 230)
    COLOR_CURSOR = (255, 255, 255)
    
    # Grid & Isometric Projection
    GRID_WIDTH, GRID_HEIGHT = 22, 14
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 24, 12
    ISO_OFFSET_X, ISO_OFFSET_Y = WIDTH // 2, 80

    # Game Mechanics
    BASE_START_HEALTH = 100
    STARTING_RESOURCES = 250
    TOTAL_WAVES = 10
    WAVE_PREP_TIME = 10 * FPS  # 10 seconds

    TOWER_SPECS = {
        0: {"name": "Cannon", "cost": 100, "range": 3.5, "damage": 10, "fire_rate": 1.0},
        1: {"name": "Sniper", "cost": 150, "range": 6.0, "damage": 35, "fire_rate": 0.33},
    }

    ENEMY_SPECS = {
        "base_health": 20, "base_speed": 0.75, "damage": 10, "bounty": 15
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 14)
        self.font_m = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.path_grid_coords = self._define_path()
        self.path_pixels = [self._project_iso(x, y) for x, y in self.path_grid_coords]
        self.path_set = set(self.path_grid_coords)
        self.base_pos_grid = self.path_grid_coords[-1]
        self.base_pos_pixels = self.path_pixels[-1]

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = self.BASE_START_HEALTH
        self.resources = self.STARTING_RESOURCES
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.current_wave = 0
        self.wave_timer = self.WAVE_PREP_TIME // 2
        self.wave_spawning = False
        self.enemies_to_spawn_in_wave = 0
        self.spawn_timer = 0
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.001  # Small time penalty
        
        if not self.game_over:
            self._handle_input(movement, space_held, shift_held)
            
            self._update_waves()
            self._update_towers()
            
            hit_reward = self._update_projectiles()
            kill_reward, resource_gain = self._update_enemies()
            reward += hit_reward + kill_reward
            self.resources += resource_gain

            self._update_particles()
        
        self.steps += 1
        self.score += reward
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.win:
                reward += 100
            elif self.base_health <= 0:
                reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Cursor movement
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Cycle tower type (on press)
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # Sound: UI_Switch

        # Place tower (on press)
        if space_held and not self.last_space_held:
            self._place_tower()
            
        self.last_space_held, self.last_shift_held = space_held, shift_held

    def _place_tower(self):
        cx, cy = self.cursor_pos
        spec = self.TOWER_SPECS[self.selected_tower_type]
        
        is_on_path = (cx, cy) in self.path_set
        is_occupied = self.grid[cy, cx] != 0
        can_afford = self.resources >= spec['cost']

        if not is_on_path and not is_occupied and can_afford:
            self.resources -= spec['cost']
            self.grid[cy, cx] = 1
            self.towers.append({
                "pos": (cx, cy),
                "type": self.selected_tower_type,
                "cooldown": 0,
                "target": None,
            })
            # Sound: Tower_Place

    def _update_waves(self):
        # If wave is over and not all waves are done, start prep timer
        if not self.wave_spawning and not self.enemies and self.current_wave < self.TOTAL_WAVES:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.current_wave += 1
                self.wave_spawning = True
                self.enemies_to_spawn_in_wave = 3 + self.current_wave * 2
                self.spawn_timer = 0
                # Sound: Wave_Start

        # Spawn enemies during a wave
        if self.wave_spawning:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0 and self.enemies_to_spawn_in_wave > 0:
                self.spawn_timer = self.FPS * 0.5  # Spawn every 0.5s
                self.enemies_to_spawn_in_wave -= 1
                
                health_mult = 1 + (self.current_wave - 1) * 0.10 # 10% per wave after 1st
                speed_mult = 1 + (self.current_wave - 1) * 0.04 # 4% per wave after 1st
                
                self.enemies.append({
                    "pos": list(self.path_pixels[0]),
                    "path_index": 1,
                    "sub_pos": 0.0,
                    "health": self.ENEMY_SPECS["base_health"] * health_mult,
                    "max_health": self.ENEMY_SPECS["base_health"] * health_mult,
                    "speed": self.ENEMY_SPECS["base_speed"] * speed_mult,
                })
            
            if self.enemies_to_spawn_in_wave == 0:
                self.wave_spawning = False
                self.wave_timer = self.WAVE_PREP_TIME

    def _update_towers(self):
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            spec = self.TOWER_SPECS[tower["type"]]

            # Find a new target if needed
            if tower["target"] is None or tower["target"] not in self.enemies:
                tower["target"] = None
                in_range_enemies = []
                for enemy in self.enemies:
                    dist = math.hypot(tower["pos"][0] - enemy["pos"][0] / self.TILE_WIDTH_HALF / 2, 
                                     tower["pos"][1] - (enemy["pos"][1] - self.ISO_OFFSET_Y) / self.TILE_HEIGHT_HALF / 2)
                    
                    dist_px = math.hypot(self._project_iso(*tower["pos"])[0] - enemy["pos"][0],
                                         self._project_iso(*tower["pos"])[1] - enemy["pos"][1])
                    if dist_px <= spec["range"] * self.TILE_WIDTH_HALF:
                        in_range_enemies.append(enemy)

                if in_range_enemies:
                    # Target enemy closest to the end of the path
                    tower["target"] = max(in_range_enemies, key=lambda e: (e["path_index"], e["sub_pos"]))

            # Fire if ready and has a target
            if tower["cooldown"] == 0 and tower["target"] is not None:
                tower["cooldown"] = self.FPS / spec["fire_rate"]
                start_pos = self._project_iso(*tower["pos"])
                self.projectiles.append({
                    "pos": list(start_pos),
                    "start_pos": list(start_pos),
                    "target": tower["target"],
                    "damage": spec["damage"],
                    "speed": 15,
                })
                # Sound: Tower_Fire

    def _update_projectiles(self):
        hit_reward = 0
        for proj in self.projectiles[:]:
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_pos = proj["target"]["pos"]
            dx = target_pos[0] - proj["pos"][0]
            dy = target_pos[1] - proj["pos"][1]
            dist = math.hypot(dx, dy)

            if dist < proj["speed"]:
                proj["target"]["health"] -= proj["damage"]
                hit_reward += 0.1
                self.projectiles.remove(proj)
                self._create_particles(target_pos, 5, self.COLOR_PROJECTILE, 1, 3, 10)
                # Sound: Enemy_Hit
            else:
                proj["pos"][0] += (dx / dist) * proj["speed"]
                proj["pos"][1] += (dy / dist) * proj["speed"]
        return hit_reward

    def _update_enemies(self):
        kill_reward = 0
        resource_gain = 0
        for enemy in self.enemies[:]:
            if enemy["health"] <= 0:
                kill_reward += 1.0
                resource_gain += self.ENEMY_SPECS["bounty"]
                self._create_particles(enemy["pos"], 15, self.COLOR_ENEMY, 2, 5, 20)
                self.enemies.remove(enemy)
                # Sound: Enemy_Explode
                continue

            if enemy["path_index"] >= len(self.path_pixels):
                self.base_health -= self.ENEMY_SPECS["damage"]
                self.enemies.remove(enemy)
                # Sound: Base_Damage
                continue

            start_node = self.path_pixels[enemy["path_index"] - 1]
            end_node = self.path_pixels[enemy["path_index"]]
            
            path_dist = math.hypot(end_node[0] - start_node[0], end_node[1] - start_node[1])
            if path_dist == 0: path_dist = 1 # Avoid division by zero
            
            enemy["sub_pos"] += enemy["speed"] / path_dist
            
            if enemy["sub_pos"] >= 1.0:
                enemy["path_index"] += 1
                enemy["sub_pos"] = 0
                if enemy["path_index"] >= len(self.path_pixels):
                    continue

            start_node = self.path_pixels[enemy["path_index"] - 1]
            end_node = self.path_pixels[enemy["path_index"]]
            
            enemy["pos"][0] = start_node[0] + (end_node[0] - start_node[0]) * enemy["sub_pos"]
            enemy["pos"][1] = start_node[1] + (end_node[1] - start_node[1]) * enemy["sub_pos"]

        return kill_reward, resource_gain

    def _update_particles(self):
        for p in self.particles[:]:
            p["lifetime"] -= 1
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["radius"] *= 0.95
            if p["lifetime"] <= 0 or p["radius"] < 0.5:
                self.particles.remove(p)

    def _check_termination(self):
        self.win = self.current_wave == self.TOTAL_WAVES and not self.enemies and not self.wave_spawning
        return self.base_health <= 0 or self.win or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_path()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave,
        }
    
    # --- Rendering Methods ---
    
    def _render_grid_and_path(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                p1 = self._project_iso(x, y)
                p2 = self._project_iso(x + 1, y)
                p3 = self._project_iso(x + 1, y + 1)
                p4 = self._project_iso(x, y + 1)
                pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p4)

        if len(self.path_pixels) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_pixels, width=self.TILE_HEIGHT_HALF * 2)

    def _render_base(self):
        px, py = self.base_pos_pixels
        pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), 18, (*self.COLOR_BASE, 100))
        pygame.gfxdraw.aacircle(self.screen, int(px), int(py), 18, self.COLOR_BASE)
        pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), 12, self.COLOR_BASE)
        
    def _render_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            px, py = self._project_iso(*tower["pos"])
            
            # Base
            pygame.draw.polygon(self.screen, (80, 90, 100), [
                (px, py),
                (px + self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF),
                (px, py + self.TILE_HEIGHT_HALF * 2),
                (px - self.TILE_WIDTH_HALF, py + self.TILE_HEIGHT_HALF)
            ])
            
            # Turret
            turret_color = (120, 130, 140)
            turret_length = 15
            turret_end_x, turret_end_y = px, py - turret_length
            if tower["target"]:
                t_dx = tower["target"]["pos"][0] - px
                t_dy = tower["target"]["pos"][1] - py
                t_dist = math.hypot(t_dx, t_dy)
                if t_dist > 0:
                    turret_end_x = px + (t_dx / t_dist) * turret_length
                    turret_end_y = py + (t_dy / t_dist) * turret_length

            pygame.draw.line(self.screen, turret_color, (px, py), (turret_end_x, turret_end_y), 5)
            pygame.draw.circle(self.screen, turret_color, (px, py), 6)

    def _render_enemies(self):
        for enemy in self.enemies:
            px, py = int(enemy["pos"][0]), int(enemy["pos"][1])
            size = 8
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, (px, py), size)
            
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            bar_w = 20
            bar_h = 4
            bar_x = px - bar_w // 2
            bar_y = py - size - 10
            pygame.draw.rect(self.screen, (80, 0, 0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (0, 200, 0), (bar_x, bar_y, bar_w * health_ratio, bar_h))

    def _render_projectiles(self):
        for proj in self.projectiles:
            px, py = int(proj["pos"][0]), int(proj["pos"][1])
            pygame.gfxdraw.filled_circle(self.screen, px, py, 4, (*self.COLOR_PROJECTILE, 100))
            pygame.gfxdraw.aacircle(self.screen, px, py, 4, self.COLOR_PROJECTILE)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / p["max_lifetime"]))
            color = (*p["color"], alpha)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), max(0, int(p["radius"])), color)

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        spec = self.TOWER_SPECS[self.selected_tower_type]
        
        is_on_path = (cx, cy) in self.path_set
        is_occupied = self.grid[cy, cy] != 0
        can_afford = self.resources >= spec['cost']
        is_valid = not is_on_path and not is_occupied

        color = (0, 255, 0) if is_valid and can_afford else (255, 0, 0)
        
        points = [
            self._project_iso(cx, cy),
            self._project_iso(cx + 1, cy),
            self._project_iso(cx + 1, cy + 1),
            self._project_iso(cx, cy + 1),
        ]
        pygame.draw.polygon(self.screen, (*color, 60), points)
        pygame.draw.lines(self.screen, color, True, points, 2)
        
        # Range indicator
        if is_valid:
            px, py = self._project_iso(cx + 0.5, cy + 0.5)
            radius = int(spec['range'] * self.TILE_WIDTH_HALF)
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius, (*color, 100))

    def _render_ui(self):
        # Base Health
        pygame.draw.rect(self.screen, (40, 40, 40), (10, 10, 200, 20))
        health_ratio = max(0, self.base_health / self.BASE_START_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_BASE, (10, 10, 200 * health_ratio, 20))
        health_text = self.font_m.render(f"BASE: {int(self.base_health)}/{self.BASE_START_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))
        
        # Resources
        res_text = self.font_m.render(f"$: {self.resources}", True, self.COLOR_GOLD)
        self.screen.blit(res_text, (230, 12))
        
        # Wave Info
        if self.current_wave > 0 and self.current_wave <= self.TOTAL_WAVES:
            wave_str = f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}"
        else:
            wave_str = f"NEXT WAVE IN: {self.wave_timer / self.FPS:.1f}s"
        wave_text = self.font_m.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 12))
        
        # Tower Info Panel
        spec = self.TOWER_SPECS[self.selected_tower_type]
        panel_y = self.HEIGHT - 70
        pygame.draw.rect(self.screen, (40, 40, 40, 200), (10, panel_y, 250, 60))
        name_text = self.font_m.render(spec['name'], True, self.COLOR_TEXT)
        cost_text = self.font_s.render(f"Cost: {spec['cost']}", True, self.COLOR_GOLD)
        dmg_text = self.font_s.render(f"Dmg: {spec['damage']}", True, self.COLOR_TEXT)
        rate_text = self.font_s.render(f"Rate: {spec['fire_rate']:.2f}/s", True, self.COLOR_TEXT)
        range_text = self.font_s.render(f"Range: {spec['range']:.1f}", True, self.COLOR_TEXT)
        
        self.screen.blit(name_text, (20, panel_y + 5))
        self.screen.blit(cost_text, (150, panel_y + 8))
        self.screen.blit(dmg_text, (20, panel_y + 30))
        self.screen.blit(rate_text, (90, panel_y + 30))
        self.screen.blit(range_text, (180, panel_y + 30))
        
        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            end_text = self.font_l.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))

    # --- Helper Methods ---
    
    def _project_iso(self, x, y):
        px = self.ISO_OFFSET_X + (x - y) * self.TILE_WIDTH_HALF
        py = self.ISO_OFFSET_Y + (x + y) * self.TILE_HEIGHT_HALF
        return px, py
        
    def _define_path(self):
        return [
            (-1, 5), (2, 5), (2, 2), (6, 2), (6, 8), 
            (10, 8), (10, 4), (15, 4), (15, 10), (19, 10), (19, 7), (22, 7)
        ]

    def _create_particles(self, pos, count, color, min_speed, max_speed, lifetime):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(min_speed, max_speed)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "radius": random.uniform(3, 7),
                "color": color,
                "lifetime": lifetime,
                "max_lifetime": lifetime,
            })

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Use a Pygame window to display the environment
    pygame.init()
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    action = np.array([0, 0, 0]) # No-op
    
    print("\n" + "="*30)
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Human Controls ---
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = np.array([movement, space, shift])
        # --- End Human Controls ---

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            # print(f"Step: {info['steps']}, Reward: {reward:.3f}, Score: {info['score']:.3f}, "
            #       f"Health: {info['base_health']}, Res: {info['resources']}, Wave: {info['wave']}")
            pass

        # Convert the observation (H, W, C) back to a Pygame surface
        # The observation is (H, W, C) but pygame wants (W, H) surface and transposes it
        # Our obs is already W,H,C from surfarray, but transposed to H,W,C for gym.
        # So we need to transpose it back.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']:.2f}")
    env.close()
    pygame.quit()