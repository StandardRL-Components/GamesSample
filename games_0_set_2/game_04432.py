
# Generated: 2025-08-28T02:24:10.958359
# Source Brief: brief_04432.md
# Brief Index: 4432

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import heapq
from collections import defaultdict
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor. Space to place selected tower. Shift to cycle tower types."
    )

    game_description = (
        "Defend your base from waves of geometric invaders by strategically placing towers. "
        "Towers alter the enemy path, so place them wisely!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    CELL_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE
    MAX_STEPS = 5000
    MAX_WAVES = 20
    INTERMISSION_TIME = 150 # 5 seconds at 30fps

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 35, 50)
    COLOR_PATH = (40, 50, 70)
    COLOR_BASE = (0, 200, 100)
    COLOR_SPAWN = (200, 50, 50)
    COLOR_ENEMY = (255, 70, 70)
    COLOR_ENEMY_HEALTH_BG = (80, 80, 80)
    COLOR_ENEMY_HEALTH = (200, 200, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR_VALID = (100, 255, 100, 150)
    COLOR_CURSOR_INVALID = (255, 100, 100, 150)

    TOWER_SPECS = {
        0: {"name": "Gatling", "cost": 50, "range": 80, "damage": 4, "fire_rate": 8, "color": (255, 255, 0)},
        1: {"name": "Cannon", "cost": 125, "range": 150, "damage": 25, "fire_rate": 1, "color": (0, 200, 255)},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        self.spawn_cell = (1, self.GRID_HEIGHT // 2)
        self.base_cell = (self.GRID_WIDTH - 2, self.GRID_HEIGHT // 2)

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = 100
        self.resources = 150
        self.current_wave = 1
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.path_grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.path_grid[self.base_cell] = 0 # Ensure base is always pathable to

        self.cursor_cell = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.wave_spawning = False
        self.enemies_to_spawn = []
        self.spawn_timer = 0
        self.intermission_timer = self.INTERMISSION_TIME // 2

        self._prepare_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        base_took_damage_this_step = False
        
        self._handle_input(action)

        # --- Game Logic Update ---
        if not self.game_over:
            self.steps += 1
            
            self._update_wave_spawning()
            
            new_projectiles = self._update_towers()
            self.projectiles.extend(new_projectiles)
            
            killed_enemies = self._update_projectiles()
            if killed_enemies > 0:
                reward += 0.1 * killed_enemies
                self.score += 10 * killed_enemies
                self.resources += 5 * killed_enemies

            damage_to_base = self._update_enemies()
            if damage_to_base > 0:
                self.base_health -= damage_to_base
                base_took_damage_this_step = True

            self._update_particles()
            
            if base_took_damage_this_step:
                 reward -= 0.01

            # Check for wave completion
            if not self.wave_spawning and not self.enemies and self.intermission_timer <= 0:
                if self.current_wave <= self.MAX_WAVES:
                    reward += 1.0
                    self.score += 100
                    self.resources += 75 + self.current_wave * 5
                    self.current_wave += 1
                    self.intermission_timer = self.INTERMISSION_TIME
                    if self.current_wave <= self.MAX_WAVES:
                        self._prepare_wave()
                
            if self.intermission_timer > 0:
                self.intermission_timer -= 1
        
        # --- Termination Check ---
        if self.base_health <= 0 and not self.game_over:
            terminated = True
            self.game_over = True
            reward -= 100
        
        if self.current_wave > self.MAX_WAVES and not self.game_over:
            terminated = True
            self.game_over = True
            reward += 100

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement == 1: self.cursor_cell[1] -= 1
        elif movement == 2: self.cursor_cell[1] += 1
        elif movement == 3: self.cursor_cell[0] -= 1
        elif movement == 4: self.cursor_cell[0] += 1
        self.cursor_cell[0] = np.clip(self.cursor_cell[0], 0, self.GRID_WIDTH - 1)
        self.cursor_cell[1] = np.clip(self.cursor_cell[1], 0, self.GRID_HEIGHT - 1)

        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: ui_chime.wav

        if space_held and not self.prev_space_held:
            self._place_tower()
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _prepare_wave(self):
        num_enemies = 5 + (self.current_wave - 1)
        enemy_health = 20 * (1.05 ** (self.current_wave - 1))
        enemy_speed = 1.0 * (1.02 ** (self.current_wave - 1))
        
        self.enemies_to_spawn = []
        for _ in range(num_enemies):
            offset_y = (self.np_random.random() - 0.5) * self.CELL_SIZE * 2
            spawn_pos = pygame.Vector2(
                self.spawn_cell[0] * self.CELL_SIZE, 
                self.spawn_cell[1] * self.CELL_SIZE + offset_y
            )
            
            enemy = {
                "pos": spawn_pos,
                "health": enemy_health,
                "max_health": enemy_health,
                "speed": enemy_speed,
                "path": self._find_path(self.spawn_cell),
                "path_node_index": 0,
            }
            self.enemies_to_spawn.append(enemy)

    def _update_wave_spawning(self):
        if self.intermission_timer > 0: return
        self.wave_spawning = True
        self.spawn_timer -= 1
        if self.spawn_timer <= 0 and self.enemies_to_spawn:
            self.enemies.append(self.enemies_to_spawn.pop(0))
            self.spawn_timer = 30 # Spawn every 1 second
        if not self.enemies_to_spawn:
            self.wave_spawning = False

    def _update_towers(self):
        new_projectiles = []
        for tower in self.towers:
            tower["cooldown"] -= 1
            if tower["cooldown"] <= 0:
                target = None
                min_dist = tower["range"] ** 2
                for enemy in self.enemies:
                    dist_sq = tower["pos"].distance_squared_to(enemy["pos"])
                    if dist_sq < min_dist:
                        min_dist = dist_sq
                        target = enemy
                
                if target:
                    tower["target"] = target
                    tower["cooldown"] = 30 / tower["fire_rate"]
                    # sfx: shoot.wav
                    projectile = {"pos": pygame.Vector2(tower["pos"]), "damage": tower["damage"], "speed": 10, "target": target, "color": self.TOWER_SPECS[tower["type"]]["color"]}
                    new_projectiles.append(projectile)
            elif tower.get("target") and (tower["target"] not in self.enemies or tower["pos"].distance_squared_to(tower["target"]["pos"]) > tower["range"] ** 2):
                tower["target"] = None
        return new_projectiles

    def _update_projectiles(self):
        killed_count = 0
        for p in self.projectiles[:]:
            if p["target"] not in self.enemies:
                self.projectiles.remove(p)
                continue
            direction = (p["target"]["pos"] - p["pos"]).normalize()
            p["pos"] += direction * p["speed"]
            if p["pos"].distance_squared_to(p["target"]["pos"]) < 10**2:
                # sfx: impact.wav
                p["target"]["health"] -= p["damage"]
                self.particles.append({"pos": pygame.Vector2(p["pos"]), "size": 10, "life": 10, "color": p["color"]})
                if p["target"]["health"] <= 0:
                    # sfx: enemy_explode.wav
                    killed_count += 1
                    self.particles.append({"pos": pygame.Vector2(p["target"]["pos"]), "size": 20, "life": 15, "color": self.COLOR_ENEMY})
                    self.enemies.remove(p["target"])
                self.projectiles.remove(p)
        return killed_count

    def _update_enemies(self):
        damage_to_base = 0
        for enemy in self.enemies[:]:
            if not enemy["path"] or enemy["path_node_index"] >= len(enemy["path"]):
                current_cell = (int(enemy["pos"].x // self.CELL_SIZE), int(enemy["pos"].y // self.CELL_SIZE))
                enemy["path"] = self._find_path(current_cell)
                enemy["path_node_index"] = 0
                if not enemy["path"]: continue

            target_node = enemy["path"][enemy["path_node_index"]]
            target_pos = pygame.Vector2(target_node[0] * self.CELL_SIZE + self.CELL_SIZE/2, target_node[1] * self.CELL_SIZE + self.CELL_SIZE/2)
            
            direction = (target_pos - enemy["pos"])
            if direction.length_squared() < (enemy["speed"] ** 2):
                enemy["path_node_index"] += 1
                if enemy["path_node_index"] >= len(enemy["path"]):
                    damage_to_base += 10
                    self.enemies.remove(enemy)
                    # sfx: base_damage.wav
                    self.particles.append({"pos": pygame.Vector2(self.base_cell[0]*self.CELL_SIZE, self.base_cell[1]*self.CELL_SIZE), "size": 30, "life": 20, "color": self.COLOR_BASE})
                    continue
            else:
                enemy["pos"] += direction.normalize() * enemy["speed"]
        return damage_to_base

    def _update_particles(self):
        for p in self.particles[:]:
            p["life"] -= 1
            p["size"] *= 0.95
            if p["life"] <= 0: self.particles.remove(p)

    def _place_tower(self):
        if self._is_placement_valid():
            x, y = self.cursor_cell
            spec = self.TOWER_SPECS[self.selected_tower_type]
            # sfx: place_tower.wav
            self.resources -= spec["cost"]
            self.path_grid[x, y] = 1
            new_tower = {"pos": pygame.Vector2(x * self.CELL_SIZE + self.CELL_SIZE/2, y * self.CELL_SIZE + self.CELL_SIZE/2), "cell": (x, y), "type": self.selected_tower_type, "range": spec["range"], "damage": spec["damage"], "fire_rate": spec["fire_rate"], "cooldown": 0, "target": None}
            self.towers.append(new_tower)
            for enemy in self.enemies:
                current_cell = (int(enemy["pos"].x // self.CELL_SIZE), int(enemy["pos"].y // self.CELL_SIZE))
                enemy["path"] = self._find_path(current_cell)
                enemy["path_node_index"] = 0

    def _find_path(self, start_cell):
        if not (0 <= start_cell[0] < self.GRID_WIDTH and 0 <= start_cell[1] < self.GRID_HEIGHT) or self.path_grid[start_cell] == 1:
             return []
        open_set = [(0, start_cell)]
        came_from, g_score, f_score = {}, defaultdict(lambda: float('inf')), defaultdict(lambda: float('inf'))
        g_score[start_cell], f_score[start_cell] = 0, abs(start_cell[0] - self.base_cell[0]) + abs(start_cell[1] - self.base_cell[1])
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == self.base_cell:
                path = []
                while current in came_from: path.append(current); current = came_from[current]
                return path[::-1]
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.GRID_WIDTH and 0 <= neighbor[1] < self.GRID_HEIGHT) or self.path_grid[neighbor] == 1: continue
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor], g_score[neighbor], f_score[neighbor] = current, tentative_g_score, tentative_g_score + abs(neighbor[0] - self.base_cell[0]) + abs(neighbor[1] - self.base_cell[1])
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE): pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE): pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        if self.enemies and self.enemies[0]['path'] and len(self.enemies[0]['path']) > 1:
            path_points = [(c[0] * self.CELL_SIZE + self.CELL_SIZE/2, c[1] * self.CELL_SIZE + self.CELL_SIZE/2) for c in self.enemies[0]['path']]
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, path_points, 3)
        pygame.draw.rect(self.screen, self.COLOR_SPAWN, pygame.Rect(self.spawn_cell[0] * self.CELL_SIZE, self.spawn_cell[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
        pygame.draw.rect(self.screen, self.COLOR_BASE, pygame.Rect(self.base_cell[0] * self.CELL_SIZE, self.base_cell[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
        for tower in self.towers:
            spec, pos = self.TOWER_SPECS[tower["type"]], (int(tower["pos"].x), int(tower["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CELL_SIZE // 2 - 2, spec["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CELL_SIZE // 2 - 2, spec["color"])
            if tower.get("target"): pygame.draw.aaline(self.screen, (255,255,255,50), pos, (int(tower["target"]["pos"].x), int(tower["target"]["pos"].y)))
        for p in self.projectiles: pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), 3, p["color"])
        for enemy in self.enemies:
            pos = (int(enemy["pos"].x), int(enemy["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            bar_w, bar_h = 16, 4
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_HEALTH_BG, (pos[0] - bar_w/2, pos[1] - 15, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_HEALTH, (pos[0] - bar_w/2, pos[1] - 15, bar_w * (enemy["health"] / enemy["max_health"]), bar_h))
        for p in self.particles: pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["size"]), p["color"] + (int(255 * (p["life"] / 15)),))
        
        cursor_pos_px, cursor_rect = (self.cursor_cell[0] * self.CELL_SIZE, self.cursor_cell[1] * self.CELL_SIZE), pygame.Rect((self.cursor_cell[0] * self.CELL_SIZE, self.cursor_cell[1] * self.CELL_SIZE), (self.CELL_SIZE, self.CELL_SIZE))
        cursor_color = self.COLOR_CURSOR_VALID if self._is_placement_valid() else self.COLOR_CURSOR_INVALID
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA); s.fill(cursor_color); self.screen.blit(s, cursor_pos_px)
        pygame.gfxdraw.aacircle(self.screen, cursor_rect.centerx, cursor_rect.centery, self.TOWER_SPECS[self.selected_tower_type]["range"], cursor_color)

    def _is_placement_valid(self):
        x, y = self.cursor_cell
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.resources < spec["cost"] or self.path_grid[x, y] == 1 or (x, y) == self.spawn_cell or (x, y) == self.base_cell: return False
        self.path_grid[x, y] = 1
        path_exists = self._find_path(self.spawn_cell)
        self.path_grid[x, y] = 0
        return bool(path_exists)

    def _render_ui(self):
        ui_bar = pygame.Surface((self.SCREEN_WIDTH, 30)); ui_bar.set_alpha(180); ui_bar.fill((10, 10, 20)); self.screen.blit(ui_bar, (0, 0))
        self.screen.blit(self.font_small.render(f"Base HP: {max(0, self.base_health)}/100", True, self.COLOR_TEXT), (10, 7))
        self.screen.blit(self.font_small.render(f"Resources: {self.resources}", True, self.COLOR_TEXT), (180, 7))
        self.screen.blit(self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT), (340, 7))
        self.screen.blit(self.font_small.render(f"Wave: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT), (480, 7))
        
        bottom_bar = pygame.Surface((self.SCREEN_WIDTH, 50)); bottom_bar.set_alpha(180); bottom_bar.fill((10, 10, 20)); self.screen.blit(bottom_bar, (0, self.SCREEN_HEIGHT - 50))
        for i, spec in self.TOWER_SPECS.items():
            x_pos = self.SCREEN_WIDTH / 2 - (len(self.TOWER_SPECS) * 100 / 2) + i * 100 + 50
            if i == self.selected_tower_type: pygame.draw.rect(self.screen, (255,255,255), (x_pos - 42, self.SCREEN_HEIGHT - 47, 84, 44), 2, border_radius=5)
            name_text, cost_text = self.font_small.render(f"{spec['name']}", True, self.COLOR_TEXT), self.font_small.render(f"Cost: {spec['cost']}", True, self.COLOR_TEXT)
            self.screen.blit(name_text, (x_pos - name_text.get_width()/2, self.SCREEN_HEIGHT - 45))
            self.screen.blit(cost_text, (x_pos - cost_text.get_width()/2, self.SCREEN_HEIGHT - 25))
            pygame.gfxdraw.filled_circle(self.screen, int(x_pos - 30), int(self.SCREEN_HEIGHT - 25), 8, spec['color'])

        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA); s.fill((0,0,0,180)); self.screen.blit(s, (0,0))
            msg = "YOU WIN!" if self.current_wave > self.MAX_WAVES else "GAME OVER"
            msg_render, score_render = self.font_large.render(msg, True, self.COLOR_TEXT), self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            self.screen.blit(msg_render, (self.SCREEN_WIDTH/2 - msg_render.get_width()/2, self.SCREEN_HEIGHT/2 - msg_render.get_height()/2 - 20))
            self.screen.blit(score_render, (self.SCREEN_WIDTH/2 - score_render.get_width()/2, self.SCREEN_HEIGHT/2 - score_render.get_height()/2 + 20))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave, "base_health": self.base_health, "resources": self.resources}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and isinstance(info, dict)
        obs, reward, term, trunc, info = self.step(self.action_space.sample())
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and isinstance(reward, (int, float)) and isinstance(term, bool) and not trunc and isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    try:
        env.validate_implementation()
    except AssertionError as e:
        print(f"Validation failed: {e}")
        exit()

    obs, info = env.reset()
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption(env.game_description)
    action = env.action_space.sample(); action.fill(0)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        
        keys = pygame.key.get_pressed()
        action[0] = 1 if keys[pygame.K_UP] else 2 if keys[pygame.K_DOWN] else 3 if keys[pygame.K_LEFT] else 4 if keys[pygame.K_RIGHT] else 0
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0: print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Info: {info}")
            pygame.time.wait(3000)
            obs, info = env.reset()

        env.clock.tick(30)
    env.close()