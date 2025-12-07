
# Generated: 2025-08-28T05:56:53.919475
# Source Brief: brief_05738.md
# Brief Index: 5738

        
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
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Shift to cycle tower types. Press Space to build a tower."
    )

    game_description = (
        "Defend your base from waves of invading enemies by strategically placing towers. "
        "Survive 10 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 15
    TILE_W_HALF, TILE_H_HALF = 20, 10
    MAX_STEPS = 5000 # Increased to allow for longer games
    MAX_WAVES = 10

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_PATH = (50, 60, 70)
    COLOR_GRID = (40, 45, 50)
    COLOR_BASE = (60, 180, 75)
    COLOR_BASE_DMG = (200, 75, 60)
    COLOR_ENEMY = (230, 25, 75)
    COLOR_TEXT = (245, 245, 245)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    
    TOWER_SPECS = {
        0: {"name": "Gatling", "cost": 40, "range": 80, "damage": 2, "fire_rate": 5, "color": (0, 200, 255), "proj_speed": 8},
        1: {"name": "Cannon", "cost": 100, "range": 150, "damage": 25, "fire_rate": 40, "color": (255, 180, 0), "proj_speed": 5},
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
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        self.iso_offset_x = self.SCREEN_WIDTH / 2
        self.iso_offset_y = 80

        self.path_grid_coords = []
        self.path_pixel_coords = []
        self._generate_path()

        self.reset()
        
        self.validate_implementation()

    def _grid_to_iso(self, x, y):
        iso_x = self.iso_offset_x + (x - y) * self.TILE_W_HALF
        iso_y = self.iso_offset_y + (x + y) * self.TILE_H_HALF
        return int(iso_x), int(iso_y)

    def _generate_path(self):
        path = []
        x, y = 0, 7
        for _ in range(5): path.append((x,y)); x+=1
        for _ in range(5): path.append((x,y)); y+=1
        for _ in range(8): path.append((x,y)); x+=1
        for _ in range(5): path.append((x,y)); y-=1
        for _ in range(5): path.append((x,y)); x+=1
        
        self.path_grid_coords = list(dict.fromkeys(path)) # Remove duplicates
        self.path_pixel_coords = [self._grid_to_iso(gx, gy) for gx, gy in self.path_grid_coords]
        self.base_pos_grid = self.path_grid_coords[-1]
        self.base_pos_pixel = self.path_pixel_coords[-1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.reward_this_step = 0
        
        self.base_health = 100
        self.resources = 80
        
        self.current_wave = 0
        self.wave_in_progress = False
        self.wave_cooldown = 150 # Ticks until next wave
        self.enemies_to_spawn = []
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.damage_flash_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        
        self._handle_input(action)
        
        if not self.game_over:
            self._update_waves()
            self._update_towers()
            self._update_enemies()
            self._update_projectiles()
        
        self._update_particles()
        
        if self.damage_flash_timer > 0:
            self.damage_flash_timer -= 1

        self.steps += 1
        
        terminated = self._check_termination()

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Debounce presses
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        if shift_pressed:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: UI_Cycle.wav

        if space_pressed and self._is_placement_valid():
            self._place_tower()

    def _is_placement_valid(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.resources < spec["cost"]: return False
        if tuple(self.cursor_pos) in self.path_grid_coords: return False
        if any(t['grid_pos'] == self.cursor_pos for t in self.towers): return False
        return True

    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        self.resources -= spec["cost"]
        
        px, py = self._grid_to_iso(self.cursor_pos[0], self.cursor_pos[1])

        self.towers.append({
            "grid_pos": list(self.cursor_pos),
            "pixel_pos": [px, py],
            "spec": spec,
            "cooldown": 0,
            "target": None,
        })
        # sfx: Tower_Place.wav

    def _update_waves(self):
        if self.wave_in_progress:
            if not self.enemies and not self.enemies_to_spawn:
                self.wave_in_progress = False
                self.wave_cooldown = 300 # Time between waves
                if self.current_wave <= self.MAX_WAVES:
                    self.reward_this_step += 1.0
                    self.score += 100
        else:
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0 and self.current_wave < self.MAX_WAVES:
                self.current_wave += 1
                self.wave_in_progress = True
                self._spawn_wave()

    def _spawn_wave(self):
        num_enemies = 3 + self.current_wave * 2
        health = 10 * (1.1 ** (self.current_wave - 1))
        speed = 0.5 * (1.05 ** (self.current_wave - 1))
        
        for i in range(num_enemies):
            self.enemies_to_spawn.append({
                "health": health,
                "speed": speed,
                "spawn_delay": i * 30 # Stagger spawns
            })

    def _update_towers(self):
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
            
            # Retarget if no target or target is dead/out of range
            if tower["target"] is None or not self._is_enemy_in_range(tower, tower["target"]):
                tower["target"] = self._find_closest_enemy(tower)

            if tower["target"] and tower["cooldown"] <= 0:
                self._fire_projectile(tower, tower["target"])
                tower["cooldown"] = tower["spec"]["fire_rate"]

    def _is_enemy_in_range(self, tower, enemy):
        dist_sq = (tower["pixel_pos"][0] - enemy["pos"][0]) ** 2 + (tower["pixel_pos"][1] - enemy["pos"][1]) ** 2
        return dist_sq <= tower["spec"]["range"] ** 2

    def _find_closest_enemy(self, tower):
        closest_enemy = None
        min_dist_sq = float('inf')
        for enemy in self.enemies:
            dist_sq = (tower["pixel_pos"][0] - enemy["pos"][0]) ** 2 + (tower["pixel_pos"][1] - enemy["pos"][1]) ** 2
            if dist_sq < min_dist_sq and dist_sq <= tower["spec"]["range"] ** 2:
                min_dist_sq = dist_sq
                closest_enemy = enemy
        return closest_enemy

    def _fire_projectile(self, tower, target_enemy):
        start_pos = list(tower["pixel_pos"])
        start_pos[1] -= self.TILE_H_HALF # Fire from top of tower
        
        self.projectiles.append({
            "pos": start_pos,
            "target": target_enemy,
            "spec": tower["spec"]
        })
        # sfx: Gatling_Fire.wav or Cannon_Fire.wav
        
    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            target = proj["target"]
            if not target["active"]:
                self.projectiles.remove(proj)
                continue

            target_pos = target["pos"]
            dx = target_pos[0] - proj["pos"][0]
            dy = target_pos[1] - proj["pos"][1]
            dist = math.hypot(dx, dy)

            if dist < proj["spec"]["proj_speed"]:
                self._hit_enemy(target, proj["spec"]["damage"])
                self.projectiles.remove(proj)
            else:
                proj["pos"][0] += (dx / dist) * proj["spec"]["proj_speed"]
                proj["pos"][1] += (dy / dist) * proj["spec"]["proj_speed"]

    def _hit_enemy(self, enemy, damage):
        enemy["health"] -= damage
        # sfx: Enemy_Hit.wav
        self._create_particles(enemy["pos"], self.COLOR_ENEMY, 3)
        if enemy["health"] <= 0:
            enemy["active"] = False
            self.reward_this_step += 0.1
            self.score += 10
            self.resources += 5
            # sfx: Enemy_Explode.wav
            self._create_particles(enemy["pos"], self.COLOR_ENEMY, 15, 2.5)

    def _update_enemies(self):
        # Spawn enemies from the queue
        if self.enemies_to_spawn:
            self.enemies_to_spawn[0]["spawn_delay"] -= 1
            if self.enemies_to_spawn[0]["spawn_delay"] <= 0:
                spawn = self.enemies_to_spawn.pop(0)
                self.enemies.append({
                    "pos": list(self.path_pixel_coords[0]),
                    "path_index": 0,
                    "health": spawn["health"],
                    "max_health": spawn["health"],
                    "speed": spawn["speed"],
                    "active": True
                })

        # Move and update active enemies
        for enemy in self.enemies[:]:
            if not enemy["active"]:
                self.enemies.remove(enemy)
                continue
            
            if enemy["path_index"] >= len(self.path_pixel_coords) - 1:
                self._damage_base(10)
                enemy["active"] = False
                continue

            target_pos = self.path_pixel_coords[enemy["path_index"] + 1]
            dx = target_pos[0] - enemy["pos"][0]
            dy = target_pos[1] - enemy["pos"][1]
            dist = math.hypot(dx, dy)

            if dist < enemy["speed"]:
                enemy["path_index"] += 1
                enemy["pos"] = list(self.path_pixel_coords[enemy["path_index"]])
            else:
                enemy["pos"][0] += (dx / dist) * enemy["speed"]
                enemy["pos"][1] += (dy / dist) * enemy["speed"]

    def _damage_base(self, amount):
        self.base_health = max(0, self.base_health - amount)
        self.reward_this_step -= amount * 0.01
        self.damage_flash_timer = 10
        # sfx: Base_Damage.wav
        self._create_particles(self.base_pos_pixel, self.COLOR_BASE_DMG, 20, 3.0)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count, power=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 1.5) * power
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(15, 30),
                "color": color
            })

    def _check_termination(self):
        if self.game_over:
            return True
            
        if self.base_health <= 0:
            self.game_over = True
            self.reward_this_step -= 100
            return True
        
        if self.current_wave >= self.MAX_WAVES and not self.enemies and not self.enemies_to_spawn:
            self.game_over = True
            self.game_won = True
            self.reward_this_step += 100
            return True
            
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_path()
        self._render_base()
        
        # Sort all drawable objects by Y-coordinate for correct isometric rendering
        renderables = []
        for t in self.towers:
            renderables.append(('tower', t['pixel_pos'], t))
        for e in self.enemies:
            renderables.append(('enemy', e['pos'], e))
        
        renderables.sort(key=lambda item: item[1][1])

        for r_type, _, item in renderables:
            if r_type == 'tower':
                self._render_tower(item)
            elif r_type == 'enemy':
                self._render_enemy(item)

        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                p1 = self._grid_to_iso(x, y)
                p2 = self._grid_to_iso(x + 1, y)
                p3 = self._grid_to_iso(x + 1, y + 1)
                p4 = self._grid_to_iso(x, y + 1)
                pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p4)

    def _render_path(self):
        for x, y in self.path_grid_coords:
            points = self._get_rhombus_points(x, y)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PATH)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PATH)

    def _render_base(self):
        color = self.COLOR_BASE if self.damage_flash_timer == 0 else self.COLOR_BASE_DMG
        points = self._get_rhombus_points(self.base_pos_grid[0], self.base_pos_grid[1], 1.2)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, tuple(min(255, c+30) for c in color))

    def _render_tower(self, tower):
        spec = tower["spec"]
        px, py = tower["pixel_pos"]
        color = spec["color"]
        
        # Base
        base_points = [
            (px, py + self.TILE_H_HALF),
            (px + self.TILE_W_HALF * 0.7, py),
            (px, py - self.TILE_H_HALF),
            (px - self.TILE_W_HALF * 0.7, py)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, base_points, tuple(c//2 for c in color))
        pygame.gfxdraw.aapolygon(self.screen, base_points, color)

        # Top
        top_rect = pygame.Rect(0, 0, self.TILE_W_HALF * 1.2, self.TILE_H_HALF * 1.2)
        top_rect.center = (px, py - self.TILE_H_HALF)
        pygame.draw.ellipse(self.screen, color, top_rect)
        pygame.draw.ellipse(self.screen, (255,255,255), top_rect, 1)

        # Flash on fire
        if tower["cooldown"] >= spec["fire_rate"] - 2:
            pygame.draw.ellipse(self.screen, (255, 255, 255), top_rect)

    def _render_enemy(self, enemy):
        px, py = int(enemy["pos"][0]), int(enemy["pos"][1])
        size = 6
        
        # Body
        pygame.gfxdraw.filled_circle(self.screen, px, py, size, self.COLOR_ENEMY)
        pygame.gfxdraw.aacircle(self.screen, px, py, size, tuple(min(255, c+30) for c in self.COLOR_ENEMY))

        # Health bar
        health_ratio = enemy["health"] / enemy["max_health"]
        bar_width = 20
        bar_height = 4
        bar_x = px - bar_width // 2
        bar_y = py - size - 8
        pygame.draw.rect(self.screen, (80, 0, 0), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, (0, 200, 0), (bar_x, bar_y, int(bar_width * health_ratio), bar_height))

    def _render_projectiles(self):
        for proj in self.projectiles:
            px, py = int(proj["pos"][0]), int(proj["pos"][1])
            color = proj["spec"]["color"]
            pygame.gfxdraw.filled_circle(self.screen, px, py, 2, color)
            pygame.gfxdraw.aacircle(self.screen, px, py, 3, (255,255,255))
            
    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 30))))
            color = p["color"] + (alpha,)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 4, 4))
            self.screen.blit(temp_surf, (int(p["pos"][0]-2), int(p["pos"][1]-2)))

    def _render_cursor(self):
        gx, gy = self.cursor_pos
        points = self._get_rhombus_points(gx, gy)
        
        valid = self._is_placement_valid()
        color = (0, 255, 0, 100) if valid else (255, 0, 0, 100)
        
        temp_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_polygon(temp_surf, points, color)
        pygame.gfxdraw.aapolygon(temp_surf, points, (color[0], color[1], color[2], 200))
        
        # Show range
        spec = self.TOWER_SPECS[self.selected_tower_type]
        px, py = self._grid_to_iso(gx, gy)
        pygame.gfxdraw.aacircle(temp_surf, px, py, spec['range'], (255, 255, 255, 50))
        
        self.screen.blit(temp_surf, (0,0))

    def _render_ui(self):
        def draw_text(text, x, y, font, color=self.COLOR_TEXT, shadow=True):
            if shadow:
                surf_s = font.render(text, True, self.COLOR_TEXT_SHADOW)
                self.screen.blit(surf_s, (x + 1, y + 1))
            surf = font.render(text, True, color)
            self.screen.blit(surf, (x, y))

        draw_text(f"♥ {int(self.base_health)}/100", 10, 10, self.font_ui)
        draw_text(f"$ {self.resources}", 10, 30, self.font_ui)
        
        wave_str = f"Wave: {self.current_wave}/{self.MAX_WAVES}"
        if not self.wave_in_progress and self.current_wave < self.MAX_WAVES:
            wave_str += f" (Next in {self.wave_cooldown//30}s)"
        draw_text(wave_str, self.SCREEN_WIDTH - 250, 10, self.font_ui)
        
        draw_text(f"Score: {self.score}", self.SCREEN_WIDTH - 250, 30, self.font_ui)

        # Tower selection UI
        spec = self.TOWER_SPECS[self.selected_tower_type]
        draw_text(f"Build: {spec['name']}", 10, self.SCREEN_HEIGHT - 45, self.font_ui)
        draw_text(f"Cost: ${spec['cost']} | Dmg: {spec['damage']}", 10, self.SCREEN_HEIGHT - 25, self.font_small, (200,200,200))

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))

        message = "VICTORY!" if self.game_won else "GAME OVER"
        text_surf = self.font_big.render(message, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
        self.screen.blit(text_surf, text_rect)

        score_surf = self.font_ui.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 30))
        self.screen.blit(score_surf, score_rect)

    def _get_rhombus_points(self, x, y, scale=1.0):
        center_x, center_y = self._grid_to_iso(x, y)
        w = self.TILE_W_HALF * scale
        h = self.TILE_H_HALF * scale
        return [
            (center_x, center_y + h),
            (center_x + w, center_y),
            (center_x, center_y - h),
            (center_x - w, center_y)
        ]

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(GameEnv.user_guide)

    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Info: {info}")
            # Wait for reset
            pass

        clock.tick(30) # Match the intended FPS

    env.close()