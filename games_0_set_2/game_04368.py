
# Generated: 2025-08-28T02:12:58.971886
# Source Brief: brief_04368.md
# Brief Index: 4368

        
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
        "Controls: Arrows to move cursor. Space to place basic tower. Shift+Space to place advanced tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by placing towers on the grid. Survive all 10 waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GAME_AREA_WIDTH = 400
    UI_WIDTH = SCREEN_WIDTH - GAME_AREA_WIDTH

    GRID_SIZE = 20
    CELL_SIZE = GAME_AREA_WIDTH // GRID_SIZE

    MAX_STEPS = 4000
    TOTAL_WAVES = 10
    WAVE_INTERVAL = 250 # Steps between wave spawns

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 55)
    COLOR_PATH = (40, 50, 70)
    COLOR_BASE = (60, 180, 75)
    COLOR_UI_BG = (25, 30, 45)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_DIM = (150, 150, 150)
    
    COLOR_ENEMY = (230, 25, 75)
    COLOR_HEALTH_FG = (70, 240, 70)
    COLOR_HEALTH_BG = (120, 20, 20)
    
    TOWER_SPECS = {
        1: {"name": "Cannon", "color": (67, 137, 225), "range": 3.5, "damage": 5, "cooldown": 30, "cost": 25, "projectile_speed": 5},
        2: {"name": "Laser", "color": (255, 225, 25), "range": 6.0, "damage": 2, "cooldown": 15, "cost": 40, "projectile_speed": 7},
    }
    COLOR_PROJECTILE = (255, 255, 255)
    
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
        
        self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 16)
        self.font_tiny = pygame.font.SysFont("Arial", 12)

        self._define_path()
        self.reset()
        
        self.validate_implementation()

    def _define_path(self):
        self.path = []
        # A simple S-like path
        for i in range(3, 17): self.path.append((i, 2))
        for i in range(3, 8): self.path.append((16, i))
        for i in range(16, 3, -1): self.path.append((i, 7))
        for i in range(8, 13): self.path.append((4, i))
        for i in range(4, 17): self.path.append((i, 12))
        for i in range(13, 18): self.path.append((16, i))
        self.path_pixels = [self._grid_to_pixel_center(p) for p in self.path]
        self.base_pos = (16, 17)
        self.spawn_pos = (3, 2)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 60 # Starting resources
        self.base_health = 100
        self.game_over = False
        self.win = False
        
        self.wave_number = 0
        self.wave_spawn_timer = self.WAVE_INTERVAL // 2

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.001 # Small penalty for existing
        
        self._handle_action(action)
        
        self._update_wave_spawner()
        self._update_enemies()
        hits_this_step = self._update_towers()
        kills_this_step, damage_to_base = self._update_projectiles()
        self._update_particles()
        
        self.score += kills_this_step
        self.base_health = max(0, self.base_health - damage_to_base)
        
        if hits_this_step > 0:
            reward = 0.1 * hits_this_step
        reward += kills_this_step * 1.0

        self.steps += 1
        terminated = False
        
        if self.base_health <= 0:
            self.game_over = True
            terminated = True
            reward = -100
        
        if self.wave_number > self.TOTAL_WAVES and not self.enemies:
            self.win = True
            terminated = True
            reward += 100

        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_press, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)

        # --- Placement ---
        if space_press:
            tower_type = 2 if shift_held else 1
            cost = self.TOWER_SPECS[tower_type]["cost"]
            if self.score >= cost and self._is_valid_placement(self.cursor_pos):
                self.score -= cost
                self.towers.append({
                    "pos": tuple(self.cursor_pos),
                    "type": tower_type,
                    "cooldown_timer": 0,
                    "target": None,
                })
                # sfx: place_tower.wav
                self._create_particles(self._grid_to_pixel_center(self.cursor_pos), 10, self.TOWER_SPECS[tower_type]['color'])

    def _update_wave_spawner(self):
        if self.wave_number > self.TOTAL_WAVES:
            return

        self.wave_spawn_timer -= 1
        if self.wave_spawn_timer <= 0:
            self.wave_number += 1
            if self.wave_number <= self.TOTAL_WAVES:
                self._spawn_wave()
                # sfx: new_wave.wav
            self.wave_spawn_timer = self.WAVE_INTERVAL

    def _spawn_wave(self):
        num_enemies = min(12, 2 + self.wave_number)
        base_health = 10 + (self.wave_number - 1) * 2
        base_speed = 0.5 + (self.wave_number - 1) * 0.05
        
        for i in range(num_enemies):
            self.enemies.append({
                "id": self.steps + i,
                "path_index": 0,
                "pixel_pos": pygame.Vector2(self.path_pixels[0]),
                "health": base_health,
                "max_health": base_health,
                "speed": base_speed * self.np_random.uniform(0.9, 1.1),
                "offset": (self.np_random.random() - 0.5) * self.CELL_SIZE * 0.6,
                "spawn_delay": i * 15 # Stagger spawns
            })

    def _update_enemies(self):
        damage_to_base = 0
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy["spawn_delay"] > 0:
                enemy["spawn_delay"] -= 1
                continue

            if enemy["path_index"] >= len(self.path_pixels) - 1:
                damage_to_base += enemy["health"]
                enemies_to_remove.append(enemy)
                self._create_particles(self._grid_to_pixel_center(self.base_pos), 20, self.COLOR_ENEMY)
                # sfx: base_damage.wav
                continue

            target_pixel = pygame.Vector2(self.path_pixels[enemy["path_index"] + 1])
            target_pixel.y += enemy["offset"] # Add jitter to path
            
            direction = (target_pixel - enemy["pixel_pos"])
            if direction.length() < enemy["speed"]:
                enemy["pixel_pos"] = target_pixel
                enemy["path_index"] += 1
            else:
                enemy["pixel_pos"] += direction.normalize() * enemy["speed"]
        
        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        return damage_to_base
        
    def _update_towers(self):
        hits_this_step = 0
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            if tower["cooldown_timer"] > 0:
                tower["cooldown_timer"] -= 1
                continue
            
            tower_pix_pos = self._grid_to_pixel_center(tower["pos"])
            target = None
            min_dist = spec["range"] * self.CELL_SIZE
            
            valid_enemies = [e for e in self.enemies if e['spawn_delay'] <= 0]
            if not valid_enemies:
                continue

            for enemy in valid_enemies:
                dist = tower_pix_pos.distance_to(enemy["pixel_pos"])
                if dist < min_dist:
                    min_dist = dist
                    target = enemy

            if target:
                self.projectiles.append({
                    "start_pos": pygame.Vector2(tower_pix_pos),
                    "pos": pygame.Vector2(tower_pix_pos),
                    "target_id": target["id"],
                    "speed": spec["projectile_speed"],
                    "damage": spec["damage"],
                    "color": spec["color"]
                })
                tower["cooldown_timer"] = spec["cooldown"]
                hits_this_step += 1
                # sfx: tower_shoot.wav
        return hits_this_step
        
    def _update_projectiles(self):
        kills = 0
        base_damage = 0
        projectiles_to_remove = []
        
        enemy_map = {e["id"]: e for e in self.enemies}
        
        for proj in self.projectiles:
            target_enemy = enemy_map.get(proj["target_id"])
            if not target_enemy:
                projectiles_to_remove.append(proj)
                continue

            target_pos = target_enemy["pixel_pos"]
            direction = (target_pos - proj["pos"])
            
            if direction.length() < proj["speed"]:
                target_enemy["health"] -= proj["damage"]
                projectiles_to_remove.append(proj)
                self._create_particles(target_pos, 5, proj["color"])
                # sfx: enemy_hit.wav

                if target_enemy["health"] <= 0:
                    if target_enemy in self.enemies:
                        self.enemies.remove(target_enemy)
                        kills += 1
                        self._create_particles(target_pos, 15, self.COLOR_ENEMY)
                        # sfx: enemy_die.wav
            else:
                proj["pos"] += direction.normalize() * proj["speed"]
        
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        
        for enemy in list(self.enemies):
            if enemy["path_index"] >= len(self.path) - 1:
                base_damage += 5
                self.enemies.remove(enemy)
                self._create_particles(self._grid_to_pixel_center(self.base_pos), 20, self.COLOR_ENEMY)
                # sfx: base_damage.wav

        return kills, base_damage

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                particles_to_remove.append(p)
        self.particles = [p for p in self.particles if p not in particles_to_remove]

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 0.5
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "life": self.np_random.integers(10, 20),
                "color": color,
                "radius": self.np_random.random() * 2 + 1
            })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        game_surface = self.screen.subsurface(pygame.Rect(0, 0, self.GAME_AREA_WIDTH, self.SCREEN_HEIGHT))

        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(game_surface, self.COLOR_GRID, (i * self.CELL_SIZE, 0), (i * self.CELL_SIZE, self.SCREEN_HEIGHT))
            pygame.draw.line(game_surface, self.COLOR_GRID, (0, i * self.CELL_SIZE), (self.GAME_AREA_WIDTH, i * self.CELL_SIZE))
        
        for pos in self.path:
            rect = pygame.Rect(pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(game_surface, self.COLOR_PATH, rect)

        base_rect = pygame.Rect(self.base_pos[0] * self.CELL_SIZE, self.base_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(game_surface, self.COLOR_BASE, base_rect, border_radius=3)

        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            center = self._grid_to_pixel_center(tower["pos"])
            if tower["type"] == 1:
                rect = pygame.Rect(tower["pos"][0] * self.CELL_SIZE, tower["pos"][1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(game_surface, spec["color"], rect, border_radius=3)
            else:
                pygame.gfxdraw.filled_circle(game_surface, int(center.x), int(center.y), self.CELL_SIZE // 2 - 2, spec["color"])
                pygame.gfxdraw.aacircle(game_surface, int(center.x), int(center.y), self.CELL_SIZE // 2 - 2, spec["color"])
            
            if tower["cooldown_timer"] > 0:
                cooldown_ratio = tower["cooldown_timer"] / spec["cooldown"]
                radius = int(self.CELL_SIZE / 3)
                pygame.draw.arc(game_surface, (0,0,0,100), (center.x-radius, center.y-radius, radius*2, radius*2), 0, 2 * math.pi * cooldown_ratio, 3)

        for enemy in self.enemies:
            if enemy["spawn_delay"] > 0: continue
            pos = (int(enemy["pixel_pos"].x), int(enemy["pixel_pos"].y))
            pygame.gfxdraw.filled_circle(game_surface, pos[0], pos[1], self.CELL_SIZE // 3, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(game_surface, pos[0], pos[1], self.CELL_SIZE // 3, self.COLOR_ENEMY)
            
            health_ratio = enemy["health"] / enemy["max_health"]
            bar_width = self.CELL_SIZE * 0.8
            bar_height = 4
            bar_x = pos[0] - bar_width / 2
            bar_y = pos[1] - self.CELL_SIZE / 2 - 5
            pygame.draw.rect(game_surface, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=1)
            pygame.draw.rect(game_surface, self.COLOR_HEALTH_FG, (bar_x, bar_y, bar_width * health_ratio, bar_height), border_radius=1)

        for proj in self.projectiles:
            pos = (int(proj["pos"].x), int(proj["pos"].y))
            pygame.gfxdraw.filled_circle(game_surface, pos[0], pos[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(game_surface, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 20.0))))
            color = p["color"] + (alpha,)
            pos = (int(p["pos"].x), int(p["pos"].y))
            try:
                pygame.gfxdraw.filled_circle(game_surface, pos[0], pos[1], int(p["radius"]), color)
            except TypeError: # Color might not have alpha
                 pygame.gfxdraw.filled_circle(game_surface, pos[0], pos[1], int(p["radius"]), p['color'])

        self._render_cursor(game_surface)
            
    def _render_ui(self):
        ui_surface = self.screen.subsurface(pygame.Rect(self.GAME_AREA_WIDTH, 0, self.UI_WIDTH, self.SCREEN_HEIGHT))
        ui_surface.fill(self.COLOR_UI_BG)

        y_offset = 20
        
        def draw_text(text, font, color, pos):
            surf = font.render(text, True, color)
            ui_surface.blit(surf, pos)
        
        draw_text("RESOURCES", self.font_main, self.COLOR_TEXT_DIM, (20, y_offset)); y_offset += 25
        draw_text(f"${int(self.score)}", self.font_main, self.COLOR_TEXT, (20, y_offset)); y_offset += 40

        draw_text("BASE HEALTH", self.font_main, self.COLOR_TEXT_DIM, (20, y_offset)); y_offset += 25
        draw_text(f"{int(self.base_health)} / 100", self.font_main, self.COLOR_TEXT, (20, y_offset)); y_offset += 30
        
        health_ratio = self.base_health / 100.0
        bar_width = self.UI_WIDTH - 40
        pygame.draw.rect(ui_surface, self.COLOR_HEALTH_BG, (20, y_offset, bar_width, 15), border_radius=3)
        pygame.draw.rect(ui_surface, self.COLOR_HEALTH_FG, (20, y_offset, bar_width * health_ratio, 15), border_radius=3)
        y_offset += 40

        draw_text("WAVE", self.font_main, self.COLOR_TEXT_DIM, (20, y_offset)); y_offset += 25
        draw_text(f"{min(self.wave_number, self.TOTAL_WAVES)} / {self.TOTAL_WAVES}", self.font_main, self.COLOR_TEXT, (20, y_offset)); y_offset += 30
        
        if self.wave_number < self.TOTAL_WAVES and not any(e['spawn_delay'] <= 0 for e in self.enemies):
            draw_text(f"Next in: {self.wave_spawn_timer // 30 + 1}s", self.font_small, self.COLOR_TEXT_DIM, (20, y_offset))
        y_offset += 40

        if self.game_over:
            self._draw_centered_text(self.screen, "GAME OVER", self.font_main, self.COLOR_ENEMY)
        elif self.win:
            self._draw_centered_text(self.screen, "VICTORY!", self.font_main, self.COLOR_BASE)

    def _render_cursor(self, surface):
        cursor_rect = pygame.Rect(self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(surface, (255, 255, 255), cursor_rect, 2)
        
        if self._is_valid_placement(self.cursor_pos):
            center = self._grid_to_pixel_center(self.cursor_pos)
            # This preview is for a human player. Let's assume they can see the range of both towers.
            for i, spec in self.TOWER_SPECS.items():
                range_surf = pygame.Surface((self.GAME_AREA_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                alpha = 30 if i == 1 else 20
                pygame.draw.circle(range_surf, spec["color"] + (alpha,), center, int(spec["range"] * self.CELL_SIZE))
                pygame.draw.circle(range_surf, spec["color"] + (80,), center, int(spec["range"] * self.CELL_SIZE), 1)
                surface.blit(range_surf, (0,0))

    def _draw_centered_text(self, surface, text, font, color):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        bg_rect = text_rect.inflate(20, 20)
        s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (0,0,0,180), s.get_rect(), border_radius=10)
        surface.blit(s, bg_rect)
        surface.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "wave": self.wave_number,
            "enemies": len(self.enemies),
        }
    
    def _grid_to_pixel_center(self, grid_pos):
        x = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return pygame.Vector2(x, y)

    def _is_valid_placement(self, grid_pos):
        if not (0 <= grid_pos[0] < self.GRID_SIZE and 0 <= grid_pos[1] < self.GRID_SIZE):
            return False
        if tuple(grid_pos) in self.path:
            return False
        if tuple(grid_pos) == self.base_pos:
            return False
        if any(tower['pos'] == tuple(grid_pos) for tower in self.towers):
            return False
        return True

    def close(self):
        pygame.font.quit()
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    action = env.action_space.sample()
    action.fill(0)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        action.fill(0)
        
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()