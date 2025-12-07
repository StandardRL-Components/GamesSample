
# Generated: 2025-08-28T06:38:17.536620
# Source Brief: brief_02985.md
# Brief Index: 2985

        
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
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 20
        self.TILE_W_HALF, self.TILE_H_HALF = 20, 10
        self.ORIGIN_X, self.ORIGIN_Y = self.WIDTH // 2 - self.TILE_W_HALF, 50

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 14)
        self.font_m = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 32, bold=True)
        
        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_PATH = (50, 60, 70)
        self.COLOR_GRID = (70, 80, 90)
        self.COLOR_BASE = (0, 150, 200)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_HEALTH_BG = (80, 80, 80)
        self.COLOR_HEALTH_GOOD = (80, 200, 80)

        # Game constants
        self.MAX_WAVES = 15
        self.MAX_STEPS = 25000
        self.INITIAL_BASE_HEALTH = 1000
        self.INITIAL_GOLD = 150
        self.WAVE_BREAK_TIME = 600  # 20 seconds at 30fps
        self.FIRST_WAVE_TIME = 300 # 10 seconds

        self._define_path_and_zones()
        self._define_towers()
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.base_health = 0
        self.gold = 0
        self.current_wave = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.game_phase = "break"
        self.wave_timer = 0
        self.cursor_pos = [0, 0]
        self.selected_tower_type_idx = 0
        self.available_tower_types = []
        self.previous_space_held = False
        self.previous_shift_held = False
        self.message = ""
        self.message_timer = 0
        
        self.reset()
        self.validate_implementation()
    
    def _define_path_and_zones(self):
        self.path_grid = [
            (0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (4, 8), (4, 7), (4, 6),
            (5, 6), (6, 6), (7, 6), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10),
            (8, 11), (8, 12), (7, 12), (6, 12), (5, 12), (4, 12), (4, 13),
            (4, 14), (4, 15), (5, 15), (6, 15), (7, 15), (8, 15), (9, 15),
            (10, 15), (11, 15), (12, 15), (12, 14), (12, 13), (12, 12),
            (12, 11), (12, 10), (12, 9), (13, 9), (14, 9), (15, 9), (16, 9),
            (17, 9), (18, 9), (19, 9)
        ]
        self.path_world = [self._grid_to_world(p[0], p[1]) for p in self.path_grid]
        self.base_pos_grid = self.path_grid[-1]
        self.base_pos_world = self.path_world[-1]
        
        self.placement_zones = {
            (r, c) for r in range(self.GRID_WIDTH) for c in range(self.GRID_HEIGHT)
        } - set(self.path_grid)

    def _define_towers(self):
        self.TOWER_SPECS = {
            0: {"name": "Cannon", "cost": 50, "range": 80, "damage": 25, "fire_rate": 45, "color": (100, 100, 255), "proj_color": (150, 150, 255), "unlocks_at": 1},
            1: {"name": "Sniper", "cost": 120, "range": 150, "damage": 80, "fire_rate": 90, "color": (255, 100, 100), "proj_color": (255, 150, 150), "unlocks_at": 5},
            2: {"name": "Gatling", "cost": 200, "range": 60, "damage": 15, "fire_rate": 15, "color": (100, 255, 100), "proj_color": (150, 255, 150), "unlocks_at": 10},
        }

    def _grid_to_world(self, r, c):
        x = self.ORIGIN_X + (r - c) * self.TILE_W_HALF
        y = self.ORIGIN_Y + (r + c) * self.TILE_H_HALF
        return [x, y]

    def _world_to_grid(self, x, y):
        x_norm = x - self.ORIGIN_X
        y_norm = y - self.ORIGIN_Y
        r = (x_norm / self.TILE_W_HALF + y_norm / self.TILE_H_HALF) / 2
        c = (y_norm / self.TILE_H_HALF - x_norm / self.TILE_W_HALF) / 2
        return [round(r), round(c)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.gold = self.INITIAL_GOLD
        self.current_wave = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.game_phase = "break"
        self.wave_timer = self.FIRST_WAVE_TIME
        
        self.cursor_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.selected_tower_type_idx = 0
        self.available_tower_types = [0]
        
        self.previous_space_held = False
        self.previous_shift_held = False

        self.message = f"Wave 1 starts in {self.wave_timer/30:.1f}s"
        self.message_timer = self.FIRST_WAVE_TIME
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.01  # Small reward for surviving
        self.steps += 1
        if self.message_timer > 0:
            self.message_timer -= 1

        # 1. HANDLE INPUT
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # 2. UPDATE GAME STATE
        if self.game_phase == "break":
            self.wave_timer -= 1
            self.message = f"Wave {self.current_wave + 1} starts in {max(0, self.wave_timer/30):.1f}s"
            self.message_timer = 90
            if self.wave_timer <= 0:
                self._start_next_wave()
        
        elif self.game_phase == "wave":
            self._update_towers()
            reward += self._update_projectiles()
            reward += self._update_enemies()
            
            if not self.enemies and self.current_wave <= self.MAX_WAVES:
                self.game_phase = "break"
                self.wave_timer = self.WAVE_BREAK_TIME
                self.gold += 100 + self.current_wave * 10
                self.message = f"Wave {self.current_wave} Cleared! Bonus: {100 + self.current_wave * 10} Gold"
                self.message_timer = 120
                
        self._update_particles()
        
        # 3. CHECK TERMINATION
        terminated = False
        if self.base_health <= 0:
            reward = -100
            terminated = True
            self.game_over = True
            self.message = "Game Over: Your base was destroyed."
            self.message_timer = 300
        elif self.current_wave > self.MAX_WAVES and not self.enemies:
            reward = 100
            terminated = True
            self.game_over = True
            self.message = "Victory! You survived all waves."
            self.message_timer = 300
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.message = "Game Over: Time limit reached."
            self.message_timer = 300
        
        self.score += reward
        self.previous_space_held = space_held
        self.previous_shift_held = shift_held

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        cursor_speed = 5
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)
        
        # Place tower on space PRESS ("fire weapon")
        if space_held and not self.previous_space_held:
            self._place_tower()

        # Cycle tower type on shift PRESS ("drift")
        if shift_held and not self.previous_shift_held:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.available_tower_types)

    def _place_tower(self):
        grid_pos = self._world_to_grid(self.cursor_pos[0], self.cursor_pos[1])
        spec = self.TOWER_SPECS[self.available_tower_types[self.selected_tower_type_idx]]
        is_valid_placement = tuple(grid_pos) in self.placement_zones and not any(t['grid_pos'] == grid_pos for t in self.towers)
        
        if self.gold >= spec['cost'] and is_valid_placement:
            self.gold -= spec['cost']
            self.towers.append({
                "type": self.available_tower_types[self.selected_tower_type_idx],
                "pos": self._grid_to_world(grid_pos[0], grid_pos[1]),
                "grid_pos": grid_pos,
                "cooldown": 0,
                "target": None,
            })
            # sfx: place_tower.wav
        elif self.gold < spec['cost']:
            self.message = "Not enough gold!"
            self.message_timer = 60
        elif not is_valid_placement:
            self.message = "Cannot build there!"
            self.message_timer = 60

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES: return

        self.game_phase = "wave"
        
        for i, spec in self.TOWER_SPECS.items():
            if self.current_wave >= spec["unlocks_at"] and i not in self.available_tower_types:
                self.available_tower_types.append(i)
                self.message = f"Unlocked: {spec['name']} Tower!"
                self.message_timer = 120
        
        num_enemies = 3 + self.current_wave
        base_health = 50 * (1 + (self.current_wave - 1) * 0.15)
        base_speed = 1.0 * (1 + (self.current_wave - 1) * 0.10)
        
        for _ in range(num_enemies):
            offset = self.np_random.uniform(-5, 5, size=2)
            self.enemies.append({
                "pos": [self.path_world[0][0] + offset[0], self.path_world[0][1] + offset[1]],
                "max_health": base_health,
                "health": base_health,
                "speed": base_speed * self.np_random.uniform(0.9, 1.1),
                "path_index": 0,
                "value": 10 + self.current_wave,
            })
            
    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue

            target = None
            min_dist_sq = spec["range"] ** 2
            for enemy in self.enemies:
                dist_sq = (tower["pos"][0] - enemy["pos"][0])**2 + (tower["pos"][1] - enemy["pos"][1])**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    target = enemy
            
            tower["target"] = target
            if target and tower["cooldown"] <= 0:
                # sfx: shoot.wav
                self.projectiles.append({
                    "pos": list(tower["pos"]),
                    "target": target,
                    "speed": 8,
                    "damage": spec["damage"],
                    "color": spec["proj_color"]
                })
                tower["cooldown"] = spec["fire_rate"]

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            target_pos = proj["target"]["pos"]
            direction = [target_pos[0] - proj["pos"][0], target_pos[1] - proj["pos"][1]]
            dist = math.hypot(*direction)
            
            if dist < proj["speed"]:
                proj["target"]["health"] -= proj["damage"]
                self._create_particles(proj["pos"], 10, (255, 255, 100))
                self.projectiles.remove(proj)
                # sfx: hit.wav
                if proj["target"]["health"] <= 0:
                    reward += 1
                    self.gold += proj["target"]["value"]
                    self._create_particles(proj["target"]["pos"], 30, self.COLOR_ENEMY)
                    self.enemies.remove(proj["target"])
                    # sfx: explosion.wav
            else:
                proj["pos"][0] += (direction[0] / dist) * proj["speed"]
                proj["pos"][1] += (direction[1] / dist) * proj["speed"]
        return reward
        
    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy["path_index"] >= len(self.path_world) - 1:
                damage = 50
                self.base_health = max(0, self.base_health - damage)
                reward -= damage
                self.enemies.remove(enemy)
                self._create_particles(self.base_pos_world, 30, self.COLOR_BASE)
                # sfx: base_damage.wav
                continue

            target_pos = self.path_world[enemy["path_index"] + 1]
            direction = [target_pos[0] - enemy["pos"][0], target_pos[1] - enemy["pos"][1]]
            dist = math.hypot(*direction)

            if dist < enemy["speed"]:
                enemy["path_index"] += 1
            else:
                enemy["pos"][0] += (direction[0] / dist) * enemy["speed"]
                enemy["pos"][1] += (direction[1] / dist) * enemy["speed"]
        return reward

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95
            p["vel"][1] *= 0.95
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for r in range(self.GRID_WIDTH):
            for c in range(self.GRID_HEIGHT):
                pos = self._grid_to_world(r, c)
                tile_points = [(pos[0], pos[1] - self.TILE_H_HALF), (pos[0] + self.TILE_W_HALF, pos[1]), (pos[0], pos[1] + self.TILE_H_HALF), (pos[0] - self.TILE_W_HALF, pos[1])]
                color = self.COLOR_PATH if (r, c) in self.path_grid else self.COLOR_GRID
                if (r,c) in self.placement_zones:
                    pygame.gfxdraw.filled_polygon(self.screen, tile_points, (*color, 50))
                pygame.gfxdraw.aapolygon(self.screen, tile_points, color)

        base_points = [(self.base_pos_world[0], self.base_pos_world[1] - self.TILE_H_HALF - 5), (self.base_pos_world[0] + self.TILE_W_HALF + 5, self.base_pos_world[1]), (self.base_pos_world[0], self.base_pos_world[1] + self.TILE_H_HALF + 5), (self.base_pos_world[0] - self.TILE_W_HALF - 5, self.base_pos_world[1])]
        pygame.gfxdraw.filled_polygon(self.screen, base_points, self.COLOR_BASE)
        pygame.gfxdraw.aapolygon(self.screen, base_points, tuple(min(255, c+30) for c in self.COLOR_BASE))
        self._draw_health_bar(self.base_pos_world, self.base_health, self.INITIAL_BASE_HEALTH, 40)

        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            pos_int = (int(tower["pos"][0]), int(tower["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, *pos_int, 8, spec["color"])
            pygame.gfxdraw.aacircle(self.screen, *pos_int, 8, tuple(min(255, c+30) for c in spec["color"]))

        for enemy in self.enemies:
            pos_int = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            size = 7
            points = [(pos_int[0], pos_int[1]-size), (pos_int[0]+size, pos_int[1]), (pos_int[0], pos_int[1]+size), (pos_int[0]-size, pos_int[1])]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.aapolygon(self.screen, points, tuple(min(255, c+30) for c in self.COLOR_ENEMY))
            self._draw_health_bar(enemy["pos"], enemy["health"], enemy["max_health"], 20)

        for proj in self.projectiles:
            pos_int = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, *pos_int, 3, proj["color"])
            pygame.gfxdraw.aacircle(self.screen, *pos_int, 3, tuple(min(255, c+50) for c in proj["color"]))

        for p in self.particles:
            alpha = max(0, min(255, int(p["life"] * 8.5)))
            if p["life"] > 0:
                 pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["life"]/4), (*p["color"], alpha))

        self._draw_cursor()

    def _draw_cursor(self):
        cursor_x, cursor_y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        spec = self.TOWER_SPECS[self.available_tower_types[self.selected_tower_type_idx]]
        grid_pos = self._world_to_grid(cursor_x, cursor_y)
        is_valid = tuple(grid_pos) in self.placement_zones and not any(t['grid_pos'] == grid_pos for t in self.towers)
        color = (0, 255, 0) if is_valid and self.gold >= spec['cost'] else (255, 0, 0)
        
        pygame.gfxdraw.aacircle(self.screen, cursor_x, cursor_y, spec["range"], (*color, 100))
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (cursor_x - 10, cursor_y), (cursor_x + 10, cursor_y), 1)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (cursor_x, cursor_y - 10), (cursor_x, cursor_y + 10), 1)

    def _draw_health_bar(self, pos, current, maximum, width):
        if maximum <= 0: return
        x, y = pos[0] - width // 2, pos[1] - 25
        ratio = max(0, current / maximum)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (x, y, width, 5))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GOOD, (x, y, int(width * ratio), 5))

    def _render_ui(self):
        bar = pygame.Surface((self.WIDTH, 30), pygame.SRCALPHA); bar.fill((20, 20, 20, 200)); self.screen.blit(bar, (0, 0))
        gold_text = self.font_m.render(f"Gold: {self.gold}", True, self.COLOR_GOLD); self.screen.blit(gold_text, (10, 5))
        wave_text = self.font_m.render(f"Wave: {min(self.current_wave, self.MAX_WAVES)}/{self.MAX_WAVES}", True, self.COLOR_TEXT); self.screen.blit(wave_text, (self.WIDTH // 2 - wave_text.get_width() // 2, 5))
        health_text = self.font_m.render(f"Base: {self.base_health}", True, self.COLOR_BASE); self.screen.blit(health_text, (self.WIDTH - health_text.get_width() - 10, 5))
        
        spec = self.TOWER_SPECS[self.available_tower_types[self.selected_tower_type_idx]]
        y_offset = self.HEIGHT - 70
        panel = pygame.Surface((220, 65), pygame.SRCALPHA); panel.fill((20, 20, 20, 200)); self.screen.blit(panel, (5, y_offset))
        name_text = self.font_m.render(spec["name"], True, spec["color"]); self.screen.blit(name_text, (15, y_offset + 5))
        cost_text = self.font_s.render(f"Cost: {spec['cost']}", True, self.COLOR_GOLD); self.screen.blit(cost_text, (15, y_offset + 25))
        dmg_text = self.font_s.render(f"Dmg: {spec['damage']}", True, self.COLOR_TEXT); self.screen.blit(dmg_text, (15, y_offset + 45))
        range_text = self.font_s.render(f"Rng: {spec['range']}", True, self.COLOR_TEXT); self.screen.blit(range_text, (100, y_offset + 45))
        rate_text = self.font_s.render(f"Rate: {30/spec['fire_rate']:.1f}/s", True, self.COLOR_TEXT); self.screen.blit(rate_text, (100, y_offset + 25))

        if self.message_timer > 0:
            msg_text = self.font_l.render(self.message, True, self.COLOR_TEXT)
            pos = (self.WIDTH // 2 - msg_text.get_width() // 2, self.HEIGHT // 2 - msg_text.get_height() // 2)
            self.screen.blit(msg_text, pos)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.current_wave,
            "base_health": self.base_health,
            "enemies_remaining": len(self.enemies),
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        movement, space_held, shift_held = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()