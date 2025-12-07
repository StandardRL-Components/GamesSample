
# Generated: 2025-08-28T06:40:40.827647
# Source Brief: brief_02999.md
# Brief Index: 2999

        
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
        "Controls: Use arrow keys to move the placement cursor. "
        "Hold Shift to cycle through tower types. Press Space to build the selected tower."
    )

    game_description = (
        "Defend your base against waves of enemies by strategically placing procedurally generated towers "
        "in this isometric 2D tower defense game. Survive 20 waves to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Pygame Setup
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 14)
        self.font_m = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 24, bold=True)

        # Spaces
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Game Constants
        self.MAX_STEPS = 3000 # Increased for longer games
        self.MAX_WAVES = 20
        self.GOLD_PER_STEP = 0.02
        self.INITIAL_GOLD = 250
        self.INITIAL_BASE_HEALTH = 100
        
        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_PATH = (60, 70, 80)
        self.COLOR_BASE = (50, 100, 200)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_HEALTH = (40, 200, 80)
        self.COLOR_HEALTH_BG = (200, 40, 40)
        
        # Isometric Grid
        self.grid_w, self.grid_h = 16, 10
        self.tile_w, self.tile_h = 32, 16
        self.origin_x, self.origin_y = self.width // 2, 80

        # Enemy Path
        self.path_coords = [
            (-1, 4), (5, 4), (5, 1), (10, 1), (10, 7),
            (3, 7), (3, 9), (self.grid_w, 9)
        ]
        self.path_points = [self._iso_to_screen(x, y) for x, y in self.path_coords]

        # Tower Definitions
        self.TOWER_TYPES = [
            {"name": "Gatling", "cost": 100, "range": 80, "damage": 6, "fire_rate": 8, "color": (0, 255, 128), "proj_speed": 8},
            {"name": "Cannon", "cost": 200, "range": 120, "damage": 25, "fire_rate": 40, "color": (255, 128, 0), "proj_speed": 6},
            {"name": "Frost", "cost": 150, "range": 70, "damage": 2, "fire_rate": 20, "color": (0, 192, 255), "proj_speed": 7, "slow": 0.5},
        ]
        
        # Initialize state variables
        self.reset()
        
        # Run validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.gold = self.INITIAL_GOLD
        self.base_health = self.INITIAL_BASE_HEALTH
        self.last_health_penalty_milestone = self.INITIAL_BASE_HEALTH

        self.wave_number = 0
        self.wave_timer = 150 # Time before first wave
        self.enemies_in_wave = 0
        self.enemies_spawned = 0
        self.spawn_timer = 0
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.grid_w // 2, self.grid_h // 2]
        self.selected_tower_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0

        self._handle_input(action)

        reward += self._update_waves()
        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        self.gold += self.GOLD_PER_STEP
        reward += self.GOLD_PER_STEP
        
        terminated, term_reward = self._check_termination()
        reward += term_reward
        if terminated:
            self.game_over = True
            
        self.score += reward
        
        self.clock.tick(30)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement: Move cursor ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.grid_w - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.grid_h - 1)

        # --- Shift: Cycle tower type ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.TOWER_TYPES)
            # sfx: ui_cycle.wav

        # --- Space: Place tower ---
        if space_held and not self.prev_space_held:
            self._place_tower()
            # sfx: handled in _place_tower

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _place_tower(self):
        selected_spec = self.TOWER_TYPES[self.selected_tower_idx]
        if self.gold < selected_spec["cost"]:
            # sfx: ui_error.wav
            return False

        x, y = self.cursor_pos
        is_on_path = any(px == x and py == y for px, py in self.path_coords)
        is_occupied = any(t["grid_pos"] == [x, y] for t in self.towers)

        if not is_on_path and not is_occupied:
            self.gold -= selected_spec["cost"]
            screen_pos = self._iso_to_screen(x, y)
            new_tower = {
                "spec": selected_spec,
                "grid_pos": [x, y],
                "pos": screen_pos,
                "cooldown": 0,
            }
            self.towers.append(new_tower)
            # sfx: build_tower.wav
            self._create_particles(screen_pos, 15, selected_spec["color"])
            return True
        # sfx: ui_error.wav
        return False

    def _update_waves(self):
        if self.wave_number > self.MAX_WAVES and not self.enemies:
            return 0
        
        self.wave_timer -= 1
        if self.wave_timer <= 0 and self.enemies_spawned >= self.enemies_in_wave:
            self._start_next_wave()

        if self.enemies_spawned < self.enemies_in_wave:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self._spawn_enemy()
                self.enemies_spawned += 1
                self.spawn_timer = 20 # Time between enemies in a wave
        return 0

    def _start_next_wave(self):
        if self.wave_number > self.MAX_WAVES: return
        self.wave_number += 1
        self.wave_timer = 300 # Time between waves
        self.enemies_in_wave = 8 + self.wave_number * 2
        self.enemies_spawned = 0

    def _spawn_enemy(self):
        health = 40 * (1.05 ** (self.wave_number - 1))
        speed = 1.0 * (1.02 ** (self.wave_number - 1))
        self.enemies.append({
            "pos": list(self.path_points[0]),
            "max_health": health,
            "health": health,
            "speed": speed,
            "path_idx": 1,
            "slow_timer": 0
        })

    def _update_towers(self):
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue
            
            target = None
            min_dist = tower["spec"]["range"] ** 2
            for enemy in self.enemies:
                dist_sq = (enemy["pos"][0] - tower["pos"][0])**2 + (enemy["pos"][1] - tower["pos"][1])**2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = enemy
            
            if target:
                tower["cooldown"] = tower["spec"]["fire_rate"]
                self.projectiles.append({
                    "pos": list(tower["pos"]),
                    "spec": tower["spec"],
                    "target": target,
                })
                # sfx: tower_fire.wav
        return 0

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_pos = proj["target"]["pos"]
            proj_pos = proj["pos"]
            dx, dy = target_pos[0] - proj_pos[0], target_pos[1] - proj_pos[1]
            dist = math.hypot(dx, dy)
            
            if dist < proj["spec"]["proj_speed"]:
                # Hit
                proj["target"]["health"] -= proj["spec"]["damage"]
                if "slow" in proj["spec"]:
                    proj["target"]["slow_timer"] = 60 # Slow for 2 seconds
                
                self._create_particles(proj["pos"], 5, proj["spec"]["color"])

                if proj["target"]["health"] <= 0:
                    # sfx: enemy_die.wav
                    self._create_particles(proj["target"]["pos"], 20, (255, 50, 50))
                    self.gold += 5 + self.wave_number
                    self.enemies.remove(proj["target"])
                    reward += 1
                else:
                    # sfx: enemy_hit.wav
                    pass
                
                self.projectiles.remove(proj)
            else:
                # Move towards target
                proj_pos[0] += (dx / dist) * proj["spec"]["proj_speed"]
                proj_pos[1] += (dy / dist) * proj["spec"]["proj_speed"]
        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy["path_idx"] >= len(self.path_points):
                self.base_health -= 10
                # sfx: base_damage.wav
                self._create_particles(enemy["pos"], 30, self.COLOR_HEALTH_BG)
                if self.base_health < self.last_health_penalty_milestone - self.INITIAL_BASE_HEALTH * 0.1:
                    reward -= 5
                    self.last_health_penalty_milestone = self.base_health
                self.enemies.remove(enemy)
                continue

            target_pos = self.path_points[enemy["path_idx"]]
            dx, dy = target_pos[0] - enemy["pos"][0], target_pos[1] - enemy["pos"][1]
            dist = math.hypot(dx, dy)
            
            speed = enemy["speed"]
            if enemy["slow_timer"] > 0:
                enemy["slow_timer"] -= 1
                speed *= 0.5 # 50% slow

            if dist < speed:
                enemy["pos"] = list(target_pos)
                enemy["path_idx"] += 1
            else:
                enemy["pos"][0] += (dx / dist) * speed
                enemy["pos"][1] += (dy / dist) * speed
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            return True, -100 # Loss
        if self.wave_number > self.MAX_WAVES and not self.enemies:
            return True, 100 # Win
        if self.steps >= self.MAX_STEPS:
            return True, 0 # Timeout
        return False, 0

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
            "gold": self.gold,
            "base_health": self.base_health,
            "wave": self.wave_number,
        }

    def _iso_to_screen(self, x, y):
        return (
            self.origin_x + (x - y) * self.tile_w / 2,
            self.origin_y + (x + y) * self.tile_h / 2
        )

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(10, 25),
                "max_life": 25,
                "color": color
            })

    def _render_game(self):
        # Grid and Path
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + 1, y)
                p3 = self._iso_to_screen(x + 1, y + 1)
                p4 = self._iso_to_screen(x, y + 1)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p4)

        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_points, 5)

        # Base
        base_pos = self.path_points[-1]
        pygame.gfxdraw.filled_circle(self.screen, int(base_pos[0]), int(base_pos[1]), 12, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, int(base_pos[0]), int(base_pos[1]), 12, tuple(c*0.8 for c in self.COLOR_BASE))

        # Towers
        for tower in self.towers:
            x, y = int(tower["pos"][0]), int(tower["pos"][1])
            color = tower["spec"]["color"]
            pygame.draw.rect(self.screen, tuple(c*0.5 for c in color), (x - 6, y - 3, 12, 6))
            pygame.gfxdraw.box(self.screen, (x - 4, y - 8, 8, 8), color)
            if tower["cooldown"] > tower["spec"]["fire_rate"] - 3: # Firing flash
                pygame.gfxdraw.filled_circle(self.screen, x, y-6, 3, (255,255,255))

        # Enemies
        for enemy in self.enemies:
            x, y = int(enemy["pos"][0]), int(enemy["pos"][1])
            color = (220, 40, 40)
            if enemy["slow_timer"] > 0:
                color = (100, 100, 255) # Blue when slowed
            pygame.gfxdraw.filled_trigon(self.screen, x, y-6, x-4, y+2, x+4, y+2, color)
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (x - 8, y - 12, 16, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH, (x - 8, y - 12, 16 * health_ratio, 3))

        # Projectiles
        for proj in self.projectiles:
            x, y = int(proj["pos"][0]), int(proj["pos"][1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, 3, proj["spec"]["color"])
            pygame.gfxdraw.aacircle(self.screen, x, y, 3, (255,255,255))
            
        # Particles
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95
            p["vel"][1] *= 0.95
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            size = int(3 * (p["life"] / p["max_life"]))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), size, color)

        # Cursor
        cx, cy = self.cursor_pos
        cursor_points = [
            self._iso_to_screen(cx, cy), self._iso_to_screen(cx + 1, cy),
            self._iso_to_screen(cx + 1, cy + 1), self._iso_to_screen(cx, cy + 1)
        ]
        is_on_path = any(px == cx and py == cy for px, py in self.path_coords)
        is_occupied = any(t["grid_pos"] == [cx, cy] for t in self.towers)
        can_afford = self.gold >= self.TOWER_TYPES[self.selected_tower_idx]["cost"]
        cursor_color = (0, 255, 0) if not is_on_path and not is_occupied and can_afford else (255, 0, 0)
        pygame.draw.aalines(self.screen, cursor_color, True, cursor_points, 2)

    def _render_ui(self):
        # Top Bar
        bar_h = 30
        pygame.draw.rect(self.screen, (10,15,20), (0, 0, self.width, bar_h))
        
        # Gold
        gold_text = self.font_m.render(f"GOLD: {int(self.gold)}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (10, 5))
        
        # Wave
        wave_str = f"WAVE: {self.wave_number}/{self.MAX_WAVES}"
        if self.wave_timer > 0 and self.enemies_spawned >= self.enemies_in_wave:
            wave_str += f" (in {self.wave_timer//30}s)"
        wave_text = self.font_m.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.width // 2 - wave_text.get_width() // 2, 5))

        # Base Health
        health_ratio = max(0, self.base_health / self.INITIAL_BASE_HEALTH)
        health_bar_w = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (self.width - health_bar_w - 10, 8, health_bar_w, 14))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (self.width - health_bar_w - 10, 8, health_bar_w * health_ratio, 14))
        health_text = self.font_s.render(f"{int(self.base_health)}/{self.INITIAL_BASE_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.width - health_bar_w/2 - 10 - health_text.get_width()/2, 8))

        # Selected Tower Info Panel
        panel_w, panel_h = 200, 85
        panel_x, panel_y = 10, self.height - panel_h - 10
        pygame.draw.rect(self.screen, (10,15,20, 200), (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(self.screen, self.COLOR_GRID, (panel_x, panel_y, panel_w, panel_h), 1)

        spec = self.TOWER_TYPES[self.selected_tower_idx]
        title = self.font_m.render(f"[Shift] {spec['name']}", True, spec["color"])
        self.screen.blit(title, (panel_x + 10, panel_y + 5))

        cost_color = self.COLOR_GOLD if self.gold >= spec["cost"] else (255, 80, 80)
        cost = self.font_s.render(f"Cost: {spec['cost']}", True, cost_color)
        self.screen.blit(cost, (panel_x + 10, panel_y + 30))

        dmg = self.font_s.render(f"DMG: {spec['damage']} | Range: {spec['range']}", True, self.COLOR_TEXT)
        self.screen.blit(dmg, (panel_x + 10, panel_y + 48))
        
        rate = self.font_s.render(f"Rate: {round(30/spec['fire_rate'],1)}/s", True, self.COLOR_TEXT)
        self.screen.blit(rate, (panel_x + 10, panel_y + 66))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Isometric Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("      HUMAN PLAYTHROUGH")
    print("="*30)
    print(env.user_guide)

    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30)

    env.close()