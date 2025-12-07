
# Generated: 2025-08-28T03:17:45.099731
# Source Brief: brief_04878.md
# Brief Index: 4878

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Ensure Pygame runs headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move selector. Space to place a Machine Gun Tower. Shift to place a Cannon Tower."
    )

    game_description = (
        "Defend your base from waves of geometric enemies by placing towers on a grid in this minimalist, real-time strategy game."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 16, 10
        self.CELL_SIZE = 40
        self.MAX_STEPS = 2500 # Increased for better gameplay balance
        self.MAX_WAVES = 20
        self.FPS = 30

        # Colors
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_GRID = (30, 45, 65)
        self.COLOR_PATH = (25, 35, 55)
        self.COLOR_BASE = (50, 205, 50)
        self.COLOR_BASE_DMG = (255, 100, 100)
        self.COLOR_ENEMY = (255, 60, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_SELECTOR = (255, 255, 0, 100)
        
        # Tower Type Config
        self.TOWER_TYPES = {
            "MACHINE_GUN": {
                "cost": 10, "color": (0, 150, 255), "range": 80, 
                "damage": 3, "fire_rate": 5, "projectile_speed": 10
            },
            "CANNON": {
                "cost": 25, "color": (150, 100, 255), "range": 120,
                "damage": 15, "fire_rate": 30, "projectile_speed": 7
            }
        }

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.game_over = False
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = 100
        self.max_base_health = 100
        self.resources = 30
        
        self.selector_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.grid = {} # Stores tower references by (gx, gy)
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self._setup_path()
        
        self.current_wave = 0
        self.wave_timer = 90 # Initial delay before first wave
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def _setup_path(self):
        self.path_coords = []
        self.path_grid_coords = []
        path_points = [(0, 2), (13, 2), (13, 7), (2, 7), (2, 4), (11, 4), (11, 5), (15, 5)]
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]
            self.path_grid_coords.append(p1)
            if p1[0] == p2[0]: # Vertical
                for y in range(min(p1[1], p2[1]), max(p1[1], p2[1])):
                    self.path_grid_coords.append((p1[0], y))
            else: # Horizontal
                for x in range(min(p1[0], p2[0]), max(p1[0], p2[0])):
                    self.path_grid_coords.append((x, p1[1]))
        self.path_grid_coords.append(path_points[-1])
        self.path_grid_coords = list(dict.fromkeys(self.path_grid_coords)) # Remove duplicates

        self.path_coords = [(gx * self.CELL_SIZE + self.CELL_SIZE / 2, gy * self.CELL_SIZE + self.CELL_SIZE / 2) for gx, gy in self.path_grid_coords]
        self.spawn_pos = self.path_coords[0]
        self.base_pos = self.path_coords[-1]
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.pending_reward = 0
        self.steps += 1
        
        self._handle_input(action)
        self._update_wave_spawner()
        self._update_towers()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        
        reward = self.pending_reward
        self.score += reward
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.base_health <= 0:
                reward -= 100
            elif self.current_wave > self.MAX_WAVES:
                reward += 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Selector Movement ---
        if movement == 1: self.selector_pos[1] = max(0, self.selector_pos[1] - 1)
        elif movement == 2: self.selector_pos[1] = min(self.GRID_H - 1, self.selector_pos[1] + 1)
        elif movement == 3: self.selector_pos[0] = max(0, self.selector_pos[0] - 1)
        elif movement == 4: self.selector_pos[0] = min(self.GRID_W - 1, self.selector_pos[0] + 1)
        
        # --- Tower Placement ---
        can_place = tuple(self.selector_pos) not in self.grid and tuple(self.selector_pos) not in self.path_grid_coords
        
        # Prioritize shift (advanced tower)
        if shift_held and not self.prev_shift_held and can_place:
            self._try_place_tower("CANNON")
        elif space_held and not self.prev_space_held and can_place:
            self._try_place_tower("MACHINE_GUN")
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _try_place_tower(self, tower_type):
        config = self.TOWER_TYPES[tower_type]
        if self.resources >= config["cost"]:
            self.resources -= config["cost"]
            
            gx, gy = self.selector_pos
            new_tower = {
                "type": tower_type, "gx": gx, "gy": gy,
                "pos": (gx * self.CELL_SIZE + self.CELL_SIZE / 2, gy * self.CELL_SIZE + self.CELL_SIZE / 2),
                "cooldown": 0, "build_anim": self.CELL_SIZE * 0.7
            }
            self.towers.append(new_tower)
            self.grid[(gx, gy)] = new_tower
            # sfx: place_tower.wav
    
    def _update_wave_spawner(self):
        if self.enemies_to_spawn == 0 and not self.enemies:
            self.wave_timer -= 1
            if self.wave_timer <= 0 and self.current_wave < self.MAX_WAVES:
                self.current_wave += 1
                self.pending_reward += 1.0 # Wave survival reward
                self.enemies_to_spawn = 5 + self.current_wave * 2
                self.spawn_timer = 0
                self.wave_timer = 150 # Cooldown between waves
        
        if self.enemies_to_spawn > 0:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self.spawn_timer = max(5, 20 - self.current_wave)
                self.enemies_to_spawn -= 1
                
                speed_mod = 1 + (self.current_wave - 1) * 0.05
                health_mod = 10 + (self.current_wave - 1) * 2
                
                new_enemy = {
                    "pos": list(self.spawn_pos), "health": health_mod, "max_health": health_mod,
                    "speed": speed_mod, "path_index": 0, "value": 2 + self.current_wave // 5,
                    "hit_timer": 0
                }
                self.enemies.append(new_enemy)
                # sfx: enemy_spawn.wav

    def _update_towers(self):
        for tower in self.towers:
            if tower["build_anim"] > 0:
                tower["build_anim"] -= 2
                continue

            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] > 0:
                continue

            config = self.TOWER_TYPES[tower["type"]]
            target = None
            max_dist = -1
            for enemy in self.enemies:
                dist_sq = (tower["pos"][0] - enemy["pos"][0])**2 + (tower["pos"][1] - enemy["pos"][1])**2
                if dist_sq < config["range"]**2:
                    # Target enemy furthest along the path
                    if enemy["path_index"] > max_dist:
                        max_dist = enemy["path_index"]
                        target = enemy
            
            if target:
                tower["cooldown"] = config["fire_rate"]
                self.projectiles.append({
                    "pos": list(tower["pos"]), "target": target, "type": tower["type"]
                })
                # sfx: shoot.wav

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            enemy["hit_timer"] = max(0, enemy["hit_timer"] - 1)
            
            if enemy["path_index"] >= len(self.path_coords) - 1:
                self.base_health -= 5
                self.pending_reward -= 0.5
                self.enemies.remove(enemy)
                # sfx: base_damage.wav
                for _ in range(10):
                    self.particles.append(self._create_particle(self.base_pos, self.COLOR_BASE_DMG))
                continue
                
            target_pos = self.path_coords[enemy["path_index"] + 1]
            dx = target_pos[0] - enemy["pos"][0]
            dy = target_pos[1] - enemy["pos"][1]
            dist = math.hypot(dx, dy)
            
            if dist < enemy["speed"]:
                enemy["path_index"] += 1
            else:
                enemy["pos"][0] += (dx / dist) * enemy["speed"]
                enemy["pos"][1] += (dy / dist) * enemy["speed"]
    
    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            config = self.TOWER_TYPES[proj["type"]]
            target_pos = proj["target"]["pos"]
            dx = target_pos[0] - proj["pos"][0]
            dy = target_pos[1] - proj["pos"][1]
            dist = math.hypot(dx, dy)
            
            if dist < config["projectile_speed"]:
                proj["target"]["health"] -= config["damage"]
                proj["target"]["hit_timer"] = 3
                if proj["target"]["health"] <= 0:
                    self.pending_reward += 0.1
                    self.resources += proj["target"]["value"]
                    # sfx: enemy_die.wav
                    for _ in range(15):
                        self.particles.append(self._create_particle(proj["target"]["pos"], self.COLOR_ENEMY))
                    self.enemies.remove(proj["target"])
                
                self.projectiles.remove(proj)
                # sfx: hit.wav
            else:
                proj["pos"][0] += (dx / dist) * config["projectile_speed"]
                proj["pos"][1] += (dy / dist) * config["projectile_speed"]

    def _create_particle(self, pos, color):
        return {
            "pos": list(pos),
            "vel": [self.np_random.uniform(-1.5, 1.5), self.np_random.uniform(-1.5, 1.5)],
            "lifespan": self.np_random.integers(15, 30),
            "color": color,
        }

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)
    
    def _check_termination(self):
        return (self.base_health <= 0 or 
                self.current_wave > self.MAX_WAVES or 
                self.steps >= self.MAX_STEPS)
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Grid and Path
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                rect = (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if (x, y) in self.path_grid_coords:
                    pygame.draw.rect(self.screen, self.COLOR_PATH, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
        
        # Base
        base_gx, base_gy = self.path_grid_coords[-1]
        base_rect = (base_gx * self.CELL_SIZE, base_gy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        
        # Towers
        for tower in self.towers:
            config = self.TOWER_TYPES[tower["type"]]
            pos_int = (int(tower["pos"][0]), int(tower["pos"][1]))
            pygame.draw.circle(self.screen, config["color"], pos_int, self.CELL_SIZE // 3)
            pygame.draw.circle(self.screen, self.COLOR_BG, pos_int, self.CELL_SIZE // 4)
            if tower["build_anim"] > 0:
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(tower["build_anim"]), (*config["color"], 150))

        # Enemies
        for enemy in self.enemies:
            pos_int = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            color = (255, 255, 255) if enemy["hit_timer"] > 0 else self.COLOR_ENEMY
            size = int(self.CELL_SIZE * 0.3 + math.sin(self.steps * 0.2) * 2)
            pygame.draw.circle(self.screen, color, pos_int, size)
            
            # Health bar
            if enemy["health"] < enemy["max_health"]:
                hp_ratio = enemy["health"] / enemy["max_health"]
                bar_w = self.CELL_SIZE * 0.6
                pygame.draw.rect(self.screen, (80,0,0), (pos_int[0] - bar_w/2, pos_int[1] - size - 8, bar_w, 4))
                pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos_int[0] - bar_w/2, pos_int[1] - size - 8, bar_w * hp_ratio, 4))

        # Projectiles
        for proj in self.projectiles:
            color = self.TOWER_TYPES[proj["type"]]["color"]
            pygame.draw.circle(self.screen, color, (int(proj["pos"][0]), int(proj["pos"][1])), 4)
        
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 20.0))))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 3, 3))
            self.screen.blit(temp_surf, (int(p["pos"][0]-1), int(p["pos"][1]-1)))

        # Selector
        sel_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        alpha = 100 + math.sin(self.steps * 0.3) * 30
        pygame.draw.rect(sel_surf, (*self.COLOR_SELECTOR[:3], alpha), (0, 0, self.CELL_SIZE, self.CELL_SIZE), 3)
        self.screen.blit(sel_surf, (self.selector_pos[0] * self.CELL_SIZE, self.selector_pos[1] * self.CELL_SIZE))

    def _render_ui(self):
        # Health bar
        health_ratio = max(0, self.base_health / self.max_base_health)
        bar_width = 200
        pygame.draw.rect(self.screen, (80, 0, 0), (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (10, 10, bar_width * health_ratio, 20))
        health_text = self.font_small.render(f"{int(self.base_health)}/{self.max_base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))
        
        # Resources
        res_text = self.font_large.render(f"${int(self.resources)}", True, self.COLOR_SELECTOR[:3])
        self.screen.blit(res_text, (220, 8))
        
        # Wave
        wave_str = f"Wave: {self.current_wave}/{self.MAX_WAVES}"
        if self.enemies_to_spawn == 0 and not self.enemies and self.current_wave < self.MAX_WAVES:
             wave_str = f"Next wave in {self.wave_timer / self.FPS:.1f}s"
        wave_text = self.font_large.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 8))

        # Score
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 35))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave,
        }

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be executed when the environment is used by Gymnasium
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Tower Defense Gym")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

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
        
        if terminated:
            print(f"Game Over! Final Score: {total_reward:.2f}, Info: {info}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()