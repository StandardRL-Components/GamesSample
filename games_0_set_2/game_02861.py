
# Generated: 2025-08-28T06:11:24.153992
# Source Brief: brief_02861.md
# Brief Index: 2861

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the placement cursor. "
        "Space to build a tower. Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers "
        "along their path. Survive all waves to win."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    # Game parameters
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 20
    MAX_STEPS = 5000 # Increased from 1000 to allow for 10 waves
    TOTAL_WAVES = 10
    
    # Initial state
    INITIAL_BASE_HEALTH = 20
    INITIAL_MONEY = 150

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_PATH = (50, 55, 68)
    COLOR_GRID = (40, 44, 55)
    COLOR_BASE = (0, 200, 100)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_PROJECTILE = (255, 200, 0)
    COLOR_TEXT = (230, 230, 230)
    COLOR_CURSOR_VALID = (100, 255, 100, 100)
    COLOR_CURSOR_INVALID = (255, 100, 100, 100)

    TOWER_SPECS = {
        0: {"name": "Cannon", "cost": 50, "range": 100, "fire_rate": 45, "damage": 2, "color": (60, 140, 255)},
        1: {"name": "Sniper", "cost": 100, "range": 200, "fire_rate": 90, "damage": 5, "color": (200, 60, 255)},
    }


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_large = pygame.font.SysFont("sans-serif", 24)
        self.font_huge = pygame.font.SysFont("sans-serif", 48)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.base_pos = (0, 0)
        self.base_health = 0
        self.money = 0
        self.wave_number = 0
        self.wave_countdown = 0
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        self.path_waypoints = []
        self.path_rects = []
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.money = self.INITIAL_MONEY
        
        self.wave_number = 0
        self.wave_countdown = 150 # Time before first wave
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self._generate_path()
        
        self.cursor_pos = [self.WIDTH // 2 // self.GRID_SIZE, self.HEIGHT // 2 // self.GRID_SIZE]
        self.selected_tower_type = 0
        
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        # 1. Handle Input
        reward += self._handle_input(action)
        
        # 2. Update Game State
        self._manage_waves()
        reward += self._update_enemies()
        reward += self._update_towers()
        reward += self._update_projectiles()
        self._update_particles()
        
        self.steps += 1
        
        # 3. Check Termination Conditions
        if self.base_health <= 0:
            terminated = True
            self.game_over = True
            reward = -100.0
        elif self.wave_number > self.TOTAL_WAVES and not self.enemies:
            terminated = True
            self.game_over = True
            reward = 100.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            # No terminal reward for timeout, let shaped rewards guide
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        if movement == 2: self.cursor_pos[1] += 1  # Down
        if movement == 3: self.cursor_pos[0] -= 1  # Left
        if movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH // self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT // self.GRID_SIZE - 1)

        # --- Place Tower (Space) ---
        if space_held and not self.last_space_held:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.money >= spec["cost"]:
                pos = (self.cursor_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2, 
                       self.cursor_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2)
                
                is_on_path = any(pygame.Rect(pos[0] - 5, pos[1] - 5, 10, 10).colliderect(r) for r in self.path_rects)
                is_on_tower = any(math.hypot(t['pos'][0]-pos[0], t['pos'][1]-pos[1]) < self.GRID_SIZE for t in self.towers)

                if not is_on_path and not is_on_tower:
                    self.money -= spec["cost"]
                    self.towers.append({
                        "pos": pos,
                        "type": self.selected_tower_type,
                        "cooldown": 0
                    })
                    # sfx: build_tower.wav
        
        self.last_space_held = space_held

        # --- Cycle Tower Type (Shift) ---
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: cycle_weapon.wav

        self.last_shift_held = shift_held

        return 0 # No direct reward for input

    def _manage_waves(self):
        if self.wave_number > self.TOTAL_WAVES:
            return

        if not self.enemies and self.enemies_to_spawn == 0:
            self.wave_countdown -= 1
            if self.wave_countdown <= 0:
                self.wave_number += 1
                if self.wave_number <= self.TOTAL_WAVES:
                    self.enemies_to_spawn = 3 + (self.wave_number - 1) * 2
                    self.spawn_timer = 0
                    self.wave_countdown = 300 # Time between waves
                    # sfx: wave_start.wav

        if self.enemies_to_spawn > 0:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self.enemies_to_spawn -= 1
                self.spawn_timer = 30 # Ticks between spawns
                speed = 1.0 + (self.wave_number - 1) * 0.05
                health = 3 + self.wave_number
                self.enemies.append({
                    "pos": list(self.path_waypoints[0]),
                    "health": health,
                    "max_health": health,
                    "speed": speed,
                    "path_index": 0,
                    "dist_traveled": 0,
                })

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy["path_index"] >= len(self.path_waypoints) - 1:
                self.base_health -= 1
                reward -= 0.1
                self.enemies.remove(enemy)
                self.particles.append({"type": "base_hit", "pos": self.base_pos, "radius": 15, "max_radius": 30, "life": 20})
                # sfx: base_damage.wav
                continue

            target_pos = self.path_waypoints[enemy["path_index"] + 1]
            dx = target_pos[0] - enemy["pos"][0]
            dy = target_pos[1] - enemy["pos"][1]
            dist = math.hypot(dx, dy)

            if dist < enemy["speed"]:
                enemy["pos"] = list(target_pos)
                enemy["path_index"] += 1
            else:
                enemy["pos"][0] += (dx / dist) * enemy["speed"]
                enemy["pos"][1] += (dy / dist) * enemy["speed"]
            
            enemy["dist_traveled"] += enemy["speed"]
        return reward

    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            
            if tower["cooldown"] == 0:
                # Target enemy that is furthest along the path
                best_target = None
                max_dist = -1
                
                for enemy in self.enemies:
                    dist = math.hypot(tower["pos"][0] - enemy["pos"][0], tower["pos"][1] - enemy["pos"][1])
                    if dist <= spec["range"] and enemy["dist_traveled"] > max_dist:
                        max_dist = enemy["dist_traveled"]
                        best_target = enemy

                if best_target:
                    self.projectiles.append({
                        "pos": list(tower["pos"]),
                        "target": best_target,
                        "damage": spec["damage"],
                        "speed": 5,
                    })
                    tower["cooldown"] = spec["fire_rate"]
                    # sfx: tower_shoot.wav
        return 0

    def _update_projectiles(self):
        reward = 0
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
                reward += 0.1 # Reward for hit
                self.particles.append({"type": "hit", "pos": proj["target"]["pos"], "radius": 0, "max_radius": 5 + proj["damage"], "life": 10})
                # sfx: enemy_hit.wav

                if proj["target"]["health"] <= 0:
                    reward += 1.0 # Reward for kill
                    self.score += 10
                    self.money += 15
                    self.enemies.remove(proj["target"])
                    # sfx: enemy_destroy.wav
                
                self.projectiles.remove(proj)
            else:
                proj["pos"][0] += (dx / dist) * proj["speed"]
                proj["pos"][1] += (dy / dist) * proj["speed"]
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                p["radius"] += (p["max_radius"] - p["radius"]) * 0.2

    def _generate_path(self):
        self.path_waypoints.clear()
        self.path_rects.clear()
        
        grid_w, grid_h = self.WIDTH // self.GRID_SIZE, self.HEIGHT // self.GRID_SIZE
        
        # Start on left, end on right
        start_y = self.np_random.integers(3, grid_h - 3)
        end_y = self.np_random.integers(3, grid_h - 3)
        
        path_grid = np.zeros((grid_w, grid_h), dtype=bool)
        
        cx, cy = 0, start_y
        self.path_waypoints.append((cx * self.GRID_SIZE, cy * self.GRID_SIZE + self.GRID_SIZE // 2))
        path_grid[cx, cy] = True
        
        direction = "R" # U, D, R
        
        while cx < grid_w - 3:
            # Move forward
            length = self.np_random.integers(3, 8)
            for _ in range(length):
                if direction == "R": cx += 1
                elif direction == "U": cy -= 1
                elif direction == "D": cy += 1
                
                cx = np.clip(cx, 0, grid_w - 1)
                cy = np.clip(cy, 0, grid_h - 1)
                if path_grid[cx, cy]: break # Avoid self-intersection
                path_grid[cx, cy] = True
            
            self.path_waypoints.append((cx * self.GRID_SIZE, cy * self.GRID_SIZE + self.GRID_SIZE // 2))

            # Turn
            if direction == "R":
                direction = "U" if (cy > end_y and cy > 2) or cy < 2 else "D"
            elif direction in ["U", "D"]:
                direction = "R"
        
        # Final segment to base
        self.base_pos = (self.WIDTH - self.GRID_SIZE, end_y * self.GRID_SIZE + self.GRID_SIZE // 2)
        self.path_waypoints.append((cx * self.GRID_SIZE, self.base_pos[1]))
        self.path_waypoints.append(self.base_pos)

        # Create rects for collision detection
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            rect = pygame.Rect(min(p1[0], p2[0]) - self.GRID_SIZE // 2, 
                               min(p1[1], p2[1]) - self.GRID_SIZE // 2,
                               abs(p1[0] - p2[0]) + self.GRID_SIZE, 
                               abs(p1[1] - p2[1]) + self.GRID_SIZE)
            self.path_rects.append(rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw path
        if len(self.path_waypoints) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, self.GRID_SIZE)
        
        # Draw base
        base_rect = pygame.Rect(self.base_pos[0], self.base_pos[1] - self.GRID_SIZE // 2, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        
        # Draw towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            pygame.gfxdraw.filled_circle(self.screen, int(tower["pos"][0]), int(tower["pos"][1]), self.GRID_SIZE // 2 - 2, spec["color"])
            pygame.gfxdraw.aacircle(self.screen, int(tower["pos"][0]), int(tower["pos"][1]), self.GRID_SIZE // 2 - 2, spec["color"])
            # Range indicator when placing or selected
            # pygame.gfxdraw.aacircle(self.screen, int(tower['pos'][0]), int(tower['pos'][1]), spec['range'], (255,255,255,50))

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            size = self.GRID_SIZE // 2 - 2
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos[0] - size, pos[1] - size, size*2, size*2))
            # Health bar
            health_pct = enemy["health"] / enemy["max_health"]
            pygame.draw.rect(self.screen, (50,50,50), (pos[0] - size, pos[1] - size - 6, size*2, 4))
            pygame.draw.rect(self.screen, (0,255,0) if health_pct > 0.5 else (255,255,0) if health_pct > 0.25 else (255,0,0), 
                             (pos[0] - size, pos[1] - size - 6, int(size * 2 * health_pct), 4))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 10)) if p["type"] == "hit" else int(255 * (p["life"] / 20))
            color = self.COLOR_PROJECTILE if p["type"] == "hit" else self.COLOR_ENEMY
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"]), (*color, alpha))

        # Draw cursor
        cursor_world_pos = (self.cursor_pos[0] * self.GRID_SIZE, self.cursor_pos[1] * self.GRID_SIZE)
        cursor_rect = pygame.Rect(cursor_world_pos[0], cursor_world_pos[1], self.GRID_SIZE, self.GRID_SIZE)
        
        spec = self.TOWER_SPECS[self.selected_tower_type]
        is_on_path = any(cursor_rect.colliderect(r) for r in self.path_rects)
        is_on_tower = any(math.hypot(t['pos'][0] - (cursor_world_pos[0] + self.GRID_SIZE//2), t['pos'][1] - (cursor_world_pos[1] + self.GRID_SIZE//2)) < self.GRID_SIZE for t in self.towers)
        can_afford = self.money >= spec["cost"]
        
        color = self.COLOR_CURSOR_VALID if can_afford and not is_on_path and not is_on_tower else self.COLOR_CURSOR_INVALID
        
        s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        s.fill(color)
        self.screen.blit(s, cursor_world_pos)
        
        # Draw tower range preview
        pygame.gfxdraw.aacircle(self.screen, cursor_world_pos[0] + self.GRID_SIZE // 2, cursor_world_pos[1] + self.GRID_SIZE // 2, spec["range"], (*self.COLOR_TEXT, 80))

    def _render_ui(self):
        # Top Bar
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        
        wave_text = self.font_large.render(f"WAVE: {self.wave_number}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 5))
        
        # Bottom Bar
        money_text = self.font_large.render(f"$ {self.money}", True, self.COLOR_PROJECTILE)
        self.screen.blit(money_text, (10, self.HEIGHT - 30))
        
        base_health_text = self.font_large.render(f"BASE HP: {self.base_health}", True, self.COLOR_BASE)
        self.screen.blit(base_health_text, (self.WIDTH - base_health_text.get_width() - 10, self.HEIGHT - 30))

        # Selected Tower Info
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_name_text = self.font_small.render(f"Selected: {spec['name']} (Cost: {spec['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(tower_name_text, (self.WIDTH // 2 - tower_name_text.get_width() // 2, self.HEIGHT - 25))
        
        # Wave countdown / Game Over
        if self.game_over:
            msg = "YOU WIN!" if self.base_health > 0 else "GAME OVER"
            color = self.COLOR_BASE if self.base_health > 0 else self.COLOR_ENEMY
            end_text = self.font_huge.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2))
        elif self.wave_countdown > 0 and self.wave_number < self.TOTAL_WAVES and not self.enemies:
            msg = f"Wave {self.wave_number + 1} in {self.wave_countdown // 30 + 1}"
            countdown_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(countdown_text, (self.WIDTH // 2 - countdown_text.get_width() // 2, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "money": self.money,
            "wave": self.wave_number,
        }
    
    def close(self):
        pygame.quit()

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Convert observation back to a Pygame surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS
        
    env.close()