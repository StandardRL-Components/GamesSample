import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the placement cursor. "
        "Hold Shift to cycle tower types. Press Space to build the selected tower."
    )

    game_description = (
        "A top-down tower defense game. Strategically place towers to defend your base "
        "from 15 waves of enemies. Manage your gold and survive to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.CELL_SIZE = 40
        self.MAX_STEPS = 15000  # Approx 8 minutes at 30fps
        self.MAX_WAVES = 15

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_PATH = (60, 60, 70)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_BASE = (0, 150, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_HEALTH_BG = (70, 0, 0)
        self.COLOR_HEALTH_FG = (0, 200, 0)
        self.COLOR_GOLD = (255, 215, 0)

        # Tower types and stats
        self.TOWER_TYPES = {
            "basic": {"cost": 50, "range": 100, "fire_rate": 20, "damage": 10, "color": (50, 150, 255), "proj_speed": 8},
            "slow": {"cost": 75, "range": 80, "fire_rate": 45, "damage": 2, "color": (255, 215, 0), "proj_speed": 6, "slow_factor": 0.5, "slow_duration": 60},
            "splash": {"cost": 125, "range": 90, "fire_rate": 60, "damage": 15, "color": (200, 50, 255), "proj_speed": 5, "splash_radius": 40},
        }
        self.TOWER_TYPE_NAMES = list(self.TOWER_TYPES.keys())

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
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        self.font_huge = pygame.font.SysFont("monospace", 48, bold=True)

        # Path definition
        self.path_waypoints = [
            (-self.CELL_SIZE, 2.5 * self.CELL_SIZE),
            (3.5 * self.CELL_SIZE, 2.5 * self.CELL_SIZE),
            (3.5 * self.CELL_SIZE, 7.5 * self.CELL_SIZE),
            (12.5 * self.CELL_SIZE, 7.5 * self.CELL_SIZE),
            (12.5 * self.CELL_SIZE, 2.5 * self.CELL_SIZE),
            (self.WIDTH + self.CELL_SIZE, 2.5 * self.CELL_SIZE),
        ]
        # FIX: Define base_pos before it's used in _get_path_grid_coords
        self.base_pos = (14, 2)
        self.path_grid_coords = self._get_path_grid_coords()

        # Initialize state variables
        self.reset()
        
    def _get_path_grid_coords(self):
        coords = set()
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            # Ensure steps is at least 1 to avoid division by zero if p1 == p2
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            steps = max(1, int(dist / (self.CELL_SIZE/4)))
            for j in range(steps + 1):
                t = j / steps
                x = p1[0] * (1-t) + p2[0] * t
                y = p1[1] * (1-t) + p2[1] * t
                gx, gy = int(x / self.CELL_SIZE), int(y / self.CELL_SIZE)
                if 0 <= gx < self.GRID_COLS and 0 <= gy < self.GRID_ROWS:
                    coords.add((gx, gy))
        coords.add(self.base_pos)
        return coords

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_won = False
        self.reward_this_step = 0.0

        self.base_health = 100.0
        self.gold = 100
        self.wave_number = 0
        self.wave_timer = 150 # Time until first wave
        self.enemies_to_spawn = []
        self.spawn_timer = 0

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_tower_index = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = -0.01 # Small penalty for existing

        self._handle_input(action)
        self._update_game_state()

        reward = self.reward_this_step
        self.score += reward
        terminated = self.game_over
        self.steps += 1
        
        truncated = False
        if self.steps >= self.MAX_STEPS and not terminated:
            truncated = True
            self.game_over = True # End the game on timeout
            self.score -= 100 # Penalty for timeout
            self.reward_this_step -= 100
        
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        if self.game_over: return

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        # --- Cycle Tower (on press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_index = (self.selected_tower_index + 1) % len(self.TOWER_TYPE_NAMES)

        # --- Place Tower (on press) ---
        if space_held and not self.prev_space_held:
            self._place_tower()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _is_valid_placement(self, grid_pos):
        if tuple(grid_pos) in self.path_grid_coords:
            return False
        for tower in self.towers:
            if tower["grid_pos"] == grid_pos:
                return False
        return True

    def _place_tower(self):
        tower_name = self.TOWER_TYPE_NAMES[self.selected_tower_index]
        tower_stats = self.TOWER_TYPES[tower_name]
        
        if self.gold >= tower_stats["cost"] and self._is_valid_placement(self.cursor_pos):
            self.gold -= tower_stats["cost"]
            new_tower = {
                "type": tower_name,
                "grid_pos": list(self.cursor_pos),
                "pos": (self.cursor_pos[0] * self.CELL_SIZE + self.CELL_SIZE/2, self.cursor_pos[1] * self.CELL_SIZE + self.CELL_SIZE/2),
                "cooldown": 0,
                "stats": tower_stats,
            }
            self.towers.append(new_tower)
            self._create_particles(new_tower["pos"], 20, tower_stats["color"], 2, 20)

    def _update_game_state(self):
        if self.game_over: return

        # Wave management
        self._update_wave_system()

        # Update Towers
        for tower in self.towers:
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            if tower["cooldown"] == 0:
                target = self._find_target(tower)
                if target:
                    self._fire_projectile(tower, target)
                    tower["cooldown"] = tower["stats"]["fire_rate"]
        
        # Update Projectiles
        for proj in self.projectiles[:]:
            self._update_projectile(proj)

        # Update Enemies
        for enemy in self.enemies[:]:
            self._update_enemy(enemy)

        # Update Particles
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

        # Check for game over
        if self.base_health <= 0:
            self.base_health = 0
            if not self.game_over:
                self.game_over = True
                self.reward_this_step -= 100
                self.score -= 100 # Final penalty
        elif self.wave_number > self.MAX_WAVES and not self.enemies and not self.enemies_to_spawn:
            if not self.game_won:
                self.game_won = True
                self.game_over = True
                self.reward_this_step += 100
                self.score += 100 # Final bonus

    def _update_wave_system(self):
        if self.wave_number > self.MAX_WAVES: return

        if not self.enemies and not self.enemies_to_spawn:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                if self.wave_number > 0:
                    self.reward_this_step += 5.0
                self.wave_number += 1
                if self.wave_number <= self.MAX_WAVES:
                    self._start_next_wave()
                    self.wave_timer = 300 # Time between waves
        
        if self.enemies_to_spawn:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self.enemies.append(self.enemies_to_spawn.pop(0))
                self.spawn_timer = 30 # Time between enemies in a wave

    def _start_next_wave(self):
        num_enemies = 3 + self.wave_number * 2
        base_health = 50 * (1.05 ** (self.wave_number - 1))
        base_speed = 1.0 * (1.05 ** (self.wave_number - 1))
        
        for _ in range(num_enemies):
            enemy_type = self.np_random.choice(["normal", "fast", "tank"], p=[0.7, 0.2, 0.1])
            health = base_health
            speed = base_speed
            if enemy_type == "fast":
                health *= 0.7
                speed *= 1.5
            elif enemy_type == "tank":
                health *= 2.0
                speed *= 0.7
            
            self.enemies_to_spawn.append({
                "pos": list(self.path_waypoints[0]),
                "health": health, "max_health": health,
                "speed": speed, "base_speed": speed,
                "waypoint_idx": 1,
                "slow_timer": 0,
            })
        self.spawn_timer = 0

    def _find_target(self, tower):
        in_range_enemies = []
        for enemy in self.enemies:
            dist = math.hypot(tower["pos"][0] - enemy["pos"][0], tower["pos"][1] - enemy["pos"][1])
            if dist <= tower["stats"]["range"]:
                in_range_enemies.append(enemy)
        
        if not in_range_enemies: return None
        return min(in_range_enemies, key=lambda e: math.hypot(e["pos"][0] - self.path_waypoints[-1][0], e["pos"][1] - self.path_waypoints[-1][1]))

    def _fire_projectile(self, tower, target):
        proj = {
            "pos": list(tower["pos"]),
            "target": target,
            "stats": tower["stats"],
            "type": tower["type"],
        }
        self.projectiles.append(proj)

    def _update_projectile(self, proj):
        if proj["target"] not in self.enemies:
            if proj in self.projectiles:
                self.projectiles.remove(proj)
            return
        
        target_pos = proj["target"]["pos"]
        proj_pos = proj["pos"]
        
        angle = math.atan2(target_pos[1] - proj_pos[1], target_pos[0] - proj_pos[0])
        speed = proj["stats"]["proj_speed"]
        proj_pos[0] += math.cos(angle) * speed
        proj_pos[1] += math.sin(angle) * speed

        if math.hypot(proj_pos[0] - target_pos[0], proj_pos[1] - target_pos[1]) < 10:
            self._handle_projectile_hit(proj)
            if proj in self.projectiles:
                self.projectiles.remove(proj)

    def _handle_projectile_hit(self, proj):
        self.reward_this_step += 0.1
        self._create_particles(proj["pos"], 10, proj["stats"]["color"], 1.5, 15)

        if proj["type"] == "splash":
            splash_radius = proj["stats"]["splash_radius"]
            for enemy in self.enemies[:]: # Iterate over a copy
                if math.hypot(proj["pos"][0] - enemy["pos"][0], proj["pos"][1] - enemy["pos"][1]) <= splash_radius:
                    self._damage_enemy(enemy, proj["stats"]["damage"])
        else:
            self._damage_enemy(proj["target"], proj["stats"]["damage"])
            if proj["type"] == "slow":
                if proj["target"] in self.enemies: # Check if target is still alive
                    proj["target"]["slow_timer"] = proj["stats"]["slow_duration"]

    def _damage_enemy(self, enemy, damage):
        enemy["health"] -= damage
        if enemy["health"] <= 0:
            if enemy in self.enemies:
                self.enemies.remove(enemy)
                self.gold += 10
                self.reward_this_step += 1.0
                self._create_particles(enemy["pos"], 30, (255, 80, 80), 3, 30)

    def _update_enemy(self, enemy):
        # Apply slow effect
        if enemy["slow_timer"] > 0:
            enemy["slow_timer"] -= 1
            enemy["speed"] = enemy["base_speed"] * self.TOWER_TYPES["slow"]["slow_factor"]
        else:
            enemy["speed"] = enemy["base_speed"]

        # Movement
        target_waypoint = self.path_waypoints[enemy["waypoint_idx"]]
        angle = math.atan2(target_waypoint[1] - enemy["pos"][1], target_waypoint[0] - enemy["pos"][0])
        enemy["pos"][0] += math.cos(angle) * enemy["speed"]
        enemy["pos"][1] += math.sin(angle) * enemy["speed"]

        if math.hypot(enemy["pos"][0] - target_waypoint[0], enemy["pos"][1] - target_waypoint[1]) < enemy["speed"]:
            enemy["waypoint_idx"] += 1
            if enemy["waypoint_idx"] >= len(self.path_waypoints):
                self.base_health -= 10
                if enemy in self.enemies:
                    self.enemies.remove(enemy)
                self.reward_this_step -= 2.0
                self._create_particles(enemy["pos"], 40, (255, 0, 0), 5, 40)

    def _create_particles(self, pos, count, color, speed_max, life_max):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            life = self.np_random.integers(10, life_max + 1)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": life,
                "color": color,
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, self.CELL_SIZE)
        
        # Draw base
        base_rect = pygame.Rect(self.base_pos[0] * self.CELL_SIZE, self.base_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

        # Draw towers
        for tower in self.towers:
            pygame.draw.circle(self.screen, tower["stats"]["color"], (int(tower["pos"][0]), int(tower["pos"][1])), self.CELL_SIZE // 3)

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, (200, 50, 50))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, (255, 100, 100))
            # Health bar
            hb_width = 20
            hb_height = 4
            health_pct = max(0, enemy["health"] / enemy["max_health"])
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (pos[0] - hb_width/2, pos[1] - 15, hb_width, hb_height))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (pos[0] - hb_width/2, pos[1] - 15, hb_width * health_pct, hb_height))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            color = proj["stats"]["color"]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, (255,255,255))
            
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 20))))
            color = p["color"]
            s = pygame.Surface((3,3), pygame.SRCALPHA)
            s.fill((color[0], color[1], color[2], alpha))
            self.screen.blit(s, (int(p["pos"][0]), int(p["pos"][1])))
        
        # Draw cursor
        if not self.game_over:
            cursor_rect = pygame.Rect(self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            is_valid = self._is_valid_placement(self.cursor_pos)
            tower_name = self.TOWER_TYPE_NAMES[self.selected_tower_index]
            tower_stats = self.TOWER_TYPES[tower_name]
            
            cursor_color = (0, 255, 0, 100) if is_valid and self.gold >= tower_stats["cost"] else (255, 0, 0, 100)
            cursor_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            cursor_surface.fill(cursor_color)
            self.screen.blit(cursor_surface, cursor_rect.topleft)
            
            # Draw range indicator
            pygame.gfxdraw.aacircle(self.screen, cursor_rect.centerx, cursor_rect.centery, tower_stats["range"], (255,255,255,100))


    def _render_ui(self):
        # Top-left UI
        health_text = self.font_small.render(f"Base Health: {int(self.base_health)}/100", True, self.COLOR_TEXT)
        gold_text = self.font_small.render(f"Gold: {self.gold}", True, self.COLOR_GOLD)
        wave_text = self.font_small.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        score_text = self.font_small.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        
        self.screen.blit(health_text, (10, 10))
        self.screen.blit(gold_text, (10, 30))
        self.screen.blit(wave_text, (10, 50))
        self.screen.blit(score_text, (10, 70))

        # Bottom-right UI: Selected Tower
        tower_name = self.TOWER_TYPE_NAMES[self.selected_tower_index]
        tower_stats = self.TOWER_TYPES[tower_name]
        tower_color = tower_stats["color"]
        
        tower_name_text = self.font_small.render(f"Selected: {tower_name.capitalize()}", True, tower_color)
        tower_cost_text = self.font_small.render(f"Cost: {tower_stats['cost']}", True, self.COLOR_GOLD)
        self.screen.blit(tower_name_text, (self.WIDTH - 180, self.HEIGHT - 40))
        self.screen.blit(tower_cost_text, (self.WIDTH - 180, self.HEIGHT - 20))
        
        # Game Over / Victory Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            message = "VICTORY!" if self.game_won else "GAME OVER"
            color = (0, 255, 0) if self.game_won else (255, 0, 0)
            
            end_text = self.font_huge.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.wave_number,
            "base_health": self.base_health,
        }
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a different screen for human rendering
    human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    terminated = False
    truncated = False
    clock = pygame.time.Clock()
    
    # Key mapping
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4
    }

    while not terminated and not truncated:
        # Human input
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement = move_val
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render to human screen
        # Need to transpose back for pygame's display surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()