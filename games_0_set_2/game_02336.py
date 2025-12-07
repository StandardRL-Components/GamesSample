
# Generated: 2025-08-27T20:04:01.156361
# Source Brief: brief_02336.md
# Brief Index: 2336

        
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
        "Controls: Arrow keys to move placement cursor. Space to build tower. Shift to cycle tower type."
    )

    game_description = (
        "Defend your base from waves of geometric invaders by strategically placing defensive towers."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2500
        self.TOTAL_WAVES = 10
        self.INITIAL_RESOURCES = 250
        self.INITIAL_BASE_HEALTH = 50

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30) # Dark blue-grey
        self.COLOR_PATH = (44, 62, 80)
        self.COLOR_BASE = (39, 174, 96)
        self.COLOR_BASE_DAMAGE = (192, 57, 43)
        self.COLOR_ENEMY = (231, 76, 60)
        self.COLOR_PROJECTILE = (241, 196, 15)
        self.COLOR_TEXT = (236, 240, 241)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TOWER_GUN = (52, 152, 219)
        self.COLOR_TOWER_CANNON = (155, 89, 182)
        
        # --- Tower Definitions ---
        self.TOWER_TYPES = [
            {
                "name": "Gun Turret", "cost": 100, "range": 100, "damage": 2, 
                "fire_rate": 15, "color": self.COLOR_TOWER_GUN, "proj_speed": 8
            },
            {
                "name": "Cannon", "cost": 225, "range": 150, "damage": 8, 
                "fire_rate": 60, "color": self.COLOR_TOWER_CANNON, "proj_speed": 6
            },
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_m = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 32, bold=True)

        # --- State Variables (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.np_random = None
        self.base_health = None
        self.resources = None
        self.wave_number = None
        self.path_waypoints = None
        self.placement_spots = None
        self.base_pos = None
        self.enemies = None
        self.towers = None
        self.projectiles = None
        self.particles = None
        self.wave_spawner = None
        self.placement_cursor_idx = None
        self.selected_tower_type_idx = None
        self.last_space_held = None
        self.last_shift_held = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.wave_number = 1
        
        self._generate_layout()
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self._setup_wave_spawner()

        # --- Initialize Control State ---
        self.placement_cursor_idx = len(self.placement_spots) // 2
        self.selected_tower_type_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def _generate_layout(self):
        # Generate a procedural path
        self.path_waypoints = []
        start_y = self.np_random.integers(self.HEIGHT * 0.2, self.HEIGHT * 0.8)
        self.path_waypoints.append(pygame.Vector2(-20, start_y))
        
        x, y = 0, start_y
        segments = 5
        for i in range(segments):
            is_last = (i == segments - 1)
            is_penultimate = (i == segments - 2)
            
            # Horizontal segment
            if is_last:
                next_x = self.WIDTH // 2
            elif is_penultimate:
                 next_x = self.np_random.integers(x + 50, self.WIDTH * 0.6)
            else:
                next_x = self.np_random.integers(x + 50, x + 100)
            self.path_waypoints.append(pygame.Vector2(next_x, y))
            x = next_x
            
            if not is_last:
                # Vertical segment
                if y > self.HEIGHT / 2:
                    next_y = self.np_random.integers(self.HEIGHT * 0.1, y - 50)
                else:
                    next_y = self.np_random.integers(y + 50, self.HEIGHT * 0.9)
                self.path_waypoints.append(pygame.Vector2(x, next_y))
                y = next_y

        self.base_pos = self.path_waypoints[-1]

        # Generate tower placement spots on a grid, avoiding the path
        self.placement_spots = []
        grid_size = 40
        path_clearance = 30
        for gx in range(grid_size, self.WIDTH - grid_size, grid_size):
            for gy in range(grid_size, self.HEIGHT - grid_size, grid_size):
                spot = pygame.Vector2(gx, gy)
                on_path = False
                for i in range(len(self.path_waypoints) - 1):
                    p1 = self.path_waypoints[i]
                    p2 = self.path_waypoints[i+1]
                    # Point-line segment distance check
                    l2 = p1.distance_squared_to(p2)
                    if l2 == 0.0:
                        if spot.distance_to(p1) < path_clearance: on_path = True; break
                    t = max(0, min(1, (spot - p1).dot(p2 - p1) / l2))
                    projection = p1 + t * (p2 - p1)
                    if spot.distance_to(projection) < path_clearance: on_path = True; break
                if not on_path and spot.distance_to(self.base_pos) > 40:
                    self.placement_spots.append(spot)

    def _setup_wave_spawner(self):
        enemy_count = 3 + (self.wave_number - 1) * 2
        enemy_speed = 1.0 + (self.wave_number - 1) * 0.05
        enemy_health = 5 + (self.wave_number - 1) // 2
        
        self.wave_spawner = {
            "count": enemy_count,
            "speed": enemy_speed,
            "health": enemy_health,
            "spawn_timer": 0,
            "spawn_delay": max(10, 60 - self.wave_number * 2),
            "enemies_spawned": 0,
            "active": True
        }

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Time penalty

        # --- Handle player input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)
        self.last_space_held, self.last_shift_held = space_held, shift_held

        # --- Update Game Logic ---
        self._update_wave_spawner()
        self._update_towers()
        hit_reward = self._update_projectiles()
        base_damage, defeated_reward = self._update_enemies()
        self._update_particles()
        
        # --- Apply Rewards & Update State ---
        reward += hit_reward + defeated_reward
        self.score += defeated_reward
        if base_damage > 0:
            self.base_health = max(0, self.base_health - base_damage)
            reward -= 10 * base_damage

        # --- Check for Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.base_health <= 0:
                reward -= 100
            elif self.wave_number > self.TOTAL_WAVES:
                reward += 100
                self.score += 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # --- Cycle Tower Type (on key press) ---
        if shift_held and not self.last_shift_held:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.TOWER_TYPES)
            # sfx: menu_click

        # --- Place Tower (on key press) ---
        if space_held and not self.last_space_held:
            tower_def = self.TOWER_TYPES[self.selected_tower_type_idx]
            pos = self.placement_spots[self.placement_cursor_idx]
            
            can_afford = self.resources >= tower_def["cost"]
            is_occupied = any(t['pos'] == pos for t in self.towers)

            if can_afford and not is_occupied:
                self.resources -= tower_def["cost"]
                self.towers.append({
                    "pos": pos,
                    "type_idx": self.selected_tower_type_idx,
                    "cooldown": 0,
                    "fire_anim": 0,
                })
                # sfx: build_tower

        # --- Move Cursor ---
        if movement != 0:
            current_pos = self.placement_spots[self.placement_cursor_idx]
            best_spot_idx = -1
            min_dist = float('inf')
            
            filtered_spots = []
            if movement == 1: # Up
                filtered_spots = [(i, s) for i, s in enumerate(self.placement_spots) if s.y < current_pos.y]
            elif movement == 2: # Down
                filtered_spots = [(i, s) for i, s in enumerate(self.placement_spots) if s.y > current_pos.y]
            elif movement == 3: # Left
                filtered_spots = [(i, s) for i, s in enumerate(self.placement_spots) if s.x < current_pos.x]
            elif movement == 4: # Right
                filtered_spots = [(i, s) for i, s in enumerate(self.placement_spots) if s.x > current_pos.x]
            
            if filtered_spots:
                for idx, spot in filtered_spots:
                    dist = current_pos.distance_squared_to(spot)
                    if dist < min_dist:
                        min_dist = dist
                        best_spot_idx = idx
                
                if best_spot_idx != -1:
                    self.placement_cursor_idx = best_spot_idx

    def _update_wave_spawner(self):
        spawner = self.wave_spawner
        if not spawner["active"]:
            if not self.enemies and self.wave_number <= self.TOTAL_WAVES:
                self.wave_number += 1
                if self.wave_number <= self.TOTAL_WAVES:
                    self._setup_wave_spawner()
            return

        spawner["spawn_timer"] -= 1
        if spawner["spawn_timer"] <= 0 and spawner["enemies_spawned"] < spawner["count"]:
            spawner["spawn_timer"] = spawner["spawn_delay"]
            spawner["enemies_spawned"] += 1
            self.enemies.append({
                "pos": self.path_waypoints[0].copy(),
                "health": spawner["health"],
                "max_health": spawner["health"],
                "speed": spawner["speed"],
                "path_idx": 1
            })
            # sfx: enemy_spawn
        
        if spawner["enemies_spawned"] >= spawner["count"]:
            spawner["active"] = False

    def _update_towers(self):
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
            if tower["fire_anim"] > 0:
                tower["fire_anim"] -= 1

            if tower["cooldown"] <= 0:
                tower_def = self.TOWER_TYPES[tower["type_idx"]]
                target = None
                max_path_dist = -1

                for enemy in self.enemies:
                    if tower["pos"].distance_to(enemy["pos"]) <= tower_def["range"]:
                        # Target enemy furthest along the path
                        enemy_path_dist = enemy["path_idx"]
                        if enemy_path_dist > max_path_dist:
                            max_path_dist = enemy_path_dist
                            target = enemy
                
                if target:
                    self.projectiles.append({
                        "pos": tower["pos"].copy(),
                        "target": target,
                        "damage": tower_def["damage"],
                        "speed": tower_def["proj_speed"]
                    })
                    tower["cooldown"] = tower_def["fire_rate"]
                    tower["fire_anim"] = 10
                    # sfx: tower_shoot

    def _update_projectiles(self):
        hit_reward = 0
        for proj in self.projectiles[:]:
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            direction = (proj["target"]["pos"] - proj["pos"]).normalize()
            proj["pos"] += direction * proj["speed"]

            if proj["pos"].distance_to(proj["target"]["pos"]) < 8:
                proj["target"]["health"] -= proj["damage"]
                hit_reward += 0.1
                self.projectiles.remove(proj)
                # sfx: projectile_hit
        return hit_reward

    def _update_enemies(self):
        base_damage = 0
        defeated_reward = 0
        for enemy in self.enemies[:]:
            if enemy["health"] <= 0:
                defeated_reward += 1
                self.resources += 25
                self._create_explosion(enemy["pos"], self.COLOR_ENEMY)
                self.enemies.remove(enemy)
                # sfx: enemy_explode
                continue

            target_waypoint = self.path_waypoints[enemy["path_idx"]]
            direction = (target_waypoint - enemy["pos"])
            dist = direction.length()

            if dist < enemy["speed"]:
                enemy["pos"] = target_waypoint.copy()
                enemy["path_idx"] += 1
                if enemy["path_idx"] >= len(self.path_waypoints):
                    base_damage += 1
                    self.enemies.remove(enemy)
                    self._create_explosion(self.base_pos, self.COLOR_BASE_DAMAGE)
                    # sfx: base_hit
            else:
                enemy["pos"] += direction.normalize() * enemy["speed"]
        return base_damage, defeated_reward

    def _create_explosion(self, pos, color):
        for _ in range(20):
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
                "radius": self.np_random.uniform(2, 5),
                "lifetime": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # friction
            p["lifetime"] -= 1
            p["radius"] *= 0.95
            if p["lifetime"] <= 0 or p["radius"] < 0.5:
                self.particles.remove(p)

    def _check_termination(self):
        win = self.wave_number > self.TOTAL_WAVES and not self.enemies
        loss = self.base_health <= 0
        timeout = self.steps >= self.MAX_STEPS
        self.game_over = win or loss or timeout
        return self.game_over

    def _get_observation(self):
        self._render_background()
        self._render_path()
        self._render_placement_spots()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)

    def _render_path(self):
        if len(self.path_waypoints) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_PATH, False, 
                                [(int(p.x), int(p.y)) for p in self.path_waypoints], 1)

    def _render_placement_spots(self):
        for spot in self.placement_spots:
            is_occupied = any(t['pos'] == spot for t in self.towers)
            color = (60, 60, 60) if is_occupied else (50, 50, 50)
            pygame.gfxdraw.filled_circle(self.screen, int(spot.x), int(spot.y), 3, color)

    def _render_base(self):
        pos = (int(self.base_pos.x), int(self.base_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, self.COLOR_BG)

    def _render_towers(self):
        for tower in self.towers:
            tower_def = self.TOWER_TYPES[tower["type_idx"]]
            pos = (int(tower["pos"].x), int(tower["pos"].y))
            
            if tower["fire_anim"] > 0:
                alpha = int(100 * (tower["fire_anim"] / 10))
                radius = int(tower_def["range"] * (1 - tower["fire_anim"] / 12))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*tower_def["color"], alpha))

            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, tower_def["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_BG)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy["pos"].x), int(enemy["pos"].y))
            size = 8
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos[0] - size//2, pos[1] - size//2, size, size))
            
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            bar_width = 12
            pygame.draw.rect(self.screen, self.COLOR_BASE_DAMAGE, (pos[0] - bar_width//2, pos[1] - 12, bar_width, 3))
            pygame.draw.rect(self.screen, self.COLOR_BASE, (pos[0] - bar_width//2, pos[1] - 12, int(bar_width * health_ratio), 3))

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj["pos"].x), int(proj["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            radius = int(p["radius"])
            if radius > 0:
                alpha = int(255 * (p["lifetime"] / 30))
                color = (*p["color"], max(0, min(255, alpha)))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_cursor(self):
        pos = self.placement_spots[self.placement_cursor_idx]
        tower_def = self.TOWER_TYPES[self.selected_tower_type_idx]
        
        can_afford = self.resources >= tower_def["cost"]
        is_occupied = any(t['pos'] == pos for t in self.towers)
        
        # Range indicator
        alpha = 80 if can_afford and not is_occupied else 30
        color = (*self.COLOR_CURSOR, alpha)
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), tower_def["range"], color)
        
        # Cursor circle
        cursor_color = self.COLOR_CURSOR if can_afford and not is_occupied else self.COLOR_ENEMY
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 12, cursor_color)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 13, cursor_color)

    def _render_ui(self):
        # --- Top Bar ---
        bar_surf = pygame.Surface((self.WIDTH, 30), pygame.SRCALPHA)
        bar_surf.fill((0,0,0,100))
        self.screen.blit(bar_surf, (0,0))

        # Health
        health_text = self.font_m.render(f"♥ {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 5))
        
        # Resources
        res_text = self.font_m.render(f"$ {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (100, 5))

        # Wave
        wave_text = self.font_m.render(f"Wave: {self.wave_number}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (220, 5))
        
        # Score
        score_text = self.font_m.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - 150, 5))
        
        # --- Bottom Bar (Tower Info) ---
        bar_surf_bottom = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        bar_surf_bottom.fill((0,0,0,100))
        self.screen.blit(bar_surf_bottom, (0, self.HEIGHT - 40))

        tower_def = self.TOWER_TYPES[self.selected_tower_type_idx]
        tower_info = f"Selected: {tower_def['name']} | Cost: ${tower_def['cost']} | Dmg: {tower_def['damage']} | Rng: {tower_def['range']}"
        tower_text = self.font_s.render(tower_info, True, self.COLOR_TEXT)
        self.screen.blit(tower_text, (10, self.HEIGHT - 28))
        
        help_text = self.font_s.render("Shift: Cycle | Space: Build", True, self.COLOR_TEXT)
        self.screen.blit(help_text, (self.WIDTH - help_text.get_width() - 10, self.HEIGHT - 28))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        win = self.wave_number > self.TOTAL_WAVES
        message = "VICTORY" if win else "GAME OVER"
        color = self.COLOR_BASE if win else self.COLOR_ENEMY
        
        text = self.font_l.render(message, True, color)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
        overlay.blit(text, text_rect)
        
        score_text = self.font_m.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
        overlay.blit(score_text, score_rect)
        
        self.screen.blit(overlay, (0,0))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
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
        assert self.base_health <= self.INITIAL_BASE_HEALTH
        assert self.resources >= 0
        assert self.wave_number <= self.TOTAL_WAVES + 1

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
    
    screen_width, screen_height = 640, 400
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Tower Defense")
    
    terminated = False
    clock = pygame.time.Clock()
    
    # Game loop
    while not terminated:
        movement, space, shift = 0, 0, 0
        
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
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for human playability
        
    env.close()