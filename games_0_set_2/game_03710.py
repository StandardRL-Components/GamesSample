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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor, Shift to cycle tower type, Space to place tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 3000 # Extended for longer games
    MAX_WAVES = 10

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_PATH = (40, 50, 60)
    COLOR_BASE = (60, 180, 75)
    COLOR_ENEMY = (230, 25, 75)
    COLOR_TEXT = (235, 235, 235)
    COLOR_UI_BG = (50, 60, 70, 180)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_PLACEMENT_VALID = (255, 255, 255, 50)
    COLOR_PLACEMENT_INVALID = (230, 25, 75, 50)

    TOWER_SPECS = {
        0: {"name": "Gatling", "cost": 100, "range": 70, "damage": 2.5, "fire_rate": 5, "color": (0, 130, 200), "proj_speed": 8},
        1: {"name": "Cannon", "cost": 250, "range": 100, "damage": 20, "fire_rate": 1, "color": (245, 130, 48), "proj_speed": 6},
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
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_large = pygame.font.SysFont("sans-serif", 28)
        
        self.game_over_font = pygame.font.SysFont("sans-serif", 60)
        self.game_over_text = ""

        # The reset method is called here to initialize the state
        # No need to call it again before the first step
        # self.reset() is called by the wrapper or user, not in __init__
        # But for standalone verification, we might need it. The traceback shows it's called.
        # Let's keep it but ensure it works.
        
        # This check is disabled for submission but useful for development
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_text = ""
        self.reward_this_step = 0
        
        # Player state
        self.money = 250
        self.base_health = 100
        self.max_base_health = 100
        
        # Wave state
        self.wave_number = 0
        self.wave_in_progress = False
        self.wave_spawn_timer = 0
        self.enemies_to_spawn_this_wave = 0
        self.wave_cooldown = 150 # Steps between waves

        # World generation
        self._generate_path_and_placements()
        
        # Entities
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        # Controls
        self.cursor_grid_pos = [self.placement_grid_dims[0] // 2, self.placement_grid_dims[1] // 2]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = 0
        self.steps += 1
        
        self._handle_actions(action)
        self._update_waves()
        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated and not terminated: # Handle timeout case
            self.game_over = True
            self.reward_this_step -= 100 # Penalize for timeout
            self.game_over_text = "TIME UP"
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_path_and_placements(self):
        self.path_points = []
        start_y = self.np_random.uniform(0.2, 0.8) * self.SCREEN_HEIGHT
        self.path_points.append(pygame.Vector2(-20, start_y))
        
        num_segments = 4
        x_increment = self.SCREEN_WIDTH / num_segments
        current_pos = pygame.Vector2(x_increment, start_y)

        for i in range(1, num_segments):
            current_pos.y = self.np_random.uniform(0.1, 0.9) * self.SCREEN_HEIGHT
            self.path_points.append(pygame.Vector2(current_pos))
            current_pos.x += x_increment

        self.base_pos = pygame.Vector2(self.SCREEN_WIDTH - 30, self.path_points[-1].y)
        self.path_points.append(self.base_pos + pygame.Vector2(30, 0))

        # Generate placement spots in a grid
        self.placement_spots = []
        self.placement_grid_dims = (6, 10) # rows, cols
        self.placement_occupied = np.zeros(self.placement_grid_dims, dtype=bool)
        cell_h = self.SCREEN_HEIGHT / self.placement_grid_dims[0]
        cell_w = self.SCREEN_WIDTH / self.placement_grid_dims[1]
        for r in range(self.placement_grid_dims[0]):
            for c in range(self.placement_grid_dims[1]):
                spot_pos = (c * cell_w + cell_w/2, r * cell_h + cell_h/2)
                # Ensure spots are not too close to the path
                min_dist_to_path = min(self._dist_point_to_segment(spot_pos, self.path_points[i], self.path_points[i+1]) for i in range(len(self.path_points)-1))
                if min_dist_to_path > 40:
                    self.placement_spots.append(spot_pos)
                else:
                    self.placement_spots.append(None) # Invalid spot


    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        if movement == 1: self.cursor_grid_pos[0] = max(0, self.cursor_grid_pos[0] - 1) # Up
        elif movement == 2: self.cursor_grid_pos[0] = min(self.placement_grid_dims[0] - 1, self.cursor_grid_pos[0] + 1) # Down
        elif movement == 3: self.cursor_grid_pos[1] = max(0, self.cursor_grid_pos[1] - 1) # Left
        elif movement == 4: self.cursor_grid_pos[1] = min(self.placement_grid_dims[1] - 1, self.cursor_grid_pos[1] + 1) # Right

        # Cycle tower type (on press)
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)

        # Place tower (on press)
        if space_held and not self.last_space_held:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            grid_idx = (self.cursor_grid_pos[0], self.cursor_grid_pos[1])
            spot_idx = grid_idx[0] * self.placement_grid_dims[1] + grid_idx[1]
            
            if self.placement_spots[spot_idx] is not None and not self.placement_occupied[grid_idx] and self.money >= spec["cost"]:
                self.money -= spec["cost"]
                pos = self.placement_spots[spot_idx]
                self.towers.append({
                    "pos": pygame.Vector2(pos), "type": self.selected_tower_type, "cooldown": 0
                })
                self.placement_occupied[grid_idx] = True
                self._create_particles(pos, spec["color"], 20, 2, 15)

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return
        
        self.wave_in_progress = True
        self.enemies_to_spawn_this_wave = 5 + (self.wave_number - 1) * 2
        self.wave_spawn_timer = 0

    def _update_waves(self):
        if self.wave_in_progress:
            self.wave_spawn_timer -= 1
            if self.wave_spawn_timer <= 0 and self.enemies_to_spawn_this_wave > 0:
                self.wave_spawn_timer = 30 # Spawn every 30 steps
                self.enemies_to_spawn_this_wave -= 1
                
                health_multiplier = 1 + (self.wave_number - 1) * 0.1
                speed_multiplier = 1 + (self.wave_number - 1) * 0.05
                
                self.enemies.append({
                    "pos": pygame.Vector2(self.path_points[0]),
                    "health": 10 * health_multiplier,
                    "max_health": 10 * health_multiplier,
                    "speed": 1.0 * speed_multiplier,
                    "path_index": 1,
                    "value": 10 + self.wave_number
                })
        elif not self.enemies:
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0:
                self.reward_this_step += 1.0
                self.wave_cooldown = 150
                self._start_next_wave()


    def _update_enemies(self):
        for enemy in reversed(self.enemies):
            if enemy["path_index"] >= len(self.path_points):
                self.base_health -= enemy["max_health"] / 2 # Damage base
                self.enemies.remove(enemy)
                self._create_particles(self.base_pos, (255,100,100), 30, 3, 20)
                continue

            target_pos = self.path_points[enemy["path_index"]]
            direction = (target_pos - enemy["pos"])
            
            if direction.length() < enemy["speed"]:
                enemy["pos"] = target_pos
                enemy["path_index"] += 1
            else:
                enemy["pos"] += direction.normalize() * enemy["speed"]

    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            tower["cooldown"] = max(0, tower["cooldown"] - 1)
            
            if tower["cooldown"] == 0:
                target = None
                min_dist = spec["range"]
                
                for enemy in self.enemies:
                    dist = tower["pos"].distance_to(enemy["pos"])
                    if dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    self.projectiles.append({
                        "pos": pygame.Vector2(tower["pos"]),
                        "target": target,
                        "speed": spec["proj_speed"],
                        "damage": spec["damage"],
                        "color": spec["color"]
                    })
                    tower["cooldown"] = 60 / spec["fire_rate"]

    def _update_projectiles(self):
        for proj in reversed(self.projectiles):
            if proj["target"] not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            direction = (proj["target"]["pos"] - proj["pos"]).normalize()
            proj["pos"] += direction * proj["speed"]
            
            if proj["pos"].distance_to(proj["target"]["pos"]) < 5:
                proj["target"]["health"] -= proj["damage"]
                self.projectiles.remove(proj)
                self._create_particles(proj["pos"], proj["color"], 5, 1, 5)
                
                if proj["target"]["health"] <= 0:
                    self.reward_this_step += 0.1
                    self.score += 1
                    self.money += proj["target"]["value"]
                    self._create_particles(proj["target"]["pos"], (255,215,0), 15, 1.5, 10)
                    self.enemies.remove(proj["target"])

    def _create_particles(self, pos, color, count, speed, lifetime):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed * self.np_random.uniform(0.5, 1.5)
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "lifetime": lifetime, "color": color})
            
    def _update_particles(self):
        for p in reversed(self.particles):
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # friction
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            self.reward_this_step -= 100
            self.game_over_text = "DEFEAT"
            return True
        if self.wave_number > self.MAX_WAVES and not self.enemies:
            self.game_over = True
            self.reward_this_step += 100
            self.game_over_text = "VICTORY!"
            return True
        return False

    def _get_observation(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        
        # Render path
        for i in range(len(self.path_points) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path_points[i], self.path_points[i+1], 35)

        # Render placement spots
        for r in range(self.placement_grid_dims[0]):
            for c in range(self.placement_grid_dims[1]):
                idx = r * self.placement_grid_dims[1] + c
                if self.placement_spots[idx] is not None:
                    color = self.COLOR_PLACEMENT_INVALID if self.placement_occupied[r,c] else self.COLOR_PLACEMENT_VALID
                    pygame.gfxdraw.filled_circle(self.screen, int(self.placement_spots[idx][0]), int(self.placement_spots[idx][1]), 10, color)
        
        # Render entities
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        
        # Render UI
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
        
    def _render_base(self):
        base_rect = pygame.Rect(0, 0, 40, 40)
        base_rect.center = self.base_pos
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)
        
        # Base Health Bar
        if self.base_health < self.max_base_health:
            health_pct = max(0, self.base_health / self.max_base_health)
            bar_width = 100
            bar_height = 10
            bar_x = self.base_pos.x - bar_width/2
            bar_y = self.SCREEN_HEIGHT - 25
            pygame.draw.rect(self.screen, (100,0,0), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_BASE, (bar_x, bar_y, bar_width * health_pct, bar_height))

    def _render_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            pygame.draw.circle(self.screen, spec["color"], (int(tower["pos"].x), int(tower["pos"].y)), 12)
            pygame.draw.circle(self.screen, self.COLOR_BG, (int(tower["pos"].x), int(tower["pos"].y)), 8)
            pygame.draw.circle(self.screen, spec["color"], (int(tower["pos"].x), int(tower["pos"].y)), 4)
            
    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy["pos"].x), int(enemy["pos"].y))
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, 7)
            
            # Health bar
            if enemy["health"] < enemy["max_health"]:
                health_pct = enemy["health"] / enemy["max_health"]
                bar_width = 14
                bar_y = pos[1] - 12
                pygame.draw.line(self.screen, (100,0,0), (pos[0] - bar_width/2, bar_y), (pos[0] + bar_width/2, bar_y), 2)
                pygame.draw.line(self.screen, (0,200,0), (pos[0] - bar_width/2, bar_y), (pos[0] - bar_width/2 + bar_width * health_pct, bar_y), 2)

    def _render_projectiles(self):
        for proj in self.projectiles:
            pygame.draw.circle(self.screen, proj["color"], (int(proj["pos"].x), int(proj["pos"].y)), 3)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifetime"] / 10))))
            color = (*p["color"], alpha)
            # Using simple circles for particles as gfxdraw doesn't support alpha well on surfaces
            temp_surf = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (3,3), int(p["lifetime"]/3))
            self.screen.blit(temp_surf, p["pos"] - pygame.Vector2(3,3))

    def _render_ui(self):
        # Cursor
        grid_idx = (self.cursor_grid_pos[0], self.cursor_grid_pos[1])
        spot_idx = grid_idx[0] * self.placement_grid_dims[1] + grid_idx[1]
        cursor_pos = self.placement_spots[spot_idx]

        if cursor_pos is not None:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            can_afford = self.money >= spec["cost"]
            is_occupied = self.placement_occupied[grid_idx]
            cursor_color = self.COLOR_CURSOR if can_afford and not is_occupied else self.COLOR_ENEMY

            pygame.gfxdraw.aacircle(self.screen, int(cursor_pos[0]), int(cursor_pos[1]), spec["range"], (*cursor_color, 100))
            pygame.draw.circle(self.screen, cursor_color, (int(cursor_pos[0]), int(cursor_pos[1])), 12, 2)
        
        # Info Panel
        panel_surf = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        panel_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(panel_surf, (0,0))
        
        # Money
        money_text = self.font_large.render(f"$ {self.money}", True, (255, 215, 0))
        self.screen.blit(money_text, (10, 5))
        
        # Wave
        wave_text = self.font_large.render(f"Wave: {self.wave_number} / {self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 5))
        
        # Selected Tower
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_info = f"Tower: {spec['name']} | Cost: ${spec['cost']}"
        tower_text = self.font_small.render(tower_info, True, self.COLOR_TEXT)
        self.screen.blit(tower_text, (160, 12))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text_surf = self.game_over_font.render(self.game_over_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "money": self.money,
            "wave": self.wave_number,
            "base_health": self.base_health,
        }

    # --- Helper methods ---
    def _dist_point_to_segment(self, p, a, b):
        p, a, b = pygame.Vector2(p), pygame.Vector2(a), pygame.Vector2(b)
        ab = b - a
        ap = p - a
        if ab.length() == 0:
            return ap.length()
        
        proj = ap.dot(ab) / ab.length_squared()
        
        if proj < 0:
            return ap.length()
        elif proj > 1:
            return (p - b).length()
        else:
            return (a + proj * ab - p).length()

    def close(self):
        pygame.quit()