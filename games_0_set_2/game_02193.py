
# Generated: 2025-08-28T04:01:28.039831
# Source Brief: brief_02193.md
# Brief Index: 2193

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Hold Shift to cycle tower types. Press Space to build a tower "
        "or to start the next wave when hovering the 'Start Wave' button."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A vibrant, geometric tower defense game. Place towers to defend your "
        "base from waves of enemies. Survive all 5 waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.CELL_SIZE = 40
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.MAX_STEPS = 3000 # Increased from 1000 to allow for longer games
        self.FPS = 30

        # Colors
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PATH = (60, 60, 80)
        self.COLOR_BASE = (0, 200, 100)
        self.COLOR_BASE_DMG = (255, 100, 100)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 0, 150)
        self.COLOR_TOWER_RANGE = (255, 255, 255, 30)
        self.TOWER_COLORS = [(80, 150, 255), (255, 150, 80)]
        self.PROJECTILE_COLORS = [(150, 200, 255), (255, 200, 150)]

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 16)
        self.font_m = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 32, bold=True)
        
        # Tower Definitions
        self.TOWER_SPECS = [
            {"name": "Cannon", "cost": 100, "range": 80, "fire_rate": 0.8, "damage": 25, "proj_speed": 8},
            {"name": "Missile", "cost": 150, "range": 120, "fire_rate": 2.0, "damage": 70, "proj_speed": 5},
        ]

        # Wave Definitions
        self.WAVE_CONFIG = [
            {"count": 5, "speed": 1.0, "health": 100, "reward": 10},
            {"count": 8, "speed": 1.1, "health": 120, "reward": 15},
            {"count": 12, "speed": 1.2, "health": 140, "reward": 20},
            {"count": 15, "speed": 1.3, "health": 160, "reward": 25},
            {"count": 20, "speed": 1.5, "health": 200, "reward": 30},
        ]
        self.TOTAL_WAVES = len(self.WAVE_CONFIG)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_pos_grid = (self.GRID_SIZE // 2, self.GRID_SIZE // 2 -1)
        self.path_grid = []
        self.path_pixels = []
        self.base_health = 0
        self.resources = 0
        self.current_wave = 0
        self.game_phase = "placement"
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.action_debounce = {"move": 0, "shift": 0, "space": 0}
        self.last_space_held = False
        self.last_shift_held = False
        self.base_hit_timer = 0

        self.start_wave_button = pygame.Rect(self.WIDTH - 150, self.HEIGHT - 55, 140, 45)
        
        self.reset()
        self.validate_implementation()
    
    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return x, y

    def _generate_path(self):
        path = []
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        
        # Mark base and surroundings as occupied
        bx, by = self.base_pos_grid
        for i in range(-1, 2):
            for j in range(-1, 2):
                if 0 <= bx + i < self.GRID_SIZE and 0 <= by + j < self.GRID_SIZE:
                    grid[bx + i][by + j] = 1

        # Random start on an edge
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            start_pos = [self.np_random.integers(self.GRID_SIZE), 0]
        elif edge == 1: # Bottom
            start_pos = [self.np_random.integers(self.GRID_SIZE), self.GRID_SIZE - 1]
        elif edge == 2: # Left
            start_pos = [0, self.np_random.integers(self.GRID_SIZE)]
        else: # Right
            start_pos = [self.GRID_SIZE - 1, self.np_random.integers(self.GRID_SIZE)]
        
        # Ensure start is not on the base
        while grid[start_pos[0]][start_pos[1]] == 1:
            start_pos[0] = (start_pos[0] + 1) % self.GRID_SIZE
        
        pos = start_pos
        path.append(tuple(pos))
        grid[pos[0]][pos[1]] = 1

        for _ in range(self.GRID_SIZE * 3): # Path length limit
            if tuple(pos) == self.base_pos_grid: break
            
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and grid[nx][ny] == 0:
                    neighbors.append((nx, ny))
            
            if not neighbors: break

            # Move towards base
            best_neighbor = min(neighbors, key=lambda n: math.hypot(n[0] - bx, n[1] - by))
            pos = list(best_neighbor)
            path.append(tuple(pos))
            grid[pos[0]][pos[1]] = 1
        
        # Final segment to base
        if path[-1] != self.base_pos_grid:
            path.append(self.base_pos_grid)
        
        self.path_grid = path
        self.path_pixels = [self._grid_to_pixel(p) for p in self.path_grid]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = 100
        self.resources = 250
        self.current_wave = 0
        self.game_phase = "placement"
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.base_hit_timer = 0
        
        self._generate_path()
        
        return self._get_observation(), self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        
        # Debounce actions
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        # Placement phase actions
        if self.game_phase == "placement":
            # Move cursor
            if movement != 0 and self.action_debounce["move"] <= 0:
                dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
                self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_SIZE - 1)
                self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_SIZE - 1)
                self.action_debounce["move"] = 5 # 5 frames cooldown
            
            # Cycle tower type
            if shift_pressed:
                self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
                
            # Place tower or start wave
            if space_pressed:
                cursor_pixel_pos = self._grid_to_pixel(self.cursor_pos)
                if self.start_wave_button.collidepoint(cursor_pixel_pos):
                    self._start_wave()
                else:
                    self._place_tower()

    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.resources >= spec["cost"]:
            is_on_path = tuple(self.cursor_pos) in self.path_grid
            is_occupied = any(t["grid_pos"] == self.cursor_pos for t in self.towers)
            
            if not is_on_path and not is_occupied:
                self.resources -= spec["cost"]
                new_tower = {
                    "grid_pos": list(self.cursor_pos),
                    "pixel_pos": self._grid_to_pixel(self.cursor_pos),
                    "type": self.selected_tower_type,
                    "spec": spec,
                    "cooldown": 0,
                    "fire_anim_timer": 0,
                }
                self.towers.append(new_tower)
                # Sound: "build_tower.wav"
                self._create_particles(new_tower["pixel_pos"], 20, self.TOWER_COLORS[self.selected_tower_type])

    def _start_wave(self):
        if self.game_phase == "placement":
            self.game_phase = "wave"
            self.current_wave += 1
            wave_info = self.WAVE_CONFIG[self.current_wave - 1]
            for i in range(wave_info["count"]):
                self.enemies.append({
                    "pos": list(self.path_pixels[0]),
                    "health": wave_info["health"],
                    "max_health": wave_info["health"],
                    "speed": wave_info["speed"],
                    "path_index": 0,
                    "spawn_delay": i * (self.FPS // 2), # 0.5 sec delay between spawns
                    "is_active": False,
                })
            # Sound: "wave_start.wav"

    def _update_game_state(self):
        reward = 0
        
        # Update cooldowns and timers
        for key in self.action_debounce: self.action_debounce[key] = max(0, self.action_debounce[key] - 1)
        self.base_hit_timer = max(0, self.base_hit_timer - 1)

        # Wave phase updates
        if self.game_phase == "wave":
            # Update enemies
            enemies_to_remove = []
            for i, enemy in enumerate(self.enemies):
                if not enemy["is_active"]:
                    enemy["spawn_delay"] -= 1
                    if enemy["spawn_delay"] <= 0: enemy["is_active"] = True
                    continue

                if enemy["path_index"] >= len(self.path_pixels) - 1:
                    self.base_health -= 10
                    self.base_hit_timer = 10 # Screen shake/flash duration
                    enemies_to_remove.append(i)
                    # Sound: "base_hit.wav"
                    continue
                
                target_pos = self.path_pixels[enemy["path_index"] + 1]
                dx = target_pos[0] - enemy["pos"][0]
                dy = target_pos[1] - enemy["pos"][1]
                dist = math.hypot(dx, dy)
                
                if dist < enemy["speed"]:
                    enemy["path_index"] += 1
                else:
                    enemy["pos"][0] += (dx / dist) * enemy["speed"]
                    enemy["pos"][1] += (dy / dist) * enemy["speed"]
            
            self.enemies = [e for i, e in enumerate(self.enemies) if i not in enemies_to_remove]

            # Update towers
            for tower in self.towers:
                tower["cooldown"] = max(0, tower["cooldown"] - 1 / self.FPS)
                tower["fire_anim_timer"] = max(0, tower["fire_anim_timer"] - 1)
                if tower["cooldown"] <= 0:
                    target = None
                    # Find first enemy in range
                    for enemy in self.enemies:
                        if not enemy["is_active"]: continue
                        dist = math.hypot(enemy["pos"][0] - tower["pixel_pos"][0], enemy["pos"][1] - tower["pixel_pos"][1])
                        if dist <= tower["spec"]["range"]:
                            target = enemy
                            break
                    
                    if target:
                        self.projectiles.append({
                            "pos": list(tower["pixel_pos"]),
                            "type": tower["type"],
                            "spec": tower["spec"],
                            "target": target,
                        })
                        tower["cooldown"] = tower["spec"]["fire_rate"]
                        tower["fire_anim_timer"] = 5
                        # Sound: "tower_fire.wav"
            
            # Update projectiles
            projectiles_to_remove = []
            enemies_to_remove = []
            for i, proj in enumerate(self.projectiles):
                if proj["target"] in self.enemies and proj["target"]["is_active"]:
                    target_pos = proj["target"]["pos"]
                    dx = target_pos[0] - proj["pos"][0]
                    dy = target_pos[1] - proj["pos"][1]
                    dist = math.hypot(dx, dy)
                    
                    if dist < proj["spec"]["proj_speed"]:
                        proj["target"]["health"] -= proj["spec"]["damage"]
                        projectiles_to_remove.append(i)
                        self._create_particles(proj["pos"], 10, self.PROJECTILE_COLORS[proj["type"]])
                        # Sound: "enemy_hit.wav"
                        
                        if proj["target"]["health"] <= 0:
                            if proj["target"] not in enemies_to_remove:
                                enemies_to_remove.append(proj["target"])
                    else:
                        proj["pos"][0] += (dx / dist) * proj["spec"]["proj_speed"]
                        proj["pos"][1] += (dy / dist) * proj["spec"]["proj_speed"]
                else: # Target is gone
                    projectiles_to_remove.append(i)
            
            # Process projectile and enemy removals
            for enemy_to_remove in enemies_to_remove:
                if enemy_to_remove in self.enemies:
                    wave_info = self.WAVE_CONFIG[self.current_wave - 1]
                    reward += 0.1
                    self.score += wave_info["reward"]
                    self.resources += wave_info["reward"]
                    self._create_particles(enemy_to_remove["pos"], 30, self.COLOR_ENEMY)
                    self.enemies.remove(enemy_to_remove)
                    # Sound: "enemy_destroy.wav"
            
            self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
            
            # Check for wave end
            if not self.enemies:
                self.game_phase = "placement"
                reward += 1
                if self.current_wave >= self.TOTAL_WAVES:
                    self.game_over = True
                # Sound: "wave_complete.wav"

        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
        
        return reward
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)
        
        reward = self._update_game_state()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.base_health <= 0:
                reward = -100
            elif self.current_wave >= self.TOTAL_WAVES:
                reward = 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.current_wave >= self.TOTAL_WAVES and self.game_phase == "placement" and not self.enemies:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": f"{self.current_wave}/{self.TOTAL_WAVES}",
            "phase": self.game_phase,
        }

    def _get_observation(self):
        # Screen shake effect
        render_offset_x, render_offset_y = 0, 0
        if self.base_hit_timer > 0:
            render_offset_x = self.np_random.integers(-5, 6)
            render_offset_y = self.np_random.integers(-5, 6)
        
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game(render_offset_x, render_offset_y)
        
        # Render UI overlay (not affected by shake)
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, ox, oy):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X + ox, self.GRID_OFFSET_Y + i * self.CELL_SIZE + oy), (self.GRID_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE + ox, self.GRID_OFFSET_Y + i * self.CELL_SIZE + oy))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X + i * self.CELL_SIZE + ox, self.GRID_OFFSET_Y + oy), (self.GRID_OFFSET_X + i * self.CELL_SIZE + ox, self.GRID_OFFSET_Y + self.GRID_SIZE * self.CELL_SIZE + oy))

        # Draw path
        if len(self.path_pixels) > 1:
            path_rects = [pygame.Rect(p[0] - self.CELL_SIZE//2 + ox, p[1] - self.CELL_SIZE//2 + oy, self.CELL_SIZE, self.CELL_SIZE) for p in self.path_pixels]
            for r in path_rects:
                pygame.draw.rect(self.screen, self.COLOR_PATH, r)

        # Draw base
        base_px, base_py = self._grid_to_pixel(self.base_pos_grid)
        base_color = self.COLOR_BASE_DMG if self.base_hit_timer > 0 else self.COLOR_BASE
        pygame.draw.rect(self.screen, base_color, (base_px - 15 + ox, base_py - 15 + oy, 30, 30))
        pygame.gfxdraw.rectangle(self.screen, (base_px - 15 + ox, base_py - 15 + oy, 30, 30), (255,255,255))

        # Draw towers and ranges
        for tower in self.towers:
            pos = (tower["pixel_pos"][0] + ox, tower["pixel_pos"][1] + oy)
            if self.game_phase == "placement":
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(tower["spec"]["range"]), self.COLOR_TOWER_RANGE)
            
            color = self.TOWER_COLORS[tower["type"]]
            if tower["fire_anim_timer"] > 0:
                color = (255, 255, 255) # Flash white when firing
            
            if tower["type"] == 0: # Cannon
                pygame.draw.rect(self.screen, color, (pos[0] - 10, pos[1] - 10, 20, 20))
            elif tower["type"] == 1: # Missile
                points = [(pos[0], pos[1] - 12), (pos[0] - 10, pos[1] + 8), (pos[0] + 10, pos[1] + 8)]
                pygame.draw.polygon(self.screen, color, points)

        # Draw enemies
        for enemy in self.enemies:
            if not enemy["is_active"]: continue
            pos = (int(enemy["pos"][0] + ox), int(enemy["pos"][1] + oy))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_ENEMY)
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            pygame.draw.rect(self.screen, (50, 50, 50), (pos[0] - 12, pos[1] - 20, 24, 4))
            pygame.draw.rect(self.screen, (0, 255, 0), (pos[0] - 12, pos[1] - 20, 24 * health_ratio, 4))
            
        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"][0] + ox), int(proj["pos"][1] + oy))
            color = self.PROJECTILE_COLORS[proj["type"]]
            pygame.draw.rect(self.screen, color, (pos[0] - 2, pos[1] - 2, 5, 5))

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / p["max_life"]))))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (p["pos"][0] - p["size"] + ox, p["pos"][1] - p["size"] + oy))
    
    def _render_ui(self):
        # Draw Base Health
        health_ratio = max(0, self.base_health / 100)
        pygame.draw.rect(self.screen, (50, 0, 0), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (10, 10, 200 * health_ratio, 20))
        health_text = self.font_s.render(f"BASE HEALTH: {self.base_health}/100", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Draw Wave Info
        wave_str = f"WAVE {self.current_wave}/{self.TOTAL_WAVES}" if self.current_wave > 0 else "PREPARING"
        wave_text = self.font_m.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Draw Resources
        resource_text = self.font_m.render(f"$: {self.resources}", True, (255, 223, 0))
        self.screen.blit(resource_text, (self.WIDTH - resource_text.get_width() - 10, 35))

        # Draw Cursor in placement phase
        if self.game_phase == "placement":
            cursor_px, cursor_py = self._grid_to_pixel(self.cursor_pos)
            
            # Cursor itself
            cursor_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            cursor_surf.fill(self.COLOR_CURSOR)
            self.screen.blit(cursor_surf, (cursor_px - self.CELL_SIZE//2, cursor_py - self.CELL_SIZE//2))
            
            # Tower range preview
            spec = self.TOWER_SPECS[self.selected_tower_type]
            pygame.gfxdraw.filled_circle(self.screen, cursor_px, cursor_py, spec["range"], self.COLOR_TOWER_RANGE)
            pygame.gfxdraw.aacircle(self.screen, cursor_px, cursor_py, spec["range"], (255,255,255, 80))

        # Bottom UI Panel
        panel_rect = pygame.Rect(0, self.HEIGHT - 70, self.WIDTH, 70)
        pygame.draw.rect(self.screen, (15, 15, 25), panel_rect)
        pygame.draw.line(self.screen, (80, 80, 100), (0, self.HEIGHT - 70), (self.WIDTH, self.HEIGHT - 70))

        # Tower Selection UI
        for i, spec in enumerate(self.TOWER_SPECS):
            x_pos = 20 + i * 150
            box_rect = pygame.Rect(x_pos, self.HEIGHT - 60, 140, 50)
            is_selected = (i == self.selected_tower_type)
            can_afford = self.resources >= spec["cost"]
            
            border_color = (255, 255, 0) if is_selected else (80, 80, 100)
            pygame.draw.rect(self.screen, (30, 30, 50), box_rect)
            pygame.draw.rect(self.screen, border_color, box_rect, 2)
            
            name_color = self.COLOR_TEXT if can_afford else (150, 150, 150)
            name_text = self.font_m.render(spec["name"], True, name_color)
            self.screen.blit(name_text, (x_pos + 5, self.HEIGHT - 55))
            
            cost_color = (255, 223, 0) if can_afford else (150, 100, 0)
            cost_text = self.font_s.render(f"Cost: ${spec['cost']}", True, cost_color)
            self.screen.blit(cost_text, (x_pos + 5, self.HEIGHT - 35))

        # Start Wave Button
        if self.game_phase == "placement":
            pygame.draw.rect(self.screen, (0, 100, 50), self.start_wave_button)
            pygame.draw.rect(self.screen, (0, 200, 100), self.start_wave_button, 2)
            start_text = self.font_m.render(f"START WAVE {self.current_wave + 1}", True, self.COLOR_TEXT)
            text_rect = start_text.get_rect(center=self.start_wave_button.center)
            self.screen.blit(start_text, text_rect)
        
        # Game Over / Win message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            msg = "YOU WIN!" if self.base_health > 0 else "GAME OVER"
            color = (0, 255, 0) if self.base_health > 0 else (255, 0, 0)
            
            end_text = self.font_l.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_text, text_rect)
            
            score_text = self.font_m.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(score_text, score_rect)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(10, 20),
                "max_life": 20,
                "color": color,
                "size": self.np_random.integers(2, 5),
            })
            
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
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

        # Pygame uses a different coordinate system for blitting
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()