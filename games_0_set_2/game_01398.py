import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place a tower or start the next wave. Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from enemy waves by strategically placing towers on the grid. Earn resources for each kill."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 16
    GRID_ROWS = 10
    CELL_SIZE = 40
    MAX_STEPS = 3000
    WIN_WAVE = 20
    FPS = 30

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_PATH = (60, 70, 100)
    COLOR_BASE = (0, 150, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_HEALTH_BAR_BG = (80, 20, 20)
    COLOR_HEALTH_BAR_FG = (50, 255, 50)
    
    CURSOR_VALID = (50, 255, 50, 100)
    CURSOR_INVALID = (255, 50, 50, 100)
    
    TOWER_SPECS = {
        0: {"name": "Gatling", "cost": 50, "range": 80, "damage": 4, "fire_rate": 6, "color": (0, 200, 255), "proj_speed": 300},
        1: {"name": "Cannon", "cost": 120, "range": 120, "damage": 25, "fire_rate": 1, "color": (255, 150, 0), "proj_speed": 200},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_large = pygame.font.SysFont("sans-serif", 36)
        
        # Game path definition (grid coordinates)
        self.path_coords = [
            (-1, 2), (1, 2), (1, 7), (4, 7), (4, 1), (7, 1), 
            (7, 8), (10, 8), (10, 2), (13, 2), (13, 5), (16, 5)
        ]
        self.path_pixels = [( (c[0] + 0.5) * self.CELL_SIZE, (c[1] + 0.5) * self.CELL_SIZE) for c in self.path_coords]

        # Initialize state variables
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        # Player state
        self.base_health = 100
        self.resources = 150
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_tower_type = 0
        
        # Game entities
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        # Wave management
        self.wave = 0
        self.game_phase = "build" # "build" or "wave"
        self.wave_spawn_timer = 0
        self.enemies_to_spawn = 0
        self.enemies_killed_in_wave = 0
        
        # Input handling
        self.prev_action = np.array([0, 0, 0])

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Handle discrete actions on press
        self._handle_input(action)
        
        if self.game_phase == "wave":
            self._update_wave_phase()
            
        # Update particles and projectiles regardless of phase
        self._update_projectiles()
        self._update_particles()
        
        # Check for wave completion
        if self.game_phase == "wave" and not self.enemies and self.enemies_to_spawn == 0:
            self.game_phase = "build"
            reward += 1.0 # Wave complete bonus
            if self.wave >= self.WIN_WAVE:
                self.game_won = True
                self.game_over = True
        
        # Calculate rewards from events this step
        reward += self._calculate_reward()

        terminated = self._check_termination()
        if terminated:
            if self.game_won:
                reward += 100
            else:
                reward -= 100
        
        self.prev_action = action
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_action, shift_action = action
        
        # --- Handle key presses (rising edge detection) ---
        move_pressed = movement != 0 and movement != self.prev_action[0]
        space_pressed = space_action == 1 and self.prev_action[1] == 0
        shift_pressed = shift_action == 1 and self.prev_action[2] == 0

        # Cursor movement
        if move_pressed:
            if movement == 1: self.cursor_pos[1] -= 1 # Up
            elif movement == 2: self.cursor_pos[1] += 1 # Down
            elif movement == 3: self.cursor_pos[0] -= 1 # Left
            elif movement == 4: self.cursor_pos[0] += 1 # Right
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # Cycle tower type
        if shift_pressed:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: UI_Bleep

        # Main action: Place tower or start wave
        if space_pressed:
            is_valid_tile = self._is_valid_build_location(self.cursor_pos[0], self.cursor_pos[1])
            spec = self.TOWER_SPECS[self.selected_tower_type]

            if is_valid_tile and self.resources >= spec["cost"]:
                # Place tower
                self.resources -= spec["cost"]
                tower_pos_px = ((self.cursor_pos[0] + 0.5) * self.CELL_SIZE, (self.cursor_pos[1] + 0.5) * self.CELL_SIZE)
                new_tower = {
                    "pos": tower_pos_px,
                    "grid_pos": list(self.cursor_pos),
                    "type": self.selected_tower_type,
                    "spec": spec,
                    "cooldown": 0,
                    "target": None,
                    "angle": 0,
                }
                self.towers.append(new_tower)
                # sfx: Build_Tower
            elif self.game_phase == "build":
                # Start next wave
                self._start_next_wave()
                # sfx: Wave_Start

    def _start_next_wave(self):
        self.game_phase = "wave"
        self.wave += 1
        self.enemies_to_spawn = 2 + self.wave
        self.enemies_killed_in_wave = 0
        self.wave_spawn_timer = 0
    
    def _update_wave_phase(self):
        # Spawn enemies
        self.wave_spawn_timer -= 1
        if self.enemies_to_spawn > 0 and self.wave_spawn_timer <= 0:
            self._spawn_enemy()
            self.enemies_to_spawn -= 1
            self.wave_spawn_timer = self.FPS # 1 second between spawns

        # Update enemies
        for enemy in self.enemies:
            self._move_enemy(enemy)

        # Update towers
        for tower in self.towers:
            self._update_tower(tower)

    def _spawn_enemy(self):
        health_multiplier = 1.05 ** (self.wave - 1)
        speed_multiplier = 1.05 ** (self.wave - 1)
        
        new_enemy = {
            "pos": list(self.path_pixels[0]),
            "path_index": 0,
            "health": 50 * health_multiplier,
            "max_health": 50 * health_multiplier,
            "speed": 20 * speed_multiplier / self.FPS, # per frame
            "reward": 10,
            "is_dead": False
        }
        self.enemies.append(new_enemy)

    def _move_enemy(self, enemy):
        if enemy["path_index"] >= len(self.path_pixels) - 1:
            # Reached base
            self.base_health -= 10
            enemy["is_dead"] = True
            self._create_particles(enemy["pos"], self.COLOR_BASE, 20)
            # sfx: Base_Damage
            return

        target_node = self.path_pixels[enemy["path_index"] + 1]
        direction = np.array(target_node) - np.array(enemy["pos"])
        distance = np.linalg.norm(direction)

        if distance < enemy["speed"]:
            enemy["pos"] = list(target_node)
            enemy["path_index"] += 1
        else:
            move_vec = (direction / distance) * enemy["speed"]
            enemy["pos"][0] += move_vec[0]
            enemy["pos"][1] += move_vec[1]
    
    def _update_tower(self, tower):
        # Cooldown
        if tower["cooldown"] > 0:
            tower["cooldown"] -= 1
            return
            
        # Find target
        tower["target"] = None
        closest_dist = tower["spec"]["range"] ** 2
        for enemy in self.enemies:
            dist_sq = (tower["pos"][0] - enemy["pos"][0])**2 + (tower["pos"][1] - enemy["pos"][1])**2
            if dist_sq < closest_dist:
                closest_dist = dist_sq
                tower["target"] = enemy
        
        # Fire
        if tower["target"]:
            tower["cooldown"] = self.FPS // tower["spec"]["fire_rate"]
            self._fire_projectile(tower)
            # sfx: Tower_Shoot
    
    def _fire_projectile(self, tower):
        spec = tower["spec"]
        proj = {
            "pos": list(tower["pos"]),
            "target": tower["target"],
            "speed": spec["proj_speed"] / self.FPS,
            "damage": spec["damage"],
            "color": spec["color"],
            "is_dead": False
        }
        self.projectiles.append(proj)
        
        # Update tower angle
        dx = tower["target"]["pos"][0] - tower["pos"][0]
        dy = tower["target"]["pos"][1] - tower["pos"][1]
        tower["angle"] = math.degrees(math.atan2(-dy, dx))

    def _update_projectiles(self):
        for p in self.projectiles:
            if p["target"]["is_dead"]:
                p["is_dead"] = True
                continue

            target_pos = p["target"]["pos"]
            direction = np.array(target_pos) - np.array(p["pos"])
            distance = np.linalg.norm(direction)

            if distance < p["speed"]:
                p["is_dead"] = True
                p["target"]["health"] -= p["damage"]
                self._create_particles(p["target"]["pos"], p["color"], 5)
                # sfx: Hit_Impact
            else:
                move_vec = (direction / distance) * p["speed"]
                p["pos"][0] += move_vec[0]
                p["pos"][1] += move_vec[1]

    def _update_particles(self):
        self.particles = [
            p for p in self.particles 
            if p["life"] > 0
        ]
        for p in self.particles:
            p["life"] -= 1
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]

    def _calculate_reward(self):
        reward = 0
        # Check for killed enemies
        surviving_enemies = []
        for enemy in self.enemies:
            if enemy["health"] <= 0 and not enemy["is_dead"]:
                enemy["is_dead"] = True
                self.score += 1
                self.resources += enemy["reward"]
                reward += 0.1 # Kill reward
                self._create_particles(enemy["pos"], self.COLOR_ENEMY, 15)
                # sfx: Enemy_Explode
            
            if not enemy["is_dead"]:
                surviving_enemies.append(enemy)
        
        self.enemies = surviving_enemies
        return reward

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        if self.game_won:
            self.game_over = True
        return self.game_over

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "base_health": self.base_health,
            "resources": self.resources,
            "game_phase": self.game_phase
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        
        # Draw path
        for i in range(len(self.path_pixels) - 1):
            start = self.path_pixels[i]
            end = self.path_pixels[i+1]
            pygame.draw.line(self.screen, self.COLOR_PATH, start, end, self.CELL_SIZE)
        
        # Draw tower placement range preview
        if self.game_phase == "build":
            spec = self.TOWER_SPECS[self.selected_tower_type]
            cursor_px = ((self.cursor_pos[0] + 0.5) * self.CELL_SIZE, (self.cursor_pos[1] + 0.5) * self.CELL_SIZE)
            pygame.gfxdraw.filled_circle(self.screen, int(cursor_px[0]), int(cursor_px[1]), spec["range"], (255, 255, 255, 30))
            pygame.gfxdraw.aacircle(self.screen, int(cursor_px[0]), int(cursor_px[1]), spec["range"], (255, 255, 255, 80))

        # Draw towers
        for tower in self.towers:
            self._render_tower(tower)

        # Draw base
        base_pos = self.path_pixels[-1]
        pygame.gfxdraw.filled_circle(self.screen, int(base_pos[0]), int(base_pos[1]), self.CELL_SIZE // 2, self.COLOR_BASE)
        # FIX: Convert generator to a tuple and clamp color values
        bright_base_color = tuple(min(c + 30, 255) for c in self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, int(base_pos[0]), int(base_pos[1]), self.CELL_SIZE // 2, bright_base_color)

        # Draw enemies
        for enemy in self.enemies:
            self._render_enemy(enemy)

        # Draw projectiles
        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 3, p["color"])
            pygame.gfxdraw.aacircle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 3, p["color"])

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / p["max_life"]))
            # FIX: Ensure alpha is an integer for the color tuple
            color = (*p["color"], int(alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["size"] * (1 - p["life"] / p["max_life"])), color)

        # Draw cursor
        self._render_cursor()

    def _render_tower(self, tower):
        pos = (int(tower["pos"][0]), int(tower["pos"][1]))
        spec = tower["spec"]
        # Base
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, self.COLOR_GRID)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, spec["color"])
        # Turret
        turret_len = 15
        end_x = pos[0] + turret_len * math.cos(math.radians(tower["angle"]))
        end_y = pos[1] - turret_len * math.sin(math.radians(tower["angle"]))
        pygame.draw.line(self.screen, spec["color"], pos, (end_x, end_y), 4)

    def _render_enemy(self, enemy):
        pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
        size = 10
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_ENEMY)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, (255, 150, 150))
        # Health bar
        if enemy["health"] < enemy["max_health"]:
            bar_w = 20
            bar_h = 4
            health_pct = enemy["health"] / enemy["max_health"]
            health_bar_rect_bg = pygame.Rect(pos[0] - bar_w // 2, pos[1] - size - bar_h - 2, bar_w, bar_h)
            health_bar_rect_fg = pygame.Rect(pos[0] - bar_w // 2, pos[1] - size - bar_h - 2, int(bar_w * health_pct), bar_h)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, health_bar_rect_bg)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, health_bar_rect_fg)
            
    def _render_cursor(self):
        is_valid = self._is_valid_build_location(self.cursor_pos[0], self.cursor_pos[1])
        spec = self.TOWER_SPECS[self.selected_tower_type]
        can_afford = self.resources >= spec["cost"]
        
        color = self.CURSOR_VALID if (is_valid and can_afford) else self.CURSOR_INVALID
        rect = pygame.Rect(self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        # Create a temporary surface for transparency
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill(color)
        self.screen.blit(s, rect.topleft)
        # FIX: Convert generator to a tuple and clamp color values
        border_color = tuple(max(c - 50, 0) for c in color[:3])
        pygame.draw.rect(self.screen, border_color, rect, 2)

    def _render_ui(self):
        # Top Bar
        bar_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 30)
        s = pygame.Surface((self.SCREEN_WIDTH, 30), pygame.SRCALPHA)
        s.fill((0, 0, 0, 150))
        self.screen.blit(s, bar_rect.topleft)
        
        # Health and Resources
        health_text = self.font_small.render(f"Base Health: {max(0, self.base_health)}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 5))
        
        resource_text = self.font_small.render(f"Resources: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(resource_text, (180, 5))

        # Wave Info
        wave_text = self.font_small.render(f"Wave: {self.wave}/{self.WIN_WAVE}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - 100, 5))

        # Tower Selection Info
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_info_text = self.font_small.render(f"Selected: {spec['name']} (Cost: {spec['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(tower_info_text, (320, 5))
        
        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
            
        elif self.game_phase == "build" and self.wave > 0:
            text_surf = self.font_large.render("WAVE CLEARED", True, (100, 255, 100))
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 30))
            self.screen.blit(text_surf, text_rect)
            
            sub_text_surf = self.font_small.render("Place towers or press SPACE to start next wave", True, self.COLOR_TEXT)
            sub_text_rect = sub_text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(sub_text_surf, sub_text_rect)

    def _is_valid_build_location(self, grid_x, grid_y):
        # Check if on path
        for i in range(len(self.path_coords) - 1):
            p1 = self.path_coords[i]
            p2 = self.path_coords[i+1]
            if p1[0] == p2[0] == grid_x and min(p1[1], p2[1]) <= grid_y <= max(p1[1], p2[1]):
                return False
            if p1[1] == p2[1] == grid_y and min(p1[0], p2[0]) <= grid_x <= max(p1[0], p2[0]):
                return False
        # Check if occupied by another tower
        for tower in self.towers:
            if tower["grid_pos"] == [grid_x, grid_y]:
                return False
        return True

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(10, 20),
                "max_life": 20,
                "color": color,
                "size": random.randint(2, 5)
            })