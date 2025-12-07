
# Generated: 2025-08-28T03:43:28.754235
# Source Brief: brief_02093.md
# Brief Index: 2093

        
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
        "Controls: Use arrow keys to move the placement cursor. Press Shift to cycle through tower types. "
        "Press Space to place the selected tower on an empty grid cell. "
        "Move the cursor over the 'START WAVE' button and press Space to begin the attack."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist tower defense game. Strategically place towers to defend your base "
        "against waves of incoming enemies. Survive all waves to win."
    )

    # auto_advance is True because waves happen in real-time.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_SIZE = 40
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        self.MAX_WAVES = 5
        self.MAX_STEPS = 15000 # Approx 8 minutes at 30fps

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_PATH = (45, 45, 65)
        self.COLOR_BASE = (0, 200, 100)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.TOWER_COLORS = [(80, 150, 255), (255, 220, 80), (200, 80, 255)]
        self.TOWER_NAMES = ["RAPID", "CANNON", "PULSE"]

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
        self.font_small = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_huge = pygame.font.SysFont("Arial", 48, bold=True)

        # Game definitions
        self._define_path()
        self._define_towers()

        # Initialize state variables
        self.state = {}
        self.reset()
        
        self.validate_implementation()


    def _define_path(self):
        # Path defined in grid coordinates
        path_coords = [
            (-1, 5), (2, 5), (2, 2), (6, 2), (6, 7),
            (10, 7), (10, 3), (13, 3), (13, 8), (16, 8)
        ]
        self.enemy_path = [(x * self.GRID_SIZE + self.GRID_SIZE // 2, y * self.GRID_SIZE + self.GRID_SIZE // 2) for x, y in path_coords]

    def _define_towers(self):
        self.tower_definitions = [
            # Type 1: Rapid
            {"cost": 75, "range": 80, "fire_rate": 0.3, "damage": 5, "projectile_speed": 8},
            # Type 2: Cannon
            {"cost": 150, "range": 120, "fire_rate": 1.5, "damage": 35, "projectile_speed": 10},
            # Type 3: Pulse (AoE)
            {"cost": 200, "range": 100, "fire_rate": 2.0, "damage": 20, "aoe_radius": 30, "projectile_speed": 6},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = {
            "steps": 0,
            "score": 0,
            "game_over": False,
            "game_phase": "PLACEMENT", # "PLACEMENT", "WAVE", "GAME_OVER", "VICTORY"
            "base_health": 100,
            "max_base_health": 100,
            "wave_number": 1,
            "resources": 200,
            "cursor_pos": [self.GRID_W // 2, self.GRID_H // 2],
            "selected_tower": 0,
            "last_space_held": False,
            "last_shift_held": False,
            "towers": [],
            "enemies": [],
            "enemies_to_spawn": [],
            "wave_spawn_timer": 0.0,
            "projectiles": [],
            "particles": [],
            "pending_rewards": [],
        }
        self._start_placement_phase()
        return self._get_observation(), self._get_info()

    def _start_placement_phase(self):
        self.state["game_phase"] = "PLACEMENT"
        if self.state["wave_number"] > 1:
            self.state["resources"] += 100 + (self.state["wave_number"] - 1) * 25

    def _start_wave(self):
        self.state["game_phase"] = "WAVE"
        wave = self.state["wave_number"]
        
        base_health = 50
        base_speed = 1.0
        num_enemies = 5 + (wave - 1) * 3

        # Difficulty scaling
        current_health = base_health * (1.2 ** (wave - 1))
        current_speed = base_speed * (1.08 ** (wave - 1))

        self.state["enemies_to_spawn"] = [
            self._create_enemy(current_health, current_speed) for _ in range(num_enemies)
        ]
        self.state["wave_spawn_timer"] = 0.0

    def _create_enemy(self, health, speed):
        return {
            "pos": list(self.enemy_path[0]),
            "health": health,
            "max_health": health,
            "speed": speed,
            "waypoint_index": 1,
            "id": self.np_random.integers(1_000_000),
        }

    def step(self, action):
        if self.state["game_over"]:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.state["steps"] += 1

        # Process player actions
        self._handle_actions(action)

        # Update game logic based on phase
        if self.state["game_phase"] == "WAVE":
            reward -= 0.001 # Small penalty for time passing
            self._update_spawner()
            self._update_enemies()
            self._update_towers()
            self._update_projectiles()
        
        self._update_particles()
        
        # Collect rewards
        reward += sum(self.state["pending_rewards"])
        self.state["pending_rewards"] = []
        self.state["score"] += reward

        # Check for termination
        terminated = self._check_termination()
        if terminated:
            self.state["game_over"] = True
            if self.state["base_health"] <= 0:
                self.state["game_phase"] = "GAME_OVER"
                reward -= 100
                self.state["score"] -= 100
            elif self.state["wave_number"] > self.MAX_WAVES:
                self.state["game_phase"] = "VICTORY"
                reward += 100
                self.state["score"] += 100
        
        if self.state["steps"] >= self.MAX_STEPS and not terminated:
            terminated = True
            self.state["game_over"] = True
            self.state["game_phase"] = "GAME_OVER"
            reward -= 100 # Penalty for running out of time
            self.state["score"] -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_actions(self, action):
        movement, space_val, shift_val = action
        space_pressed = space_val == 1 and not self.state["last_space_held"]
        shift_pressed = shift_val == 1 and not self.state["last_shift_held"]
        self.state["last_space_held"] = space_val == 1
        self.state["last_shift_held"] = shift_val == 1

        # --- Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -1
        elif movement == 2: dy = 1
        elif movement == 3: dx = -1
        elif movement == 4: dx = 1
        
        if dx != 0 or dy != 0:
            self.state["cursor_pos"][0] = np.clip(self.state["cursor_pos"][0] + dx, 0, self.GRID_W - 1)
            self.state["cursor_pos"][1] = np.clip(self.state["cursor_pos"][1] + dy, 0, self.GRID_H - 1)

        # --- Shift: Cycle Tower ---
        if shift_pressed:
            self.state["selected_tower"] = (self.state["selected_tower"] + 1) % len(self.tower_definitions)
            # sfx: ui_cycle.wav

        # --- Space: Place Tower / Start Wave ---
        if space_pressed and self.state["game_phase"] == "PLACEMENT":
            cx, cy = self.state["cursor_pos"]
            start_button_rect = pygame.Rect(self.WIDTH - 140, self.HEIGHT - 35, 130, 25)
            cursor_pixel_pos = (cx * self.GRID_SIZE + self.GRID_SIZE // 2, cy * self.GRID_SIZE + self.GRID_SIZE // 2)

            if start_button_rect.collidepoint(cursor_pixel_pos):
                self._start_wave()
                # sfx: wave_start.wav
            else:
                self._place_tower(cx, cy)

    def _place_tower(self, gx, gy):
        tower_def = self.tower_definitions[self.state["selected_tower"]]
        if self.state["resources"] < tower_def["cost"]:
            # sfx: error.wav
            return

        is_occupied = any(t["grid_pos"] == [gx, gy] for t in self.state["towers"])
        if is_occupied:
            # sfx: error.wav
            return
            
        self.state["resources"] -= tower_def["cost"]
        new_tower = {
            "type": self.state["selected_tower"],
            "grid_pos": [gx, gy],
            "pixel_pos": (gx * self.GRID_SIZE + self.GRID_SIZE // 2, gy * self.GRID_SIZE + self.GRID_SIZE // 2),
            "cooldown": 0.0,
            "target_id": None,
        }
        self.state["towers"].append(new_tower)
        # sfx: place_tower.wav
        
    def _update_spawner(self):
        if not self.state["enemies_to_spawn"]:
            return

        self.state["wave_spawn_timer"] -= 1 / self.FPS
        if self.state["wave_spawn_timer"] <= 0:
            self.state["enemies"].append(self.state["enemies_to_spawn"].pop(0))
            self.state["wave_spawn_timer"] = 0.5 # Time between spawns

    def _update_enemies(self):
        for enemy in reversed(self.state["enemies"]):
            if enemy["waypoint_index"] >= len(self.enemy_path):
                self.state["base_health"] -= 10
                self.state["base_health"] = max(0, self.state["base_health"])
                self.state["enemies"].remove(enemy)
                # sfx: base_damage.wav
                continue

            target_pos = self.enemy_path[enemy["waypoint_index"]]
            direction = np.array(target_pos) - np.array(enemy["pos"])
            distance = np.linalg.norm(direction)

            if distance < enemy["speed"]:
                enemy["waypoint_index"] += 1
            else:
                move_vec = (direction / distance) * enemy["speed"]
                enemy["pos"][0] += move_vec[0]
                enemy["pos"][1] += move_vec[1]

    def _update_towers(self):
        for tower in self.state["towers"]:
            tower["cooldown"] = max(0, tower["cooldown"] - 1 / self.FPS)
            if tower["cooldown"] > 0:
                continue

            tower_def = self.tower_definitions[tower["type"]]
            
            # Retain target if still valid
            target = None
            if tower.get("target_id"):
                for e in self.state["enemies"]:
                    if e["id"] == tower["target_id"]:
                        dist_sq = (e["pos"][0] - tower["pixel_pos"][0])**2 + (e["pos"][1] - tower["pixel_pos"][1])**2
                        if dist_sq <= tower_def["range"]**2:
                            target = e
                        break
            
            # Find new target if necessary
            if not target:
                tower["target_id"] = None
                # Find closest enemy in range
                min_dist_sq = float('inf')
                for enemy in self.state["enemies"]:
                    dist_sq = (enemy["pos"][0] - tower["pixel_pos"][0])**2 + (enemy["pos"][1] - tower["pixel_pos"][1])**2
                    if dist_sq <= tower_def["range"]**2 and dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        target = enemy

            if target:
                tower["target_id"] = target["id"]
                tower["cooldown"] = tower_def["fire_rate"]
                self._create_projectile(tower, target)
                # sfx: shoot_tower_X.wav

    def _create_projectile(self, tower, target):
        tower_def = self.tower_definitions[tower["type"]]
        proj = {
            "pos": list(tower["pixel_pos"]),
            "type": tower["type"],
            "target_id": target["id"],
            "speed": tower_def["projectile_speed"],
            "damage": tower_def["damage"],
        }
        if tower["type"] == 2: # AoE
            proj["aoe_radius"] = tower_def["aoe_radius"]
        
        self.state["projectiles"].append(proj)

    def _update_projectiles(self):
        for proj in reversed(self.state["projectiles"]):
            target = None
            for e in self.state["enemies"]:
                if e["id"] == proj["target_id"]:
                    target = e
                    break
            
            if not target:
                self.state["projectiles"].remove(proj)
                continue

            direction = np.array(target["pos"]) - np.array(proj["pos"])
            distance = np.linalg.norm(direction)

            if distance < proj["speed"]:
                self._hit_target(proj, target)
                self.state["projectiles"].remove(proj)
            else:
                move_vec = (direction / distance) * proj["speed"]
                proj["pos"][0] += move_vec[0]
                proj["pos"][1] += move_vec[1]
    
    def _hit_target(self, proj, target):
        # sfx: impact.wav
        self.state["pending_rewards"].append(0.01) # Reward for hitting
        
        if proj["type"] == 2: # AoE
            self._create_explosion(proj["pos"], proj["damage"], proj["aoe_radius"])
        else:
            self._damage_enemy(target, proj["damage"])
        
        self._create_particles(target["pos"], 5, self.TOWER_COLORS[proj["type"]])

    def _damage_enemy(self, enemy, damage):
        enemy["health"] -= damage
        if enemy["health"] <= 0:
            self.state["pending_rewards"].append(1.0) # Reward for kill
            self._create_particles(enemy["pos"], 20, self.COLOR_ENEMY)
            if enemy in self.state["enemies"]:
                self.state["enemies"].remove(enemy)
            # sfx: enemy_die.wav
            
            # Check for wave completion
            if self.state["game_phase"] == "WAVE" and not self.state["enemies"] and not self.state["enemies_to_spawn"]:
                self.state["pending_rewards"].append(10.0) # Wave clear bonus
                self.state["wave_number"] += 1
                if self.state["wave_number"] > self.MAX_WAVES:
                    pass # Victory will be handled in step
                else:
                    self._start_placement_phase()
                # sfx: wave_clear.wav

    def _create_explosion(self, pos, damage, radius):
        # sfx: explosion.wav
        self._create_particles(pos, 30, self.TOWER_COLORS[2])
        for enemy in self.state["enemies"]:
            dist_sq = (enemy["pos"][0] - pos[0])**2 + (enemy["pos"][1] - pos[1])**2
            if dist_sq <= radius**2:
                self._damage_enemy(enemy, damage)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.uniform(0.3, 0.8)
            self.state["particles"].append({"pos": list(pos), "vel": vel, "lifetime": lifetime, "color": color})
            
    def _update_particles(self):
        for p in reversed(self.state["particles"]):
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifetime"] -= 1 / self.FPS
            if p["lifetime"] <= 0:
                self.state["particles"].remove(p)

    def _check_termination(self):
        return self.state["base_health"] <= 0 or self.state["wave_number"] > self.MAX_WAVES

    def _get_info(self):
        return {
            "score": self.state["score"],
            "steps": self.state["steps"],
            "wave": self.state["wave_number"],
            "health": self.state["base_health"],
            "resources": self.state["resources"],
            "enemies_left": len(self.state["enemies"]) + len(self.state["enemies_to_spawn"]),
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.state["game_phase"] in ["GAME_OVER", "VICTORY"]:
            self._render_end_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_grid_and_path()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        if self.state["game_phase"] == "PLACEMENT":
            self._render_cursor()

    def _render_grid_and_path(self):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        if len(self.enemy_path) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.enemy_path, 20)

        # Base
        base_rect = pygame.Rect(self.enemy_path[-1][0] - 15, self.enemy_path[-1][1] - 15, 30, 30)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

    def _render_towers(self):
        for tower in self.state["towers"]:
            pos = tower["pixel_pos"]
            tower_def = self.tower_definitions[tower["type"]]
            color = self.TOWER_COLORS[tower["type"]]
            
            # Range indicator
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(tower_def["range"]), (*color, 60))

            if tower["type"] == 0: # Triangle
                points = [
                    (pos[0], pos[1] - 12),
                    (pos[0] - 10, pos[1] + 8),
                    (pos[0] + 10, pos[1] + 8),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            elif tower["type"] == 1: # Square
                rect = pygame.Rect(pos[0] - 10, pos[1] - 10, 20, 20)
                pygame.draw.rect(self.screen, color, rect)
            elif tower["type"] == 2: # Pentagon
                points = []
                for i in range(5):
                    angle = math.pi / 2 + 2 * math.pi * i / 5
                    points.append((pos[0] + 12 * math.cos(angle), pos[1] + 12 * math.sin(angle)))
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_enemies(self):
        for enemy in self.state["enemies"]:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            radius = 10
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            
            # Health bar
            if enemy["health"] < enemy["max_health"]:
                health_pct = enemy["health"] / enemy["max_health"]
                bar_w = 20
                bar_h = 4
                bar_x = pos[0] - bar_w // 2
                bar_y = pos[1] - radius - bar_h - 2
                pygame.draw.rect(self.screen, (80, 0, 0), (bar_x, bar_y, bar_w, bar_h))
                pygame.draw.rect(self.screen, self.COLOR_ENEMY, (bar_x, bar_y, int(bar_w * health_pct), bar_h))


    def _render_projectiles(self):
        for proj in self.state["projectiles"]:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            color = self.TOWER_COLORS[proj["type"]]
            pygame.draw.rect(self.screen, color, (pos[0] - 2, pos[1] - 2, 5, 5))

    def _render_particles(self):
        for p in self.state["particles"]:
            alpha = max(0, min(255, int(255 * (p["lifetime"] / 0.8))))
            color = (*p["color"], alpha)
            size = max(1, int(3 * p["lifetime"]))
            # Pygame doesn't handle alpha well on basic shapes, this is an approximation
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), size)

    def _render_cursor(self):
        cx, cy = self.state["cursor_pos"]
        pixel_pos = (cx * self.GRID_SIZE, cy * self.GRID_SIZE)
        rect = pygame.Rect(pixel_pos[0], pixel_pos[1], self.GRID_SIZE, self.GRID_SIZE)
        
        tower_def = self.tower_definitions[self.state["selected_tower"]]
        can_afford = self.state["resources"] >= tower_def["cost"]
        color = (255, 255, 255, 100) if can_afford else (255, 0, 0, 100)
        
        # Transparent overlay for cursor
        s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        s.fill(color)
        self.screen.blit(s, rect.topleft)
        
        # Range indicator for potential tower
        center_pos = (rect.centerx, rect.centery)
        pygame.gfxdraw.aacircle(self.screen, center_pos[0], center_pos[1], tower_def["range"], color)
        
    def _render_ui(self):
        # Top Left: Wave Info
        wave_text = self.font_large.render(f"WAVE {self.state['wave_number']}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        if self.state["game_phase"] == "WAVE":
            enemies_text = self.font_small.render(f"ENEMIES: {len(self.state['enemies']) + len(self.state['enemies_to_spawn'])}", True, self.COLOR_TEXT)
            self.screen.blit(enemies_text, (10, 35))

        # Top Right: Base Health
        health_pct = self.state["base_health"] / self.state["max_base_health"]
        health_bar_w = 150
        health_bar_rect_bg = pygame.Rect(self.WIDTH - health_bar_w - 10, 15, health_bar_w, 20)
        health_bar_rect_fg = pygame.Rect(self.WIDTH - health_bar_w - 10, 15, int(health_bar_w * health_pct), 20)
        pygame.draw.rect(self.screen, (80,0,0), health_bar_rect_bg)
        pygame.draw.rect(self.screen, self.COLOR_BASE, health_bar_rect_fg)
        health_text = self.font_small.render("BASE HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (health_bar_rect_bg.x, health_bar_rect_bg.y - 14))

        # Bottom Left: Resources
        resource_text = self.font_large.render(f"$ {self.state['resources']}", True, self.TOWER_COLORS[1])
        self.screen.blit(resource_text, (10, self.HEIGHT - 35))
        
        # Bottom Center: Tower Selection
        total_w = len(self.tower_definitions) * 60
        start_x = self.WIDTH // 2 - total_w // 2
        for i, tower_def in enumerate(self.tower_definitions):
            is_selected = i == self.state["selected_tower"]
            can_afford = self.state["resources"] >= tower_def["cost"]
            
            box_rect = pygame.Rect(start_x + i * 60, self.HEIGHT - 55, 50, 50)
            
            # Border for selection
            if is_selected:
                pygame.draw.rect(self.screen, (255, 255, 255), box_rect.inflate(4, 4), 2)
            
            # Background
            bg_color = self.COLOR_GRID if can_afford else (50, 20, 20)
            pygame.draw.rect(self.screen, bg_color, box_rect)
            
            # Tower name and cost
            name_color = self.COLOR_TEXT if can_afford else (150, 150, 150)
            name_surf = self.font_small.render(self.TOWER_NAMES[i], True, name_color)
            self.screen.blit(name_surf, (box_rect.centerx - name_surf.get_width() // 2, box_rect.y + 5))
            cost_surf = self.font_small.render(f"${tower_def['cost']}", True, name_color)
            self.screen.blit(cost_surf, (box_rect.centerx - cost_surf.get_width() // 2, box_rect.y + 25))

        # Bottom Right: Start Wave Button
        if self.state["game_phase"] == "PLACEMENT":
            button_rect = pygame.Rect(self.WIDTH - 140, self.HEIGHT - 35, 130, 25)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, button_rect)
            start_text = self.font_large.render("START WAVE", True, self.COLOR_TEXT)
            self.screen.blit(start_text, (button_rect.centerx - start_text.get_width() // 2, button_rect.centery - start_text.get_height() // 2))

    def _render_end_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.state["game_phase"] == "VICTORY":
            end_text = self.font_huge.render("VICTORY", True, self.COLOR_BASE)
        else:
            end_text = self.font_huge.render("GAME OVER", True, self.COLOR_ENEMY)

        self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2 - 20))
        
        score_text = self.font_large.render(f"Final Score: {int(self.state['score'])}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, self.HEIGHT // 2 + 30))

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
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Manual Play Example ---
    # This requires a window to be created.
    # Replace env.screen with a display surface.
    pygame.display.set_caption("Tower Defense")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    env.screen = display_screen
    
    # Game loop for manual play
    while not terminated:
        # Action mapping from keyboard to MultiDiscrete
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # The environment returns the rendered frame in `obs`, 
        # but for manual play we've already drawn to the screen.
        # So we just need to update the display.
        pygame.display.flip()
        
        # We are manually controlling the clock here, which is what
        # an external runner would do.
        env.clock.tick(env.FPS)

    env.close()