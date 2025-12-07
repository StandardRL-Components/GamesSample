
# Generated: 2025-08-28T02:45:31.711813
# Source Brief: brief_01799.md
# Brief Index: 1799

        
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
        "Press Space to build the selected tower. Hold Shift to cycle tower types."
    )

    game_description = (
        "Defend your base from waves of invading enemies by strategically placing "
        "towers in this isometric tower defense game."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 15
    ISO_OFFSET_X, ISO_OFFSET_Y = SCREEN_WIDTH // 2, 80
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 24, 12

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (40, 45, 58)
    COLOR_PATH = (60, 68, 85)
    COLOR_BASE = (0, 150, 255)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_VALID_CURSOR = (50, 255, 50, 150)
    COLOR_INVALID_CURSOR = (255, 50, 50, 150)
    
    TOWER_SPECS = [
        {"name": "Gatling", "cost": 100, "range": 4, "damage": 5, "fire_rate": 5, "color": (0, 200, 200), "proj_speed": 15},
        {"name": "Cannon", "cost": 250, "range": 6, "damage": 40, "fire_rate": 30, "color": (255, 150, 0), "proj_speed": 10},
    ]

    MAX_WAVES = 10
    MAX_STEPS = 5000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)
        
        # Path definition (in grid coordinates)
        self.path_nodes = [
            (0, 7), (4, 7), (4, 3), (9, 3), (9, 11), (15, 11), (15, 7), (19, 7)
        ]
        self.path_cells = self._generate_path_cells()
        
        self.reset()
        self.validate_implementation()

    def _generate_path_cells(self):
        cells = set()
        for i in range(len(self.path_nodes) - 1):
            p1 = self.path_nodes[i]
            p2 = self.path_nodes[i+1]
            x1, y1 = p1
            x2, y2 = p2
            for x in range(min(x1, x2), max(x1, x2) + 1):
                cells.add((x, y1))
            for y in range(min(y1, y2), max(y1, y2) + 1):
                cells.add((x2, y))
        return cells

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_win = False
        
        self.base_health = 1000
        self.max_base_health = 1000
        self.resources = 200
        self.wave = 0
        self.wave_in_progress = False
        self.next_wave_timer = 240 # 8s at 30fps
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        
        self._space_pressed_last_frame = False
        self._shift_pressed_last_frame = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01 # Time penalty
        self.steps += 1
        
        # --- Handle Input ---
        if not self.game_over:
            self._handle_input(movement, space_action, shift_action)

        # --- Update Game State ---
        if not self.game_over:
            reward += self._update_wave_logic()
            self._update_towers()
            reward += self._update_projectiles()
            reward += self._update_enemies()
        self._update_particles()
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.wave_win:
                reward = 100.0
            else: # Lost base or max steps
                reward = -100.0

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_action, shift_action):
        # Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Cycle tower (on press, not hold)
        if shift_action and not self._shift_pressed_last_frame:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
        self._shift_pressed_last_frame = shift_action

        # Place tower (on press, not hold)
        if space_action and not self._space_pressed_last_frame:
            self._place_tower()
        self._space_pressed_last_frame = space_action

    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.resources < spec["cost"]:
            return # sfx: error

        cx, cy = self.cursor_pos
        if (cx, cy) in self.path_cells:
            return # sfx: error

        for tower in self.towers:
            if tower["pos"] == [cx, cy]:
                return # sfx: error

        self.resources -= spec["cost"]
        self.towers.append({
            "pos": [cx, cy],
            "type": self.selected_tower_type,
            "cooldown": 0,
        })
        # sfx: build_tower

    def _update_wave_logic(self):
        # Check for win condition
        if self.wave >= self.MAX_WAVES and not self.enemies and self.wave_in_progress:
            self.wave_win = True
            self.game_over = True
            return 0

        # Start next wave
        if not self.wave_in_progress:
            self.next_wave_timer -= 1
            if self.next_wave_timer <= 0 and self.wave < self.MAX_WAVES:
                self.wave += 1
                self.wave_in_progress = True
                self.enemies_to_spawn = 10 + self.wave * 2
                self.spawn_timer = 0
        
        # Spawn enemies
        if self.wave_in_progress and self.enemies_to_spawn > 0:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self._spawn_enemy()
                self.enemies_to_spawn -= 1
                self.spawn_timer = 15 # Spawn every 0.5s
        
        # Check if wave is over
        if self.wave_in_progress and self.enemies_to_spawn == 0 and not self.enemies:
            self.wave_in_progress = False
            self.next_wave_timer = 300 # 10s until next wave
        
        return 0

    def _spawn_enemy(self):
        health_multiplier = 1 + (self.wave - 1) * 0.1
        speed_multiplier = 1 + (self.wave - 1) * 0.05
        
        start_pos = list(self.path_nodes[0])
        self.enemies.append({
            "pos": [start_pos[0] - 0.1, start_pos[1]], # Start slightly off-screen
            "health": 50 * health_multiplier,
            "max_health": 50 * health_multiplier,
            "speed": 0.05 * speed_multiplier,
            "path_index": 0,
        })

    def _update_towers(self):
        for tower in self.towers:
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue
            
            spec = self.TOWER_SPECS[tower["type"]]
            
            # Find target: enemy closest to the base
            best_target = None
            max_dist_traveled = -1

            for enemy in self.enemies:
                dist_sq = (enemy["pos"][0] - tower["pos"][0])**2 + (enemy["pos"][1] - tower["pos"][1])**2
                if dist_sq <= spec["range"]**2:
                    # Calculate distance traveled along path
                    dist_traveled = self._get_enemy_path_distance(enemy)
                    if dist_traveled > max_dist_traveled:
                        max_dist_traveled = dist_traveled
                        best_target = enemy
            
            if best_target:
                tower["cooldown"] = spec["fire_rate"]
                self.projectiles.append({
                    "start_pos": list(tower["pos"]),
                    "pos": list(tower["pos"]),
                    "target": best_target,
                    "spec": spec
                })
                # sfx: fire_weapon

    def _get_enemy_path_distance(self, enemy):
        dist = 0
        for i in range(enemy["path_index"]):
            p1 = self.path_nodes[i]
            p2 = self.path_nodes[i+1]
            dist += abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        p_current = enemy["pos"]
        p_next_node = self.path_nodes[enemy["path_index"]]
        dist += abs(p_current[0] - p_next_node[0]) + abs(p_current[1] - p_next_node[1])
        return dist

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            target = proj["target"]
            spec = proj["spec"]
            
            # If target is gone, fizzle
            if target not in self.enemies:
                self.projectiles.remove(proj)
                continue

            direction = np.array(target["pos"]) - np.array(proj["pos"])
            dist = np.linalg.norm(direction)
            
            if dist < 0.5: # Hit
                target["health"] -= spec["damage"]
                reward += 0.1 # Reward for hitting
                self._create_particles(target["pos"], spec["color"], 5, 2)
                self.projectiles.remove(proj)
                # sfx: hit_enemy
                continue

            direction /= dist
            proj["pos"][0] += direction[0] * spec["proj_speed"] / 30.0
            proj["pos"][1] += direction[1] * spec["proj_speed"] / 30.0

        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            # Move
            if enemy["path_index"] >= len(self.path_nodes) - 1:
                # Reached base
                self.base_health -= enemy["health"]
                self.enemies.remove(enemy)
                self._create_particles(self._grid_to_iso(*self.path_nodes[-1]), self.COLOR_BASE_DMG, 20, 5)
                # sfx: base_damage
                continue

            target_node = self.path_nodes[enemy["path_index"] + 1]
            direction = np.array(target_node) - np.array(enemy["pos"])
            dist = np.linalg.norm(direction)

            if dist < enemy["speed"]:
                enemy["pos"] = list(target_node)
                enemy["path_index"] += 1
            else:
                move = direction / dist * enemy["speed"]
                enemy["pos"][0] += move[0]
                enemy["pos"][1] += move[1]
            
            # Check for death
            if enemy["health"] <= 0:
                reward += 1.0 # Reward for kill
                self.resources += 25
                self._create_particles(enemy["pos"], self.COLOR_ENEMY, 15, 4)
                self.enemies.remove(enemy)
                # sfx: enemy_explode
        
        self.base_health = max(0, self.base_health)
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.05 # Gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, grid_pos, color, count, speed_factor):
        screen_pos = self._grid_to_iso(*grid_pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_factor)
            self.particles.append({
                "pos": list(screen_pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed - 2],
                "life": self.np_random.integers(10, 20),
                "color": color
            })

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_iso(self, x, y):
        iso_x = (x - y) * self.TILE_WIDTH_HALF + self.ISO_OFFSET_X
        iso_y = (x + y) * self.TILE_HEIGHT_HALF + self.ISO_OFFSET_Y
        return int(iso_x), int(iso_y)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                p1 = self._grid_to_iso(x, y)
                p2 = self._grid_to_iso(x + 1, y)
                p3 = self._grid_to_iso(x + 1, y + 1)
                p4 = self._grid_to_iso(x, y + 1)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p4)

        # Draw path
        for x, y in self.path_cells:
            points = [
                self._grid_to_iso(x, y),
                self._grid_to_iso(x + 1, y),
                self._grid_to_iso(x + 1, y + 1),
                self._grid_to_iso(x, y + 1)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PATH)
        
        # Draw base
        bx, by = self.path_nodes[-1]
        base_points = [
            self._grid_to_iso(bx, by),
            self._grid_to_iso(bx + 1, by),
            self._grid_to_iso(bx + 1, by + 1),
            self._grid_to_iso(bx, by + 1)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, base_points, self.COLOR_BASE)
        
        # Combine and sort all drawable entities for correct Z-ordering
        draw_list = []
        for tower in self.towers:
            draw_list.append(("tower", tower))
        for enemy in self.enemies:
            draw_list.append(("enemy", enemy))
        
        draw_list.sort(key=lambda item: item[1]["pos"][0] + item[1]["pos"][1])

        for item_type, item in draw_list:
            if item_type == "tower":
                self._render_tower(item)
            elif item_type == "enemy":
                self._render_enemy(item)

        # Draw projectiles (on top)
        for proj in self.projectiles:
            spec = proj["spec"]
            p = self._grid_to_iso(*proj["pos"])
            pygame.gfxdraw.filled_circle(self.screen, p[0], p[1] - 10, 3, spec["color"])
            pygame.gfxdraw.aacircle(self.screen, p[0], p[1] - 10, 3, spec["color"])
        
        # Draw cursor
        self._render_cursor()

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), int(p["life"] / 5))

    def _render_tower(self, tower):
        spec = self.TOWER_SPECS[tower["type"]]
        x, y = tower["pos"]
        
        # Base
        base_points = [
            self._grid_to_iso(x, y),
            self._grid_to_iso(x + 1, y),
            self._grid_to_iso(x + 1, y + 1),
            self._grid_to_iso(x, y + 1)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, base_points, (30,30,30))
        
        # Tower structure
        p = self._grid_to_iso(x + 0.5, y + 0.5)
        if spec["name"] == "Gatling":
            pygame.draw.rect(self.screen, spec["color"], (p[0] - 5, p[1] - 25, 10, 20))
        elif spec["name"] == "Cannon":
            pygame.draw.circle(self.screen, spec["color"], (p[0], p[1] - 15), 8)


    def _render_enemy(self, enemy):
        p = self._grid_to_iso(*enemy["pos"])
        
        # Body
        size = 8
        pygame.gfxdraw.box(self.screen, (p[0] - size//2, p[1] - int(size*1.5), size, size), self.COLOR_ENEMY)

        # Health bar
        bar_w = 20
        bar_h = 4
        health_pct = enemy["health"] / enemy["max_health"]
        pygame.draw.rect(self.screen, (50,0,0), (p[0] - bar_w//2, p[1] - 30, bar_w, bar_h))
        pygame.draw.rect(self.screen, (0,200,0), (p[0] - bar_w//2, p[1] - 30, int(bar_w * health_pct), bar_h))

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        points = [
            self._grid_to_iso(cx, cy),
            self._grid_to_iso(cx + 1, cy),
            self._grid_to_iso(cx + 1, cy + 1),
            self._grid_to_iso(cx, cy + 1)
        ]
        
        spec = self.TOWER_SPECS[self.selected_tower_type]
        is_valid = self.resources >= spec["cost"] and (cx, cy) not in self.path_cells and all(t["pos"] != [cx, cy] for t in self.towers)
        color = self.COLOR_VALID_CURSOR if is_valid else self.COLOR_INVALID_CURSOR
        
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color[:3])

    def _render_text(self, text, font, pos, color=COLOR_TEXT):
        shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow, (pos[0] + 1, pos[1] + 1))
        content = font.render(text, True, color)
        self.screen.blit(content, pos)

    def _render_ui(self):
        # Base Health
        health_pct = self.base_health / self.max_base_health
        health_color = (int(255 * (1-health_pct)), int(255 * health_pct), 0)
        pygame.draw.rect(self.screen, (50,50,50), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, health_color, (10, 10, int(200 * health_pct), 20))
        self._render_text(f"Base: {int(self.base_health)}", self.font_s, (15, 12))

        # Resources
        self._render_text(f"$: {self.resources}", self.font_m, (220, 8))

        # Wave Info
        self._render_text(f"Wave: {self.wave}/{self.MAX_WAVES}", self.font_m, (self.SCREEN_WIDTH - 150, 8))
        if not self.wave_in_progress and self.wave < self.MAX_WAVES and not self.game_over:
            secs = self.next_wave_timer // 30
            self._render_text(f"Next wave in: {secs}", self.font_s, (self.SCREEN_WIDTH - 150, 38))
        
        # Selected Tower
        spec = self.TOWER_SPECS[self.selected_tower_type]
        cost_color = self.COLOR_TEXT if self.resources >= spec["cost"] else (255,100,100)
        self._render_text(f"Selected: {spec['name']}", self.font_s, (10, self.SCREEN_HEIGHT - 30))
        self._render_text(f"Cost: {spec['cost']}", self.font_s, (160, self.SCREEN_HEIGHT - 30), cost_color)
        
        # Game Over / Win message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            if self.wave_win:
                self._render_text("VICTORY", self.font_l, (self.SCREEN_WIDTH//2 - 100, self.SCREEN_HEIGHT//2 - 50))
                self._render_text("All waves defeated!", self.font_m, (self.SCREEN_WIDTH//2 - 120, self.SCREEN_HEIGHT//2 + 10))
            else:
                self._render_text("GAME OVER", self.font_l, (self.SCREEN_WIDTH//2 - 140, self.SCREEN_HEIGHT//2 - 50))
                if self.base_health <= 0:
                    msg = "Your base has been destroyed."
                else:
                    msg = "You ran out of time."
                self._render_text(msg, self.font_m, (self.SCREEN_WIDTH//2 - 150, self.SCREEN_HEIGHT//2 + 10))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "resources": self.resources,
            "base_health": self.base_health,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # To run headlessly, we don't create a display. For manual play, we do.
    pygame.display.set_caption("Isometric Tower Defense")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    while not done:
        # --- Action gathering from human input ---
        movement = 0
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render to the display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Control FPS ---
        env.clock.tick(30)

    env.close()