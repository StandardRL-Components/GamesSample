
# Generated: 2025-08-28T02:14:23.352756
# Source Brief: brief_04382.md
# Brief Index: 4382

        
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
        "Controls: Arrow keys to move cursor, Space to place a tower, Shift to cycle tower types."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing defensive towers in an isometric 2D environment."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 12, 12
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 24, 12
    ORIGIN_X, ORIGIN_Y = SCREEN_WIDTH // 2, 60

    MAX_STEPS = 3000  # Approx 100 seconds at 30fps
    MAX_WAVES = 20

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (45, 50, 62)
    COLOR_PATH = (68, 76, 92)
    COLOR_BASE = (68, 166, 116)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_ENEMY = (217, 74, 74)
    COLOR_CURSOR_VALID = (200, 200, 255, 100)
    COLOR_CURSOR_INVALID = (255, 100, 100, 100)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    TOWER_SPECS = {
        0: {"name": "Gatling", "cost": 50, "range": 80, "damage": 5, "fire_rate": 5, "color": (80, 144, 255), "proj_speed": 8},
        1: {"name": "Cannon", "cost": 100, "range": 120, "damage": 25, "fire_rate": 20, "color": (255, 165, 0), "proj_speed": 6},
        2: {"name": "Slower", "cost": 75, "range": 60, "damage": 0, "fire_rate": 10, "color": (137, 207, 240), "proj_speed": 10, "slow_factor": 0.5, "slow_duration": 60},
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
        self.font_s = pygame.font.Font(None, 20)
        self.font_m = pygame.font.Font(None, 28)
        self.font_l = pygame.font.Font(None, 48)
        
        self.path_waypoints = [
            (-1, 5), (2, 5), (2, 2), (5, 2), (5, 8), (9, 8), (9, 4), (12, 4)
        ]
        self.path_pixels = []
        for i in range(len(self.path_waypoints) - 1):
            p1 = self._grid_to_world(self.path_waypoints[i][0], self.path_waypoints[i][1])
            p2 = self._grid_to_world(self.path_waypoints[i+1][0], self.path_waypoints[i+1][1])
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            for j in range(int(dist)):
                x = p1[0] + (p2[0] - p1[0]) * j / dist
                y = p1[1] + (p2[1] - p1[1]) * j / dist
                self.path_pixels.append((x, y))

        self.tower_placement_zones = set()
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                on_path = False
                tile_center = self._grid_to_world(x,y)
                for p in self.path_pixels:
                    if math.hypot(tile_center[0]-p[0], tile_center[1]-p[1]) < self.TILE_WIDTH_HALF:
                        on_path = True
                        break
                if not on_path:
                    self.tower_placement_zones.add((x, y))

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_health = 100
        self.money = 150
        
        self.current_wave = 0
        self.wave_cooldown = 0
        self.enemies_to_spawn = []

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.cursor_move_timer = 0
        self.selected_tower_type = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.tower_grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        self._start_next_wave()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.001 # Small penalty for time passing
        
        self._handle_actions(action)
        
        # --- Game Logic Update ---
        self._update_wave_spawner()
        reward += self._update_towers()
        self._update_projectiles()
        spawned_reward, health_penalty = self._update_enemies()
        reward += spawned_reward + health_penalty
        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.win:
                reward += 100
            else: # Loss
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        if self.cursor_move_timer > 0:
            self.cursor_move_timer -= 1
        
        if movement != 0 and self.cursor_move_timer == 0:
            if movement == 1: self.cursor_pos[1] -= 1
            elif movement == 2: self.cursor_pos[1] += 1
            elif movement == 3: self.cursor_pos[0] -= 1
            elif movement == 4: self.cursor_pos[0] += 1
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)
            self.cursor_move_timer = 4 # Cooldown for cursor movement

        # --- Cycle Tower ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: UI_Bleep

        # --- Place Tower ---
        if space_held and not self.prev_space_held:
            x, y = self.cursor_pos
            spec = self.TOWER_SPECS[self.selected_tower_type]
            can_place = (x, y) in self.tower_placement_zones and self.tower_grid[x, y] == 0
            if can_place and self.money >= spec["cost"]:
                self.money -= spec["cost"]
                self.tower_grid[x, y] = 1
                self.towers.append({
                    "pos": (x, y), "type": self.selected_tower_type, "cooldown": 0,
                    "target": None
                })
                # sfx: Tower_Place

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_wave_spawner(self):
        if not self.enemies and not self.enemies_to_spawn and self.current_wave <= self.MAX_WAVES:
            if self.wave_cooldown == 0:
                self.wave_cooldown = 150 # 5 seconds
            else:
                self.wave_cooldown -= 1
                if self.wave_cooldown == 0:
                    self._start_next_wave()
        
        if self.enemies_to_spawn and self.wave_cooldown == 0:
            self.enemies.append(self.enemies_to_spawn.pop(0))
            self.wave_cooldown = 15 # spawn interval

    def _update_towers(self):
        wave_completion_reward = 0
        if self.current_wave > 0 and not self.enemies and not self.enemies_to_spawn and self.wave_cooldown > 1:
            if not hasattr(self, 'wave_reward_given') or self.wave_reward_given < self.current_wave:
                 wave_completion_reward = 1.0 * self.current_wave
                 self.wave_reward_given = self.current_wave
                 # sfx: Wave_Complete

        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            if tower["cooldown"] > 0:
                tower["cooldown"] -= 1
                continue

            # Find new target if current is invalid
            if tower.get("target") not in self.enemies:
                tower["target"] = None
                closest_enemy = None
                min_dist = spec["range"] ** 2
                tower_pos_world = self._grid_to_world(*tower["pos"])
                for enemy in self.enemies:
                    dist_sq = (enemy["pos"][0] - tower_pos_world[0])**2 + (enemy["pos"][1] - tower_pos_world[1])**2
                    if dist_sq < min_dist:
                        min_dist = dist_sq
                        closest_enemy = enemy
                if closest_enemy:
                    tower["target"] = closest_enemy
            
            # Fire projectile
            if tower["target"] and tower["cooldown"] == 0:
                tower["cooldown"] = spec["fire_rate"]
                self.projectiles.append({
                    "start_pos": self._grid_to_world(*tower["pos"]),
                    "pos": self._grid_to_world(*tower["pos"]),
                    "target": tower["target"],
                    "spec": spec
                })
                # sfx: Pew or Thump

        return wave_completion_reward

    def _update_projectiles(self):
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
                if "slow_factor" in proj["spec"]:
                    proj["target"]["slow_timer"] = proj["spec"]["slow_duration"]
                self.projectiles.remove(proj)
                self._create_particles(target_pos, proj["spec"]["color"], 5)
                # sfx: Hit_Impact
            else:
                # Move
                proj["pos"] = (proj_pos[0] + dx/dist * proj["spec"]["proj_speed"], 
                               proj_pos[1] + dy/dist * proj["spec"]["proj_speed"])

    def _update_enemies(self):
        reward = 0
        health_penalty = 0
        for enemy in self.enemies[:]:
            # --- Handle death ---
            if enemy["health"] <= 0:
                self.money += enemy["value"]
                reward += 0.5 # Reward for kill
                self.enemies.remove(enemy)
                self._create_particles(enemy["pos"], self.COLOR_ENEMY, 15)
                # sfx: Enemy_Explode
                continue
            
            # --- Handle movement ---
            if enemy["path_idx"] >= len(self.path_pixels):
                # Reached base
                self.base_health -= enemy["damage"]
                health_penalty -= 10
                self.enemies.remove(enemy)
                self._create_particles((self.SCREEN_WIDTH - 50, self.SCREEN_HEIGHT - 50), self.COLOR_BASE_DMG, 20)
                # sfx: Base_Damage
                continue

            speed = enemy["speed"]
            if enemy["slow_timer"] > 0:
                enemy["slow_timer"] -= 1
                speed *= self.TOWER_SPECS[2]["slow_factor"]

            enemy["path_idx"] = min(len(self.path_pixels) - 1, enemy["path_idx"] + speed)
            enemy["pos"] = self.path_pixels[int(enemy["path_idx"])]
        
        return reward, health_penalty

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] = (p["pos"][0] + p["vel"][0], p["pos"][1] + p["vel"][1])
            p["vel"] = (p["vel"][0] * 0.95, p["vel"][1] * 0.95)
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            return

        num_enemies = 3 + self.current_wave * 2
        health_multiplier = 1 + (self.current_wave - 1) * 0.1
        speed_multiplier = 1 + (self.current_wave - 1) * 0.05
        
        self.enemies_to_spawn = []
        for _ in range(num_enemies):
            self.enemies_to_spawn.append({
                "pos": self.path_pixels[0],
                "path_idx": 0,
                "health": 50 * health_multiplier,
                "max_health": 50 * health_multiplier,
                "speed": 1.0 * speed_multiplier,
                "damage": 10,
                "value": 10,
                "slow_timer": 0
            })

    def _check_termination(self):
        if self.game_over:
            return True
        
        if self.base_health <= 0:
            self.game_over = True
            self.win = False
            return True
        
        if self.current_wave > self.MAX_WAVES and not self.enemies and not self.enemies_to_spawn:
            self.game_over = True
            self.win = True
            return True
            
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            return True
            
        return False
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid and path
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                is_path_tile = False
                tile_center = self._grid_to_world(x,y)
                for p in self.path_pixels:
                    if math.hypot(tile_center[0]-p[0], tile_center[1]-p[1]) < self.TILE_WIDTH_HALF:
                        is_path_tile = True
                        break
                
                points = self._get_tile_points(x, y)
                color = self.COLOR_PATH if is_path_tile else self.COLOR_GRID
                pygame.gfxdraw.aapolygon(self.screen, points, color)

        # Draw base
        base_points = [self._grid_to_world(11, 3), self._grid_to_world(12, 3), self._grid_to_world(12, 4), self._grid_to_world(11, 4)]
        pygame.gfxdraw.filled_polygon(self.screen, base_points, self.COLOR_BASE)
        pygame.gfxdraw.aapolygon(self.screen, base_points, tuple(c*1.2 for c in self.COLOR_BASE[:3]))

        # Draw towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            center = self._grid_to_world(*tower["pos"])
            # Base
            points = self._get_tile_points(*tower["pos"])
            pygame.gfxdraw.filled_polygon(self.screen, points, (40,40,55))
            # Turret
            pygame.gfxdraw.filled_circle(self.screen, int(center[0]), int(center[1] - 5), 6, spec["color"])
            pygame.gfxdraw.aacircle(self.screen, int(center[0]), int(center[1] - 5), 6, tuple(c*0.8 for c in spec["color"][:3]))

        # Draw projectiles
        for proj in self.projectiles:
            pygame.draw.aaline(self.screen, proj["spec"]["color"], proj["pos"], proj["target"]["pos"], 2)

        # Draw enemies
        for enemy in self.enemies:
            x, y = int(enemy["pos"][0]), int(enemy["pos"][1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, 7, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, x, y, 7, (255, 120, 120))
            # Health bar
            health_ratio = enemy["health"] / enemy["max_health"]
            pygame.draw.rect(self.screen, (255,0,0), (x - 8, y - 15, 16, 3))
            pygame.draw.rect(self.screen, (0,255,0), (x - 8, y - 15, 16 * health_ratio, 3))
            if enemy["slow_timer"] > 0:
                pygame.gfxdraw.aacircle(self.screen, x, y, 9, self.TOWER_SPECS[2]["color"])

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(p["life"] * p["alpha_decay"])))
            color = (*p["color"], alpha)
            radius = int(p["life"] * p["radius_decay"])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), radius, color)

        # Draw cursor
        x, y = self.cursor_pos
        points = self._get_tile_points(x, y)
        spec = self.TOWER_SPECS[self.selected_tower_type]
        can_place = (x, y) in self.tower_placement_zones and self.tower_grid[x, y] == 0 and self.money >= spec["cost"]
        cursor_color = self.COLOR_CURSOR_VALID if can_place else self.COLOR_CURSOR_INVALID
        pygame.gfxdraw.filled_polygon(self.screen, points, cursor_color)
        if can_place:
            center = self._grid_to_world(x,y)
            pygame.gfxdraw.aacircle(self.screen, int(center[0]), int(center[1]), spec["range"], (*spec["color"], 80))

    def _render_ui(self):
        # --- Top Left: Stats ---
        self._draw_text(f"HP: {self.base_health}", (10, 10), self.font_m, self.COLOR_BASE)
        self._draw_text(f"$ {self.money}", (10, 35), self.font_m, (255, 215, 0))
        self._draw_text(f"Wave: {self.current_wave}/{self.MAX_WAVES}", (10, 60), self.font_m)
        self._draw_text(f"Score: {self.score}", (10, 85), self.font_m)

        # --- Bottom Right: Tower Selection ---
        spec = self.TOWER_SPECS[self.selected_tower_type]
        ui_box = pygame.Rect(self.SCREEN_WIDTH - 160, self.SCREEN_HEIGHT - 90, 150, 80)
        pygame.draw.rect(self.screen, (20, 20, 30, 200), ui_box, border_radius=5)
        self._draw_text(spec["name"], (ui_box.x + 10, ui_box.y + 5), self.font_m, spec["color"])
        self._draw_text(f"Cost: {spec['cost']}", (ui_box.x + 10, ui_box.y + 30), self.font_s)
        self._draw_text(f"Dmg: {spec['damage']}", (ui_box.x + 10, ui_box.y + 45), self.font_s)
        self._draw_text(f"Rng: {spec['range']}", (ui_box.x + 80, ui_box.y + 30), self.font_s)
        self._draw_text(f"Rate: {spec['fire_rate']}", (ui_box.x + 80, ui_box.y + 45), self.font_s)
        self._draw_text("Shift to cycle", (ui_box.x + 10, ui_box.y + 62), self.font_s, (150,150,150))

        # --- Game Over Screen ---
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            message = "VICTORY!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            self._draw_text(message, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20), self.font_l, color, center=True)
            self._draw_text(f"Final Score: {self.score}", (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20), self.font_m, self.COLOR_TEXT, center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "money": self.money,
            "wave": self.current_wave,
            "enemies_left": len(self.enemies) + len(self.enemies_to_spawn)
        }

    # --- Helper Methods ---
    def _grid_to_world(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return screen_x, screen_y
    
    def _get_tile_points(self, x, y):
        cx, cy = self._grid_to_world(x, y)
        return [
            (cx, cy - self.TILE_HEIGHT_HALF),
            (cx + self.TILE_WIDTH_HALF, cy),
            (cx, cy + self.TILE_HEIGHT_HALF),
            (cx - self.TILE_WIDTH_HALF, cy)
        ]

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow=True, center=False):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        if shadow:
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surf, (text_rect.x + 1, text_rect.y + 1))
        
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "pos": pos,
                "vel": (math.cos(angle) * speed, math.sin(angle) * speed),
                "life": random.randint(15, 30),
                "color": color,
                "alpha_decay": 255 / 30,
                "radius_decay": 5 / 30
            })

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        else: action[0] = 0
            
        # Buttons
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.3f}, Score: {info['score']}, Money: {info['money']}, Wave: {info['wave']}")
        
        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()